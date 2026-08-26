#include <torch/extension.h>

#include "feature_ragged_allgather_common.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

#define BASIS_CUDA_CHECK(expr)                                                                    \
  do {                                                                                            \
    const cudaError_t _status = (expr);                                                           \
    TORCH_CHECK(_status == cudaSuccess, #expr, " failed: ", cudaGetErrorString(_status));         \
  } while (0)

#define BASIS_NCCL_CHECK(expr)                                                                    \
  do {                                                                                            \
    const ncclResult_t _status = (expr);                                                          \
    TORCH_CHECK(_status == ncclSuccess, #expr, " failed: ", ncclGetErrorString(_status));         \
  } while (0)

#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22900
#define BASISSERVE_NCCL_RMA_COMPILED 1
#else
#define BASISSERVE_NCCL_RMA_COMPILED 0
#endif

constexpr int64_t kRmaSlotCount = 2;

struct RaggedPlan {
  std::vector<int64_t> widths;
  std::vector<int64_t> offsets;
  int64_t total_width = 0;
};

RaggedPlan make_plan(const std::vector<int64_t>& widths, int world_size) {
  TORCH_CHECK(
      static_cast<int>(widths.size()) == world_size,
      "widths must have one entry per rank; got ",
      widths.size(),
      " entries for world_size=",
      world_size);

  RaggedPlan plan;
  plan.widths = widths;
  plan.offsets.resize(widths.size() + 1, 0);
  for (size_t i = 0; i < widths.size(); ++i) {
    TORCH_CHECK(widths[i] > 0, "all source widths must be positive; widths[", i, "]=", widths[i]);
    TORCH_CHECK(
        plan.offsets[i] <= std::numeric_limits<int64_t>::max() - widths[i],
        "width sum overflow");
    plan.offsets[i + 1] = plan.offsets[i] + widths[i];
  }
  plan.total_width = plan.offsets.back();
  return plan;
}

size_t checked_bytes(int64_t rows, int64_t cols, size_t element_size) {
  TORCH_CHECK(rows > 0 && cols > 0, "workspace dimensions must be positive");
  const uint64_t n = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
  TORCH_CHECK(
      n <= std::numeric_limits<size_t>::max() / element_size,
      "workspace byte count overflows size_t");
  return static_cast<size_t>(n) * element_size;
}

size_t checked_offset_bytes(int64_t rows, int64_t cols, size_t element_size) {
  TORCH_CHECK(rows >= 0 && cols > 0, "offset dimensions must be nonnegative/positive");
  const uint64_t n = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
  TORCH_CHECK(
      n <= std::numeric_limits<size_t>::max() / element_size,
      "byte offset overflows size_t");
  return static_cast<size_t>(n) * element_size;
}

ncclDataType_t to_nccl_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
      return ncclFloat16;
    case at::kFloat:
      return ncclFloat32;
    case at::kDouble:
      return ncclFloat64;
    case at::kInt:
      return ncclInt32;
    case at::kLong:
      return ncclInt64;
    case at::kChar:
      return ncclInt8;
    case at::kByte:
      return ncclUint8;
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 21000
    case at::kBFloat16:
      return ncclBfloat16;
#endif
    default:
      TORCH_CHECK(false, "unsupported NCCL tensor dtype: ", dtype);
  }
  return ncclFloat32;
}

at::ScalarType dtype_from_code(int64_t code) {
  switch (code) {
    case 0:
      return at::kHalf;
    case 1:
      return at::kBFloat16;
    case 2:
      return at::kFloat;
    default:
      TORCH_CHECK(false, "dtype_code must be 0 (fp16), 1 (bf16), or 2 (fp32); got ", code);
  }
  return at::kFloat;
}

void validate_cuda_tensor(const at::Tensor& tensor, int device, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.get_device() == device, name, " must be on CUDA device ", device);
  TORCH_CHECK(tensor.layout() == c10::kStrided, name, " must be a strided tensor");
}

std::pair<at::Tensor, int64_t> as_feature_major(
    const at::Tensor& local,
    int64_t expected_width,
    bool local_is_feature_major,
    int device) {
  validate_cuda_tensor(local, device, "local");
  TORCH_CHECK(local.dim() == 2, "local must be rank-2");
  TORCH_CHECK(!local.requires_grad(), "feature ragged transport is inference-only");

  if (local_is_feature_major) {
    TORCH_CHECK(
        local.size(0) == expected_width,
        "feature-major local must have shape [local_width, batch]; expected first dimension ",
        expected_width,
        ", got ",
        local.sizes());
    return {local.contiguous(), local.size(1)};
  }

  TORCH_CHECK(
      local.size(1) == expected_width,
      "token-major local must have shape [batch, local_width]; expected second dimension ",
      expected_width,
      ", got ",
      local.sizes());
  return {local.transpose(0, 1).contiguous(), local.size(0)};
}

class FeatureRaggedCommunicator {
 public:
  FeatureRaggedCommunicator(
      const py::bytes& unique_id_bytes,
      int rank,
      int world_size,
      int device)
      : rank_(rank), world_size_(world_size), device_(device) {
    TORCH_CHECK(world_size_ > 0, "world_size must be positive");
    TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "invalid rank ", rank_, " for world_size ", world_size_);
    int device_count = 0;
    BASIS_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    TORCH_CHECK(device_ >= 0 && device_ < device_count, "invalid CUDA device ", device_);

    std::string raw = unique_id_bytes;
    TORCH_CHECK(
        raw.size() == sizeof(ncclUniqueId),
        "invalid NCCL unique-id byte length: expected ",
        sizeof(ncclUniqueId),
        ", got ",
        raw.size());

    ncclUniqueId unique_id;
    std::memcpy(&unique_id, raw.data(), sizeof(unique_id));
    c10::cuda::CUDAGuard guard(device_);
    BASIS_NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, unique_id, rank_));
  }

  ~FeatureRaggedCommunicator() {
    // close() is the clean, collective shutdown path.  Destructors across Python
    // processes are not ordered, so never perform collective window deregistration
    // here.  Abort releases communicator resources without waiting for peers.
    if (comm_ != nullptr) {
      (void)ncclCommAbort(comm_);
      comm_ = nullptr;
    }
#if BASISSERVE_NCCL_RMA_COMPILED
    rma_window_ = nullptr;
    rma_arena_ = at::Tensor();
#endif
  }

  FeatureRaggedCommunicator(const FeatureRaggedCommunicator&) = delete;
  FeatureRaggedCommunicator& operator=(const FeatureRaggedCommunicator&) = delete;

  at::Tensor gather_feature_direct(
      const at::Tensor& local,
      const std::vector<int64_t>& widths,
      bool local_is_feature_major,
      const std::optional<at::Tensor>& workspace) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_open();
    c10::cuda::CUDAGuard guard(device_);

    const RaggedPlan plan = make_plan(widths, world_size_);
    validate_cuda_tensor(local, device_, "local");
    TORCH_CHECK(local.dim() == 2, "local must be rank-2");
    TORCH_CHECK(local.is_contiguous(), "local coordinates must be contiguous");
    TORCH_CHECK(!local.requires_grad(), "feature ragged transport is inference-only");
    const int64_t batch = local_is_feature_major ? local.size(1) : local.size(0);
    const int64_t observed_width =
        local_is_feature_major ? local.size(0) : local.size(1);
    TORCH_CHECK(
        observed_width == plan.widths[rank_],
        "local coordinate width ",
        observed_width,
        " differs from plan width ",
        plan.widths[rank_]);
    const auto dtype = local.scalar_type();
    const ncclDataType_t nccl_dtype = to_nccl_dtype(dtype);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_).stream();

    at::Tensor arena;
    if (workspace.has_value()) {
      arena = *workspace;
      validate_cuda_tensor(arena, device_, "feature-direct workspace");
      TORCH_CHECK(arena.is_contiguous(), "feature-direct workspace must be contiguous");
      TORCH_CHECK(
          arena.scalar_type() == dtype,
          "feature-direct workspace and local dtypes must match");
      TORCH_CHECK(
          arena.dim() == 2 && arena.size(0) == plan.total_width && arena.size(1) == batch,
          "feature-direct workspace must have shape [",
          plan.total_width,
          ", ",
          batch,
          "]; got ",
          arena.sizes());
      TORCH_CHECK(!arena.requires_grad(), "feature-direct workspace must not require gradients");
    } else {
      arena = at::empty({plan.total_width, batch}, local.options());
    }
    auto own = arena.narrow(0, plan.offsets[rank_], plan.widths[rank_]);
    if (local_is_feature_major) {
      if (local.data_ptr() != own.data_ptr()) {
        own.copy_(local, /*non_blocking=*/true);
      }
    } else {
      basisserve::feature_ag::launch_token_to_feature_pack(local, own, stream);
    }

    if (world_size_ > 1) {
      const bool uniform_width = std::all_of(
          plan.widths.begin() + 1,
          plan.widths.end(),
          [&](int64_t width) { return width == plan.widths.front(); });
      if (uniform_width) {
        // NCCL explicitly supports in-place AllGather when sendbuff points at
        // rank * sendcount inside recvbuff. Each source is already feature-major,
        // so its native rank concatenation is exactly [sum(widths), batch].
        const size_t count =
            static_cast<size_t>(plan.widths.front()) * static_cast<size_t>(batch);
        BASIS_NCCL_CHECK(ncclAllGather(
            own.data_ptr(),
            arena.data_ptr(),
            count,
            nccl_dtype,
            comm_,
            stream));
      } else {
        // Exact-width ring AllGather. Step one sends the local source to the
        // right neighbor; later steps forward the source received previously.
        // This avoids the all-peer broadcast bottleneck while retaining a
        // compact arena for genuinely ragged source widths.
        const int right = (rank_ + 1) % world_size_;
        const int left = (rank_ - 1 + world_size_) % world_size_;
        for (int step = 1; step < world_size_; ++step) {
          const int send_source = (rank_ - step + 1 + world_size_) % world_size_;
          const int recv_source = (rank_ - step + world_size_) % world_size_;
          void* send_ptr = static_cast<char*>(arena.data_ptr()) +
              checked_offset_bytes(
                  plan.offsets[send_source], batch, local.element_size());
          void* recv_ptr = static_cast<char*>(arena.data_ptr()) +
              checked_offset_bytes(
                  plan.offsets[recv_source], batch, local.element_size());
          const size_t send_count =
              static_cast<size_t>(plan.widths[send_source]) *
              static_cast<size_t>(batch);
          const size_t recv_count =
              static_cast<size_t>(plan.widths[recv_source]) *
              static_cast<size_t>(batch);
          BASIS_NCCL_CHECK(ncclGroupStart());
          const ncclResult_t recv_status = ncclRecv(
              recv_ptr, recv_count, nccl_dtype, left, comm_, stream);
          const ncclResult_t send_status = ncclSend(
              send_ptr, send_count, nccl_dtype, right, comm_, stream);
          const ncclResult_t group_status = ncclGroupEnd();
          TORCH_CHECK(
              recv_status == ncclSuccess,
              "NCCL ragged-ring receive failed: ",
              ncclGetErrorString(recv_status));
          TORCH_CHECK(
              send_status == ncclSuccess,
              "NCCL ragged-ring send failed: ",
              ncclGetErrorString(send_status));
          TORCH_CHECK(
              group_status == ncclSuccess,
              "NCCL ragged-ring group failed: ",
              ncclGetErrorString(group_status));
        }
      }
    }

    // arena is already the exact [sum(widths), batch] feature-major matrix.
    // No rank-major -> token-major pack follows this operation.
    return arena;
  }

  bool rma_compiled() const {
#if BASISSERVE_NCCL_RMA_COMPILED
    return true;
#else
    return false;
#endif
  }

  bool rma_runtime_version_ok() const {
    int version = 0;
    if (ncclGetVersion(&version) != ncclSuccess) {
      return false;
    }
    return version >= 22900;
  }

  int nccl_version() const {
    int version = 0;
    BASIS_NCCL_CHECK(ncclGetVersion(&version));
    return version;
  }

  void prepare_rma(int64_t batch, int64_t max_total_width, int64_t dtype_code) {
#if BASISSERVE_NCCL_RMA_COMPILED
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_open();
    TORCH_CHECK(batch > 0, "batch must be positive");
    TORCH_CHECK(max_total_width > 0, "max_total_width must be positive");
    TORCH_CHECK(
        rma_runtime_version_ok(),
        "feature_rma requires NCCL >= 2.29 at runtime; detected version ",
        nccl_version());

    c10::cuda::CUDAGuard guard(device_);
    const at::ScalarType dtype = dtype_from_code(dtype_code);
    (void)to_nccl_dtype(dtype);  // validate before entering collective registration
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_).stream();

    if (rma_prepared_ && batch == rma_batch_ && max_total_width == rma_max_width_ && dtype == rma_dtype_ &&
        stream == rma_stream_) {
      return;
    }

    // Reconfiguration is a cold-path collective.  Every rank in this communicator
    // must call prepare_rma with the same shape/dtype and in the same order.
    release_rma_locked();

    TORCH_CHECK(
        max_total_width <= std::numeric_limits<int64_t>::max() / kRmaSlotCount,
        "RMA workspace row count overflows int64_t");
    const int64_t arena_rows = kRmaSlotCount * max_total_width;
    const size_t bytes = checked_bytes(arena_rows, batch, c10::elementSize(dtype));
    void* allocation = nullptr;
    BASIS_NCCL_CHECK(ncclMemAlloc(&allocation, bytes));
    const auto options = at::TensorOptions().device(at::kCUDA, device_).dtype(dtype);
    const int allocation_device = device_;
    rma_arena_ = at::from_blob(
        allocation,
        {kRmaSlotCount, max_total_width, batch},
        [allocation_device](void* pointer) {
          if (pointer == nullptr) {
            return;
          }
          int previous_device = -1;
          const cudaError_t get_status = cudaGetDevice(&previous_device);
          (void)cudaSetDevice(allocation_device);
          (void)ncclMemFree(pointer);
          if (get_status == cudaSuccess && previous_device >= 0) {
            (void)cudaSetDevice(previous_device);
          }
        },
        options);
    BASIS_CUDA_CHECK(cudaMemsetAsync(rma_arena_.mutable_data_ptr(), 0, bytes, stream));
    BASIS_CUDA_CHECK(cudaStreamSynchronize(stream));

    const ncclResult_t register_status = ncclCommWindowRegister(
        comm_,
        rma_arena_.mutable_data_ptr(),
        bytes,
        &rma_window_,
        NCCL_WIN_COLL_SYMMETRIC);
    if (register_status != ncclSuccess || rma_window_ == nullptr) {
      rma_window_ = nullptr;
      rma_arena_ = at::Tensor();
      TORCH_CHECK(
          false,
          "ncclCommWindowRegister failed: ",
          register_status == ncclSuccess ? "returned a null window" : ncclGetErrorString(register_status),
          ". The NCCL build, driver, or topology does not support one-sided RMA.");
    }

    rma_batch_ = batch;
    rma_max_width_ = max_total_width;
    rma_dtype_ = dtype;
    rma_stream_ = stream;
    rma_epoch_ = 0;
    rma_prepared_ = true;
#else
    (void)batch;
    (void)max_total_width;
    (void)dtype_code;
    TORCH_CHECK(
        false,
        "feature_rma was not compiled because the build-time NCCL headers are older than 2.29. ",
        "Use feature_direct or rebuild against NCCL 2.29+.");
#endif
  }

  at::Tensor rma_local_feature_view(const std::vector<int64_t>& widths) {
#if BASISSERVE_NCCL_RMA_COMPILED
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_open();
    ensure_rma_prepared();
    c10::cuda::CUDAGuard guard(device_);
    ensure_rma_stream();
    const RaggedPlan plan = make_plan(widths, world_size_);
    TORCH_CHECK(
        plan.total_width <= rma_max_width_,
        "plan total width ",
        plan.total_width,
        " exceeds prepared RMA width ",
        rma_max_width_);
    auto arena = rma_slot_tensor(next_rma_slot());
    return arena.narrow(0, plan.offsets[rank_], plan.widths[rank_]);
#else
    (void)widths;
    TORCH_CHECK(false, "feature_rma is not compiled");
    return at::Tensor();
#endif
  }

  at::Tensor gather_feature_rma(
      const at::Tensor& local,
      const std::vector<int64_t>& widths,
      bool local_is_feature_major) {
#if BASISSERVE_NCCL_RMA_COMPILED
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_open();
    ensure_rma_prepared();
    c10::cuda::CUDAGuard guard(device_);
    ensure_rma_stream();

    const RaggedPlan plan = make_plan(widths, world_size_);
    TORCH_CHECK(
        plan.total_width <= rma_max_width_,
        "plan total width ",
        plan.total_width,
        " exceeds prepared RMA width ",
        rma_max_width_);

    auto [local_fm, batch] = as_feature_major(local, plan.widths[rank_], local_is_feature_major, device_);
    TORCH_CHECK(
        batch == rma_batch_,
        "RMA workspace was prepared for batch ",
        rma_batch_,
        ", but local has batch ",
        batch,
        ". Re-run prepare_rma collectively outside the hot path.");
    TORCH_CHECK(
        local_fm.scalar_type() == rma_dtype_,
        "RMA workspace dtype mismatch: prepared ",
        rma_dtype_,
        ", local ",
        local_fm.scalar_type());

    const ncclDataType_t nccl_dtype = to_nccl_dtype(rma_dtype_);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_).stream();
    const int64_t slot = next_rma_slot();
    auto full_arena = rma_slot_tensor(slot);
    auto own = full_arena.narrow(0, plan.offsets[rank_], plan.widths[rank_]);

    // This copy disappears when the upstream attention kernel writes directly into
    // rma_local_feature_view().  It is source-local and never becomes a global pack.
    if (local_fm.data_ptr() != own.data_ptr()) {
      own.copy_(local_fm, /*non_blocking=*/true);
    }

    const size_t count = static_cast<size_t>(plan.widths[rank_]) * static_cast<size_t>(batch);
    const int64_t remote_row_offset = slot * rma_max_width_ + plan.offsets[rank_];
    const size_t remote_offset_bytes =
        checked_offset_bytes(remote_row_offset, batch, local_fm.element_size());

    // One-sided source push.  A low-width source can execute these puts as soon as
    // its local compact attention finishes; receivers do not have to post matching
    // ncclRecv calls.  The local source is itself inside the registered window, as
    // required by NCCL's host-RMA path.
    if (world_size_ > 1) {
      BASIS_NCCL_CHECK(ncclGroupStart());
      ncclResult_t first_error = ncclSuccess;
      for (int peer = 0; peer < world_size_; ++peer) {
        if (peer == rank_) {
          continue;
        }
        const ncclResult_t status = ncclPutSignal(
            own.data_ptr(),
            count,
            nccl_dtype,
            peer,
            rma_window_,
            remote_offset_bytes,
            /*sigIdx=*/0,
            /*ctx=*/0,
            /*flags=*/0u,
            comm_,
            stream);
        if (first_error == ncclSuccess && status != ncclSuccess) {
          first_error = status;
        }
      }
      const ncclResult_t group_status = ncclGroupEnd();
      TORCH_CHECK(
          first_error == ncclSuccess,
          "NCCL grouped PutSignal operation failed: ",
          ncclGetErrorString(first_error));
      TORCH_CHECK(
          group_status == ncclSuccess,
          "ncclGroupEnd for PutSignal failed: ",
          ncclGetErrorString(group_status));
    }

    if (world_size_ > 1) {
      std::vector<ncclWaitSignalDesc_t> descs;
      descs.reserve(static_cast<size_t>(world_size_ - 1));
      for (int peer = 0; peer < world_size_; ++peer) {
        if (peer == rank_) {
          continue;
        }
        ncclWaitSignalDesc_t desc{};
        desc.peer = peer;
        desc.sigIdx = 0;
        desc.ctx = 0;
        // opCnt is incremental, not an absolute epoch. NCCL internally advances
        // the expected sequence value, so opCnt=1 is correct on every reuse.
        desc.opCnt = 1;
        descs.push_back(desc);
      }
      BASIS_NCCL_CHECK(ncclWaitSignal(
          static_cast<int>(descs.size()), descs.data(), comm_, stream));
    }

    // A peer may enter the next layer while this rank is still decoding the
    // current arena. Alternating slots prevents that peer's next remote write
    // from racing this rank's GEMM without adding a post-decode barrier.
    ++rma_epoch_;
    return full_arena.narrow(0, 0, plan.total_width);
#else
    (void)local;
    (void)widths;
    (void)local_is_feature_major;
    TORCH_CHECK(false, "feature_rma is not compiled; use feature_direct or NCCL 2.29+");
    return at::Tensor();
#endif
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (comm_ == nullptr) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_);
#if BASISSERVE_NCCL_RMA_COMPILED
    release_rma_locked();
#endif
    BASIS_NCCL_CHECK(ncclCommDestroy(comm_));
    comm_ = nullptr;
  }

  int rank() const {
    return rank_;
  }

  int world_size() const {
    return world_size_;
  }

  int device() const {
    return device_;
  }

 private:
  void ensure_open() const {
    TORCH_CHECK(comm_ != nullptr, "FeatureRaggedCommunicator is closed");
  }

#if BASISSERVE_NCCL_RMA_COMPILED
  void ensure_rma_prepared() const {
    TORCH_CHECK(rma_prepared_, "prepare_rma must be called collectively before feature_rma");
  }

  void ensure_rma_stream() const {
    const cudaStream_t current = c10::cuda::getCurrentCUDAStream(device_).stream();
    TORCH_CHECK(
        current == rma_stream_,
        "feature_rma must run on the CUDA stream used by prepare_rma. ",
        "Use one communicator/context per concurrently active stream.");
  }

  at::Tensor rma_arena_tensor() const {
    TORCH_CHECK(rma_arena_.defined(), "RMA arena is null");
    return rma_arena_;
  }

  int64_t next_rma_slot() const {
    return static_cast<int64_t>(rma_epoch_ % static_cast<uint64_t>(kRmaSlotCount));
  }

  at::Tensor rma_slot_tensor(int64_t slot) const {
    TORCH_CHECK(slot >= 0 && slot < kRmaSlotCount, "invalid RMA slot ", slot);
    return rma_arena_tensor().select(0, slot);
  }

  void release_rma_locked() {
    if (!rma_prepared_ && !rma_arena_.defined()) {
      return;
    }
    const cudaError_t sync_status = cudaDeviceSynchronize();
    TORCH_CHECK(sync_status == cudaSuccess, "cudaDeviceSynchronize failed: ", cudaGetErrorString(sync_status));

    ncclResult_t deregister_status = ncclSuccess;
    if (rma_window_ != nullptr && comm_ != nullptr) {
      deregister_status = ncclCommWindowDeregister(comm_, rma_window_);
      rma_window_ = nullptr;
    }
    // Drop the communicator's reference after deregistration. Views returned to
    // Python keep the owning Storage alive, so close/reconfigure cannot leave a
    // dangling CUDA Tensor. The deleter calls ncclMemFree at the final release.
    rma_arena_ = at::Tensor();
    rma_prepared_ = false;
    rma_batch_ = 0;
    rma_max_width_ = 0;
    rma_stream_ = nullptr;
    rma_epoch_ = 0;
    TORCH_CHECK(
        deregister_status == ncclSuccess,
        "ncclCommWindowDeregister failed: ",
        ncclGetErrorString(deregister_status));
  }
#endif

  int rank_ = -1;
  int world_size_ = 0;
  int device_ = -1;
  ncclComm_t comm_ = nullptr;
  mutable std::mutex mutex_;

#if BASISSERVE_NCCL_RMA_COMPILED
  at::Tensor rma_arena_;
  ncclWindow_t rma_window_ = nullptr;
  bool rma_prepared_ = false;
  int64_t rma_batch_ = 0;
  int64_t rma_max_width_ = 0;
  at::ScalarType rma_dtype_ = at::kHalf;
  cudaStream_t rma_stream_ = nullptr;
  uint64_t rma_epoch_ = 0;
#endif
};

py::bytes get_nccl_unique_id() {
  ncclUniqueId id;
  BASIS_NCCL_CHECK(ncclGetUniqueId(&id));
  return py::bytes(reinterpret_cast<const char*>(&id), sizeof(id));
}

int get_nccl_unique_id_size() {
  return static_cast<int>(sizeof(ncclUniqueId));
}

int get_nccl_version() {
  int version = 0;
  BASIS_NCCL_CHECK(ncclGetVersion(&version));
  return version;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() =
      "Feature-major ragged TP transport: two-sided direct baseline and NCCL one-sided RMA";
  m.def("get_unique_id", &get_nccl_unique_id);
  m.def("get_unique_id_size", &get_nccl_unique_id_size);
  m.def("nccl_version", &get_nccl_version);

  py::class_<FeatureRaggedCommunicator, std::shared_ptr<FeatureRaggedCommunicator>>(
      m, "FeatureRaggedCommunicator")
      .def(py::init<const py::bytes&, int, int, int>())
      .def(
          "gather_feature_direct",
          &FeatureRaggedCommunicator::gather_feature_direct,
          py::arg("local"),
          py::arg("widths"),
          py::arg("local_is_feature_major") = false,
          py::arg("workspace") = py::none())
      .def("rma_compiled", &FeatureRaggedCommunicator::rma_compiled)
      .def("rma_runtime_version_ok", &FeatureRaggedCommunicator::rma_runtime_version_ok)
      .def("nccl_version", &FeatureRaggedCommunicator::nccl_version)
      .def(
          "prepare_rma",
          &FeatureRaggedCommunicator::prepare_rma,
          py::arg("batch"),
          py::arg("max_total_width"),
          py::arg("dtype_code"))
      .def(
          "rma_local_feature_view",
          &FeatureRaggedCommunicator::rma_local_feature_view,
          py::arg("widths"))
      .def(
          "gather_feature_rma",
          &FeatureRaggedCommunicator::gather_feature_rma,
          py::arg("local"),
          py::arg("widths"),
          py::arg("local_is_feature_major") = false)
      .def("close", &FeatureRaggedCommunicator::close)
      .def_property_readonly("rank", &FeatureRaggedCommunicator::rank)
      .def_property_readonly("world_size", &FeatureRaggedCommunicator::world_size)
      .def_property_readonly("device", &FeatureRaggedCommunicator::device);
}
