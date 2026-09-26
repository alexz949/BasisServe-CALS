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
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
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
constexpr size_t kIpcAlignment = 256;

class UniformAllGatherPlan;

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

size_t align_up(size_t value, size_t alignment) {
  TORCH_CHECK(alignment > 0 && (alignment & (alignment - 1)) == 0, "alignment must be a power of two");
  TORCH_CHECK(
      value <= std::numeric_limits<size_t>::max() - (alignment - 1),
      "aligned byte count overflows size_t");
  return (value + alignment - 1) & ~(alignment - 1);
}

basisserve::feature_ag::UniformIpcAlgorithm parse_ipc_algorithm(const std::string& name) {
  if (name == "auto") {
    return basisserve::feature_ag::UniformIpcAlgorithm::kAuto;
  }
  if (name == "fanout") {
    return basisserve::feature_ag::UniformIpcAlgorithm::kFanout;
  }
  if (name == "fanout_warp") {
    return basisserve::feature_ag::UniformIpcAlgorithm::kFanoutWarp;
  }
  if (name == "recursive_doubling") {
    return basisserve::feature_ag::UniformIpcAlgorithm::kRecursiveDoubling;
  }
  if (name == "ring") {
    return basisserve::feature_ag::UniformIpcAlgorithm::kRing;
  }
  TORCH_CHECK(
      false,
      "unknown IPC AllGather algorithm '",
      name,
      "'; expected auto, fanout, fanout_warp, recursive_doubling, or ring");
  return basisserve::feature_ag::UniformIpcAlgorithm::kAuto;
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
    case 3:
      return at::kByte;
    default:
      TORCH_CHECK(
          false,
          "dtype_code must be 0 (fp16), 1 (bf16), 2 (fp32), or 3 (uint8); got ",
          code);
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

class FeatureRaggedCommunicator : public std::enable_shared_from_this<FeatureRaggedCommunicator> {
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
    // close() is the clean shutdown path. Destructors across Python processes
    // are not ordered, so never perform collective window deregistration here.
    // CUDA IPC mappings are process-local and may be closed independently.
    int previous_device = -1;
    const cudaError_t get_device_status = cudaGetDevice(&previous_device);
    (void)cudaSetDevice(device_);
    close_ipc_peer_mappings_locked();
    ipc_peer_bases_ = at::Tensor();
    ipc_arena_ = at::Tensor();
    if (comm_ != nullptr) {
      (void)ncclCommAbort(comm_);
      comm_ = nullptr;
    }
#if BASISSERVE_NCCL_RMA_COMPILED
    rma_window_ = nullptr;
    rma_arena_ = at::Tensor();
#endif
    if (get_device_status == cudaSuccess && previous_device >= 0) {
      (void)cudaSetDevice(previous_device);
    }
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

  void prepare_ipc(int64_t batch, int64_t max_total_width, int64_t dtype_code) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_open();
    TORCH_CHECK(batch > 0, "IPC batch/tokens must be positive");
    TORCH_CHECK(max_total_width > 0, "IPC maximum total width must be positive");
    TORCH_CHECK(
        world_size_ == 2 || world_size_ == 4 || world_size_ == 8,
        "uniform IPC AllGather supports TP2/TP4/TP8; got TP",
        world_size_);

    c10::cuda::CUDAGuard guard(device_);
    const at::ScalarType dtype = dtype_from_code(dtype_code);
    const size_t element_size = c10::elementSize(dtype);
    const size_t slot_stride_bytes = checked_bytes(max_total_width, batch, element_size);
    TORCH_CHECK(
        slot_stride_bytes <= std::numeric_limits<size_t>::max() / basisserve::feature_ag::kIpcSlotCount,
        "IPC arena byte count overflows size_t");
    const size_t data_bytes =
        static_cast<size_t>(basisserve::feature_ag::kIpcSlotCount) * slot_stride_bytes;
    const size_t flags_offset_bytes = align_up(data_bytes, kIpcAlignment);
    const size_t flag_count =
        static_cast<size_t>(basisserve::feature_ag::kIpcSlotCount) *
        static_cast<size_t>(basisserve::feature_ag::kIpcMaxChannels) *
        static_cast<size_t>(basisserve::feature_ag::kIpcMaxPhases) *
        static_cast<size_t>(world_size_);
    TORCH_CHECK(
        flag_count <= (std::numeric_limits<size_t>::max() - flags_offset_bytes) / sizeof(uint64_t),
        "IPC flag byte count overflows size_t");
    const size_t allocation_bytes = flags_offset_bytes + flag_count * sizeof(uint64_t);

    if (ipc_prepared_ && batch == ipc_batch_ && max_total_width == ipc_max_width_ &&
        dtype == ipc_dtype_) {
      return;
    }
    TORCH_CHECK(
        !ipc_connected_,
        "disconnect_ipc must be called collectively before reconfiguring an exported IPC arena");
    release_ipc_locked();

    void* allocation = nullptr;
    BASIS_CUDA_CHECK(cudaMalloc(&allocation, allocation_bytes));
    const int allocation_device = device_;
    const auto options = at::TensorOptions().device(at::kCUDA, device_).dtype(dtype);
    ipc_arena_ = at::from_blob(
        allocation,
        {basisserve::feature_ag::kIpcSlotCount, max_total_width, batch},
        [allocation_device](void* pointer) {
          if (pointer == nullptr) {
            return;
          }
          int previous_device = -1;
          const cudaError_t get_status = cudaGetDevice(&previous_device);
          (void)cudaSetDevice(allocation_device);
          (void)cudaFree(pointer);
          if (get_status == cudaSuccess && previous_device >= 0) {
            (void)cudaSetDevice(previous_device);
          }
        },
        options);
    BASIS_CUDA_CHECK(cudaMemset(ipc_arena_.mutable_data_ptr(), 0, allocation_bytes));
    BASIS_CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handle_, allocation));

    ipc_batch_ = batch;
    ipc_max_width_ = max_total_width;
    ipc_dtype_ = dtype;
    ipc_slot_stride_bytes_ = slot_stride_bytes;
    ipc_flags_offset_bytes_ = flags_offset_bytes;
    ipc_allocation_bytes_ = allocation_bytes;
    ipc_epoch_ = 0;
    ++ipc_generation_;
    TORCH_CHECK(ipc_generation_ != 0, "IPC generation counter wrapped");
    ipc_prepared_ = true;
    ipc_connected_ = false;
  }

  py::bytes ipc_handle() const {
    TORCH_CHECK(ipc_prepared_, "prepare_ipc must be called before requesting an IPC handle");
    return py::bytes(reinterpret_cast<const char*>(&ipc_handle_), sizeof(ipc_handle_));
  }

  int ipc_handle_size() const {
    return static_cast<int>(sizeof(cudaIpcMemHandle_t));
  }

  void connect_ipc(const py::list& handles) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_open();
    TORCH_CHECK(ipc_prepared_, "prepare_ipc must be called before connect_ipc");
    TORCH_CHECK(
        static_cast<int>(py::len(handles)) == world_size_,
        "connect_ipc requires one handle per rank");
    c10::cuda::CUDAGuard guard(device_);

    close_ipc_peer_mappings_locked();
    ipc_peer_allocations_.assign(static_cast<size_t>(world_size_), nullptr);
    std::vector<int64_t> peer_addresses(static_cast<size_t>(world_size_), 0);
    for (int peer = 0; peer < world_size_; ++peer) {
      std::string raw = py::cast<py::bytes>(handles[peer]);
      TORCH_CHECK(
          raw.size() == sizeof(cudaIpcMemHandle_t),
          "invalid CUDA IPC handle byte length for peer ",
          peer,
          ": expected ",
          sizeof(cudaIpcMemHandle_t),
          ", got ",
          raw.size());
      cudaIpcMemHandle_t handle{};
      std::memcpy(&handle, raw.data(), sizeof(handle));
      void* pointer = nullptr;
      if (peer == rank_) {
        pointer = ipc_arena_.mutable_data_ptr();
      } else {
        const cudaError_t status =
            cudaIpcOpenMemHandle(&pointer, handle, cudaIpcMemLazyEnablePeerAccess);
        if (status != cudaSuccess) {
          close_ipc_peer_mappings_locked();
          TORCH_CHECK(
              false,
              "cudaIpcOpenMemHandle failed for peer ",
              peer,
              ": ",
              cudaGetErrorString(status),
              ". The selected GPUs may not have a usable CUDA P2P path.");
        }
      }
      ipc_peer_allocations_[static_cast<size_t>(peer)] = pointer;
      peer_addresses[static_cast<size_t>(peer)] =
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(pointer));
    }

    auto cpu_addresses = at::empty(
        {world_size_},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    std::memcpy(
        cpu_addresses.mutable_data_ptr<int64_t>(),
        peer_addresses.data(),
        peer_addresses.size() * sizeof(int64_t));
    ipc_peer_bases_ = cpu_addresses
                          .to(
                              at::TensorOptions()
                                  .device(at::Device(at::kCUDA, device_))
                                  .dtype(at::kLong),
                              /*non_blocking=*/false,
                              /*copy=*/true)
                          .contiguous();
    BASIS_CUDA_CHECK(cudaDeviceSynchronize());
    ipc_connected_ = true;
  }

  void disconnect_ipc() {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_open();
    c10::cuda::CUDAGuard guard(device_);
    if (!ipc_connected_) {
      return;
    }
    BASIS_CUDA_CHECK(cudaDeviceSynchronize());
    // This closes only imported peer mappings. The local exported allocation
    // remains alive until every rank has disconnected and crossed a host-side
    // process-group barrier, after which prepare_ipc()/close() may free it.
    close_ipc_peer_mappings_locked();
  }

  bool ipc_prepared() const {
    return ipc_prepared_;
  }

  bool ipc_connected() const {
    return ipc_connected_;
  }

  std::shared_ptr<UniformAllGatherPlan> make_uniform_plan(
      int64_t local_width,
      int64_t batch,
      int64_t dtype_code,
      const std::string& backend,
      const std::optional<at::Tensor>& workspace,
      const std::string& ipc_algorithm,
      int ipc_channels);

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
    release_ipc_locked();
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
  friend class UniformAllGatherPlan;

  void ensure_open() const {
    TORCH_CHECK(comm_ != nullptr, "FeatureRaggedCommunicator is closed");
  }

  void ensure_ipc_ready() const {
    TORCH_CHECK(ipc_prepared_, "prepare_ipc must be called before uniform_ipc");
    TORCH_CHECK(ipc_connected_, "connect_ipc must be called before uniform_ipc");
    TORCH_CHECK(ipc_arena_.defined(), "IPC arena is unavailable");
    TORCH_CHECK(ipc_peer_bases_.defined(), "IPC peer table is unavailable");
  }

  at::Tensor ipc_slot_tensor(int64_t slot) const {
    TORCH_CHECK(slot >= 0 && slot < basisserve::feature_ag::kIpcSlotCount, "invalid IPC slot ", slot);
    TORCH_CHECK(ipc_arena_.defined(), "IPC arena is unavailable");
    return ipc_arena_.select(0, slot);
  }

  void close_ipc_peer_mappings_locked() {
    for (int peer = 0; peer < static_cast<int>(ipc_peer_allocations_.size()); ++peer) {
      void* pointer = ipc_peer_allocations_[static_cast<size_t>(peer)];
      if (pointer != nullptr && peer != rank_) {
        (void)cudaIpcCloseMemHandle(pointer);
      }
    }
    ipc_peer_allocations_.clear();
    ipc_peer_bases_ = at::Tensor();
    ipc_connected_ = false;
  }

  void release_ipc_locked() {
    if (!ipc_prepared_ && !ipc_arena_.defined() && ipc_peer_allocations_.empty()) {
      return;
    }
    const cudaError_t sync_status = cudaDeviceSynchronize();
    TORCH_CHECK(
        sync_status == cudaSuccess,
        "cudaDeviceSynchronize before IPC release failed: ",
        cudaGetErrorString(sync_status));
    close_ipc_peer_mappings_locked();
    ipc_arena_ = at::Tensor();
    ipc_prepared_ = false;
    ipc_batch_ = 0;
    ipc_max_width_ = 0;
    ipc_dtype_ = at::kHalf;
    ipc_slot_stride_bytes_ = 0;
    ipc_flags_offset_bytes_ = 0;
    ipc_allocation_bytes_ = 0;
    ipc_epoch_ = 0;
    std::memset(&ipc_handle_, 0, sizeof(ipc_handle_));
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

  at::Tensor ipc_arena_;
  at::Tensor ipc_peer_bases_;
  std::vector<void*> ipc_peer_allocations_;
  cudaIpcMemHandle_t ipc_handle_{};
  bool ipc_prepared_ = false;
  bool ipc_connected_ = false;
  int64_t ipc_batch_ = 0;
  int64_t ipc_max_width_ = 0;
  at::ScalarType ipc_dtype_ = at::kHalf;
  size_t ipc_slot_stride_bytes_ = 0;
  size_t ipc_flags_offset_bytes_ = 0;
  size_t ipc_allocation_bytes_ = 0;
  uint64_t ipc_epoch_ = 0;
  uint64_t ipc_generation_ = 0;

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

class UniformAllGatherPlan {
 public:
  UniformAllGatherPlan(
      std::shared_ptr<FeatureRaggedCommunicator> owner,
      int64_t local_width,
      int64_t batch,
      at::ScalarType dtype,
      std::string backend,
      const std::optional<at::Tensor>& workspace,
      basisserve::feature_ag::UniformIpcAlgorithm ipc_algorithm,
      std::string ipc_algorithm_name,
      int ipc_channels)
      : owner_(std::move(owner)),
        local_width_(local_width),
        batch_(batch),
        total_width_(0),
        dtype_(dtype),
        backend_(std::move(backend)),
        ipc_algorithm_(ipc_algorithm),
        ipc_algorithm_name_(std::move(ipc_algorithm_name)),
        ipc_channels_(ipc_channels) {
    TORCH_CHECK(owner_ != nullptr, "uniform AllGather plan requires a communicator");
    owner_->ensure_open();
    TORCH_CHECK(local_width_ > 0 && batch_ > 0, "uniform AllGather dimensions must be positive");
    TORCH_CHECK(
        ipc_channels_ == 0 || ipc_channels_ == 1 || ipc_channels_ == 2 ||
            ipc_channels_ == 4 || ipc_channels_ == 8,
        "IPC channel count must be 0 (auto), 1, 2, 4, or 8; got ",
        ipc_channels_);
    TORCH_CHECK(
        local_width_ <= std::numeric_limits<int64_t>::max() / owner_->world_size_,
        "uniform AllGather total width overflows int64_t");
    total_width_ = local_width_ * owner_->world_size_;
    stream_ = c10::cuda::getCurrentCUDAStream(owner_->device_).stream();

    if (backend_ == "uniform_nccl") {
      TORCH_CHECK(workspace.has_value(), "uniform_nccl requires a preallocated workspace");
      arena_ = *workspace;
      validate_cuda_tensor(arena_, owner_->device_, "uniform NCCL workspace");
      TORCH_CHECK(arena_.is_contiguous(), "uniform NCCL workspace must be contiguous");
      TORCH_CHECK(arena_.scalar_type() == dtype_, "uniform NCCL workspace dtype mismatch");
      TORCH_CHECK(
          arena_.dim() == 2 && arena_.size(0) == total_width_ && arena_.size(1) == batch_,
          "uniform NCCL workspace must have shape [",
          total_width_,
          ", ",
          batch_,
          "]; got ",
          arena_.sizes());
      own_ = arena_.narrow(0, owner_->rank_ * local_width_, local_width_);
      nccl_dtype_ = to_nccl_dtype(dtype_);
      count_ = checked_bytes(local_width_, batch_, /*element_size=*/1);
      return;
    }

    TORCH_CHECK(
        backend_ == "uniform_ipc",
        "unknown uniform AllGather backend '",
        backend_,
        "'; expected uniform_nccl or uniform_ipc");
    TORCH_CHECK(!workspace.has_value(), "uniform_ipc owns its symmetric arena internally");
    owner_->ensure_ipc_ready();
    ipc_generation_ = owner_->ipc_generation_;
    TORCH_CHECK(owner_->ipc_batch_ == batch_, "uniform_ipc batch differs from prepared IPC arena");
    TORCH_CHECK(owner_->ipc_dtype_ == dtype_, "uniform_ipc dtype differs from prepared IPC arena");
    TORCH_CHECK(
        total_width_ <= owner_->ipc_max_width_,
        "uniform_ipc plan width exceeds prepared IPC arena");
    block_bytes_ = checked_bytes(local_width_, batch_, c10::elementSize(dtype_));
    for (int64_t slot = 0; slot < basisserve::feature_ag::kIpcSlotCount; ++slot) {
      ipc_arenas_[static_cast<size_t>(slot)] =
          owner_->ipc_slot_tensor(slot).narrow(0, 0, total_width_);
      ipc_own_[static_cast<size_t>(slot)] =
          ipc_arenas_[static_cast<size_t>(slot)].narrow(
              0,
              owner_->rank_ * local_width_,
              local_width_);
    }
  }

  at::Tensor local_view() const {
    owner_->ensure_open();
    if (backend_ == "uniform_nccl") {
      return own_;
    }
    ensure_current_ipc_plan();
    return local_view_fast();
  }

  at::Tensor local_view_fast() const {
    if (backend_ == "uniform_nccl") {
      return own_;
    }
    const int64_t slot = static_cast<int64_t>(
        owner_->ipc_epoch_ % static_cast<uint64_t>(basisserve::feature_ag::kIpcSlotCount));
    return ipc_own_[static_cast<size_t>(slot)];
  }

  at::Tensor gather_inplace() {
    owner_->ensure_open();
    c10::cuda::CUDAGuard guard(owner_->device_);
    const cudaStream_t current_stream =
        c10::cuda::getCurrentCUDAStream(owner_->device_).stream();
    TORCH_CHECK(
        current_stream == stream_,
        "prepared uniform AllGather plan belongs to another CUDA stream; "
        "create one plan per active stream");
    if (backend_ == "uniform_ipc") {
      ensure_current_ipc_plan();
      cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
      BASIS_CUDA_CHECK(cudaStreamIsCapturing(stream_, &capture_status));
      TORCH_CHECK(
          capture_status == cudaStreamCaptureStatusNone,
          "uniform_ipc v1 is not CUDA-graph capturable because its epoch advances on the host; ",
          "use uniform_nccl for graph experiments");
    }
    return gather_inplace_impl(/*check_launch=*/true);
  }

  at::Tensor gather_inplace_fast() {
    // Serving-only fast path: the caller guarantees that the communicator is
    // open and the current CUDA device/stream still match plan construction.
    // It also guarantees that an IPC plan still belongs to the current arena
    // generation and that capture is inactive. The checked gather_inplace()
    // validates those invariants and requests an immediate CUDA launch check.
    return gather_inplace_impl(/*check_launch=*/false);
  }

  at::Tensor gather(const at::Tensor& local, bool local_is_feature_major) {
    validate_cuda_tensor(local, owner_->device_, "uniform local coordinates");
    TORCH_CHECK(local.dim() == 2, "uniform local coordinates must be rank-2");
    TORCH_CHECK(local.is_contiguous(), "uniform local coordinates must be contiguous");
    TORCH_CHECK(local.scalar_type() == dtype_, "uniform local coordinate dtype mismatch");
    TORCH_CHECK(!local.requires_grad(), "uniform AllGather is inference-only");
    if (local_is_feature_major) {
      TORCH_CHECK(
          local.size(0) == local_width_ && local.size(1) == batch_,
          "feature-major uniform local must have shape [",
          local_width_,
          ", ",
          batch_,
          "]; got ",
          local.sizes());
    } else {
      TORCH_CHECK(
          local.size(0) == batch_ && local.size(1) == local_width_,
          "token-major uniform local must have shape [",
          batch_,
          ", ",
          local_width_,
          "]; got ",
          local.sizes());
    }

    auto destination = local_view();
    const cudaStream_t current_stream =
        c10::cuda::getCurrentCUDAStream(owner_->device_).stream();
    TORCH_CHECK(
        current_stream == stream_,
        "prepared uniform AllGather plan belongs to another CUDA stream; "
        "create one plan per active stream");
    if (local_is_feature_major) {
      if (local.data_ptr() != destination.data_ptr()) {
        destination.copy_(local, /*non_blocking=*/true);
      }
    } else {
      basisserve::feature_ag::launch_token_to_feature_pack(local, destination, stream_);
    }
    return gather_inplace();
  }

  int64_t local_width() const {
    return local_width_;
  }

  int64_t batch() const {
    return batch_;
  }

  int64_t total_width() const {
    return total_width_;
  }

  const std::string& backend() const {
    return backend_;
  }

  const std::string& ipc_algorithm() const {
    return ipc_algorithm_name_;
  }

  int ipc_channels() const {
    return ipc_channels_;
  }

 private:
  at::Tensor gather_inplace_impl(bool check_launch) {
    if (backend_ == "uniform_nccl") {
      if (owner_->world_size_ > 1) {
        BASIS_NCCL_CHECK(ncclAllGather(
            own_.data_ptr(),
            arena_.data_ptr(),
            count_,
            nccl_dtype_,
            owner_->comm_,
            stream_));
      }
      return arena_;
    }

    // IPC uses two symmetric slots. The host epoch chooses the producer slot;
    // the device kernel publishes and waits on a system-scope epoch for every
    // source before the stream may consume the returned global arena.
    const uint64_t epoch = owner_->ipc_epoch_ + 1;
    const int64_t slot = static_cast<int64_t>(
        owner_->ipc_epoch_ % static_cast<uint64_t>(basisserve::feature_ag::kIpcSlotCount));
    basisserve::feature_ag::launch_uniform_ipc_allgather(
        owner_->ipc_peer_bases_,
        owner_->ipc_arena_.mutable_data_ptr(),
        owner_->rank_,
        owner_->world_size_,
        slot,
        epoch,
        owner_->ipc_slot_stride_bytes_,
        block_bytes_,
        owner_->ipc_flags_offset_bytes_,
        ipc_algorithm_,
        ipc_channels_,
        stream_,
        check_launch);
    owner_->ipc_epoch_ = epoch;
    return ipc_arenas_[static_cast<size_t>(slot)];
  }

  void ensure_current_ipc_plan() const {
    owner_->ensure_ipc_ready();
    TORCH_CHECK(
        owner_->ipc_generation_ == ipc_generation_,
        "uniform_ipc plan belongs to a previous symmetric arena generation; "
        "recreate the plan after prepare_ipc");
  }

  std::shared_ptr<FeatureRaggedCommunicator> owner_;
  int64_t local_width_ = 0;
  int64_t batch_ = 0;
  int64_t total_width_ = 0;
  at::ScalarType dtype_ = at::kHalf;
  std::string backend_;
  at::Tensor arena_;
  at::Tensor own_;
  std::array<at::Tensor, basisserve::feature_ag::kIpcSlotCount> ipc_arenas_{};
  std::array<at::Tensor, basisserve::feature_ag::kIpcSlotCount> ipc_own_{};
  size_t count_ = 0;
  size_t block_bytes_ = 0;
  uint64_t ipc_generation_ = 0;
  ncclDataType_t nccl_dtype_ = ncclFloat16;
  cudaStream_t stream_ = nullptr;
  basisserve::feature_ag::UniformIpcAlgorithm ipc_algorithm_ =
      basisserve::feature_ag::UniformIpcAlgorithm::kAuto;
  std::string ipc_algorithm_name_ = "auto";
  int ipc_channels_ = 0;
};

std::shared_ptr<UniformAllGatherPlan> FeatureRaggedCommunicator::make_uniform_plan(
    int64_t local_width,
    int64_t batch,
    int64_t dtype_code,
    const std::string& backend,
    const std::optional<at::Tensor>& workspace,
    const std::string& ipc_algorithm,
    int ipc_channels) {
  ensure_open();
  const at::ScalarType dtype = dtype_from_code(dtype_code);
  return std::make_shared<UniformAllGatherPlan>(
      shared_from_this(),
      local_width,
      batch,
      dtype,
      backend,
      workspace,
      parse_ipc_algorithm(ipc_algorithm),
      ipc_algorithm,
      ipc_channels);
}

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
      "Feature-major TP transport: ragged NCCL, prepared uniform NCCL, and experimental CUDA-IPC collectives";
  m.def("get_unique_id", &get_nccl_unique_id);
  m.def("get_unique_id_size", &get_nccl_unique_id_size);
  m.def("nccl_version", &get_nccl_version);

  py::class_<UniformAllGatherPlan, std::shared_ptr<UniformAllGatherPlan>>(
      m, "UniformAllGatherPlan")
      .def("local_view", &UniformAllGatherPlan::local_view)
      .def("local_view_fast", &UniformAllGatherPlan::local_view_fast)
      .def("gather_inplace", &UniformAllGatherPlan::gather_inplace)
      .def("gather_inplace_fast", &UniformAllGatherPlan::gather_inplace_fast)
      .def(
          "gather",
          &UniformAllGatherPlan::gather,
          py::arg("local"),
          py::arg("local_is_feature_major") = false)
      .def_property_readonly("local_width", &UniformAllGatherPlan::local_width)
      .def_property_readonly("batch", &UniformAllGatherPlan::batch)
      .def_property_readonly("total_width", &UniformAllGatherPlan::total_width)
      .def_property_readonly("backend", &UniformAllGatherPlan::backend)
      .def_property_readonly("ipc_algorithm", &UniformAllGatherPlan::ipc_algorithm)
      .def_property_readonly("ipc_channels", &UniformAllGatherPlan::ipc_channels);

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
      .def(
          "prepare_ipc",
          &FeatureRaggedCommunicator::prepare_ipc,
          py::arg("batch"),
          py::arg("max_total_width"),
          py::arg("dtype_code"))
      .def("ipc_handle", &FeatureRaggedCommunicator::ipc_handle)
      .def("ipc_handle_size", &FeatureRaggedCommunicator::ipc_handle_size)
      .def("connect_ipc", &FeatureRaggedCommunicator::connect_ipc, py::arg("handles"))
      .def("disconnect_ipc", &FeatureRaggedCommunicator::disconnect_ipc)
      .def("ipc_prepared", &FeatureRaggedCommunicator::ipc_prepared)
      .def("ipc_connected", &FeatureRaggedCommunicator::ipc_connected)
      .def(
          "make_uniform_plan",
          &FeatureRaggedCommunicator::make_uniform_plan,
          py::arg("local_width"),
          py::arg("batch"),
          py::arg("dtype_code"),
          py::arg("backend"),
          py::arg("workspace") = py::none(),
          py::arg("ipc_algorithm") = "auto",
          py::arg("ipc_channels") = 0)
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
