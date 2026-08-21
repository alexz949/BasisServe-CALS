#include "ragged_allgather_common.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace basisserve::ragged_ag {
namespace {

enum class GatherAlgorithm {
  kDirect,
  kPairwise,
  kRing,
  kBiRingGrouped,
};

void check_nccl(ncclResult_t result, const char* expression) {
  TORCH_CHECK(
      result == ncclSuccess,
      expression,
      " failed: ",
      ncclGetErrorString(result));
}

ncclDataType_t nccl_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return ncclFloat32;
    case at::ScalarType::Half:
      return ncclFloat16;
    case at::ScalarType::BFloat16:
      return ncclBfloat16;
    default:
      TORCH_CHECK(
          false,
          "ragged NCCL AllGather supports float32, float16, and bfloat16; got ",
          dtype);
  }
}

GatherAlgorithm gather_algorithm(const std::string& value) {
  if (value == "direct") {
    return GatherAlgorithm::kDirect;
  }
  if (value == "pairwise") {
    return GatherAlgorithm::kPairwise;
  }
  if (value == "ring") {
    return GatherAlgorithm::kRing;
  }
  if (value == "biring_grouped") {
    return GatherAlgorithm::kBiRingGrouped;
  }
  TORCH_CHECK(
      false,
      "unknown ragged AllGather algorithm '",
      value,
      "'; expected direct, pairwise, ring, or biring_grouped");
  return GatherAlgorithm::kDirect;
}

bool is_power_of_two(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int64_t checked_total_width(const std::vector<int64_t>& widths) {
  TORCH_CHECK(!widths.empty(), "source widths cannot be empty");
  TORCH_CHECK(
      widths.size() <= static_cast<size_t>(kMaximumWorldSize),
      "at most ",
      kMaximumWorldSize,
      " sources are supported");
  int64_t total = 0;
  for (size_t source = 0; source < widths.size(); ++source) {
    TORCH_CHECK(
        widths[source] > 0,
        "source width ",
        source,
        " must be positive, got ",
        widths[source]);
    TORCH_CHECK(
        total <= std::numeric_limits<int64_t>::max() - widths[source],
        "source widths overflow int64");
    total += widths[source];
  }
  return total;
}

std::vector<int64_t> rank_major_offsets(
    const std::vector<int64_t>& widths,
    int64_t batch) {
  std::vector<int64_t> offsets(widths.size());
  int64_t prefix = 0;
  for (size_t source = 0; source < widths.size(); ++source) {
    offsets[source] = prefix;
    TORCH_CHECK(
        widths[source] <= std::numeric_limits<int64_t>::max() / batch,
        "ragged buffer size overflows int64");
    prefix += batch * widths[source];
  }
  return offsets;
}

at::Tensor pack_rank_major_impl(
    const at::Tensor& rank_major,
    const std::vector<int64_t>& widths,
    int64_t batch,
    cudaStream_t stream) {
  const int64_t total_width = checked_total_width(widths);
  TORCH_CHECK(
      rank_major.numel() == batch * total_width,
      "rank-major buffer has the wrong number of elements");
  if (batch == 1 || widths.size() == 1) {
    return rank_major.view({batch, total_width});
  }
  at::Tensor packed = at::empty({batch, total_width}, rank_major.options());
  launch_ragged_pack(rank_major, packed, widths, batch, stream);
  return packed;
}

}  // namespace

class RaggedNcclCommunicator {
 public:
  RaggedNcclCommunicator(
      const std::string& unique_id_bytes,
      int64_t rank,
      int64_t world_size,
      int64_t device_index)
      : rank_(rank), world_size_(world_size), device_index_(device_index) {
    TORCH_CHECK(
        unique_id_bytes.size() == sizeof(ncclUniqueId),
        "NCCL unique ID has ",
        unique_id_bytes.size(),
        " bytes, expected ",
        sizeof(ncclUniqueId));
    TORCH_CHECK(world_size_ > 0, "NCCL world size must be positive");
    TORCH_CHECK(
        world_size_ <= kMaximumWorldSize,
        "NCCL world size exceeds supported maximum ",
        kMaximumWorldSize);
    TORCH_CHECK(
        rank_ >= 0 && rank_ < world_size_,
        "NCCL rank is outside the communicator");
    TORCH_CHECK(device_index_ >= 0, "CUDA device index must be non-negative");

    ncclUniqueId unique_id{};
    std::memcpy(&unique_id, unique_id_bytes.data(), sizeof(unique_id));
    c10::cuda::CUDAGuard device_guard(device_index_);
    check_nccl(
        ncclCommInitRank(
            &communicator_,
            static_cast<int>(world_size_),
            unique_id,
            static_cast<int>(rank_)),
        "ncclCommInitRank");
  }

  RaggedNcclCommunicator(const RaggedNcclCommunicator&) = delete;
  RaggedNcclCommunicator& operator=(const RaggedNcclCommunicator&) = delete;

  ~RaggedNcclCommunicator() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (communicator_ != nullptr) {
      c10::cuda::CUDAGuard device_guard(device_index_);
      // Destructors must not block or throw during interpreter/CUDA teardown.
      (void)ncclCommAbort(communicator_);
      communicator_ = nullptr;
    }
    registered_workspace_pointers_.clear();
    registered_workspaces_.clear();
  }

  int64_t rank() const {
    return rank_;
  }

  int64_t world_size() const {
    return world_size_;
  }

  int64_t device_index() const {
    return device_index_;
  }

  bool is_closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return communicator_ == nullptr;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (communicator_ == nullptr) {
      return;
    }
    c10::cuda::CUDAGuard device_guard(device_index_);
    for (RegisteredWorkspace& workspace : registered_workspaces_) {
      check_nccl(
          ncclCommDeregister(communicator_, workspace.registration),
          "ncclCommDeregister");
      workspace.registration = nullptr;
    }
    registered_workspace_pointers_.clear();
    registered_workspaces_.clear();
    check_nccl(ncclCommDestroy(communicator_), "ncclCommDestroy");
    communicator_ = nullptr;
  }

  at::Tensor create_registered_workspace(
      const at::Tensor& prototype,
      int64_t elements) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(communicator_ != nullptr, "ragged NCCL communicator is closed");
    TORCH_CHECK(prototype.is_cuda(), "workspace prototype must be CUDA");
    TORCH_CHECK(
        prototype.get_device() == device_index_,
        "workspace prototype is on the wrong CUDA device");
    TORCH_CHECK(elements > 0, "registered workspace must be non-empty");
    (void)nccl_dtype(prototype.scalar_type());
    const size_t element_size = prototype.element_size();
    TORCH_CHECK(
        static_cast<uint64_t>(elements) <=
            std::numeric_limits<size_t>::max() / element_size,
        "registered workspace byte size overflows size_t");
    const size_t bytes = static_cast<size_t>(elements) * element_size;

    c10::cuda::CUDAGuard device_guard(device_index_);
    void* pointer = nullptr;
    check_nccl(ncclMemAlloc(&pointer, bytes), "ncclMemAlloc");
    at::Tensor tensor;
    try {
      const int64_t allocation_device = device_index_;
      tensor = at::from_blob(
          pointer,
          {elements},
          [allocation_device](void* allocation) {
            c10::cuda::CUDAGuard allocation_guard(allocation_device);
            (void)ncclMemFree(allocation);
          },
          prototype.options());
      void* registration = nullptr;
      check_nccl(
          ncclCommRegister(communicator_, pointer, bytes, &registration),
          "ncclCommRegister");
      registered_workspace_pointers_.insert(pointer);
      registered_workspaces_.push_back(
          RegisteredWorkspace{tensor, registration});
    } catch (...) {
      if (!tensor.defined()) {
        (void)ncclMemFree(pointer);
      }
      throw;
    }
    return tensor;
  }

  at::Tensor all_gather(
      const at::Tensor& local_latent,
      const std::vector<int64_t>& source_widths,
      const std::string& algorithm,
      const std::optional<at::Tensor>& workspace) {
    std::lock_guard<std::mutex> lock(mutex_);
    return all_gather_locked(
        local_latent,
        source_widths,
        gather_algorithm(algorithm),
        workspace,
        true);
  }

  at::Tensor all_gather_rank_major(
      const at::Tensor& local_latent,
      const std::vector<int64_t>& source_widths,
      const std::string& algorithm,
      const std::optional<at::Tensor>& workspace) {
    std::lock_guard<std::mutex> lock(mutex_);
    return all_gather_locked(
        local_latent,
        source_widths,
        gather_algorithm(algorithm),
        workspace,
        false);
  }

  at::Tensor all_gather_decode(
      const at::Tensor& local_latent,
      const at::Tensor& decoder,
      const std::vector<int64_t>& source_widths,
      const std::optional<at::Tensor>& bias,
      const std::string& algorithm,
      const std::optional<at::Tensor>& workspace) {
    std::lock_guard<std::mutex> lock(mutex_);
    at::Tensor gathered = all_gather_locked(
        local_latent,
        source_widths,
        gather_algorithm(algorithm),
        workspace,
        true);
    TORCH_CHECK(decoder.is_cuda(), "ragged decoder must be a CUDA tensor");
    TORCH_CHECK(
        decoder.get_device() == device_index_,
        "ragged decoder is on CUDA device ",
        decoder.get_device(),
        ", communicator is on device ",
        device_index_);
    TORCH_CHECK(decoder.dim() == 2, "ragged decoder must be a matrix");
    TORCH_CHECK(decoder.is_contiguous(), "ragged decoder must be contiguous");
    TORCH_CHECK(
        decoder.scalar_type() == gathered.scalar_type(),
        "ragged decoder and latent dtypes must match");
    TORCH_CHECK(
        decoder.size(0) == gathered.size(1),
        "ragged decoder input width ",
        decoder.size(0),
        " does not match gathered width ",
        gathered.size(1));
    TORCH_CHECK(
        !decoder.requires_grad() && !local_latent.requires_grad(),
        "ragged NCCL decode is inference-only");

    if (bias.has_value()) {
      const at::Tensor& value = *bias;
      TORCH_CHECK(value.is_cuda(), "ragged decoder bias must be CUDA");
      TORCH_CHECK(value.get_device() == device_index_, "bias is on the wrong device");
      TORCH_CHECK(value.is_contiguous(), "ragged decoder bias must be contiguous");
      TORCH_CHECK(value.scalar_type() == decoder.scalar_type(), "bias dtype differs");
      TORCH_CHECK(
          value.dim() == 1 && value.size(0) == decoder.size(1),
          "ragged decoder bias has the wrong shape");
      TORCH_CHECK(!value.requires_grad(), "ragged NCCL decode is inference-only");
      return at::addmm(value, gathered, decoder);
    }
    return at::matmul(gathered, decoder);
  }

 private:
  struct RegisteredWorkspace {
    at::Tensor tensor;
    void* registration;
  };

  at::Tensor all_gather_locked(
      const at::Tensor& local_latent,
      const std::vector<int64_t>& source_widths,
      GatherAlgorithm algorithm,
      const std::optional<at::Tensor>& workspace,
      bool pack_output) {
    TORCH_CHECK(communicator_ != nullptr, "ragged NCCL communicator is closed");
    TORCH_CHECK(
        source_widths.size() == static_cast<size_t>(world_size_),
        "received ",
        source_widths.size(),
        " source widths for world size ",
        world_size_);
    const int64_t total_width = checked_total_width(source_widths);
    TORCH_CHECK(local_latent.is_cuda(), "local ragged latent must be CUDA");
    TORCH_CHECK(
        local_latent.get_device() == device_index_,
        "local latent is on CUDA device ",
        local_latent.get_device(),
        ", communicator is on device ",
        device_index_);
    TORCH_CHECK(local_latent.dim() == 2, "local ragged latent must be a matrix");
    TORCH_CHECK(local_latent.is_contiguous(), "local ragged latent must be contiguous");
    TORCH_CHECK(local_latent.size(0) > 0, "local ragged batch must be positive");
    TORCH_CHECK(
        local_latent.size(1) == source_widths[rank_],
        "local latent width ",
        local_latent.size(1),
        " does not match source schedule width ",
        source_widths[rank_]);
    TORCH_CHECK(!local_latent.requires_grad(), "ragged NCCL gather is inference-only");

    c10::cuda::CUDAGuard device_guard(device_index_);
    const int64_t batch = local_latent.size(0);
    const std::vector<int64_t> source_offsets =
        rank_major_offsets(source_widths, batch);
    at::Tensor rank_major;
    if (workspace.has_value()) {
      rank_major = *workspace;
      TORCH_CHECK(rank_major.is_cuda(), "ragged workspace must be CUDA");
      TORCH_CHECK(
          rank_major.get_device() == device_index_,
          "ragged workspace is on the wrong CUDA device");
      TORCH_CHECK(rank_major.is_contiguous(), "ragged workspace must be contiguous");
      TORCH_CHECK(
          rank_major.scalar_type() == local_latent.scalar_type(),
          "ragged workspace and latent dtypes must match");
      TORCH_CHECK(
          rank_major.numel() == batch * total_width,
          "ragged workspace has ",
          rank_major.numel(),
          " elements, expected ",
          batch * total_width);
      TORCH_CHECK(
          registered_workspace_pointers_.count(rank_major.mutable_data_ptr()) != 0,
          "ragged workspace was not registered by this communicator");
      rank_major = rank_major.view({batch * total_width});
    } else {
      rank_major = at::empty({batch * total_width}, local_latent.options());
    }
    const size_t element_size = local_latent.element_size();
    const int64_t local_elements = batch * source_widths[rank_];
    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(device_index_).stream();
    char* receive_base = static_cast<char*>(rank_major.mutable_data_ptr());
    char* local_destination =
        receive_base + source_offsets[rank_] * element_size;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        local_destination,
        local_latent.const_data_ptr(),
        static_cast<size_t>(local_elements) * element_size,
        cudaMemcpyDeviceToDevice,
        stream));

    if (world_size_ > 1) {
      const ncclDataType_t datatype = nccl_dtype(local_latent.scalar_type());
      if (algorithm == GatherAlgorithm::kDirect) {
        check_nccl(ncclGroupStart(), "ncclGroupStart(direct)");
        for (int64_t peer = 0; peer < world_size_; ++peer) {
          if (peer == rank_) {
            continue;
          }
          void* destination =
              receive_base + source_offsets[peer] * element_size;
          const size_t receive_count =
              static_cast<size_t>(batch * source_widths[peer]);
          check_nccl(
              ncclRecv(
                  destination,
                  receive_count,
                  datatype,
                  static_cast<int>(peer),
                  communicator_,
                  stream),
              "ncclRecv(direct)");
        }
        for (int64_t peer = 0; peer < world_size_; ++peer) {
          if (peer == rank_) {
            continue;
          }
          check_nccl(
              ncclSend(
                  local_destination,
                  static_cast<size_t>(local_elements),
                  datatype,
                  static_cast<int>(peer),
                  communicator_,
                  stream),
              "ncclSend(direct)");
        }
        check_nccl(ncclGroupEnd(), "ncclGroupEnd(direct)");
      } else if (algorithm == GatherAlgorithm::kPairwise) {
        TORCH_CHECK(
            is_power_of_two(world_size_),
            "pairwise ragged AllGather currently requires a power-of-two world size");
        for (int64_t round = 1; round < world_size_; ++round) {
          const int64_t peer = rank_ ^ round;
          void* destination =
              receive_base + source_offsets[peer] * element_size;
          const size_t receive_count =
              static_cast<size_t>(batch * source_widths[peer]);
          check_nccl(ncclGroupStart(), "ncclGroupStart(pairwise)");
          check_nccl(
              ncclRecv(
                  destination,
                  receive_count,
                  datatype,
                  static_cast<int>(peer),
                  communicator_,
                  stream),
              "ncclRecv(pairwise)");
          check_nccl(
              ncclSend(
                  local_destination,
                  static_cast<size_t>(local_elements),
                  datatype,
                  static_cast<int>(peer),
                  communicator_,
                  stream),
              "ncclSend(pairwise)");
          check_nccl(ncclGroupEnd(), "ncclGroupEnd(pairwise)");
        }
      } else if (
          algorithm == GatherAlgorithm::kBiRingGrouped &&
          batch >= 2 && world_size_ >= 3) {
        const int64_t next = (rank_ + 1) % world_size_;
        const int64_t previous = (rank_ + world_size_ - 1) % world_size_;
        const int64_t clockwise_batch = (batch + 1) / 2;
        const int64_t counterclockwise_batch = batch - clockwise_batch;
        for (int64_t step = 0; step < world_size_ - 1; ++step) {
          const int64_t clockwise_send_source =
              (rank_ + world_size_ - step) % world_size_;
          const int64_t clockwise_receive_source =
              (rank_ + world_size_ - step - 1) % world_size_;
          const int64_t counterclockwise_send_source =
              (rank_ + step) % world_size_;
          const int64_t counterclockwise_receive_source =
              (rank_ + step + 1) % world_size_;

          const void* clockwise_send_pointer =
              receive_base +
              source_offsets[clockwise_send_source] * element_size;
          void* clockwise_receive_pointer =
              receive_base +
              source_offsets[clockwise_receive_source] * element_size;
          const void* counterclockwise_send_pointer =
              receive_base +
              (source_offsets[counterclockwise_send_source] +
               clockwise_batch *
                   source_widths[counterclockwise_send_source]) *
                  element_size;
          void* counterclockwise_receive_pointer =
              receive_base +
              (source_offsets[counterclockwise_receive_source] +
               clockwise_batch *
                   source_widths[counterclockwise_receive_source]) *
                  element_size;

          const size_t clockwise_send_count = static_cast<size_t>(
              clockwise_batch * source_widths[clockwise_send_source]);
          const size_t clockwise_receive_count = static_cast<size_t>(
              clockwise_batch * source_widths[clockwise_receive_source]);
          const size_t counterclockwise_send_count = static_cast<size_t>(
              counterclockwise_batch *
              source_widths[counterclockwise_send_source]);
          const size_t counterclockwise_receive_count = static_cast<size_t>(
              counterclockwise_batch *
              source_widths[counterclockwise_receive_source]);

          check_nccl(ncclGroupStart(), "ncclGroupStart(biring_grouped)");
          check_nccl(
              ncclRecv(
                  clockwise_receive_pointer,
                  clockwise_receive_count,
                  datatype,
                  static_cast<int>(previous),
                  communicator_,
                  stream),
              "ncclRecv(biring_grouped_clockwise)");
          check_nccl(
              ncclRecv(
                  counterclockwise_receive_pointer,
                  counterclockwise_receive_count,
                  datatype,
                  static_cast<int>(next),
                  communicator_,
                  stream),
              "ncclRecv(biring_grouped_counterclockwise)");
          check_nccl(
              ncclSend(
                  clockwise_send_pointer,
                  clockwise_send_count,
                  datatype,
                  static_cast<int>(next),
                  communicator_,
                  stream),
              "ncclSend(biring_grouped_clockwise)");
          check_nccl(
              ncclSend(
                  counterclockwise_send_pointer,
                  counterclockwise_send_count,
                  datatype,
                  static_cast<int>(previous),
                  communicator_,
                  stream),
              "ncclSend(biring_grouped_counterclockwise)");
          check_nccl(ncclGroupEnd(), "ncclGroupEnd(biring_grouped)");
        }
      } else {
        const int64_t next = (rank_ + 1) % world_size_;
        const int64_t previous = (rank_ + world_size_ - 1) % world_size_;
        for (int64_t step = 0; step < world_size_ - 1; ++step) {
          const int64_t send_source =
              (rank_ + world_size_ - step) % world_size_;
          const int64_t receive_source =
              (rank_ + world_size_ - step - 1) % world_size_;
          const void* send_pointer =
              receive_base + source_offsets[send_source] * element_size;
          void* receive_pointer =
              receive_base + source_offsets[receive_source] * element_size;
          const size_t send_count =
              static_cast<size_t>(batch * source_widths[send_source]);
          const size_t receive_count =
              static_cast<size_t>(batch * source_widths[receive_source]);
          check_nccl(ncclGroupStart(), "ncclGroupStart(ring)");
          check_nccl(
              ncclRecv(
                  receive_pointer,
                  receive_count,
                  datatype,
                  static_cast<int>(previous),
                  communicator_,
                  stream),
              "ncclRecv(ring)");
          check_nccl(
              ncclSend(
                  send_pointer,
                  send_count,
                  datatype,
                  static_cast<int>(next),
                  communicator_,
                  stream),
              "ncclSend(ring)");
          check_nccl(ncclGroupEnd(), "ncclGroupEnd(ring)");
        }
      }
    }

    if (pack_output) {
      return pack_rank_major_impl(
          rank_major,
          source_widths,
          batch,
          stream);
    }
    return rank_major;
  }

  int64_t rank_;
  int64_t world_size_;
  int64_t device_index_;
  ncclComm_t communicator_{nullptr};
  mutable std::mutex mutex_;
  std::vector<RegisteredWorkspace> registered_workspaces_;
  std::unordered_set<void*> registered_workspace_pointers_;
};

std::string nccl_unique_id() {
  ncclUniqueId unique_id{};
  check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
  return std::string(
      reinterpret_cast<const char*>(&unique_id),
      sizeof(unique_id));
}

int64_t nccl_version() {
  int version = 0;
  check_nccl(ncclGetVersion(&version), "ncclGetVersion");
  return version;
}

at::Tensor pack_rank_major(
    const at::Tensor& rank_major,
    const std::vector<int64_t>& source_widths,
    int64_t batch) {
  TORCH_CHECK(rank_major.is_cuda(), "rank-major test input must be CUDA");
  c10::cuda::CUDAGuard device_guard(rank_major.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(rank_major.get_device()).stream();
  return pack_rank_major_impl(
      rank_major.contiguous(),
      source_widths,
      batch,
      stream);
}

}  // namespace basisserve::ragged_ag

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using basisserve::ragged_ag::RaggedNcclCommunicator;
  module.doc() = "BasisServe static-ragged NCCL AllGather kernels";
  module.def("nccl_unique_id", []() {
    const std::string value = basisserve::ragged_ag::nccl_unique_id();
    return py::bytes(value);
  });
  module.def("nccl_version", &basisserve::ragged_ag::nccl_version);
  module.def("pack_rank_major", &basisserve::ragged_ag::pack_rank_major);
  py::class_<RaggedNcclCommunicator, std::shared_ptr<RaggedNcclCommunicator>>(
      module,
      "RaggedNcclCommunicator")
      .def(
          py::init<const std::string&, int64_t, int64_t, int64_t>(),
          py::arg("unique_id"),
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("device_index"))
      .def_property_readonly("rank", &RaggedNcclCommunicator::rank)
      .def_property_readonly("world_size", &RaggedNcclCommunicator::world_size)
      .def_property_readonly("device_index", &RaggedNcclCommunicator::device_index)
      .def_property_readonly("is_closed", &RaggedNcclCommunicator::is_closed)
      .def("close", &RaggedNcclCommunicator::close)
      .def(
          "create_registered_workspace",
          &RaggedNcclCommunicator::create_registered_workspace,
          py::arg("prototype"),
          py::arg("elements"))
      .def(
          "all_gather",
          &RaggedNcclCommunicator::all_gather,
          py::arg("local_latent"),
          py::arg("source_widths"),
          py::arg("algorithm") = "direct",
          py::arg("workspace") = std::nullopt)
      .def(
          "all_gather_rank_major",
          &RaggedNcclCommunicator::all_gather_rank_major,
          py::arg("local_latent"),
          py::arg("source_widths"),
          py::arg("algorithm") = "direct",
          py::arg("workspace") = std::nullopt)
      .def(
          "all_gather_decode",
          &RaggedNcclCommunicator::all_gather_decode,
          py::arg("local_latent"),
          py::arg("decoder"),
          py::arg("source_widths"),
          py::arg("bias") = std::nullopt,
          py::arg("algorithm") = "direct",
          py::arg("workspace") = std::nullopt);
}
