#pragma once

namespace basisserve::compressed_v_decode {

template <int kMajorValue, int kMinorValue>
struct ArchitectureTraits {
  static constexpr int kMajor = kMajorValue;
  static constexpr int kMinor = kMinorValue;
  static constexpr int kWarpSize = 32;
  static constexpr int kWarpsPerBlock = 4;
  static constexpr int kTokensPerWarpStep = 4;
};

using Sm80Traits = ArchitectureTraits<8, 0>;
using Sm89Traits = ArchitectureTraits<8, 9>;
using Sm90Traits = ArchitectureTraits<9, 0>;

}  // namespace basisserve::compressed_v_decode
