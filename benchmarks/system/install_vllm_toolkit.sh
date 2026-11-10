#!/usr/bin/env bash
set -euo pipefail
toolkit=/home/yl1139/basis_artifacts/cuda/12.9
downloads=/home/yl1139/basis_artifacts/cuda/downloads
mkdir -p "$toolkit" "$downloads"
while read -r component version digest; do
    archive="$component-linux-x86_64-$version-archive.tar.xz"
    curl --fail --location --retry 3 \
        "https://developer.download.nvidia.com/compute/cuda/redist/$component/linux-x86_64/$archive" \
        -o "$downloads/$archive"
    printf '%s  %s\n' "$digest" "$downloads/$archive" | sha256sum --check
    tar -xf "$downloads/$archive" --strip-components=1 -C "$toolkit"
done <<'PACKAGES'
cuda_nvcc 12.9.86 7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35
cuda_cudart 12.9.79 1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99
cuda_cccl 12.9.27 8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa
libcurand 10.3.10.19 48281b4caadb1cf790d44ac76b23c77d06f474c0b1799814f314aafec9258ad6
PACKAGES
if [[ ! -e "$toolkit/lib64" ]]; then ln -s lib "$toolkit/lib64"; fi
"$toolkit/bin/nvcc" --version
test -f "$toolkit/include/cuda_runtime.h"
test -f "$toolkit/include/crt/host_defines.h"
test -f "$toolkit/include/nv/target"
test -f "$toolkit/include/curand.h"
test -f "$toolkit/include/curand_kernel.h"
test -f "$toolkit/lib64/libcudart.so"
