"""Build the unchanged upstream CPU gather extension for the active PyTorch."""
from pathlib import Path
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
build = Path('/home/zhangal/.cache/torch_extensions/lrqk_gather')
build.mkdir(parents=True, exist_ok=True)
module = load(name='take_along_dim_grouped',
    sources=[str(root/'external/LRQK/cpp_kernel/take_along_dim_grouped.cpp')],
    build_directory=str(build), extra_cflags=['-O3', '-fopenmp', '-mavx2'],
    extra_ldflags=['-fopenmp'], with_cuda=False, verbose=True)
print(module.__file__, flush=True)
