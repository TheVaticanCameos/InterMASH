from pathlib import Path
import os
import torch
from platform import system
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

# -----------------------------------------------------------------------------
# Paths (IMPORTANT: keep them RELATIVE for setuptools / editable install)
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent  # ❗Do not call resolve()

# Relative paths (relative to setup.py)
MASH_ROOT = Path("ma_sh/Cpp")
MASH_SRC = MASH_ROOT / "src"
MASH_INCLUDE = MASH_ROOT / "include"

assert (THIS_DIR / MASH_SRC).exists(), f"Source directory not found: {MASH_SRC}"

# -----------------------------------------------------------------------------
# Collect sources (AS RELATIVE PATH STRINGS)
# -----------------------------------------------------------------------------
mash_sources = [
    str(MASH_SRC / p.name)
    for p in (THIS_DIR / MASH_SRC).glob("*.cpp")
]

SYSTEM = system()

# -----------------------------------------------------------------------------
# Common compile args
# -----------------------------------------------------------------------------
mash_extra_compile_args = [
    "-O3",
    "-DCMAKE_BUILD_TYPE=Release",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-DTORCH_USE_CUDA_DSA",
    "-std=c++17",
]

# -----------------------------------------------------------------------------
# CUDA / CPU extension selection
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    # CUDA architecture
    cc = torch.cuda.get_device_capability()
    arch_str = f"{cc[0]}.{cc[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str

    # Add CUDA sources (still RELATIVE)
    mash_sources += [
        str(MASH_SRC / p.name)
        for p in (THIS_DIR / MASH_SRC).glob("*.cu")
    ]

    assert len(mash_sources) > 0, f"No C++/CUDA sources found in {MASH_SRC}"

    extra_compile_args = {
        "cxx": mash_extra_compile_args + [
            "-DUSE_CUDA",
            "-DTORCH_USE_CUDA_DSA",
        ],
        "nvcc": [
            "-O3",
            "-Xfatbin",
            "-compress-all",
            "-DUSE_CUDA",
            "-std=c++17",
            "-DTORCH_USE_CUDA_DSA",
        ],
    }

    mash_module = CUDAExtension(
        name="mash_cpp",
        sources=mash_sources,                     # ✅ Relative path
        include_dirs=[str(MASH_INCLUDE.resolve())],          # ✅ Absolute include path
        extra_compile_args=extra_compile_args,
    )

else:
    assert len(mash_sources) > 0, f"No C++ sources found in {MASH_SRC}"

    mash_module = CppExtension(
        name="mash_cpp",
        sources=mash_sources,                     # ✅ Relative path
        include_dirs=[str(MASH_INCLUDE.resolve())],          # ✅ Absolute include path
        extra_compile_args=mash_extra_compile_args,
    )

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup(
    name="mash-cpp",
    version="1.5.0",
    packages=find_packages(include=["ma_sh", "ma_sh.*"]),
    ext_modules=[mash_module],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
