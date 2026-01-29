#!/usr/bin/env bash
set -euo pipefail

# 1) System deps
apt-get update
apt-get install -y ninja-build

# 2) Clone repo
git clone https://github.com/rhombus19/envgs.git
cd envgs

# 2.5) Overwrite pyproject.toml
cat > pyproject.toml <<'PYPROJECT'
[project]
name = "envgs"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "addict>=2.4.0",
    "autopep8>=2.3.2",
    "av>=16.0.1",
    "chumpy>=0.70",
    "clang-format>=21.1.6",
    "commentjson>=0.9.0",
    "cuda-python>=12.2",
    "diff-surfel-rasterization-wet",
    "diff-surfel-rasterization-wet-ch05",
    "diff-surfel-rasterization-wet-ch07",
    "dotmap>=1.3.30",
    "einops>=0.8.1",
    "fast-autocomplete>=0.9.0",
    "fast-gauss>=0.0.9",
    "func-timeout>=4.3.5",
    "glfw>=2.10.0",
    "gpustat",
    "h5py>=3.15.1",
    "imageio>=2.37.2",
    "imgui-bundle[full]==1.6",
    "ipdb>=0.13.13",
    "ipython>=9.7.0",
    "jupyter>=1.1.1",
    "kornia>=0.8.2",
    "lpips>=0.1.4",
    "matplotlib>=3.10.7",
    "memory-tempfile>=2.2.3",
    "msgpack>=1.1.2",
    "ninja>=1.13.0",
    "numpy>=2.3.5",
    "nvitop>=1.6.0",
    "open3d>=0.19.0",
    "opencv-contrib-python>=4.11.0.86",
    "opencv-python>=4.11.0.86",
    "openpyxl>=3.1.5",
    "pandas>=2.3.3",
    "pdbr>=0.9.2",
    "pillow>=12.0.0",
    "pip>=25.3",
    "plyfile>=1.1.3",
    "psutil>=7.1.3",
    "pybind11>=3.0.1",
    "pycocotools>=2.0.10",
    "pycolmap>=3.13.0",
    "pyglm>=2.8.3",
    "pymcubes>=0.1.6",
    "pymeshlab>=2025.7",
    "pyntcloud>=0.3.1",
    "pyopengl>=3.1.10",
    "pyperclip>=1.11.0",
    "pypiwin32>=223 ; sys_platform == 'win32'",
    "pytorch-memlab>=0.3.0",
    "pytorch-msssim>=1.0.0",
    "pyturbojpeg>=1.8.2",
    "pyyaml>=6.0.3",
    "regex>=2025.11.3 ; sys_platform == 'win32'",
    "rich>=14.2.0",
    "ruamel-yaml>=0.18.16",
    "scikit-build-core>=0.11.6",
    "scikit-image>=0.25.2",
    "scikit-learn>=1.7.2",
    "scipy>=1.16.3",
    "setuptools-scm>=9.2.2",
    "shtab>=1.8.0",
    "sympy>=1.14.0",
    "tensorboard>=2.20.0",
    "tensorboardx>=2.6.4",
    "termcolor>=3.2.0",
    "timg>=1.1.6",
    "torch>=2.8.0",
    "torch-scatter>=2.1.2",
    "torch-tb-profiler>=0.4.3",
    "torchdiffeq>=0.2.5",
    "torchvision",
    "tqdm>=4.67.1",
    "trimesh>=4.10.0",
    "tyro>=0.9.35",
    "ujson>=5.11.0",
    "websockets>=15.0.1",
    "xatlas>=0.0.11",
    "yacs>=0.1.8",
    "yapf>=0.43.0",
    "tinycudann",
    "simple-knn",
    "torchmcubes",
    "nvdiffrast",
    "diff-surfel-tracing",
    "pytorch3d",
    "nerfview>=0.1.3",
    "splines>=0.3.3",
    "jaxtyping>=0.3.5",
    "hf>=1.1.0",
]

[tool.uv.sources]
gpustat = { git = "https://github.com/wookayin/gpustat" }
nvdiffrast = { git = "https://github.com/NVlabs/nvdiffrast" }
torchmcubes = { git = "https://github.com/tatsy/torchmcubes" }
pytorch3d = { git = "https://github.com/facebookresearch/pytorch3d" }
tinycudann = { git = "https://github.com/NVlabs/tiny-cuda-nn/", subdirectory = "bindings/torch" }
simple-knn = { git = "https://gitlab.inria.fr/bkerbl/simple-knn" }
diff-surfel-tracing = { path = "submodules/diff-surfel-tracing" }
diff-surfel-rasterization-wet = { path = "submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet" }
diff-surfel-rasterization-wet-ch05 = { path = "submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch05" }
diff-surfel-rasterization-wet-ch07 = { path = "submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch07" }

[tool.uv.extra-build-dependencies]
diff-surfel-tracing = ["torch", "setuptools"]
diff-surfel-rasterization-wet = ["torch"]
diff-surfel-rasterization-wet-ch05 = ["torch"]
diff-surfel-rasterization-wet-ch07 = ["torch"]
simple-knn = ["torch"]
pytorch3d = ["torch"]
tinycudann = ["torch"]
torchmcubes = ["torch"]
nvdiffrast = ["torch"]
chumpy = ["pip"]
torch-scatter = ["torch"]
PYPROJECT

# 3) Submodules
git submodule update --init --recursive

# 4) Python tooling
pip install -U uv gdown

# 5) Patch rasterizer_impl.h headers (add stdint.h + cstdint if missing)
find . -type f -name 'rasterizer_impl.h' -print0 | xargs -0 -I{} sh -c '
f="{}"
need=""
grep -qE "^[[:space:]]*#include[[:space:]]*<stdint\.h>" "$f" || need="${need}#include <stdint.h>\n"
grep -qE "^[[:space:]]*#include[[:space:]]*<cstdint>"    "$f" || need="${need}#include <cstdint>\n"
[ -z "$need" ] && exit 0
if grep -qE "^[[:space:]]*#pragma[[:space:]]+once" "$f"; then
  sed -i "/^[[:space:]]*#pragma[[:space:]]\+once/a\\
$need" "$f"
else
  sed -i "1i\\
$need" "$f"
fi
'

# 6) Sync deps (limit parallel builds)
export MAKEFLAGS="-j8"
export MAX_JOBS="4"
uv sync

# 7) Editable install
uv pip install -e easyvolcap/
