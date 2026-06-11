*Forked from [EnvGS](https://github.com/zju3dv/EnvGS)*

# Installation
The repository is built on top of [easyvolcap](https://github.com/zju3dv/EasyVolcap)
```bash
# Clone submodules
git submodule update --init --recursive

# Install uv package manager globally, if you haven't already
pip install uv

# Build environment
# NOTE: Pytorch3D takes a long time to build and eats a lot of RAM. To tame him, use the MAX_JOBS env variable and make sure you have ninja installed
MAX_JOBS=6 uv sync

# A workaround to install the local copy of easyvolcap, as it has its own pyproject.toml If you know a better way to do this, fix this
uv pip install -e easyvolcap/

# Launch the webgui
uv run webui.py
```

Installation in AWS:


If you want to run on recent cuda versions (higher than cuda 11.8)
you have to include stdint.h/cstint in all occurances of rasterizer_impl.h

Monkey-patch one-liner:
```bash
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
```


# Datasets
An EnvGS dataset requires:
- images in the easyvolcap format (1 folder per image in our case)
- normal maps in the easyvolcap format
- camera poses in intri.yaml, extri.yaml
- initial base point cloud
- metadata to generate the initial env point cloud

Initially the scene we use is structured like this:
1. images/ folder with all gt frames ordered by name
2. stablenormal_normals/ folder with normal map images (be aware of the normal convention, tagent space normal maps used are inverted and don't contain the standard magenta-blue-light green)
The same folders exist for other normal map predictors
3. colmap_sparse/0/ folder for the sparse point cloud and camera poses
The same folder exist for other SfM algorithms, they all adhere to the standard colmap scheme

The next step is to generate a training dataset in the correct format. This is done using the colmap_to_easyvolcap.py script

```bash
uv run easyvolcap/scripts/preprocess/colmap_to_easyvolcap.py --data_root data/datasets/original/ref_real --src_normals_dir stblenormal_normals/ --colmap colmap_sparse/0/ --output data/datasets/ --multi-scene
```
You can provide custom paths for different normal maps and colmap models to generate the training dataset you need. Use --multi-scene if your data_root contains multiple scenes in subfolders of data_root

NOTE: This script symlinks images and normals from the original folder. To archive the dataset, use the -h flag in tar to follow symlinks `tar -czhf dataset.tar.gz dataset_dir/`

The second step is generating the metadata and the dataset config. This is done using another script
```bash
uv run easyvolcap/scripts/preprocess/tools/generate_metadata.py --data_root data/datasets/original/ref_real --scenes sedan --output_yaml config/ref_real
```
Note that the equivalent to the 


The datasets we provided are structured

Convert from colmap to easyvolcap



# Issues encountered:


# Training

# Envgs project structure
The meat and potatoes of the project live in `envgs/easyvolcap/easyvolcap/models/samplers/envgs_sampler.py` and `envgs/easyvolcap/easyvolcap/models/supervisors/envgs_supervisor.py`
The other relevant parts are located in `envgs/easyvolcap/easyvolcap/utils/gaussian2d_utils.py` and `envgs/easyvolcap/easyvolcap/utils/optix_utils.py`


This is what the built model looks like:
```python
>>> self.model
VolumetricVideoModel(
  (camera): NoopCamera()
  (network): NoopNetwork()
  (sampler): EnvGSSampler(
    (pcd): GaussianModel()
    (diffop): HardwareRendering(
      (tracer): SurfelTracer()
    )
    (env): GaussianModel()
  )
  (renderer): NoopRenderer()
  (supervisor): SequentialSupervisor(
    (supervisors): ModuleList(
      (0): VolumetricVideoSupervisor()
      (1): EnvGSSupervisor()
    )
  )
)
```

This is what a test batch can look like:
```python
>>> batch
    {
    'H': tensor([768], device='cuda:0', dtype=torch.int32),
    'W': tensor([1366], device='cuda:0', dtype=torch.int32),
    'K': tensor([[[1.3660e+03, 0.0000e+00, 6.8300e+02],
         [0.0000e+00, 1.3660e+03, 3.8400e+02],
         [0.0000e+00, 0.0000e+00, 1.0000e+00]]], device='cuda:0'),
    'R': tensor([[[ 0.7071, -0.7071,  0.0000],
         [ 0.4082,  0.4082, -0.8165],
         [ 0.5774,  0.5774,  0.5774]]], device='cuda:0'),
    'T': tensor([[[ 0.0000],
         [ 0.0000],
         [25.9808]]], device='cuda:0'),
    'n': tensor([0.0200], device='cuda:0'),
    'f': tensor([100.], device='cuda:0'),
    't': tensor([0.], device='cuda:0'),
    'v': tensor([0.], device='cuda:0'),
    'bounds': tensor([[[-10., -10., -10.],
         [ 10.,  10.,  10.]]], device='cuda:0'),
    'meta': {
        'H': tensor([768], dtype=torch.int32),
        'W': tensor([1366], dtype=torch.int32),
        'K': tensor([[[1.3660e+03, 0.0000e+00, 6.8300e+02],
         [0.0000e+00, 1.3660e+03, 3.8400e+02],
         [0.0000e+00, 0.0000e+00, 1.0000e+00]]]),
        'R': tensor([[[ 0.7071, -0.7071,  0.0000],
         [ 0.4082,  0.4082, -0.8165],
         [ 0.5774,  0.5774,  0.5774]]]),
        'T': tensor([[[ 0.0000],
         [ 0.0000],
         [25.9808]]]),
        'n': tensor([0.0200]),
        'f': tensor([100.]),
        't': tensor([0.]),
        'v': tensor([0.]),
        'bounds': tensor([[[-10., -10., -10.],
         [ 10.,  10.,  10.]]]),
        'view_index': tensor([0]),
        'frame_index': tensor([0]),
        'camera_index': tensor([0]),
        'latent_index': tensor([0]),
        'crop_x': tensor([0], dtype=torch.int32),
        'crop_y': tensor([0], dtype=torch.int32),
        'orig_w': tensor([1366]),
        'orig_h': tensor([768]),
        'iter': 60000,
        'frac': 1.0
    },
    'view_index': tensor([0], device='cuda:0'),
    'frame_index': tensor([0], device='cuda:0'),
    'camera_index': tensor([0], device='cuda:0'),
    'latent_index': tensor([0], device='cuda:0'),
    'tar_ixt': tensor([[[1.3660e+03, 0.0000e+00, 6.8300e+02],
         [0.0000e+00, 1.3660e+03, 3.8400e+02],
         [0.0000e+00, 0.0000e+00, 1.0000e+00]]], device='cuda:0'),
    'crop_x': tensor([0], device='cuda:0', dtype=torch.int32),
    'crop_y': tensor([0], device='cuda:0', dtype=torch.int32),
    'orig_w': tensor([1366], device='cuda:0'),
    'orig_h': tensor([768], device='cuda:0'),
    'iter': 60000,
    'frac': 1.0
}
```

# Analysis

Launch Tensorboard:
```bash
uv run tensorboard --logdir data/record/ --bind_all
```