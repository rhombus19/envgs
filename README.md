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

# Datasets
Convert from colmap to easyvolcap
```bash
uv run scripts/preprocess/colmap_to_easyvolcap.py --data_root data/datasets/original/ref_real --output data/datasets/
```
NOTE: This script symlinks images from the original folder. To archive the dataset, use `tar -czhf dataset.tar.gz dataset_dir/`

An EnvGS dataset requires a couple of things:
- initial pointcloud for the base gaussian: e.g. colmap's points3D.ply
- initial env gaussian: e.g. random point cloud within certain bounds
- intri.yaml, extri.yaml: can be obtained from a colmap model using colmap_to_easyvolcap.py
- spatial_scale: 5.231606340408326

model_cfg:
    sampler_cfg:
        # Base Gaussian
        densify_until_iter: 30000
        normal_prop_until_iter: 24000
        color_sabotage_until_iter: 24000
        sh_start_iter: 10000 # let the base Gaussian be view-independent first
        # Environment Gaussian
        env_densify_until_iter: 30000
        init_specular: 0.1 # large initial specular
    supervisor_cfg:
        perc_loss_weight: 0.1


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