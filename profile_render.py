import math
import os
from contextlib import contextmanager

import numpy as np
import torch

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.gaussian2d_utils import GaussianModel, render
from easyvolcap.utils.optix_utils import HardwareRendering
from easyvolcap.utils.ray_utils import get_rays


# To Profile:
# EVG_NVTX=1 EVG_PROFILE_ONCE=1 nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop -o /tmp/envgs_profile uv run profile_render.py
# nsys-gui {path_to_profile}

_USE_NVTX = os.getenv("EVG_NVTX", "1") == "1"
_PROFILE_ONCE = os.getenv("EVG_PROFILE_ONCE", "1") == "1"
_WARMUP = int(os.getenv("EVG_WARMUP", "1"))
_PROFILE_DONE = False
try:
    from torch.cuda import nvtx as _nvtx
except Exception:
    _nvtx = None


@contextmanager
def nvtx_range(name: str):
    if not _USE_NVTX or _nvtx is None:
        yield
        return
    _nvtx.range_push(name)
    try:
        yield
    finally:
        _nvtx.range_pop()


def _profiled_call(fn, *args, range_name: str | None = None, **kwargs):
    global _PROFILE_DONE

    def _run():
        if range_name:
            with nvtx_range(range_name):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    if not _PROFILE_ONCE or _PROFILE_DONE or not torch.cuda.is_available():
        return _run()

    torch.cuda.profiler.start()
    try:
        return _run()
    finally:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        _PROFILE_DONE = True


# Copied from easyvolcap/easyvolcap/utils/gaussian2d_utils.py
def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


@torch.jit.script
def getWorld2View(R: torch.Tensor, t: torch.Tensor):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = torch.eye(4, dtype=R.dtype, device=R.device)  # 4, 4
    for i in range(len(sh)):
        T = T.unsqueeze(0)
    T = T.expand(sh + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    return T


@torch.jit.script
def getProjectionMatrix(fovx: torch.Tensor, fovy: torch.Tensor, znear: torch.Tensor, zfar: torch.Tensor):
    tanfovy = math.tan((fovy / 2))
    tanfovx = math.tan((fovx / 2))

    t = tanfovy * znear
    b = -t
    r = tanfovx * znear
    l = -r

    P = torch.zeros(4, 4, dtype=znear.dtype, device=znear.device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (r - l)
    P[1, 1] = 2.0 * znear / (t - b)

    P[0, 2] = (r + l) / (r - l)
    P[1, 2] = (t + b) / (t - b)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)

    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def make_camera_batch(H, W, K, R, T, n=0.1, f=100.0):
    # Input camera params should be on CPU
    output = dotdict()

    output.image_height = H
    output.image_width = W

    fl_x = K[0, 0]  # use cpu K
    fl_y = K[1, 1]  # use cpu K
    FoVx = focal2fov(fl_x, W)
    FoVy = focal2fov(fl_y, H)

    output.K = K.cuda()
    output.R = R.cuda()
    output.T = T.cuda()
    n = torch.tensor(n)
    f = torch.tensor(f)

    output.world_view_transform = getWorld2View(R, T).transpose(0, 1).cuda()
    output.projection_matrix = getProjectionMatrix(FoVx, FoVy, n, f).transpose(0, 1).cuda()
    output.full_proj_transform = torch.matmul(output.world_view_transform, output.projection_matrix).cuda()
    output.camera_center = (-R.mT @ T)[..., 0].cuda()  # B, 3, 1 -> 3,

    output.FoVx = FoVx.cuda()
    output.FoVy = FoVy.cuda()
    output.tanfovx = math.tan(FoVx * 0.5)
    output.tanfovy = math.tan(FoVy * 0.5)

    output.znear = n.cuda()
    output.zfar = f.cuda()

    return output


def load_splats(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")["model"]

    # Dummy GaussianModel
    pcd = GaussianModel(
        xyz=torch.empty(1, 3, device="cuda"),
        colors=None,
        sh_degree=3,
        render_reflection=True,
        specular_channels=True,
        xyz_lr_scheduler=None,  # key fix: avoid lr_init/lr_final access
    )

    # Dummy GaussianModel
    env = GaussianModel(
        xyz=torch.empty(1, 3, device="cuda"),
        colors=None,
        sh_degree=3,
        render_reflection=False,
        specular_channels=True,
        xyz_lr_scheduler=None,  # key fix: avoid lr_init/lr_final access
    )

    # Separate parameters of pcd and env based on prefix. NOTE: Pretty smelly code
    loaded_params_pcd = {k[len("sampler.pcd."):]: v for k, v in state.items() if k.startswith("sampler.pcd.")}
    loaded_params_env = {k[len("sampler.env."):]: v for k, v in state.items() if k.startswith("sampler.env.")}

    pcd.load_state_dict(loaded_params_pcd, strict=False)
    env.load_state_dict(loaded_params_env, strict=False)

    return pcd.cuda(), env.cuda()


TRACER = HardwareRendering().cuda()

torch.set_grad_enabled(False)


# Copied from easyvolcap/easyvolcap/utils/math_utils.py
@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def render_frame(pcd, env, cam):
    with nvtx_range("render_frame"):
        # Diffuse pass (rasterizer) for pcd gaussians
        pipe_dif = dotdict(convert_SHs_python=True, compute_cov3D_python=False, depth_ratio=0.0)
        bg = torch.zeros(3, device="cuda")
        with nvtx_range("diffuse_render"):
            dif = render(cam, pcd, pipe_dif, bg)
        dif_rgb = dif.render.permute(1, 2, 0)
        norm_map = dif.rend_normal.permute(1, 2, 0)
        dpt_map = dif.surf_depth.permute(1, 2, 0)
        spec_map = dif.specular.permute(1, 2, 0)

        # Calculate ray directions per pixel
        w, h = cam.image_width.item(), cam.image_height.item()
        with nvtx_range("ray_gen"):
            ray_o, ray_d = get_rays(h, w, cam.K, cam.R, cam.T, correct_pix=True, z_depth=True)

        # Calculate reflected rays in screen space
        with nvtx_range("reflect"):
            norm = normalize(norm_map)
            ref_d = ray_d - 2 * (ray_d * norm).sum(-1, keepdim=True) * norm
            ref_o = ray_o + ray_d * dpt_map

        # Trace reflected rays through the env gaussians using OptiX
        pipe_env = dotdict(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0)
        env_bg = torch.zeros(3, device="cuda")
        with nvtx_range("trace_env"):
            env_out = TRACER.render_gaussians(
                cam, ref_o, ref_d, env, pipe_env, env_bg,
                max_trace_depth=0,
                specular_threshold=0.0,
                start_from_first=False,
                batch=cam,
            )
        env_rgb = env_out.render.permute(1, 2, 0)

        # Blend
        with nvtx_range("blend"):
            rgb = (1.0 - spec_map) * dif_rgb + spec_map * env_rgb
            rgb8 = (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        return rgb8


def main():
    ckpt_path = os.getenv("EVG_CKPT_PATH", "/home/roman/ba/envgs/data/trained_model/envgs_sedan/latest.pt")
    pcd, env = load_splats(ckpt_path)

    H, W = 480 * 2, 640 * 2
    fx = fy = 500.0
    K = [
        [fx, 0.0, W * 0.5],
        [0.0, fy, H * 0.5],
        [0.0, 0.0, 1.0],
    ]
    R = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    T = [[0.0], [0.0], [3.0]]
    cam = make_camera_batch(
        torch.tensor(H, dtype=torch.float32),
        torch.tensor(W, dtype=torch.float32),
        torch.tensor(K, dtype=torch.float32),
        torch.tensor(R, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
    )

    for _ in range(_WARMUP):
        render_frame(pcd, env, cam)
    torch.cuda.synchronize()

    _profiled_call(render_frame, pcd, env, cam, range_name="render_frame")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
