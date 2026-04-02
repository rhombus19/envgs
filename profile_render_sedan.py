from pathlib import Path

import numpy as np
import torch

from easyvolcap.utils.easy_utils import read_camera
from profile_render import dotdict, get_rays, get_tracer, load_splats, make_camera_batch, normalize, render


CKPT_PATH = "data/trained_model/envgs_sedan_mast3r/latest.pt"
CAMERA_ROOT = Path("/home/roman/ba/datasets/ref_real/sedan_mast3r")
VAL_VIEWS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152]
RATIO = 0.25
WARMUP = 3
REPEATS = 5


def make_sedan_camera(cam):
    h = int(cam.H * RATIO)
    w = int(cam.W * RATIO)
    k = cam.K.copy()
    k[0:1] *= w / cam.W
    k[1:2] *= h / cam.H
    return make_camera_batch(
        torch.tensor(h, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(cam.R, dtype=torch.float32),
        torch.tensor(cam.T, dtype=torch.float32),
    )


@torch.inference_mode()
def timed_render(pcd, env, cam, tracer, bg):
    pipe_dif = dotdict(convert_SHs_python=True, compute_cov3D_python=False, depth_ratio=0.0)
    pipe_env = dotdict(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    ev[0].record()
    dif = render(cam, pcd, pipe_dif, bg)
    ev[1].record()

    h = int(cam.image_height.item())
    w = int(cam.image_width.item())
    dif_rgb = dif.render.permute(1, 2, 0)
    norm = normalize(dif.rend_normal.permute(1, 2, 0))
    dpt = dif.surf_depth.permute(1, 2, 0)
    spec = dif.specular.permute(1, 2, 0)
    ray_o, ray_d = get_rays(h, w, cam.K, cam.R, cam.T, correct_pix=True, z_depth=True)
    ref_d = ray_d - 2 * (ray_d * norm).sum(-1, keepdim=True) * norm
    ref_o = ray_o + ray_d * dpt

    ev[2].record()
    env_out = tracer.render_gaussians(
        cam,
        ref_o,
        ref_d,
        env,
        pipe_env,
        bg,
        max_trace_depth=0,
        specular_threshold=0.0,
        start_from_first=False,
        batch=cam,
    )
    ev[3].record()

    _ = (1.0 - spec) * dif_rgb + spec * env_out.render.permute(1, 2, 0)
    ev[4].record()
    torch.cuda.synchronize()

    return {
        "frame_ms": ev[0].elapsed_time(ev[4]),
        "raster_ms": ev[0].elapsed_time(ev[1]),
        "raytrace_ms": ev[2].elapsed_time(ev[3]),
    }


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr.mean(), arr.std()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(CKPT_PATH)

    cams = read_camera(CAMERA_ROOT / "intri.yml", CAMERA_ROOT / "extri.yml")
    val_cams = [make_sedan_camera(cams[f"{view:04d}"]) for view in VAL_VIEWS]
    pcd, env = load_splats(CKPT_PATH)
    tracer = get_tracer()
    bg = torch.zeros(3, device="cuda")

    per_view = {key: [] for key in ("frame_ms", "raster_ms", "raytrace_ms")}
    for cam in val_cams:
        for _ in range(WARMUP):
            timed_render(pcd, env, cam, tracer, bg)
        samples = {key: [] for key in per_view}
        for _ in range(REPEATS):
            stats = timed_render(pcd, env, cam, tracer, bg)
            for key, value in stats.items():
                samples[key].append(value)
        for key in per_view:
            per_view[key].append(np.mean(samples[key]))

    print(f"checkpoint: {CKPT_PATH}")
    print(f"views: {len(val_cams)}  repeats_per_view: {REPEATS}  resolution: {int(val_cams[0].image_height.item())}x{int(val_cams[0].image_width.item())}")
    for key, label in [("frame_ms", "frame"), ("raster_ms", "rasterization"), ("raytrace_ms", "raytracing")]:
        mean, std = mean_std(per_view[key])
        print(f"{label}: {mean:.3f} ms +- {std:.3f} ms")


if __name__ == "__main__":
    main()
