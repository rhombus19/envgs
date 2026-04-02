import argparse
import math
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import add_batch, to_cuda, to_x
from easyvolcap.utils.gaussian2d_utils import GaussianModel, prepare_gaussian_camera, render
from easyvolcap.utils.optix_utils import HardwareRendering
from easyvolcap.utils.ray_utils import get_rays


# To Profile:
# EVG_NVTX=1 EVG_PROFILE_ONCE=1 nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop -o /tmp/envgs_profile uv run profile_render.py
# nsys-gui {path_to_profile}

_USE_NVTX = os.getenv("EVG_NVTX", "1") == "0"
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


TRACER = None

torch.set_grad_enabled(False)


# Copied from easyvolcap/easyvolcap/utils/math_utils.py
@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def get_tracer():
    global TRACER
    if TRACER is None:
        TRACER = HardwareRendering().cuda()
    return TRACER


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
            env_out = get_tracer().render_gaussians(
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

def parse_hw(text: str) -> tuple[int, int]:
    h, w = [int(x.strip()) for x in text.split(",")]
    return h, w


def parse_view_list(text: str, n_views: int) -> list[int]:
    text = text.strip()
    if not text:
        return [0]
    if text.lower() == "all":
        return list(range(n_views))
    return [int(x.strip()) for x in text.split(",")]


def parse_local_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bench-source", choices=["dummy", "viewer_init", "val"], default=None)
    parser.add_argument("--bench-views", default="")
    parser.add_argument("--bench-frame", type=int, default=None)
    parser.add_argument("--bench-window-size", default="1080,729")
    parser.add_argument("--bench-iters", type=int, default=10)
    parser.add_argument("--bench-warmup", type=int, default=_WARMUP)
    parser.add_argument("--bench-compare-model", action="store_true")
    return parser.parse_known_args(argv)


def prime_batch(batch, iter_value: int, total_iter: int):
    batch.iter = iter_value
    batch.frac = iter_value / total_iter if total_iter else 0.0
    batch.meta.iter = batch.iter
    batch.meta.frac = batch.frac
    return batch


def load_config_runner(engine_argv: list[str]):
    import importlib
    import importlib.util
    from pathlib import Path

    engine_argv = list(engine_argv)
    if not any(arg in {"-t", "--type"} for arg in engine_argv):
        engine_argv = ["-t", "test", *engine_argv]

    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *engine_argv]
    try:
        from easyvolcap.engine import cfg as evc_cfg
        import easyvolcap
        import easyvolcap.utils

        inner_pkg_root = Path(easyvolcap.utils.__file__).resolve().parents[1]
        if str(inner_pkg_root) not in easyvolcap.__path__:
            easyvolcap.__path__.append(str(inner_pkg_root))

        module_path = inner_pkg_root / "scripts" / "main.py"
        spec = importlib.util.spec_from_file_location("evc_main_module", module_path)
        evc_main = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(evc_main)

        runner = evc_main.test(
            evc_cfg,
            dry_run=True,
            print_test_progress=False,
            record_images_to_tb=False,
            base_device="cuda",
        )
        epoch = runner.load_network()
        runner.model.eval()
        return runner, evc_cfg, epoch
    finally:
        sys.argv = old_argv


def make_viewer_style_batch(dataset, window_size: tuple[int, int], source: str, view_index: int | None, frame_index: int | None):
    from easyvolcap.utils.viewer_utils import Camera

    H, W = window_size
    M = max(H, W)
    frame_index = dataset.frame_min if frame_index is None else frame_index
    latent_index = dataset.frame_to_latent(frame_index)

    if source == "viewer_init":
        ref_view_index = getattr(dataset, "init_viewer_index", 0)
        ref_H = int(dataset.Hs[ref_view_index, latent_index].item())
        ref_W = int(dataset.Ws[ref_view_index, latent_index].item())
        K = dataset.Ks[ref_view_index, latent_index].clone()
        K[:2] *= M / max(ref_H, ref_W)
        R = dataset.Rv.clone()
        T = dataset.Tv.clone()
        camera_index = dataset.view_min
        label = f"viewer_init frame={frame_index}"
    else:
        if view_index is None:
            raise ValueError("view_index is required for val view benchmarking")
        ref_H = int(dataset.Hs[view_index, latent_index].item())
        ref_W = int(dataset.Ws[view_index, latent_index].item())
        K = dataset.Ks[view_index, latent_index].clone()
        K[:2] *= M / max(ref_H, ref_W)
        R = dataset.Rs[view_index, latent_index].clone()
        T = dataset.Ts[view_index, latent_index].clone()
        camera_index = int(dataset.view_min + view_index * dataset.view_int)
        label = f"val view={view_index} frame={frame_index}"

    t = dataset.frame_to_t(frame_index)
    v = dataset.camera_to_v(camera_index)
    camera = Camera(H, W, K, R, T, dataset.near, dataset.far, t, v, dataset.bounds.clone())
    batch = camera.to_batch(is_dataset=True)
    batch = dataset.get_viewer_batch(batch, requires_bounds=True)
    batch = add_batch(batch)
    batch = to_cuda(batch)
    return batch, label


def render_frame_split(sampler, batch):
    viewpoint_camera = to_x(prepare_gaussian_camera(batch), torch.float)
    ray_o, ray_d, coords, _, _, _ = sampler.get_camera_rays(
        batch,
        n_rays=sampler.n_rays,
        patch_size=sampler.patch_size,
    )
    dif_output = sampler.render_gaussians(
        viewpoint_camera,
        sampler.pcd,
        sampler.pipe,
        sampler.bg_color,
        sampler.scale_mod,
        override_color=None,
    )
    output = sampler.store_dif_gaussian_output(dif_output, batch)
    ref_o, ref_d = sampler.get_reflect_rays(ray_o, ray_d, coords, output, batch)
    env_output = sampler.diffop.render_gaussians(
        viewpoint_camera,
        ref_o,
        ref_d,
        sampler.env,
        sampler.pipe_env,
        sampler.env_bg_color,
        0,
        start_from_first=False,
        scaling_modifier=sampler.scale_mod,
        override_color=None,
        batch=batch,
    )
    return sampler.store_env_gaussian_output(env_output, output, batch)


def render_frame_split_timed(sampler, batch):
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    evs[0].record()
    viewpoint_camera = to_x(prepare_gaussian_camera(batch), torch.float)
    ray_o, ray_d, coords, _, _, _ = sampler.get_camera_rays(
        batch,
        n_rays=sampler.n_rays,
        patch_size=sampler.patch_size,
    )
    evs[1].record()
    dif_output = sampler.render_gaussians(
        viewpoint_camera,
        sampler.pcd,
        sampler.pipe,
        sampler.bg_color,
        sampler.scale_mod,
        override_color=None,
    )
    output = sampler.store_dif_gaussian_output(dif_output, batch)
    evs[2].record()
    ref_o, ref_d = sampler.get_reflect_rays(ray_o, ray_d, coords, output, batch)
    evs[3].record()
    env_output = sampler.diffop.render_gaussians(
        viewpoint_camera,
        ref_o,
        ref_d,
        sampler.env,
        sampler.pipe_env,
        sampler.env_bg_color,
        0,
        start_from_first=False,
        scaling_modifier=sampler.scale_mod,
        override_color=None,
        batch=batch,
    )
    output = sampler.store_env_gaussian_output(env_output, output, batch)
    evs[4].record()
    _ = output.rgb_map
    evs[5].record()
    torch.cuda.synchronize()
    return {
        "cam_rays_ms": evs[0].elapsed_time(evs[1]),
        "dif_ms": evs[1].elapsed_time(evs[2]),
        "reflect_ms": evs[2].elapsed_time(evs[3]),
        "env_ms": evs[3].elapsed_time(evs[4]),
        "post_ms": evs[4].elapsed_time(evs[5]),
        "direct_ms": evs[0].elapsed_time(evs[5]),
    }


def benchmark_case(make_batch, runner, compare_model: bool, warmup: int, iters: int):
    sampler = runner.model.sampler
    total_iter = runner.total_iter
    iter_value = getattr(runner, "_bench_iter", 0)

    for _ in range(warmup):
        batch = prime_batch(make_batch(), iter_value, total_iter)
        render_frame_split(sampler, batch)
        if compare_model:
            batch = prime_batch(make_batch(), iter_value, total_iter)
            _ = runner.model(batch)
    torch.cuda.synchronize()

    totals = dotdict(batch_ms=0.0, cam_rays_ms=0.0, dif_ms=0.0, reflect_ms=0.0, env_ms=0.0, post_ms=0.0, direct_ms=0.0)
    if compare_model:
        totals.model_ms = 0.0

    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = make_batch()
        torch.cuda.synchronize()
        totals.batch_ms += (time.perf_counter() - t0) * 1000.0

        batch = prime_batch(batch, iter_value, total_iter)
        direct_stats = render_frame_split_timed(sampler, batch)
        for key, value in direct_stats.items():
            totals[key] += value

        if compare_model:
            batch = prime_batch(make_batch(), iter_value, total_iter)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = runner.model(batch)
            end.record()
            torch.cuda.synchronize()
            totals.model_ms += start.elapsed_time(end)

    return dotdict({k: v / iters for k, v in totals.items()})


def run_config_benchmark(local_args, engine_argv):
    runner, _, epoch = load_config_runner(engine_argv)
    reflection_start = getattr(runner.model.sampler, "render_reflection_start_iter", 0)
    runner._bench_iter = max(epoch * runner.ep_iter, reflection_start + 1)
    dataset = runner.val_dataloader.dataset
    window_size = parse_hw(local_args.bench_window_size)

    source = local_args.bench_source or "viewer_init"
    if source == "viewer_init":
        cases = [dotdict(source="viewer_init", view_index=None, frame_index=local_args.bench_frame)]
    else:
        views = parse_view_list(local_args.bench_views, dataset.Hs.shape[0])
        cases = [dotdict(source="val", view_index=view_index, frame_index=local_args.bench_frame) for view_index in views]

    print(f"benchmark window_size={window_size[0]},{window_size[1]} source={source} warmup={local_args.bench_warmup} iters={local_args.bench_iters}")
    if local_args.bench_compare_model:
        print("comparing split direct path against full model(batch) on the same camera batches")

    for case in cases:
        def make_batch():
            batch, _ = make_viewer_style_batch(dataset, window_size, case.source, case.view_index, case.frame_index)
            return batch

        _, label = make_viewer_style_batch(dataset, window_size, case.source, case.view_index, case.frame_index)
        stats = benchmark_case(make_batch, runner, local_args.bench_compare_model, local_args.bench_warmup, local_args.bench_iters)
        stats_str = " ".join(f"{key}={value:8.3f}ms" for key, value in stats.items())
        print(f"{label}: {stats_str}")


def run_dummy_benchmark():
    ckpt_path = os.getenv("EVG_CKPT_PATH", "/home/roman/ba/envgs/data/trained_model/envgs_sedan_mast3r/latest.pt")
    pcd, env = load_splats(ckpt_path)

    H, W = 1080, 729
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
    t1 = time.perf_counter()
    render_frame(pcd, env, cam)
    print(time.perf_counter() - t1)


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    local_args, engine_argv = parse_local_args(argv)
    use_config_benchmark = local_args.bench_source is not None or any(arg in {"-c", "--config"} for arg in engine_argv)
    if use_config_benchmark:
        run_config_benchmark(local_args, engine_argv)
    else:
        run_dummy_benchmark()

if __name__ == "__main__":
    main()

# frametime is 150-160ms
