import numpy as np
import math
import os
from contextlib import contextmanager
import torch
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.gaussian2d_utils import GaussianModel
from easyvolcap.utils.gaussian2d_utils import render
from easyvolcap.utils.optix_utils import HardwareRendering
from easyvolcap.utils.ray_utils import get_rays

import viser
import nerfview
import time

# Profile python: sudo py-spy top --pid {PID}
#
#TODO for optimization:
# 1. [x]Implement bounds for the pcd splat to render only the car itself
# 2. Trace rays only where the specular mask is above a threshold
# 3. Benchmark CPU/GPU time using Nsight
# 4. Look into making a custom renderer, reimplementing everything in Vulkanб D3D12 or CUDA completely. How does 2d surfel rasterization work right now?
# 5. Bake in the env gaussian to an. image, from the center of the car
#
# Speed up compilation in cloud: ensure ninja (apt-get install ninja-build)
#
# IDEAS: (not verified)
# Normals are problematic
# Understand the training process and training metrics better
# Disable SH at train time or penalize SH values. The reflection_mask is very noisy and patchy, but the surface isn't.
# Maybe SH deal with reflections partially and make it difficult for gradients to flow to the env gaussian and the reflection_mask
# Try 3 specular channels?
# 
#
# TODO:
# 1. Train with all our datasets using the sedan config
# 2. Better priors, add depth, mask priors??
# 3. Train with full resolution (lots of RAM needed)
# 4. Adapt gaussian density???? To have more gaussians close up without that much sacrifices, maybe do LODs
# 5. Check if maybe the trained surface normals are still the problem (Solution: better surface, for example SolidGS instead of 2dgs)
# 6. [x] Dellensegel instead of ENV gs
# 7. Change initial specular value.
# 8. Implement segmented car loss (specular mask or general loss weight)

#TODO GUI:
# 1. [x] Find minimal code to load the model and render a frame
# 2. [x] Stick it into nerfview, see what fps we get compared to native evc-gui
# 3. [x] Setup up good camera
# 4. [x] Add UI elements for switching between render modes
# 5. Render Train cameras overlay
# 6. Enable switching between gt cameras/views


#TODO 26-30.01.2026
# [x] Deal with init envgs point clouds (Open3D)
# [x] Generate data configs for all HUK datasets with stablenormal
# [x] Create a list of experiments to run (ex. sedan with metric3d, our datasets with sedan config, ...)
# [x] Run experiments
# [] Get Metric3D VIT giant to work (using the training checkpoint. Handle vertical high-res images correctly)
# [] Test training in AWS
# [x] Print out and read envgs, materialrefgs, 2dgs papers
# [] Understand and redocument the training process and differnet losses at all stages of envgs

# Experiments
# [x] test sedan -> huk_scenes with sedan config
# [x] test sedan 0.25x -> sedan 0.5x
# [] test sedan stablenormal -> sedan metric3d/DTK
# [] test sedan init_specular 0.001 -> sedan init_specular 0.1/0.01
# [] test sedan envgs -> sedan materialrefgs ref mask

torch.set_grad_enabled(False)

TRACER = HardwareRendering().cuda()

def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cx, -sx],
                   [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0],
                   [sz, cz, 0.0],
                   [0.0, 0.0, 1.0]])

    return Rz @ Ry @ Rx


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

def apply_origin_transform(
    c2w: np.ndarray,
    rot_xyz_deg: tuple[float, float, float],
    origin_offset: tuple[float, float, float],
) -> np.ndarray:
    
    if rot_xyz_deg is None and origin_offset is None:
        return c2w
    
    rx, ry, rz = rot_xyz_deg
    ox, oy, oz = origin_offset

    if rx == 0.0 and ry == 0.0 and rz == 0.0 and ox == 0.0 and oy == 0.0 and oz == 0.0:
        return c2w
    
    R = rotation_matrix_xyz(rx, ry, rz)
    t = np.array([ox, oy, oz], dtype=c2w.dtype)

    c2w = c2w.copy()
    c2w[:3, :3] = R @ c2w[:3, :3]
    c2w[:3, 3] = R @ c2w[:3, 3] + t

    return c2w

def get_render_inputs(W, H, c2w, K):
    Rc = c2w[:3, :3]
    tc = c2w[:3, 3]

    R = Rc.T  #w2c
    T = (-R @ tc).reshape(3, 1)  #w2c

    cam_dict = make_camera_batch(
        torch.tensor(H, dtype=torch.float32), 
        torch.tensor(W, dtype=torch.float32), 
        torch.tensor(K, dtype=torch.float32), 
        torch.tensor(R, dtype=torch.float32), 
        torch.tensor(T, dtype=torch.float32), 
    )
    
    return cam_dict

def make_camera_batch(H, W, K, R, T, n=0.1, f=10.0):
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

    # if 'msk' in batch: output.gt_alpha_mask = batch.msk[0]  # FIXME: whatever for now

    output.world_view_transform = getWorld2View(R, T).transpose(0, 1).cuda()
    output.projection_matrix = getProjectionMatrix(FoVx, FoVy, n, f).transpose(0, 1).cuda()
    output.full_proj_transform = torch.matmul(output.world_view_transform, output.projection_matrix).cuda()
    output.camera_center = (-R.mT @ T)[..., 0].cuda()  # B, 3, 1 -> 3,

    # Set up rasterization configuration
    output.FoVx = FoVx.cuda()
    output.FoVy = FoVy.cuda()
    output.tanfovx = math.tan(FoVx * 0.5)
    output.tanfovy = math.tan(FoVy * 0.5)

    output.znear = n.cuda()
    output.zfar = f.cuda()

    return output

def load_splats(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")["model"]
    
    # initialize GaussianModel
    pcd = GaussianModel(
            xyz=torch.empty(1, 3, device="cuda"),
            colors=None,
            sh_degree=3,
            render_reflection=True,
            specular_channels=True,
            xyz_lr_scheduler=None,  # key fix: avoid lr_init/lr_final access
        )

    # initialize GaussianModel
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

    # Dump trained parameters to the GaussianModel instances
    pcd.load_state_dict(loaded_params_pcd, strict=False)
    env.load_state_dict(loaded_params_env, strict=False)

    return pcd.cuda(), env.cuda()

# Copied from easyvolcap/easyvolcap/utils/math_utils.py
@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

from enum import Enum, auto
class RenderPass(Enum):
    COMBINED = auto()
    DIFFUSE = auto()
    SPECULAR = auto()
    SPECULAR_MASK = auto()
    NORMAL = auto()
    SURF_DEPTH = auto()
    SURF_NORMAL_FROM_DEPTH = auto()
    DISTORTION = auto()
    ALPHA = auto()

def render_frame(pcd, env, cam, renderpass=RenderPass.COMBINED, render_stripped_env_only=False, stripes_freq=100):
    # Diffuse pass (rasterizer) for base gaussians (pcd gaussians)
    pipe_dif = dotdict(convert_SHs_python=True, compute_cov3D_python=False, depth_ratio=0.0)
    bg = torch.zeros(3, device="cuda")
    dif = render(cam, pcd, pipe_dif, bg)
    dif_rgb = dif.render.permute(1, 2, 0)
    norm_map = dif.rend_normal.permute(1, 2, 0)
    dpt_map = dif.surf_depth.permute(1, 2, 0)
    spec_map = dif.specular.permute(1, 2, 0)
    surf_norm_map = dif.surf_normal.permute(1, 2, 0)
    distortion_map = dif.rend_dist.permute(1, 2, 0)
    alpha_map = dif.rend_alpha.permute(1, 2, 0)


    # Calculate ray directions per pixel
    w, h = cam.image_width.item(), cam.image_height.item()
    ray_o, ray_d = get_rays(h, w, cam.K, cam.R, cam.T, correct_pix=True, z_depth=True)

    # Calculate reflected rays in screen space
    norm = normalize(norm_map)
    ref_d = ray_d - 2 * (ray_d * norm).sum(-1, keepdim=True) * norm
    ref_o = ray_o + ray_d * dpt_map
    
    if not render_stripped_env_only:
        # Trace reflected rays through the env gaussians using OptiX
        pipe_env = dotdict(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0)
        env_bg = torch.zeros(3, device="cuda")
        env_out = TRACER.render_gaussians(
            cam, ref_o, ref_d, env, pipe_env, env_bg,
            max_trace_depth=0,
            specular_threshold=0.0,
            start_from_first=False,
            batch=cam,
        )
        env_rgb = env_out.render.permute(1, 2, 0)

    else:
        #TODO: Check math. Code written by ChatGPT
        x, y, z = ref_d[...,0], ref_d[...,1], ref_d[...,2]
        
        lon = torch.atan2(x, z)              # [-pi, pi], seam at pi/-pi
        s = 0.5 + 0.5 * torch.sin(stripes_freq * lon)   # N = stripe frequency around 360

        # sharpen (smooth binary)
        sharp = 10.0
        s = s**sharp / (s**sharp + (1-s)**sharp + 1e-8)

        # Other stripes. TODO: Check what fits us best
        # a = torch.tensor([1.0, 0.0, 0.0], device=ref_d.device)  # choose stripe axis
        # t = (ref_d * a).sum(dim=-1).clamp(-1, 1)                # [-1, 1]
        # s = 0.5 + 0.5 * torch.sin(stripes_freq * t)                        # K controls density

        env_rgb = torch.Tensor([0,0,0]).cuda()*(1-s)[...,None] + torch.Tensor([1,1,1]).cuda()*s[...,None]


    # extra logic for debugging
    match renderpass:
        case RenderPass.DIFFUSE:
            return (dif_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        case RenderPass.SPECULAR:
            return (env_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        case RenderPass.NORMAL:
            return (((norm_map + 1) * 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        case RenderPass.SURF_NORMAL_FROM_DEPTH:
            return (((surf_norm_map + 1) * 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        case RenderPass.SPECULAR_MASK:
            return (spec_map.expand(*spec_map.shape[:2], 3).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        case RenderPass.SURF_DEPTH:
            return (dpt_map.expand(*dpt_map.shape[:2], 3).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        case RenderPass.DISTORTION:
            return (distortion_map.expand(*distortion_map.shape[:2], 3).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        case RenderPass.ALPHA:
            return (alpha_map.expand(*alpha_map.shape[:2], 3).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        case _:
            pass

    # Blend
    rgb = (1.0 - spec_map) * dif_rgb + spec_map * env_rgb
    rgb8 = (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return rgb8

def main():
    pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/envgs_sedan/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/envgs_sedan_50/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/audi_silver/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/mitsubishi_totaled_rerun/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/vw_rain/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/bmw_rain/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/hyundai_white/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/mercedes_orbit_1/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/mercedes_outside_orbit_2/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/audi_nbg/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/bmw_nbg/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/renault_white_rain/latest.pt")

    # Baselines (sedan config):
    # [x] audi_nbg
    # [x] bmw_nbg
    # [x] mercedes_outside_orbit_2
    # [x] renault_white_rain
    # [x] audi_silver
    # [x] bmw_rain
    # [x] hyundai_white
    # [x] mercedes_outside_orbit_1
    # [x] mitsubishi_totaled
    # [x] vw_rain

    # Initialize a viser server and our viewer.
    server = viser.ViserServer(verbose=True)

    # GUI
    with server.gui.add_folder("Camera Origin"):
        rot_x=server.gui.add_slider("Origin Rot X (deg)", -180.0, 180.0, 1.0, 116.0)
        rot_y=server.gui.add_slider("Origin Rot Y (deg)", -180.0, 180.0, 1.0, 0.0)
        rot_z=server.gui.add_slider("Origin Rot Z (deg)", -180.0, 180.0, 1.0, 0.0)
        off_x=server.gui.add_slider("Origin X", -10.0, 10.0, 0.01, 0.0)
        off_y=server.gui.add_slider("Origin Y", -10.0, 10.0, 0.01, 1.0)
        off_z=server.gui.add_slider("Origin Z", -10.0, 10.0, 0.01, 1.0)
    
    with server.gui.add_folder("Splat Controls"):
        aabb_x = server.gui.add_multi_slider("AABB X", -10, 10, 0.1, (-2.5, 2.5))
        aabb_y = server.gui.add_multi_slider("AABB Y", -10, 10, 0.1, (-2.5, 2.5))
        aabb_z = server.gui.add_multi_slider("AABB Z", -10, 10, 0.1, (-2.5, 2.5))
        render_pass = server.gui.add_dropdown("Render Pass", [render_pass.name for render_pass in RenderPass], initial_value=RenderPass.COMBINED.name)
        sh_toggle = server.gui.add_checkbox("Disable SH", initial_value=False)
        stripped_env = server.gui.add_checkbox("Enable stripes", initial_value=False)
        stripped_env_freq = server.gui.add_slider("Stripe frequency", min=0, max=500, step=1, initial_value=100)

    def render_fn(
        camera_state, render_tab_state
    ) -> np.ndarray:
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        
        c2w = camera_state.c2w
        K = camera_state.get_K([width, height])

        # Camera Origin Transform
        rot_xyz = (rot_x.value, rot_y.value, rot_z.value)
        origin_offset = (off_x.value, off_y.value, off_z.value)

        c2w = apply_origin_transform(c2w, rot_xyz, origin_offset)

        cam = get_render_inputs(width, height, c2w, K)
        img = render_frame(pcd, env, cam, RenderPass[render_pass.value], render_stripped_env_only=stripped_env.value, stripes_freq=stripped_env_freq.value)
        return img

    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode='rendering')

    # GUI LOGIC
    def _apply_aabb(_=None):
        xmin, xmax = aabb_x.value
        ymin, ymax = aabb_y.value
        zmin, zmax = aabb_z.value
        
        with viewer.lock:
            pcd.set_aabb_mask(
                min(-0.5, xmin),
                max(0.5, xmax), 
                min(-0.5, ymin), 
                max(0.5, ymax), 
                min(-0.5, zmin), 
                max(0.5, zmax))
        
        viewer.rerender(_)
    
    def toggle_sh_effects(_=None):
        with viewer.lock:
            pcd.toggle_sh()
        viewer.rerender(_)

    aabb_x.on_update(_apply_aabb)
    aabb_y.on_update(_apply_aabb)
    aabb_z.on_update(_apply_aabb)

    render_pass.on_update(lambda _: viewer.rerender(_))
    stripped_env.on_update(lambda _: viewer.rerender(_))
    stripped_env_freq.on_update(lambda _: viewer.rerender(_))
    sh_toggle.on_update(toggle_sh_effects)

    # NerfView GUI overrides
    viewer._rendering_tab_handles["viewer_res_slider"].value = 1024
    viewer._rendering_folder.expand_by_default = False
    
    # Apply AABB from the start
    _apply_aabb()

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
