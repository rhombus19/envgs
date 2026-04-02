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
# 4. Look into making a custom renderer, reimplementing everything in Vulkan, D3D12 or CUDA completely. How does 2d surfel rasterization work right now?
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
# 1. [x]Train with all our datasets using the sedan config
# 3. [x]Train with full resolution (lots of RAM needed)
# 6. [x] Dellensegel instead of ENV gs
# 2. Better priors, add depth, mask priors??
# 4. Adapt gaussian density???? To have more gaussians close up without that much sacrifices, maybe do LODs
# 5. Check if maybe the trained surface normals are still the problem (Solution: better surface, for example SolidGS instead of 2dgs)
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
# [x] Print out and read envgs, materialrefgs, 2dgs papers
#
#
# TODO 02-06.02.2026:
# [x] Training in AWS
# [x] Do sedan baseline again to confirm what changed in repo
# [] Get Metric3D VIT giant to work (using the training checkpoint. Handle vertical high-res images correctly)
# [] (1.2) -> 1. Find a good test scene (e.g. mercedes_orbit) 2. Explicitly cut out a full orbit out of the video, sample frames from it, run SFM
# [] Understand and redocument the training process and differnet losses at all stages of envgs
#
# Maybe the performance is worse because we have objects near the car that reflect, but aren't in env gaussians
# We have a lot of reflective objects in the background, compared to sedan, which only has trees
# Segment background from car, use only segmented car gt images for training, change loss to ignore background. This way we only have the reflections to learn the env gaussian from

# But there is still a difference between sedan,hatchback and our datasets
# get a better understanding of our dataset (view all frames, maybe try colmap instead of mast3r, maybe the cam positions are wrong)



# Choose 1 dataset from our data to test -> VW Rain
# Hypotheses:
# 1. [x] Ray tracer code has changed for our scenes -> 0.5db change can explain bad performance?
# 2. [x] Reflective objects in the background (close to main object) make it harder to split base/env gaussian -> it has worked for hatchback
# 3. [x] Neural SfM is wrong. Use classical SfM or sampling is bad
# 4. Capture method is too bad. Test: drop some gt views in sedan dataset, see what happens, or recapture a car the same way they've done in the paper
# 5. User is reflected in car -> Inpaint myself in gt views or do another capture
# 6. Normal map too smooth, norm_loss is always active, which means the optim can't introduce surface details
#
# Symptoms:
# 1. Very noisy trained surface normals
# 2. Artifacts in trained geometry (Reflective surfaces become holes, weird artifacts)
# 3. Specular mask is almost 0
#
# NOTE: Try to disable norm_loss at certain level
# NOTE: Add depth prior?
# TODO:
# 0. [x] Train sedan with Mast3r +0.4db
# 1. [x] Train VW with old raytracer +0.3db
# 2. [x]Sample more views (300), try to fix optical flow or sample differently
# 3. [x] Try denser init env gs (look at init_env_points() in envgs_sampler.py. The args to sample_points_subgrid()) -0.1db
# 4. Segment car in gt frames, train with that as the base gaussian
# 5. Inpaint myself in gt images
# 6. Redo capture in an open field, test different capture methods, maybe train mercedes orbit 1/2
# 7. [x] Increase lr, more time for optim to figure it out
# 8. Document findings, the entire process with screenshots and so on
#
#
# What explains the fact that our datasets have such bad quality -> Env gaussian and reflections don't contribute Why?
# -> Specular mask is way too low valued, reflections don't attribute anything to the final frame. Why? -> 
# The raytraced reflections don't explain ground-truth reflections. Why? -> The reflection direction is incosistent/wrong. Why?
# -> The trained normals per gaussian are wrong/incosistent. Why? -> because the prior normals are very inconsistent and they are strong supervisors at train time. Why?
# -> because the gt frames have captured only a small portion of a car and the normal generation model can't figure out what it is 
#
#
# Metric3D, DKT, StableNormal(up diffusion steps, implement tiles?)
#
# The issue with the video: Droplets, fast pan moves, too close perspectives
# Synthetic data: Test white/black cars, test capture methods, person vs no-person
#
# Test capture methods:
# 1. Vertical orbit vs Horizontal orbit
# 2. 3 layers vs 5 layers
# 3. further away vs close-up 
#
# What is the right BRDF for us? One that has clearcoat, PBR Reflections.
# Is Fresnel really that important for cars? In gt views we try not to take pictures with strong angles
# What is our limiting factor? Why is the ref mask so bad?
#
# TODO 19.02-25.02
# 1. [x] Train with matthias_car, look if the problem was actually vertical format
# 2. [x] Process matthias_car, sedan, vw_rain through DKT/Metric3D
# 1. [x] Document DKT/Metric3D/StableNormal for matthias_car, sedan, vw_rain
# 2. [x] Analyze matthias_car and radu_vw
# 3. [x] Can we save one of the huk_scenes datasets with damaged cars? -> Probably no, we can try to train with metric3D on known-good datasets to see its impact, but Metric3D is not as sharp as StableNormal. Maybe we can try the normal map consistency notebook
# 4. [x] Do we have a good synthetic dataset with damage we can train with? -> red car
# 5. [x] What do we need from Experts? Show them the idea, demos, ask what their pain points are, what's difficult for them, how can they use it, our ideas at the end. (Formative study, future outlook, goal: use cases for 3DGS in insurance)
# 6. Can we train with higher res or by cutting out the car out of images to achieve higher detail?
# 7. How do we clearly define the scope of the BA, what to research, what not to research until the deadline?
# 8. Document findings, future directions
#
#
# TODO 26.02-05.03
# 1. [x] Text Jaro Dobry, schedule new capture, use constant wb, exposure settings
# 2. [x] Render red glossy cars
# 3. [x] Invite Prof. Meyer to the meetup at wednesday
# 4. [x] Train vw_rain, sedan with Metric3D normal priors
# 5. [x] Run StableNormal with same seed, same input noise latent
# 6. [] Run StableNormal with same seed, denoised latent from prev frame taken as start noise latent (maybe add some noise to it)
# 7. [x] Train sedan with 0.1 init specular value
# 8. [x] Train sedan with 0.5 ratio and set max number of gaussians from 1.8M to 3.6M
# 9. [o] Cutout car from background in sedan, enable transparent bg in envgs, train sedan, compare details
# 10.[] Train sedan without bg on mast3r capture (more initial points on the actual car)
# 11.[] Implement spec mask prior loss like the normal prior one, run a segmentation model on car body panels, ommit a mask for reflective body panels, train
# 12.[] Implement a PBR BRDF that supports clearcoat, eyeball car paint material properties, generate material priors for them
# 13.[] Implement annealing of the normal prior loss, so that after 30-40k steps it stays small
# 14.[] Look at SpotlessSplats, figure out how to apply it to sedan (DINO features -> seg mask -> loss weight) and test it
#   
# Before finishing:
# 1. Freeze env afer 30k steps, disable norm prior loss, train normals with freezed env
# 2. Research normal estimation for multi-view/ SVFormer
# 3. Test BRDF
# 4. Termine (HUK Autowelt, SP + Prof in die Werkstatt)
# 5. Formative evaluation
# 6. Test what happens when you add closeups of damaged areas
# 7. Run matthias without badly estimated views, maybe the problem is really the mast3r script and that's why synthetic data didn't work that well
# 8. Why don't synthetic captures work? 
# 9. Implement disk view like in 2DGS paper (no transparency, smaller)
#
# TODO 5.03-12.03
# [] Deep Dive: Normal map consistency metric
# [] Deep Dive: car part segmentation model
# [o] Deep dive: Read papers, figure out how to include reflection distortion information in training
# [] Start training with sedan no bg COLMAP and MAST3R
# [x] Map out the Bachelorarbeit, what to write
#
#
# TODO:
# [] What are the most important arguments in the discussion that I want to make
# [] What and how many experiments do I need to run to make those arguments (3 datasets per test)
# [] Reduce scope
# Work backwards from discussion, arguments back to experiments, preliminaries
#
# Maybe we try materialrefgs once again, but we use segmented mask as reflection score
#
# How to deal with dents/surface imperfections?
# People do it by looking at the reflections and seeing how they move, this principle is used in deflectometry, where we know already how the reflection supposed to look like
# In EnvGS our surface is basically defined by the single-view normal estimation model, which kinda defeats the purpose of finding dents. 
# It works to get a general sense of how the surface looks like, estimating the environment reflected in the scene, but we are still relying on single view inaccurate normal maps
# The other source of signal we have are the actual rendered gt views. However the photometric loss delivers too little signal to the normals, because we don't know the environment
# So the problem is underconstrained, if we only have single rendered gt views to compare our rendered views against (Can we measure that? Magnitude of gradients?)
# We can try to freeze the env and let normals be fitted, but we also don't know if our environment is correct. And our surface parameters are also underconstrained
# Righ now we basically use AI generated surface estimation, which uses single views. We could train a better model that does detect surfaces better, but doing it with a single view is inherently ambigious and difficult
#
# Another option is to track points/lines in the reflections and do deflectometry on them or train a model that understands how objects should look like and detects when they are deformed by reflection, then it could estimate the surface
# Do people look at objects themselves or just the distorted "look" or probably more like the change in motion
#
# GUI?
#
# 
# BONUS:
# [] Render out a car with specifically small dents on the door, train with it
# [] Generate an error map to see what areas of the car contribute to the bad PSNR -> already kinda there with the results folder in data/ , but not helpful for me
# [x] Train sedan with max sh degree set to 0
# [x] Train sedan with no sh and 0.1 init specular
# [x] Train red_vw_synthetic with envgs_synth model config
#  
#
# ANALYSIS
# [] Why didn't training with 0.1 init specular improve performance? Did it impact training at all? Was there a mistake?  -> the num_pts for both pcd and env spike to max in this run
# [] Why vw_metric3d didn't improve performance?
# [] Why sedan_metric3d improved performance? Is it because of the windows? 
#
# For dent detection: Basically we model dents pretty much based on the normal map predictions, which are inherently flawed. But, can we really make the optimizer do a dent pass that explicitly tries to correct normals locally and detect dents
#
# It would be cool to test how consistent the normals are for each tracked point from mast3r, mb segment it into parts, visualize 
#
# We need a normal estimation model that understands multi-view for sure. A lot of the things might not be visible from just one view. And this is kind the whole point of what we are doing
# The other option is to use priors really only for the start and train for a lot of steps without priors, but then the problem will be kinda underconstrainted
#
# A scene with a car in the center can be constrained more to achieve better quality. Things like reflective strength 
# Or maybe we should explicitly factor out the car paint material and have things like global roughness, metallness. 
# Nah, for things like dirt the optim should be able to vary these things locally. But also not due to noise, duh \_0_O_/ 
#
# Are camera parameters at fault? Is that the difference between sedan and our data? https://github.com/nv-tlabs/ppisp
# BRDF/segment car/camera parameters/init specular mask/increase max num of gaussians/ Solve BA for exposure and white balance?
# Spherical harmonics do capture white balance and exposure changes, but its still confusing for the model
#
# We can project the estimated normal maps into 3d from gt frames and look at their consistency
#
# Talk with someone from HUK Autowelt
#
# PLAN:
# 1. [x] Bring our datasets to sedan baseline performance
#       1.1 Initial env bounds: tune them manually, retrain
#           Env gaussian didn't learn at all, it also appears to not cover the entire scene
#       1.2 Cameras: fix blurred frame detection and optical flow sampling, maybe sample by hand first
#           Our frame sampling is very inconsistent, not a smooth ellipse around the car like sedan
#       1.3 Normals: Try other normal estimators, compare, tune StableNormal/run stablenormal on sedan imgs
#           The normal maps in our dataset are significantly less detailed and inconsistent, learned normals are also very bad
# 2. Improve sedan baseline performance
#       2.1 Better normals: test Metric3D and DKT (maybe add more weight to them if they are good, but they also have to be multi-view consistent)
#           - Start with same latent noise for all frames
#           - Start with frame[n-1] result as latent noise
#           - Test Metric3D
#       2.2 Higher initial spec value
#       2.3 car mask/body part mask as specular value loss
#       2.4 Figure out why sedan 0.5 ratio performed worse. Increase max number of gaussians??(curerntly 1.800.000)
#       2.5 Cutout background, train only on the car
#       2.6 Anneal normal map loss
#       2.7 PBR BRDF with clear coat + material map priors
#       2.8 integrate PPISP or take video with constant camera params
#       2.9 User is reflected in car, so Inpaint myself in gt views or do another capture or use (DINO features -> seg mask -> loss weight)
#       ~~~~~~Check if maybe the trained surface normals are still the problem (Solution: better surface, for example SolidGS instead of 2dgs)
#       ~~~~~~Play with training process and co-optimization of SH and Ray tracing. e.g. disable SH completly, see what happens, or set weights. 
#       ~~~~~~Idea is, to not let the optimizer fall into a local minimum by optimizing SH at first (Already kinda implemented, still SH take on some reflections. Maybe because of bad normals and spec map)
#       ~~~~~~Multi-view consistency loss like materialrefgs, maybe implement other parts of materialrefgs
# 3. GUI
#       3.1 Implement switching of scenes
#       3.2 Implement gt cameras and gt images overlay
#       3.3 Bake env gaussian into an env map
#       3.4 Cast rays only in tiles where spec mask is bigger than T
#       3.5 [Stretch goal] Develop a custom renderer for 2dgs, implement splat streaming and LODs
#       3.6 [Stretch goal] Develop a dashboard with visualizations for each step in the pipeline and make each step run in the background
#
# 4. General
#       4.1 Come up with a video + SFM points-based GUI to compete with envgs
#       4.2 Test images vs SFM video vs full splat. Our goal is a better teleexpertise. Splats themselves can be used for difficult claims or for huk autowelt
#
#
# 
#
# Experiments
# [x] test sedan -> huk_scenes with sedan config
# [x] test sedan 0.25x -> sedan 0.5x
# [x] test sedan stablenormal -> sedan metric3d/DTK
# [x] test sedan init_specular 0.001 -> sedan init_specular 0.1/0.01
# [] test sedan envgs -> sedan materialrefgs ref mask
#
# TO CHECK:
# 1. Initial ENV gaussian:
#       - initial bounds are too small
#       - env gaussian didn't learn at all
# 2. Colmap capture (views are too sparse)
#       - sampling is not consistent
# 3. Normal priors
#       - Metric3D or DKT?
# 4. Vertical video?


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
            sh_degree=state['sampler.pcd.active_sh_degree'],
            render_reflection=True,
            specular_channels=True,
            xyz_lr_scheduler=None,  # avoid lr_init/lr_final access
        )

    # initialize GaussianModel
    env = GaussianModel(
            xyz=torch.empty(1, 3, device="cuda"),
            colors=None,
            sh_degree=state['sampler.env.active_sh_degree'],
            render_reflection=False,
            specular_channels=True,
            xyz_lr_scheduler=None,  # avoid lr_init/lr_final access
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
        #TODO: Refactor, add comments
        x, y, z = ref_d[...,0], ref_d[...,1], ref_d[...,2]
        
        lon = torch.atan2(x, z)
        s = 0.5 + 0.5 * torch.sin(stripes_freq * lon)   # N - stripe frequency around 360

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
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/matthias_vw_metric3d_fixed_50/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/red_vw_envgs_synth_model_conf/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/sedan_no_sh_01_init_specular/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/envgs_sedan/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/envgs_sedan_mast3r/latest.pt")
    pcd, env = load_splats("/mnt/e/BA/results/data/trained_model/yellow_van_small_dent_lotusv2/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/sedan_no_bg_01_spec/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/mercedes_a_class_stablenormal/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/white_van_stablenormal/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/yellow_van_stablenormal/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/radu_mercedes/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/matthias_vw/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/envgs_sedan/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/envgs_sedan_rerun/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/envgs_sedan_50/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/audi_silver/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/mitsubishi_totaled_rerun/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/bmw_rain/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/hyundai_white/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/mercedes_orbit_1/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/mercedes_orbit_2/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/audi_nbg/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/bmw_nbg/latest.pt")
    # pcd, env = load_splats("/home/roman/ba/envgs/data/trained_model/renault_white_rain/latest.pt")

    # Observation: 
    #
    # sedan          res: 336x220; frame time: 0.032398
    # sedan          res: 1024x669; frame time: 0.114060
    # renault white: res: 45x30;   frame time: 0.216973
    # renault white: res: 1024x669; frame time: 0.066036   <--- wtf?
    #
    # When I disable raytracing by enabling stripes, the frame times stay the same
    # sedan         has 1 800 000 base gaussians, 507497 env gaussians
    # renault white has 1 800 000 base gaussians, 81510 env gaussians
    #
    # Interpretation: renault white concentrates the splats on the car to reproduce reflections instead of relying on the raytracing
    #
    # Baselines (sedan config):
    #[x] audi_nbg
    #[x] bmw_nbg
    #[x] mercedes_outside_orbit_2
    #[x] renault_white_rain
    #[x] audi_silver
    #[x] bmw_rain
    #[x] hyundai_white
    #[x] mercedes_outside_orbit_1
    #[x] mitsubishi_totaled
    #[x] vw_rain

    # Initialize a viser server and our viewer.
    server = viser.ViserServer(verbose=True)

    # GUI
    with server.gui.add_folder("Splat Stats"):
        base_splats_rendered = server.gui.add_number("Base splats rendered", initial_value=0, min=0, step=1, disabled=True)
        env_splats_total = server.gui.add_number("Env splats total", initial_value=0, min=0, step=1, disabled=True)

    with server.gui.add_folder("Camera Origin"):
        rot_x=server.gui.add_slider("Origin Rot X (deg)", -180.0, 180.0, 1.0, 116.0)
        rot_y=server.gui.add_slider("Origin Rot Y (deg)", -180.0, 180.0, 1.0, 0.0)
        rot_z=server.gui.add_slider("Origin Rot Z (deg)", -180.0, 180.0, 1.0, 0.0)
        off_x=server.gui.add_slider("Origin X", -10.0, 10.0, 0.01, 0.0)
        off_y=server.gui.add_slider("Origin Y", -10.0, 10.0, 0.01, 1.0)
        off_z=server.gui.add_slider("Origin Z", -10.0, 10.0, 0.01, 1.0)
    
    with server.gui.add_folder("Splat Controls"):
        aabb_x = server.gui.add_multi_slider("AABB X", -50, 50, 0.1, (-2.5, 2.5))
        aabb_y = server.gui.add_multi_slider("AABB Y", -50, 50, 0.1, (-2.5, 2.5))
        aabb_z = server.gui.add_multi_slider("AABB Z", -50, 50, 0.1, (-2.5, 2.5))
        render_pass = server.gui.add_dropdown("Render Pass", [render_pass.name for render_pass in RenderPass], initial_value=RenderPass.COMBINED.name)
        sh_toggle = server.gui.add_checkbox("Disable SH", initial_value=False)
        stripped_env = server.gui.add_checkbox("Enable stripes", initial_value=False)
        stripped_env_freq = server.gui.add_slider("Stripe frequency", min=0, max=500, step=1, initial_value=100)

    def update_splat_stats():
        base_splats_rendered.value = int(pcd.get_xyz.shape[0])
        env_splats_total.value = int(env.get_xyz.shape[0])

    def render_fn(
        camera_state, render_tab_state
    ) -> np.ndarray:
        # t1 = time.perf_counter()
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
        # torch.cuda.synchronize()
        # print(f"res: {width}x{height}; frame time: {(time.perf_counter() - t1):04f}")
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
            update_splat_stats()
        
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
    # _apply_aabb()

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
