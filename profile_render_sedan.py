import math
from pathlib import Path

import numpy as np
import torch
import cv2

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.gaussian2d_utils import GaussianModel, render
from easyvolcap.utils.optix_utils import HardwareRendering
from easyvolcap.utils.ray_utils import get_rays


CKPT_PATH = "/workspace/latest.pt"
WARMUP = 3
REPEATS = 5

IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
REFERENCE_IMAGE_WIDTH = 1292
REFERENCE_IMAGE_HEIGHT = 839
REFERENCE_CX = 646.0
REFERENCE_CY = 419.5

# Extracted once from the sedan Mast3r dataset at ratio 0.25.
# These intrinsics correspond to the 1292x839 resized images, so output
# resolution changes should rescale pixel density without changing FoV.
# Each entry is (fx, fy, R, T) for one validation view.
VAL_CAMERAS = [
    (955.5135224407, 955.1292059117, [[0.9890409119983057, -0.04107428115427937, 0.1418131792931075], [-0.026269708136568256, 0.8962380276514441, 0.44279487375744175], [-0.14528584524302207, -0.4416676365991835, 0.8853370668581046]], [0.2719437216, -1.4898544212, 3.7450671647]),
    (952.6354103539, 952.2522514286, [[-0.4838536283677983, -0.5346608631845148, 0.6928372303031254], [0.08365683192920184, 0.7597965910128726, 0.6447562909788371], [-0.8711413207895065, 0.36992823853283663, -0.3229023034125135]], [0.4101435257, -1.7512063532, 4.2867763174]),
    (955.0497672388, 954.6656372365, [[-0.727662060104992, 0.29743821732054765, -0.6180925765295469], [-0.29061529980538103, 0.6825392666415064, 0.6705839969843472], [0.6213297625277479, 0.6675856921546846, -0.41021783216672003]], [-0.2920603347, -1.5380300791, 3.2056417014]),
    (953.4961191895, 953.1126140790, [[0.9684371196324216, 0.16040531684861198, -0.19078700072267807], [-0.055211272084469264, 0.884451328702487, 0.4633546833595405], [0.24306637108591034, -0.4381962819138265, 0.8653916788160289]], [0.5368830213, -1.3875679225, 4.1158580865]),
    (954.6239145650, 954.2399558446, [[0.0034893568992069197, -0.5442121394288143, 0.8389403862532444], [0.28293768101176486, 0.805201615090771, 0.5211493334150001], [-0.9591319476756651, 0.23554937137153859, 0.15678775651930948]], [0.0655863271, -1.8465377135, 3.8569830488]),
    (954.2419630034, 953.8581579075, [[-0.9921417881132474, -0.058993696612275444, -0.11033773624407461], [-0.124393546189975, 0.5598886962248496, 0.8191769610443889], [0.01345057417847783, 0.8264649972017306, -0.5628274073413737]], [0.1180907078, -1.5752337180, 2.9399157091]),
    (950.6581905349, 950.2758268660, [[0.019421248636355422, 0.5549552832468979, -0.8316534426657297], [-0.3814700155984036, 0.7729981436697879, 0.5069067932888801], [0.9241771704066047, 0.307406088878241, 0.2267113897002628]], [-0.3047622984, -1.8485157650, 3.3056796128]),
    (969.1459767125, 968.7561770821, [[0.7601781642288667, -0.3272525520020889, 0.561279721562921], [0.18387158074080245, 0.9369294802577706, 0.2972446649139099], [-0.6231535929652565, -0.12275551405954892, 0.772405776352344]], [0.0760343203, -1.4271242895, 3.4515435343]),
    (964.8934376516, 964.5053484324, [[-0.9148147932227955, -0.30021729517290563, 0.27015452944527607], [0.045434158188821044, 0.588162661378161, 0.8074654302384918], [-0.40130989443372467, 0.7509555642244246, -0.5244197833701252]], [0.9477972708, -1.4048906849, 3.5110262353]),
    (964.6637699814, 964.2757731367, [[-0.21431934153137527, 0.5001843952081165, -0.8389772289137819], [-0.4825621058287013, 0.6925707275796582, 0.5361712425316675], [0.8492355585252699, 0.5197704859744093, 0.09293873275550632]], [-0.2608593044, -1.6871044465, 3.5820331033]),
    (964.0108190521, 963.6230848304, [[0.982071754499825, -0.043556721595372244, 0.1834063276375674], [0.016214000267412425, 0.9888514391959208, 0.14802005740940863], [-0.18780887947315975, -0.1423925672358621, 0.9718292965264098]], [0.1950983853, -1.5023044601, 3.2704394802]),
    (966.4513303531, 966.0626145348, [[-0.22944729479916393, -0.5120430939972408, 0.8277474305602393], [0.34072880618243656, 0.7543683410601709, 0.561099177190287], [-0.9117334147363674, 0.4107800821557178, 0.001380056993251777]], [-0.1114050752, -1.4730931787, 3.6360458656]),
    (967.4382738868, 967.0491611105, [[-0.9938424000535866, -0.04659021144742611, -0.10053176638760006], [-0.11016394649512543, 0.5127766607023544, 0.8514246890545056], [0.011882287168134324, 0.8572569325681925, -0.5147517487249824]], [0.0386400282, -1.1680429117, 3.3079980916]),
    (965.1753620014, 964.7871593895, [[-0.14692406575819686, 0.5619790171714767, -0.8139980977619433], [-0.4791486537328604, 0.6794985645381177, 0.5556062170428127], [0.8653495747540808, 0.47165801705220556, 0.1694367977250304]], [-0.5951984540, -1.5212031954, 3.2319164236]),
    (968.2689086856, 967.8794618202, [[0.9490616738645841, -0.05075979050240852, 0.31097489105956055], [0.030094098056469273, 0.9970294171243953, 0.07089913011283068], [-0.31364993936476626, -0.05792913823577976, 0.9477699776210146]], [-0.0717152687, -1.1743867260, 2.9298378163]),
    (965.5775009654, 965.1891366095, [[-0.6423667965053309, -0.5152298358898713, 0.5673650632144835], [0.258431809975013, 0.5513229760313586, 0.7932565635990443], [-0.7215108442293279, 0.6561868577690522, -0.22099979491090088]], [0.2992909972, -1.4918841671, 3.2591067881]),
    (965.4919235887, 965.1035936529, [[-0.33765281666447344, 0.5255313025432876, -0.780901674633691], [-0.4880166545009702, 0.6116704602426479, 0.622654794405558], [0.8048790718662202, 0.5913341678896142, 0.04993577433132751]], [-0.4104588007, -1.4586543196, 3.8976388374]),
    (966.6921551643, 966.3033424840, [[0.9764613820530738, 0.1939017427794729, -0.09447371859985332], [-0.1856790669722819, 0.9785470782261924, 0.08926869430704672], [0.10975633670683435, -0.06962564069410947, 0.991516927092337]], [0.2506696052, -1.3687991288, 3.3002120123]),
    (964.4029639619, 964.0150720158, [[-0.09586295278454732, -0.47401136655665466, 0.8752848214487214], [0.5494084861139535, 0.7080600300517309, 0.44362293586909507], [-0.8300365110459909, 0.5234159132260254, 0.19254914207122364]], [-0.2408142726, -1.4032851387, 3.1457995450]),
    (965.7759608055, 965.3875166272, [[-0.2776018120709547, 0.5262829237485366, -0.8037185565268569], [-0.5461025066841335, 0.6018432327016309, 0.5827150036206005], [0.7903855300600008, 0.6006754593138816, 0.12033165190351658]], [-0.6493958832, -1.4941923665, 3.3770697516]),
]

PIPE_DIF = dotdict(convert_SHs_python=True, compute_cov3D_python=False, depth_ratio=0.0)
PIPE_ENV = dotdict(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0)
TRACER = None

torch.set_grad_enabled(False)


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def get_world_to_view(r, t):
    world_to_view = torch.eye(4, dtype=r.dtype, device=r.device)
    world_to_view[:3, :3] = r
    world_to_view[:3, 3:] = t
    return world_to_view


def get_projection_matrix(fovx, fovy, znear, zfar):
    tanfovx = math.tan(fovx / 2)
    tanfovy = math.tan(fovy / 2)
    top = tanfovy * znear
    right = tanfovx * znear
    proj = torch.zeros(4, 4, dtype=torch.float32, device=znear.device)
    proj[0, 0] = znear * 2 / (right - -right)
    proj[1, 1] = znear * 2 / (top - -top)
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0
    return proj


def make_camera_batch(fx, fy, r, t, near=0.1, far=100.0):
    scale_x = IMAGE_WIDTH / REFERENCE_IMAGE_WIDTH
    scale_y = IMAGE_HEIGHT / REFERENCE_IMAGE_HEIGHT
    scaled_fx = fx * scale_x
    scaled_fy = fy * scale_y
    scaled_cx = REFERENCE_CX * scale_x
    scaled_cy = REFERENCE_CY * scale_y

    k = torch.tensor(
        [[scaled_fx, 0.0, scaled_cx], [0.0, scaled_fy, scaled_cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    r = torch.tensor(r, dtype=torch.float32, device="cuda")
    t = torch.tensor(t, dtype=torch.float32, device="cuda").view(3, 1)
    znear = torch.tensor(near, dtype=torch.float32, device="cuda")
    zfar = torch.tensor(far, dtype=torch.float32, device="cuda")
    fovx = focal2fov(scaled_fx, IMAGE_WIDTH)
    fovy = focal2fov(scaled_fy, IMAGE_HEIGHT)
    world_view = get_world_to_view(r, t).T
    proj = get_projection_matrix(fovx, fovy, znear, zfar).T

    return dotdict(
        image_height=torch.tensor(IMAGE_HEIGHT, dtype=torch.float32, device="cuda"),
        image_width=torch.tensor(IMAGE_WIDTH, dtype=torch.float32, device="cuda"),
        K=k,
        R=r,
        T=t,
        world_view_transform=world_view,
        projection_matrix=proj,
        full_proj_transform=world_view @ proj,
        camera_center=(-r.mT @ t)[:, 0],
        FoVx=torch.tensor(fovx, dtype=torch.float32, device="cuda"),
        FoVy=torch.tensor(fovy, dtype=torch.float32, device="cuda"),
        tanfovx=math.tan(fovx * 0.5),
        tanfovy=math.tan(fovy * 0.5),
        znear=znear,
        zfar=zfar,
    )


def load_splats(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    pcd = GaussianModel(
        xyz=torch.empty(1, 3, device="cuda"),
        colors=None,
        sh_degree=3,
        render_reflection=True,
        specular_channels=True,
        xyz_lr_scheduler=None,
    )
    env = GaussianModel(
        xyz=torch.empty(1, 3, device="cuda"),
        colors=None,
        sh_degree=3,
        render_reflection=False,
        specular_channels=True,
        xyz_lr_scheduler=None,
    )
    pcd.load_state_dict({k[12:]: v for k, v in state.items() if k.startswith("sampler.pcd.")}, strict=False)
    env.load_state_dict({k[12:]: v for k, v in state.items() if k.startswith("sampler.env.")}, strict=False)
    return pcd.cuda(), env.cuda()


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def get_tracer():
    global TRACER
    if TRACER is None:
        TRACER = HardwareRendering().cuda()
    return TRACER


@torch.inference_mode()
def render_frame(pcd, env, cam, tracer, bg, ev=None):
    if ev is not None:
        ev[0].record()
    dif = render(cam, pcd, PIPE_DIF, bg)
    if ev is not None:
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

    if ev is not None:
        ev[2].record()
    env_out = tracer.render_gaussians(
        cam,
        ref_o,
        ref_d,
        env,
        PIPE_ENV,
        bg,
        max_trace_depth=0,
        specular_threshold=0.0,
        start_from_first=False,
        batch=cam,
    )
    if ev is not None:
        ev[3].record()

    rgb = (1.0 - spec) * dif_rgb + spec * env_out.render.permute(1, 2, 0)
    if ev is not None:
        ev[4].record()
    return (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def profile_view(pcd, env, cam, tracer, bg):
    for _ in range(WARMUP):
        render_frame(pcd, env, cam, tracer, bg)
    torch.cuda.synchronize()

    samples = {"frame_ms": [], "raster_ms": [], "raytrace_ms": []}
    for i in range(REPEATS):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        img = render_frame(pcd, env, cam, tracer, bg, ev)
        torch.cuda.synchronize()
        samples["frame_ms"].append(ev[0].elapsed_time(ev[4]))
        samples["raster_ms"].append(ev[0].elapsed_time(ev[1]))
        samples["raytrace_ms"].append(ev[2].elapsed_time(ev[3]))
    cv2.imwrite(f"/workspace/{abs(hash(cam.world_view_transform))}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return {key: float(np.mean(values)) for key, values in samples.items()}


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(CKPT_PATH)

    pcd, env = load_splats(CKPT_PATH)
    pcd.set_aabb_mask(
                min(-0.5, -2.5),
                max(0.5, 2.5), 
                min(-0.5, -2.5), 
                max(0.5, 2.5), 
                min(-0.5, -2.5), 
                max(0.5, 2.5))
    tracer = get_tracer()
    bg = torch.zeros(3, device="cuda")
    cams = [make_camera_batch(*cam) for cam in VAL_CAMERAS]

    per_view = {"frame_ms": [], "raster_ms": [], "raytrace_ms": []}
    for cam in cams:
        stats = profile_view(pcd, env, cam, tracer, bg)
        for key, value in stats.items():
            per_view[key].append(value)

    print(f"checkpoint: {CKPT_PATH}")
    print("base gaussians: ", int(pcd.get_xyz.shape[0]))
    print("env gaussians: ", int(env.get_xyz.shape[0]))
    print(f"views: {len(cams)}  repeats_per_view: {REPEATS}  resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    for key, label in [("frame_ms", "frame"), ("raster_ms", "rasterization"), ("raytrace_ms", "raytracing")]:
        mean, std = mean_std(per_view[key])
        print(f"{label}: {mean:.3f} ms +- {std:.3f} ms")


if __name__ == "__main__":
    main()
