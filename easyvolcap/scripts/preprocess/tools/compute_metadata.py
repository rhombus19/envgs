import os
import cv2
import copy
import torch
import argparse
import numpy as np
from pathlib import Path

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.colmap_utils import (
    read_points3D_binary_custom,
    read_points3D_text_custom,
    load_sfm_ply,
    save_sfm_ply,
    read_points3D_binary,
    read_points3D_text,
    write_points3D_ply,
)


def calculate_bounding_box(xyz, lower_percentile=5, upper_percentile=95):
    # Convert to tensor
    xyz_tensor = torch.tensor(xyz)
    # Compute min and max values
    min_values = torch.quantile(xyz_tensor, lower_percentile / 100.0, dim=0)
    max_values = torch.quantile(xyz_tensor, upper_percentile / 100.0, dim=0)

    # Filter out points outside the range
    mask = ((xyz_tensor >= min_values) & (xyz_tensor <= max_values)).all(dim=1)
    filtered_xyz = xyz_tensor[mask]

    # Compute min and max values
    min_xyz = torch.min(filtered_xyz, dim=0).values
    max_xyz = torch.max(filtered_xyz, dim=0).values
    return min_xyz.numpy(), max_xyz.numpy()


def load_points(points_file: str = None):
    xyz = None
    if exists(points_file):
        xyz, _ = load_sfm_ply(points_file)
    else:
        try:
            xyz, _, _ = read_points3D_binary_custom(points_file.replace(".ply", ".bin"))
        except:
            xyz, _, _ = read_points3D_text_custom(points_file.replace(".ply", ".txt"))
    return xyz


def ensure_points3d_ply(points_path: str):
    if exists(points_path):
        return points_path
    base, _ = splitext(points_path)
    bin_path = base + ".bin"
    txt_path = base + ".txt"
    if exists(bin_path):
        log(f'{yellow("Converting points3D.bin to PLY")}: {cyan(bin_path)}')
        points3D = read_points3D_binary(bin_path)
        write_points3D_ply(points3D, points_path)
        return points_path
    if exists(txt_path):
        log(f'{yellow("Converting points3D.txt to PLY")}: {cyan(txt_path)}')
        points3D = read_points3D_text(txt_path)
        write_points3D_ply(points3D, points_path)
        return points_path
    return points_path


def format_list_lines(values, prefix, max_line_length=120):
    if not values:
        return [prefix + "[]"]
    indent = " " * len(prefix)
    lines = []
    line = prefix + "["
    for i, value in enumerate(values):
        token = f"{value}"
        if i < len(values) - 1:
            token += ", "
        else:
            token += "]"
        if len(line) + len(token) > max_line_length and line != prefix + "[":
            lines.append(line.rstrip())
            line = indent + token
        else:
            line += token
    lines.append(line.rstrip())
    return lines


def format_vec(values):
    return ", ".join(repr(float(v)) for v in values)


def format_bounds(min_xyz, max_xyz):
    return f"[[{format_vec(min_xyz)}], [{format_vec(max_xyz)}]]"


def build_yaml_config(data_root, view_sample, val_view_sample, preload_gs, preload_env_gs_path, spatial_scale, min_xyz, max_xyz):
    lines = []
    lines.append("dataloader_cfg:")
    lines.append("    dataset_cfg: &dataset_cfg")
    lines.append(f"        data_root: {data_root}")
    lines.extend(format_list_lines(view_sample, "        view_sample: "))
    lines.append("")
    lines.append("val_dataloader_cfg:")
    lines.append("    dataset_cfg:")
    lines.append("        <<: *dataset_cfg")
    lines.extend(format_list_lines(val_view_sample, "        view_sample: "))
    lines.append("")
    lines.append("model_cfg:")
    lines.append("    sampler_cfg:")
    lines.append(f"        preload_gs: {preload_gs}")
    lines.append(f"        spatial_scale: {repr(float(spatial_scale))}")
    lines.append(f"        env_bounds: {format_bounds(min_xyz, max_xyz)}")
    lines.append(f"        env_preload_gs: {preload_env_gs_path}")
    return "\n".join(lines) + "\n"


def resolve_yaml_output_path(scene, output_arg, multi_scene):
    if not output_arg:
        return None
    if output_arg == "-":
        return "-"
    if multi_scene and output_arg.endswith(".yaml"):
        out_dir = dirname(output_arg) or "."
        os.makedirs(out_dir, exist_ok=True)
        return join(out_dir, f"{scene}.yaml")
    if (not multi_scene) and output_arg.endswith(".yaml"):
        out_dir = dirname(output_arg)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return output_arg
    os.makedirs(output_arg, exist_ok=True)
    return join(output_arg, f"{scene}.yaml")


def detect_camera_outliers(cameras, camera_names, mad_mult=8.0, ratio=10.0, min_inliers=1):
    if len(camera_names) <= 1:
        return [], None

    Rs = np.stack([cameras[name]['R'] for name in camera_names], axis=0)  # (N, 3, 3)
    Ts = np.stack([cameras[name]['T'] for name in camera_names], axis=0)  # (N, 3, 1)
    Cs = -Rs.mT @ Ts  # (N, 3, 1)
    centers = Cs.reshape(len(camera_names), 3)

    median_center = np.median(centers, axis=0)
    dists = np.linalg.norm(centers - median_center[None], axis=1)
    med_dist = np.median(dists)
    mad_dist = np.median(np.abs(dists - med_dist))

    if mad_dist > 0:
        threshold = med_dist + mad_mult * mad_dist
        method = f"median+{mad_mult}*MAD"
    else:
        nonzero = dists[dists > 0]
        base = np.median(nonzero) if nonzero.size > 0 else med_dist
        threshold = base * ratio
        method = f"median_nonzero*{ratio}"

    outlier_mask = dists > threshold
    outlier_indices = np.where(outlier_mask)[0].tolist()
    if len(camera_names) - len(outlier_indices) < min_inliers:
        log(yellow("Outlier filter would remove too many cameras; skipping filter."))
        return [], None

    if outlier_indices:
        outlier_names = [camera_names[i] for i in outlier_indices]
        log(f'{yellow("Filtering outlier cameras")}: {cyan(outlier_names)}')
        log(f'{yellow("Outlier filter method")}: {cyan(method)}, {yellow("threshold")}: {cyan(threshold)}')
    return outlier_indices, dists


@catch_throw
def main(args):
    # Define the per-scene processing function
    def process_scene(scene):
        log(f'{yellow("Processing scene")}: {cyan(scene)}')

        # Data paths
        data_root = join(args.data_root, scene)
        points_rel = splitext(args.points_file)[0] + ".ply"
        if os.path.isabs(points_rel):
            points_path = points_rel
        else:
            points_path = join(data_root, points_rel)
        preload_gs_path = points_path
        if args.output_yaml:
            preload_gs_path = ensure_points3d_ply(preload_gs_path)

        # Load all cameras
        cameras = read_camera(data_root)
        camera_names = sorted(cameras.keys())  # sort the camera names
        outlier_indices = []
        if not args.disable_camera_outlier_filter:
            outlier_indices, _ = detect_camera_outliers(
                cameras,
                camera_names,
                mad_mult=args.center_outlier_mad_mult,
                ratio=args.center_outlier_ratio,
                min_inliers=args.center_outlier_min_inliers,
            )
        outlier_index_set = set(outlier_indices)
        valid_indices = [i for i in range(len(camera_names)) if i not in outlier_index_set]

        # Split the cameras into two groups if `args.eval` is True
        if args.eval:
            view_sample = [i for i in valid_indices if i % args.skip != 0]
            val_view_sample = [i for i in valid_indices if i % args.skip == 0]
        else:
            view_sample = valid_indices
            val_view_sample = []
        log(f'dataloader_cfg.dataset_cfg.view_sample: {view_sample}')
        log(f'val_dataloader_cfg.dataset_cfg.view_sample: {val_view_sample}')

        # Compute the center and radius for the training set
        Rs = np.stack([cameras[camera_names[i]]['R'] for i in view_sample], axis=0)  # (N, 3, 3)
        Ts = np.stack([cameras[camera_names[i]]['T'] for i in view_sample], axis=0)  # (N, 3, 1)
        Cs = -Rs.mT @ Ts  # (N, 3, 1)
        center = np.mean(Cs, axis=0)  # (3, 1)

        radius = np.linalg.norm(Cs - center[None], axis=1).max()  # scalar
        radius = radius * 1.1  # follow the original 3DGS
        log(f"model_cfg.sampler_cfg.spatial_scale: {radius}")

        # Load SfM points
        xyz = load_points(points_path)
        if xyz is not None:
            min_xyz, max_xyz = calculate_bounding_box(xyz, lower_percentile=args.lower_percentile, upper_percentile=args.upper_percentile)
        else:
            min_xyz = args.bounds[:3]
            max_xyz = args.bounds[3:]
        log(f"model_cfg.sampler_cfg.env_bounds: [[{min_xyz[0]}, {min_xyz[1]}, {min_xyz[2]}], [{max_xyz[0]}, {max_xyz[1]}, {max_xyz[2]}]]")
        preload_env_gs_path= Path(data_root) / "env" / "points3D.ply"
        preload_env_gs_path.parent.mkdir(exist_ok=True, parents=True)

        if args.output_yaml:
            yaml_text = build_yaml_config(
                data_root=data_root,
                view_sample=view_sample,
                val_view_sample=val_view_sample,
                preload_gs=preload_gs_path,
                preload_env_gs_path=str(preload_env_gs_path),
                spatial_scale=radius,
                min_xyz=min_xyz,
                max_xyz=max_xyz,
            )
            out_path = resolve_yaml_output_path(scene, args.output_yaml, len(scenes) > 1)
            if out_path == "-":
                print(yaml_text, end="")
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(yaml_text)
                log(f'{yellow("Wrote yaml config")}: {cyan(out_path)}')
        log()

    # Find all scenes
    if len(args.scenes):
        scenes = [f for f in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, f)) and f in args.scenes]
    else:
        scenes = [f for f in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, f))]

    # Process each scene
    parallel_execution(scenes, action=process_scene, sequential=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/datasets/refnerf/ref_real')
    parser.add_argument('--images_dir', type=str, default='images')
    parser.add_argument('--points_file', type=str, default='sparse/0/points3D.ply')
    parser.add_argument('--output_yaml', nargs='?', const='-', default='',
                        help='Output yaml config (use "-" for stdout or pass a directory/path)')
    parser.add_argument('--scenes', nargs='+', default=[])
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--skip', type=int, default=8)
    parser.add_argument('--lower_percentile', type=float, default=1)
    parser.add_argument('--upper_percentile', type=float, default=99)
    parser.add_argument('--bounds', type=float, nargs='+', default=[-20., -20., -20., 20., 20., 20.])
    parser.add_argument('--disable_camera_outlier_filter', action='store_true', default=False)
    parser.add_argument('--center_outlier_mad_mult', type=float, default=8.0)
    parser.add_argument('--center_outlier_ratio', type=float, default=10.0)
    parser.add_argument('--center_outlier_min_inliers', type=int, default=1)
    args = parser.parse_args()
    main(args)
