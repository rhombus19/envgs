import os
import cv2
import copy
import torch
import argparse
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.colmap_utils import read_points3D_binary_custom, read_points3D_text_custom, load_sfm_ply, save_sfm_ply


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
        points_path = join(data_root, args.points_file)

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
