#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from easyvolcap.utils.colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
)


def image_size(images_dir: Path, image_name: str):
    with Image.open(images_dir / image_name) as image:
        return image.size  # W, H


def scaled_params(camera, target_w: int, target_h: int):
    sx, sy = target_w / camera.width, target_h / camera.height
    params = camera.params.astype(float).copy()

    if camera.model == "PINHOLE":
        params[[0, 2]] *= sx
        params[[1, 3]] *= sy
    elif camera.model == "SIMPLE_PINHOLE":
        params = np.array([params[0] * sx, params[0] * sy, params[1] * sx, params[2] * sy])
        camera.model = "PINHOLE"
    else:
        raise NotImplementedError(f"Only PINHOLE/SIMPLE_PINHOLE are handled, got {camera.model}")

    camera.width, camera.height, camera.params = target_w, target_h, params
    return sx, sy


def main():
    ba_root = Path(__file__).resolve().parents[2]
    parser = ArgumentParser()
    parser.add_argument("--sparse", type=Path, default=ba_root / "datasets/ref_real/sedan_dataset/mapanything_sparse/0")
    parser.add_argument("--images", type=Path, default=ba_root / "datasets/ref_real/sedan_dataset/images")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sparse = args.sparse.resolve()
    images_dir = args.images.resolve()
    out = (Path(str(sparse) + "_rescaled") if args.out is None else args.out).resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite or choose --out")
        shutil.rmtree(out)
    shutil.copytree(sparse, out)

    cameras = read_cameras_binary(str(out / "cameras.bin"))
    images = read_images_binary(str(out / "images.bin"))

    target_by_camera = {}
    for image in images.values():
        target = image_size(images_dir, image.name)
        old = target_by_camera.setdefault(image.camera_id, target)
        if old != target:
            raise SystemExit(f"camera {image.camera_id} is used by multiple image sizes: {old} and {target}")

    scale_by_camera = {}
    for camera_id, camera in cameras.items():
        target_w, target_h = target_by_camera.get(camera_id, next(iter(target_by_camera.values())))
        scale_by_camera[camera_id] = scaled_params(camera, target_w, target_h)

    for image in images.values():
        sx, sy = scale_by_camera[image.camera_id]
        if len(image.xys):
            image.xys[:, 0] *= sx
            image.xys[:, 1] *= sy

    write_cameras_binary(cameras, str(out / "cameras.bin"))
    write_images_binary(images, str(out / "images.bin"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
