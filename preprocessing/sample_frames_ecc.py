import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import shutil
from functools import partial



def choose_metric_size(first_frame_bgr, target_w=960, target_h=540):
    """
    Pick one fixed metric size for the whole scene, swapping for portrait scenes. 
    Assuming the orientation stays the same across all frames
    """
    h, w = first_frame_bgr.shape[:2]
    if h > w:
        return (int(target_h), int(target_w))  # swap if portrait
    return (int(target_w), int(target_h))

def warp_motion_magnitude(warp, shape_hw, grid=40):
    """
    Computes median pixel displacement induced by a transformation matrix on a grid.
    """
    H, W = shape_hw

    xs = np.linspace(0, W - 1, grid, dtype=np.float32)
    ys = np.linspace(0, H - 1, grid, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)  # (N,2)

    if warp.shape == (2, 3):
        A = warp[:, :2]
        t = warp[:, 2]
        pts_w = (pts @ A.T) + t
    elif warp.shape == (3, 3):
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)  # homogenize to (N,3)
        proj = (pts_h @ warp.T)
        pts_w = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)  # projective transform
    else:
        raise ValueError("warp must be 2x3 or 3x3")

    disp = np.linalg.norm(pts_w - pts, axis=1)
    return float(np.median(disp)), float(np.mean(disp))

def ecc_warp(frames, model="homography", iters=200, eps=1e-6, gauss=5, metric_wh=(960, 540)):
    """
    Find a transformation matrix between a pair of frames
    frames: tuple of paths to a pair of images
    model: "translation" | "euclidean" | "affine" | "homography"
    Returns: median warp-induced pixel motion or NaN on failure
    """
    frame1_path, frame2_path = frames
    img1_bgr = cv2.imread(str(frame1_path))
    img2_bgr = cv2.imread(str(frame2_path))

    img1_bgr = cv2.resize(img1_bgr, metric_wh, interpolation=cv2.INTER_AREA)
    img2_bgr = cv2.resize(img2_bgr, metric_wh, interpolation=cv2.INTER_AREA)

    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    # ECC expects float32, typically normalized
    prev = img1_gray.astype(np.float32) / 255.0
    curr = img2_gray.astype(np.float32) / 255.0

    if model == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
    elif model == "euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp = np.eye(2, 3, dtype=np.float32)
    elif model == "affine":
        warp_mode = cv2.MOTION_AFFINE
        warp = np.eye(2, 3, dtype=np.float32)
    elif model == "homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        raise ValueError("unknown model")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    
    try:
        err, warp = cv2.findTransformECC(
            templateImage=prev,
            inputImage=curr,
            warpMatrix=warp,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=gauss
        )
    except cv2.error:
        return np.nan

    med_px, _ = warp_motion_magnitude(warp, img1_gray.shape, grid=40)
    return float(med_px)

def laplacian_var_downsize(image_path, metric_wh=(960, 540)):
    """
    Computes variance of the Laplacian of a downsized image 
    """
    bgr = cv2.imread(str(image_path))

    metric_bgr = cv2.resize(bgr, metric_wh, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(metric_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def pick_best_if_worth_it(g, value_col="sharpness", upside_thresh=0.5):
    x = g[value_col].astype(float).to_numpy()
    n = len(g)
    
    mid_i = n // 2
    best_i = int(np.nanargmax(x))

    # typical variation inside the group
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = (q3 - q1)

    upside = (x[best_i] - x[mid_i]) / iqr # "how many IQRs better than middle?"

    return g.iloc[[best_i]] if upside >= upside_thresh else g.iloc[[mid_i]]

def sample_local_distance(df, dist_interval, sharpness_column="sharpness"):
    df = df.copy()
    df["min_d_idx"] = (df["dist"].cumsum() / dist_interval).astype(int)
    
    df = df.groupby("min_d_idx", group_keys=False)\
    .apply(lambda g: pick_best_if_worth_it(g, value_col=sharpness_column) \
        .assign(dist=g["dist"].sum()).assign(n_frames=len(g))) \
        .reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser(prog='Frame Sampler')
    parser.add_argument('--data_root', required=True, help='Source directory containing frames to be subsampled, frames have to be named in alphabetical order')
    parser.add_argument('--output', required=False, help='Destination directory to copy sampled frames over. Will be created if it does not exist')
    parser.add_argument('--motion_segment', type=int, help='Amount of motion per sampled frame. also correlates with the final number of frames', default=60)
    parser.add_argument('--sharpness_quantile', type=float, help='What percentile of sharp frames to keep', default=0.8)
    parser.add_argument('--accept_all', action='store_true', help='Copy frames over without confirmation prompt')
    parser.add_argument('--metric_width', type=int, default=960, help='Canonical metric canvas width (landscape base).')
    parser.add_argument('--metric_height', type=int, default=540, help='Canonical metric canvas height (landscape base).')
    parser.add_argument('--output_raw_pickle_only', action="store_true", default=False, help='Write only a pickle with sharpness and ecc distance in the output folder without any filtering in place')
    
    args = parser.parse_args()
    
    print("The frames in the framedir are expected to be named in alphabetical order")

    frames_root = Path(args.data_root)
    subdirs = [subdir for subdir in frames_root.iterdir() if subdir.is_dir()]
    if any(subdirs):
        scenes = subdirs
    else:
        scenes = [frames_root]
    
    for scene in scenes:
        print(scene)
        frames = sorted([p for p in scene.iterdir() if p.is_file()])
        if not frames:
            print("No frames found, skipping scene.")
            continue

        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            print(f"Could not read first frame {frames[0]}, skipping scene.")
            continue
        metric_wh = choose_metric_size(
            first_frame_bgr=first_frame,
            target_w=args.metric_width,
            target_h=args.metric_height,
        )
        print(f"metric evaluation size (W,H): {metric_wh}")

        pairs = list(zip(frames,frames[1:]))

        lap_fn = partial(
            laplacian_var_downsize,
            metric_wh=metric_wh,
        )
        with ThreadPoolExecutor() as executor:
            sharpness = list(tqdm(executor.map(lap_fn, frames), total=len(frames)))

        ecc_fn = partial(
            ecc_warp,
            model="homography",
            metric_wh=metric_wh,
        )
        with ThreadPoolExecutor() as executor:
            ecc_dist = list(tqdm(executor.map(ecc_fn, pairs), total=len(pairs)))
        
        # One motion value per frame: distance to the next frame (last frame has no successor).
        df = pd.DataFrame({"name": frames, "sharpness": sharpness, "dist": ecc_dist + [0.0]})

        if args.output_raw_pickle_only:
            if not args.output:
                raise ValueError("No output directory provided")
            output_path = Path(args.output) / (scene.name + ".pkl")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_pickle(output_path)
            continue
        
        # Filter
        sharp_thresh = df["sharpness"].quantile(1 - args.sharpness_quantile)
        print("filtered out based on sharpness", int((df["sharpness"] <= sharp_thresh).sum()))

        df = df[df["sharpness"] > sharp_thresh]  # Filter out blurrier images.
        df = sample_local_distance(df, args.motion_segment, sharpness_column="sharpness")
        
        print(f"Final number of frames {len(df)}")
        
        if not args.accept_all:
            choice = input("Copy frames to dest dir? (y/n)")
            if choice != 'y':
                return

        if not args.output:
            output = frames_root.parent / (frames_root.name + "_sampled") / (scene.name + f"_sampled_{len(df)}/")
        else:
            output = Path(args.output)
        output.mkdir(exist_ok=True, parents=True)
        for _, row in df.iterrows():
            shutil.copy(row["name"], output / Path(row["name"]).name) 

if __name__ == "__main__":
    main()