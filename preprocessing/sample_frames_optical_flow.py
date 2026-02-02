from pathlib import Path
import cv2
import numpy as np
import math
from tqdm import tqdm
import pandas as pd
import shutil
from concurrent.futures import ThreadPoolExecutor
import argparse
from diskcache import Cache

cache_dir = Path.home() / ".frame_sampler_cache"
cache = Cache(cache_dir)

@cache.memoize()
def get_optical_flow_between_frames(frames):
    frame1_path, frame2_path = frames
    img1_bgr = cv2.imread(frame1_path)
    img2_bgr = cv2.imread(frame2_path)
    
    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    
    p0 = cv2.goodFeaturesToTrack(
        img1_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        img1_gray,
        img2_gray,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # Keep only successfully tracked points
    st = st.reshape(-1)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    if len(good_new) < 10:
        print("Low features:", len(good_new))
        print(frame1_path)
        print(frame2_path)
    
    # good_old, good_new from KLT sparse flow snippet
    pts1 = good_old.reshape(-1, 2).astype(np.float32)
    pts2 = good_new.reshape(-1, 2).astype(np.float32)

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=2.0,
        confidence=0.99
    )  #RANSAC

    if F is not None and mask is not None:
        inlier_mask = mask.ravel().astype(bool)
        inliers1 = pts1[inlier_mask]
        inliers2 = pts2[inlier_mask]

        disp = inliers2 - inliers1
        mag = np.linalg.norm(disp, axis=1)
        median_disp = float(np.median(mag))
        return median_disp

    else:
        print("RANSAC failed for KLT points")
        return -1


def select_frames_by_global_motion(df, n_samples,
                                   distance_col="dist",
                                   clearness_col="clearness"):
    n = len(df)
    if n == 0 or n_samples <= 0:
        return []

    d = df[distance_col].to_numpy().astype(float)
    c = df[clearness_col].to_numpy().astype(float)

    # Only finite distances contribute to motion
    finite = np.isfinite(d)
    total_motion = d[finite].sum()
    if total_motion == 0:
        # fallback: uniform in index space
        idx = np.linspace(0, n - 1, min(n_samples, n), dtype=int)
        return sorted(set(idx.tolist()))

    T = total_motion / n_samples  # target motion per sample

    # cumulative motion before each frame
    cum = np.cumsum(np.where(finite, d, 0.0))
    pos = np.concatenate([[0.0], cum[:-1]])  # motion position of frame i

    # assign each frame to a motion segment
    seg_id = np.floor(pos / T).astype(int)
    seg_id = np.clip(seg_id, 0, n_samples - 1)

    best_idx  = np.full(n_samples, -1, dtype=int)
    best_clear = np.full(n_samples, -np.inf, dtype=float)

    for i, s in enumerate(seg_id):
        if c[i] > best_clear[s]:
            best_clear[s] = c[i]
            best_idx[s] = i

    # keep selected frames (one per non-empty segment)
    selected = [int(i) for i in best_idx if i >= 0]

    # make sure first and last are kept
    if 0 not in selected:
        selected.append(0)
    last_idx = n - 1
    if last_idx not in selected:
        selected.append(last_idx)

    selected = sorted(set(selected))

    # enforce exactly n_samples
    if len(selected) > n_samples:
        idx = np.linspace(0, len(selected) - 1, n_samples, dtype=int)
        selected = [selected[i] for i in idx]
    elif len(selected) < n_samples:
        remaining = sorted(
            set(range(n)) - set(selected),
            key=lambda i: c[i],
            reverse=True
        )
        extra = remaining[: max(0, n_samples - len(selected))]
        selected = sorted(selected + extra)

    return selected


@cache.memoize()
def clearness_score(image_path):
    return cv2.Laplacian(cv2.imread(image_path), cv2.CV_64F).var()


def main():
    parser = argparse.ArgumentParser(prog='Frame Sampler')
    parser.add_argument('--framedir', help='Source directory containing frames to be subsampled, frames have to be named in alphabetical order')
    parser.add_argument('--n', type=int, help='Number of frames to sample uniformly based on optical flow', default=200)
    args = parser.parse_args()
    
    print("The frames in the framedir are expected to be named in alphabetical order")
    
    n_out_frames = args.n
    
    src_dir = Path(args.framedir)
    dst_dir = src_dir.parent / f"{src_dir.name}_sampled_{n_out_frames}"
    
    frames = sorted(src_dir.iterdir())
    
    print("Calculating optical flow mean displacement between all pairs of frames")
    pairs = list(zip(frames,frames[1:]))
    with ThreadPoolExecutor() as executor:
        distances = list(tqdm(executor.map(get_optical_flow_between_frames, pairs), total=len(pairs)))
    # distances = calculate_optical_flow(pairs)
    
    print("Calculating clearness score for all frames")
    # clearness = calculate_clearness_score(frames)
    with ThreadPoolExecutor() as executor:
        clearness = list(tqdm(executor.map(clearness_score, frames), total=len(frames)))
    
    distances.append(math.inf) # Last frame has infinite distance to the next frame 
    
    df = pd.DataFrame({"image": frames, "dist": distances, "clearness": clearness})
    
    quant_95 = df["dist"][:-1].quantile(0.95)  # considered to be fails of optical flow
    df = df[df["dist"] < quant_95]
    
    sampled_indices = select_frames_by_global_motion(df, n_samples=n_out_frames)
    sampled_frames = df.iloc[sampled_indices]
    
    out_paths = sampled_frames["image"].tolist()
    
    dst_dir.mkdir(exist_ok=True)
    
    for src in out_paths:
        shutil.copy2(src,  dst_dir / src.name)
    
if __name__ == "__main__":
    main()