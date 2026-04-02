from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
THUMB_CACHE_DIR = APP_DIR / ".thumb_cache"

DEFAULT_DF_PATH = PROJECT_ROOT / "distances_vw_rain_all_frames.pkl"
DEFAULT_BLUR_PATH = PROJECT_ROOT / "blur_score_vw_rain_all_frames.pkl"

REQUIRED_INPUT_COLUMNS = [
    "frame1",  # absolute or resolvable path to frame 1
    "dist",  # motion/distance scalar for each pair
]

# Minimal API aliases accepted from user dataframes.
FRAME1_ALIASES = ("name",)
CLEARNESS_F1_ALIASES = ("clearness_frame1", "clearness", "sharpness")
CLEARNESS_F2_ALIASES = ("clearness_frame2", "frame2_clearness")


def _has_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if _has_streamlit_context():
    cache_data = st.cache_data
    cache_resource = st.cache_resource
else:
    # In bare python mode, avoid Streamlit runtime warnings from cache decorators.
    def cache_data(*_args, **_kwargs):
        def _identity(func):
            return func

        return _identity

    def cache_resource(*_args, **_kwargs):
        def _identity(func):
            return func

        return _identity


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


@cache_data(show_spinner=False)
def load_table(path_str: str) -> pd.DataFrame:
    path = _resolve_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)

    # Fallback: try pickle first.
    return pd.read_pickle(path)


def _promote_alias_column(df: pd.DataFrame, target_col: str, aliases: tuple[str, ...]) -> None:
    if target_col in df.columns:
        return
    for alias in aliases:
        if alias in df.columns:
            df[target_col] = df[alias]
            return


def ensure_expected_columns(df: pd.DataFrame, blur_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = df.copy()

    _promote_alias_column(df, "frame1", FRAME1_ALIASES)
    missing_core = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_core:
        raise ValueError(f"Input dataframe is missing required columns: {missing_core}")

    df["dist"] = pd.to_numeric(df["dist"], errors="coerce")
    if df["dist"].isna().all():
        raise ValueError("Column 'dist' has no valid numeric values.")

    # Normalize clearness aliases to canonical internal names.
    if "frame1_clearness" not in df.columns:
        for c in CLEARNESS_F1_ALIASES:
            if c in df.columns:
                df["frame1_clearness"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "frame2_clearness" not in df.columns:
        for c in CLEARNESS_F2_ALIASES:
            if c in df.columns:
                df["frame2_clearness"] = pd.to_numeric(df[c], errors="coerce")
                break

    # Optional fallback for legacy workflows: derive clearness from blur dataframe.
    if ("frame1_clearness" not in df.columns or "frame2_clearness" not in df.columns) and (
        blur_df is not None and {"name", "clearness"}.issubset(blur_df.columns)
    ):
        blur_df = blur_df.copy()
        blur_df["basename"] = blur_df["name"].astype(str).map(lambda x: Path(x).name)
        clearness_map = (
            blur_df.drop_duplicates(subset=["basename"]).set_index("basename")["clearness"].to_dict()
        )
        if "frame1_clearness" not in df.columns:
            df["frame1_clearness"] = df["frame1"].astype(str).map(
                lambda x: clearness_map.get(Path(x).name, np.nan)
            )
        if "frame2_clearness" not in df.columns and "frame2" in df.columns:
            df["frame2_clearness"] = df["frame2"].astype(str).map(
                lambda x: clearness_map.get(Path(x).name, np.nan)
            )

    if "frame1_clearness" not in df.columns:
        df["frame1_clearness"] = np.nan
    if "frame2_clearness" not in df.columns:
        df["frame2_clearness"] = np.nan

    if "clearness_sum" not in df.columns:
        df["clearness_sum"] = df["frame1_clearness"] + df["frame2_clearness"]

    # drop is optional; if absent, default all False.
    if "drop" not in df.columns:
        df["drop"] = False
    else:
        df["drop"] = df["drop"].astype(bool)

    df = df.reset_index(drop=True)
    df["pair_idx"] = np.arange(len(df))
    return df


def build_frame_lookup(
    df: pd.DataFrame, frames_dir_str: str, blur_df: Optional[pd.DataFrame]
) -> Dict[str, Optional[str]]:
    frames_dir = _resolve_path(frames_dir_str) if frames_dir_str.strip() else None
    if frames_dir is not None and not frames_dir.exists():
        frames_dir = None

    blur_path_map: Dict[str, str] = {}
    if blur_df is not None and "name" in blur_df.columns:
        for p in blur_df["name"].astype(str).tolist():
            pp = Path(p)
            if pp.exists():
                blur_path_map[pp.name] = str(pp)

    frame_cols = ["frame1"] + (["frame2"] if "frame2" in df.columns else [])
    all_names = pd.unique(df[frame_cols].astype(str).values.ravel("K")).tolist()
    lookup: Dict[str, Optional[str]] = {}
    for n in all_names:
        p = Path(n)
        if p.exists():
            lookup[n] = str(p)
            continue

        basename = p.name
        if frames_dir is not None:
            candidate = frames_dir / basename
            if candidate.exists():
                lookup[n] = str(candidate)
                continue

        if basename in blur_path_map:
            lookup[n] = blur_path_map[basename]
            continue

        lookup[n] = None

    return lookup


@cache_resource(show_spinner=False)
def get_thumb_server_port() -> int:
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    class ThumbHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(THUMB_CACHE_DIR), **kwargs)

        def log_message(self, format: str, *args) -> None:
            return

    ThreadingTCPServer.allow_reuse_address = True
    server = ThreadingTCPServer(("0.0.0.0", 0), ThumbHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return int(server.server_address[1])


@cache_data(show_spinner=False, max_entries=26000)
def ensure_thumbnail_file(path_str: str, max_width: int, jpg_quality: int = 72) -> Optional[str]:
    try:
        quality = int(np.clip(jpg_quality, 35, 95))
        key_raw = f"{Path(path_str).resolve()}|w={int(max_width)}|q={quality}"
        key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()
        THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_jpg = THUMB_CACHE_DIR / f"{key}.jpg"
    except Exception:
        return None

    if cache_jpg.exists():
        return cache_jpg.name

    img_bgr = cv2.imread(path_str)
    if img_bgr is None:
        return None

    if max_width > 0 and img_bgr.shape[1] > max_width:
        scale = max_width / float(img_bgr.shape[1])
        new_size = (max(1, int(img_bgr.shape[1] * scale)), max(1, int(img_bgr.shape[0] * scale)))
        img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None

    try:
        cache_jpg.write_bytes(encoded.tobytes())
        return cache_jpg.name
    except Exception:
        return None


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if np.isfinite(out):
        return out
    return float("nan")


def _fmt(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _motion_bounds(series: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0

    lo = float(vals.quantile(0.05))
    hi = float(vals.quantile(0.95))
    if not np.isfinite(lo):
        lo = float(vals.min())
    if not np.isfinite(hi):
        hi = float(vals.max())
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _motion_norm(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip((value - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def _motion_color(norm: float) -> str:
    hue = int(round((1.0 - norm) * 120.0))
    return f"hsl({hue}, 82%, 43%)"


def _motion_level(norm: float) -> str:
    if norm >= 0.67:
        return "high"
    if norm >= 0.34:
        return "medium"
    return "low"


def _clearness_thresholds(df: pd.DataFrame) -> tuple[float, float]:
    vals = pd.concat([df["frame1_clearness"], df["frame2_clearness"]], ignore_index=True)
    vals = pd.to_numeric(vals, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    q1 = float(vals.quantile(0.33))
    q2 = float(vals.quantile(0.66))
    if q2 <= q1:
        q1 = float(vals.min())
        q2 = float(vals.max())
    if q2 <= q1:
        q2 = q1 + 1e-6
    return q1, q2


def _clearness_label_color(value: float, q1: float, q2: float) -> tuple[str, str]:
    if not np.isfinite(value):
        return "unknown", "#6b7280"
    if value <= q1:
        return "blurry", "#dc2626"
    if value <= q2:
        return "medium", "#d97706"
    return "clear", "#16a34a"


def _clean_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _compute_sharpness_prefilter_threshold(
    sharpness_values: pd.Series, keep_fraction: float
) -> tuple[bool, float]:
    keep = float(np.clip(keep_fraction, 0.1, 1.0))
    if keep >= 1.0:
        return False, float("nan")

    vals = _clean_numeric(sharpness_values)
    if vals.empty:
        return False, float("nan")

    threshold = float(vals.quantile(1.0 - keep))
    if not np.isfinite(threshold):
        return False, float("nan")
    return True, threshold


def build_distribution_histograms(
    df: pd.DataFrame,
    dist_col: str,
    sharpness_col: str,
    keep_fraction: float,
    threshold_available: bool,
    sharpness_threshold: float,
) -> None:
    st.subheader("Distributions")

    dist_vals = _clean_numeric(df[dist_col])
    sharp_vals = _clean_numeric(df[sharpness_col])
    if dist_vals.empty and sharp_vals.empty:
        st.caption("No valid numeric values for `dist` or frame1 sharpness in this window.")
        return

    dist_q05 = float("nan")
    dist_q95 = float("nan")
    if not dist_vals.empty:
        dist_q05 = float(dist_vals.quantile(0.05))
        dist_q95 = float(dist_vals.quantile(0.95))

    dist_bins = int(np.clip(round(np.sqrt(max(1, len(dist_vals))) * 2), 20, 120))
    sharp_bins = int(np.clip(round(np.sqrt(max(1, len(sharp_vals))) * 2), 20, 120))

    if HAS_PLOTLY:
        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.12,
            subplot_titles=("Motion (`dist`)", "Frame1 Sharpness"),
        )

        if dist_vals.empty:
            fig.add_annotation(
                text="No valid values",
                x=0.23,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 11, "color": "#6b7280"},
            )
        else:
            fig.add_trace(
                go.Histogram(
                    x=dist_vals,
                    nbinsx=dist_bins,
                    name="dist",
                    marker={"color": "#3b82f6"},
                    opacity=0.85,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            if np.isfinite(dist_q05):
                fig.add_vline(
                    x=dist_q05,
                    line_dash="dot",
                    line_color="#1d4ed8",
                    annotation_text="q05",
                    annotation_position="top left",
                    row=1,
                    col=1,
                )
            if np.isfinite(dist_q95):
                fig.add_vline(
                    x=dist_q95,
                    line_dash="dot",
                    line_color="#1d4ed8",
                    annotation_text="q95",
                    annotation_position="top right",
                    row=1,
                    col=1,
                )

        if sharp_vals.empty:
            fig.add_annotation(
                text="No valid values",
                x=0.77,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 11, "color": "#6b7280"},
            )
        else:
            if threshold_available and np.isfinite(sharpness_threshold):
                kept_vals = sharp_vals[sharp_vals > sharpness_threshold]
                fig.add_trace(
                    go.Histogram(
                        x=sharp_vals,
                        nbinsx=sharp_bins,
                        name="all",
                        marker={"color": "#94a3b8"},
                        opacity=0.60,
                    ),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Histogram(
                        x=kept_vals,
                        nbinsx=sharp_bins,
                        name=f"kept top {keep_fraction:.0%}",
                        marker={"color": "#16a34a"},
                        opacity=0.80,
                    ),
                    row=1,
                    col=2,
                )
                fig.add_vline(
                    x=sharpness_threshold,
                    line_dash="dash",
                    line_color="#dc2626",
                    annotation_text=f"q-thresh {sharpness_threshold:.2f}",
                    annotation_position="top left",
                    row=1,
                    col=2,
                )
            else:
                fig.add_trace(
                    go.Histogram(
                        x=sharp_vals,
                        nbinsx=sharp_bins,
                        name="sharpness",
                        marker={"color": "#0ea5e9"},
                        opacity=0.85,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

        fig.update_xaxes(title_text="dist", row=1, col=1)
        fig.update_xaxes(title_text="frame1_clearness", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_layout(
            barmode="overlay",
            height=360,
            margin={"l": 20, "r": 20, "t": 40, "b": 10},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.caption("Motion (`dist`) histogram")
            if dist_vals.empty:
                st.caption("No valid values.")
            else:
                hist, edges = np.histogram(dist_vals.to_numpy(dtype=float), bins=dist_bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                hist_df = pd.DataFrame({"bin_center": centers, "count": hist})
                st.bar_chart(hist_df.set_index("bin_center")["count"])
                if np.isfinite(dist_q05) and np.isfinite(dist_q95):
                    st.caption(f"q05={dist_q05:.3f}, q95={dist_q95:.3f}")

        with c2:
            st.caption("Frame1 sharpness histogram")
            if sharp_vals.empty:
                st.caption("No valid values.")
            else:
                hist, edges = np.histogram(sharp_vals.to_numpy(dtype=float), bins=sharp_bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                hist_df = pd.DataFrame({"bin_center": centers, "count": hist})
                st.bar_chart(hist_df.set_index("bin_center")["count"])
                if threshold_available and np.isfinite(sharpness_threshold):
                    st.caption(
                        f"Quantile threshold (top {keep_fraction:.0%} kept): > {sharpness_threshold:.2f}"
                    )


def _prefetch_thumb_names(
    paths: List[Optional[str]], thumb_width: int, image_workers: int
) -> Dict[str, Optional[str]]:
    unique_paths = sorted({p for p in paths if p})
    encoded: Dict[str, Optional[str]] = {}
    if not unique_paths:
        return encoded

    max_workers = max(1, int(image_workers))
    if max_workers == 1 or len(unique_paths) == 1:
        for p in unique_paths:
            try:
                encoded[p] = ensure_thumbnail_file(p, max_width=thumb_width)
            except Exception:
                encoded[p] = None
        return encoded

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(ensure_thumbnail_file, p, max_width=thumb_width): p for p in unique_paths
        }
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                encoded[p] = fut.result()
            except Exception:
                encoded[p] = None
    return encoded


@cache_data(show_spinner=False, max_entries=12)
def _build_pair_entries(
    df: pd.DataFrame,
    frame_lookup: Dict[str, Optional[str]],
    dist_col: str,
    motion_lo: float,
    motion_hi: float,
    clear_q1: float,
    clear_q2: float,
    thumb_width: int,
    image_workers: int,
    first_frame_only: bool,
) -> tuple[List[dict], Dict[str, Optional[str]]]:
    rows = list(df.itertuples(index=False))
    frame_key_to_path: Dict[str, Optional[str]] = {}
    for row in rows:
        f1 = str(row.frame1)
        frame_key_to_path[f1] = frame_lookup.get(f1)
        if not first_frame_only:
            f2 = str(row.frame2)
            frame_key_to_path[f2] = frame_lookup.get(f2)

    frame_paths = list(frame_key_to_path.values())
    encoded_by_path = _prefetch_thumb_names(
        paths=frame_paths, thumb_width=thumb_width, image_workers=image_workers
    )
    image_by_frame_key: Dict[str, Optional[str]] = {}
    for k, p in frame_key_to_path.items():
        image_by_frame_key[k] = encoded_by_path.get(p) if p else None

    entries: List[dict] = []
    if first_frame_only:
        for i, row in enumerate(rows[:-1]):
            next_row = rows[i + 1]
            frame1 = str(row.frame1)
            next_frame1 = str(next_row.frame1)
            dist_value = _safe_float(getattr(row, dist_col))
            norm = _motion_norm(dist_value, motion_lo, motion_hi)
            motion_color = _motion_color(norm)
            motion_level = _motion_level(norm)
            drop_value = bool(getattr(row, "drop", False))

            c1 = _safe_float(getattr(row, "frame1_clearness", np.nan))
            next_c1 = _safe_float(getattr(next_row, "frame1_clearness", np.nan))
            c1_label, c1_color = _clearness_label_color(c1, clear_q1, clear_q2)
            c2_label, c2_color = _clearness_label_color(next_c1, clear_q1, clear_q2)
            clearness_pair = (
                c1 + next_c1 if np.isfinite(c1) and np.isfinite(next_c1) else float("nan")
            )

            entries.append(
                {
                    "pair_idx": int(row.pair_idx),
                    "drop": drop_value,
                    "frame1": frame1,
                    "frame2": next_frame1,
                    "next_frame1": next_frame1,
                    "dist": _fmt(dist_value, 3),
                    "motion_norm": norm,
                    "motion_color": motion_color,
                    "motion_level": motion_level,
                    "c1": _fmt(c1, 2),
                    "c2": _fmt(next_c1, 2),
                    "c1_label": c1_label,
                    "c2_label": c2_label,
                    "c1_color": c1_color,
                    "c2_color": c2_color,
                    "clearness_sum": _fmt(clearness_pair, 2),
                }
            )
    else:
        for row in rows:
            frame1 = str(row.frame1)
            frame2 = str(row.frame2)
            dist_value = _safe_float(getattr(row, dist_col))
            norm = _motion_norm(dist_value, motion_lo, motion_hi)
            motion_color = _motion_color(norm)
            motion_level = _motion_level(norm)

            c1 = _safe_float(getattr(row, "frame1_clearness", np.nan))
            c2 = _safe_float(getattr(row, "frame2_clearness", np.nan))
            c1_label, c1_color = _clearness_label_color(c1, clear_q1, clear_q2)
            c2_label, c2_color = _clearness_label_color(c2, clear_q1, clear_q2)

            entries.append(
                {
                    "pair_idx": int(row.pair_idx),
                    "drop": bool(getattr(row, "drop", False)),
                    "frame1": frame1,
                    "frame2": frame2,
                    "next_frame1": frame2,
                    "dist": _fmt(dist_value, 3),
                    "motion_norm": norm,
                    "motion_color": motion_color,
                    "motion_level": motion_level,
                    "c1": _fmt(c1, 2),
                    "c2": _fmt(c2, 2),
                    "c1_label": c1_label,
                    "c2_label": c2_label,
                    "c1_color": c1_color,
                    "c2_color": c2_color,
                    "clearness_sum": _fmt(_safe_float(getattr(row, "clearness_sum", np.nan)), 2),
                }
            )
    return entries, image_by_frame_key


def _pick_best_if_worth_it(
    group_df: pd.DataFrame,
    value_col: str = "sharpness",
    upside_thresh: float = 0.5,
    eps: float = 1e-12,
) -> pd.DataFrame:
    # Mirrors sample_frames_ecc.py::pick_best_if_worth_it.
    values = pd.to_numeric(group_df[value_col], errors="coerce").to_numpy(dtype=float)
    n = len(group_df)
    if n == 0:
        return group_df

    mid_idx = n // 2
    valid_mask = np.isfinite(values)
    if not valid_mask.any():
        return group_df.iloc[[mid_idx]]

    valid_indices = np.flatnonzero(valid_mask)
    best_idx = int(valid_indices[np.argmax(values[valid_mask])])

    mid_val = values[mid_idx]
    if not np.isfinite(mid_val):
        mid_val = float(np.nanmedian(values[valid_mask]))

    q1, q3 = np.nanpercentile(values[valid_mask], [25, 75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr < eps:
        upside = 0.0
    else:
        upside = float((values[best_idx] - mid_val) / (iqr + eps))

    chosen_idx = best_idx if upside >= upside_thresh else mid_idx
    if not np.isfinite(values[chosen_idx]):
        chosen_idx = best_idx
    return group_df.iloc[[int(chosen_idx)]]


def sample_frames_by_cumulative_motion(
    df: pd.DataFrame,
    motion_threshold: float,
    dist_col: str,
    sharpness_col: str,
    upside_thresh: float = 0.5,
) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()

    threshold = max(float(motion_threshold), 1e-6)
    work_df = df.copy().sort_values("pair_idx").reset_index(drop=True)
    work_df["dist_for_sampling"] = (
        pd.to_numeric(work_df[dist_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    work_df["sharpness_for_sampling"] = pd.to_numeric(work_df[sharpness_col], errors="coerce")
    # Mirrors sample_frames_ecc.py::sample_local_distance:
    # min_d_idx = (dist.cumsum() / dist_interval).astype(int)
    work_df["cum_motion"] = work_df["dist_for_sampling"].cumsum()
    work_df["min_d_idx"] = (work_df["cum_motion"] / threshold).astype(int)
    work_df["motion_bucket"] = work_df["min_d_idx"]

    sampled_df = (
        work_df.groupby("min_d_idx", group_keys=False)
        .apply(
            lambda g: _pick_best_if_worth_it(
                g, value_col="sharpness_for_sampling", upside_thresh=upside_thresh
            ).assign(
                dist=g["dist_for_sampling"].sum(),
                n_frames=len(g),
                bucket_motion=g["dist_for_sampling"].sum(),
                bucket_frames=len(g),
            )
        )
        .reset_index(drop=True)
    )
    sampled_df = sampled_df.sort_values("pair_idx").reset_index(drop=True)
    return sampled_df


@cache_data(show_spinner=False, max_entries=24)
def _build_sampled_frame_entries(
    sampled_df: pd.DataFrame,
    frame_lookup: Dict[str, Optional[str]],
    thumb_width: int,
    image_workers: int,
) -> tuple[List[dict], Dict[str, Optional[str]]]:
    rows = list(sampled_df.itertuples(index=False))
    frame_key_to_path: Dict[str, Optional[str]] = {}
    for row in rows:
        frame_key = str(row.frame1)
        frame_key_to_path[frame_key] = frame_lookup.get(frame_key)

    frame_paths = list(frame_key_to_path.values())
    encoded_by_path = _prefetch_thumb_names(
        paths=frame_paths, thumb_width=thumb_width, image_workers=image_workers
    )
    image_by_frame_key: Dict[str, Optional[str]] = {}
    for key, path in frame_key_to_path.items():
        image_by_frame_key[key] = encoded_by_path.get(path) if path else None

    entries: List[dict] = []
    for row in rows:
        entries.append(
            {
                "pair_idx": int(row.pair_idx),
                "frame1": str(row.frame1),
                "motion_bucket": int(getattr(row, "motion_bucket", -1)),
                "bucket_motion": _fmt(_safe_float(getattr(row, "bucket_motion", np.nan)), 3),
                "bucket_frames": int(getattr(row, "bucket_frames", 1)),
                "cum_motion": _fmt(_safe_float(getattr(row, "cum_motion", np.nan)), 3),
                "sharpness": _fmt(_safe_float(getattr(row, "sharpness_for_sampling", np.nan)), 2),
            }
    )
    return entries, image_by_frame_key


def copy_sampled_frames_to_output(
    sampled_df: pd.DataFrame,
    frame_lookup: Dict[str, Optional[str]],
    output_dir_str: str,
) -> Dict[str, int]:
    output_clean = output_dir_str.strip()
    if not output_clean:
        raise ValueError("Output directory is empty.")

    output_dir = _resolve_path(output_clean)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    failed = 0
    overwritten = 0

    for frame_key in sampled_df["frame1"].astype(str).tolist():
        src_path_str = frame_lookup.get(frame_key)
        if not src_path_str:
            missing += 1
            continue

        src_path = Path(src_path_str)
        if not src_path.exists():
            missing += 1
            continue

        dst_path = output_dir / src_path.name
        if dst_path.exists():
            overwritten += 1

        try:
            # Keep copy behavior aligned with sample_frames_ecc.py.
            shutil.copy(src_path, dst_path)
            copied += 1
        except Exception:
            failed += 1

    return {
        "copied": int(copied),
        "missing": int(missing),
        "failed": int(failed),
        "overwritten": int(overwritten),
    }


def render_horizontal_pair_ribbon(
    entries: List[dict],
    thumb_name_by_frame_key: Dict[str, Optional[str]],
    thumb_server_port: int,
    heatmap_cell_width: int = 2,
    scroll_sync_ratio: int = 50,
    show_first_frame_only: bool = False,
    height: int = 830,
) -> None:
    if not entries:
        st.info("No pairs available for the ribbon.")
        return

    entries_json = json.dumps(entries, ensure_ascii=True)
    image_json = json.dumps(thumb_name_by_frame_key, ensure_ascii=True)
    ribbon_title = (
        "Horizontal scrollable first-frame ribbon (stats use next frame1)."
        if show_first_frame_only
        else "Horizontal scrollable frame-pair ribbon."
    )

    base_heat_cell_width = max(1, int(heatmap_cell_width))
    sync_ratio = max(1, int(scroll_sync_ratio))

    html_block = """
    <style>
      .ribbon-root {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 8px;
        background: #ffffff;
      }
      .ribbon-root, .ribbon-root * {
        box-sizing: border-box;
      }
      .ribbon-title {
        font-size: 12px;
        color: #374151;
        margin: 2px 2px 8px 2px;
      }
      .pair-ribbon {
        position: relative;
        overflow-x: auto;
        overflow-y: hidden;
        height: 620px;
        padding: 8px 4px 0 4px;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #f8fafc;
      }
      .pair-track {
        position: relative;
        height: 606px;
      }
      .pair-layer {
        position: absolute;
        inset: 0 0 0 0;
      }
      .pair-card {
        position: absolute;
        top: 0;
        box-sizing: border-box;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 10px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.07);
        overflow: hidden;
      }
      .pair-card-drop {
        border-color: #ef4444;
        box-shadow: 0 0 0 1px #ef4444 inset;
      }
      .pair-meta {
        font-size: 12px;
        color: #374151;
        margin-bottom: 8px;
        font-weight: 600;
      }
      .pair-frames {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 8px;
        min-width: 0;
      }
      .pair-frames-single {
        grid-template-columns: minmax(0, 1fr);
      }
      .frame-cell {
        display: flex;
        flex-direction: column;
        gap: 6px;
        min-width: 0;
      }
      .frame-cell img, .img-missing {
        display: block;
        width: 100%;
        height: 390px;
        max-width: 100%;
        object-fit: contain;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background: #f3f4f6;
      }
      .img-missing {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6b7280;
        font-size: 12px;
      }
      .frame-name {
        font-size: 11px;
        color: #374151;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .clear-tag {
        display: inline-block;
        border-radius: 999px;
        color: #fff;
        font-size: 11px;
        padding: 2px 8px;
        width: fit-content;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .motion-row {
        margin-top: 10px;
        display: grid;
        grid-template-columns: 18px minmax(0, 1fr) auto auto;
        gap: 8px;
        align-items: center;
        min-width: 0;
      }
      .motion-arrow {
        font-size: 16px;
        font-weight: 700;
      }
      .motion-track {
        height: 8px;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
        min-width: 0;
      }
      .motion-fill {
        height: 100%;
      }
      .motion-chip {
        color: #ffffff;
        font-size: 11px;
        border-radius: 999px;
        padding: 2px 8px;
        font-weight: 600;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .motion-level {
        font-size: 11px;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .sum-line {
        margin-top: 8px;
        color: #6b7280;
        font-size: 11px;
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .link-line {
        margin-top: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        min-width: 0;
      }
      .link-title {
        color: #4b5563;
        font-size: 11px;
        min-width: 0;
        flex: 1 1 auto;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .link-target {
        color: #111827;
        display: inline-block;
        max-width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        vertical-align: bottom;
      }
      .heat-title {
        margin: 10px 2px 6px 2px;
        font-size: 12px;
        color: #374151;
      }
      .heat-scroll {
        overflow-x: auto;
        overflow-y: hidden;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 4px;
        background: #f8fafc;
      }
      .heat-strip {
        position: relative;
        display: flex;
        gap: 1px;
      }
      .heat-cell {
        flex: 0 0 __BASE_HEAT_CELL_WIDTH__px;
        height: 26px;
        border-radius: 2px;
      }
      .heat-cell.drop {
        outline: 1px solid #111827;
      }
      .heat-window {
        position: absolute;
        top: 0;
        height: 26px;
        border: 2px solid rgba(17, 24, 39, 0.9);
        background: rgba(17, 24, 39, 0.12);
        border-radius: 3px;
        pointer-events: none;
        box-sizing: border-box;
      }
      .heat-legend {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 8px;
        font-size: 11px;
        color: #4b5563;
      }
      .heat-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
      }
      .heat-axis {
        margin-top: 6px;
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 8px;
        align-items: center;
        color: #6b7280;
        font-size: 11px;
      }
      .heat-axis-mid {
        text-align: center;
      }
      .heat-status {
        margin-top: 4px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        color: #6b7280;
        font-size: 11px;
      }
      .heat-status-right {
        text-align: right;
      }
    </style>
    <div class="ribbon-root">
      <div class="ribbon-title">__RIBBON_TITLE__</div>
      <div id="pair-ribbon" class="pair-ribbon">
        <div id="pair-track" class="pair-track">
          <div id="pair-layer" class="pair-layer"></div>
        </div>
      </div>
      <div class="heat-title">Motion heatmap timeline (synced with ribbon scroll).</div>
      <div id="heat-scroll" class="heat-scroll">
        <div id="heat-strip" class="heat-strip"></div>
      </div>
      <div class="heat-legend">
        <span><span class="heat-dot" style="background:hsl(120,82%,43%);"></span> low motion</span>
        <span><span class="heat-dot" style="background:hsl(60,82%,43%);"></span> medium</span>
        <span><span class="heat-dot" style="background:hsl(0,82%,43%);"></span> high motion</span>
        <span><span class="heat-dot" style="background:#111827;"></span> outlined = drop</span>
      </div>
      <div class="heat-axis">
        <span id="heat-start"></span>
        <span id="heat-mid" class="heat-axis-mid"></span>
        <span id="heat-end"></span>
      </div>
      <div class="heat-status">
        <span id="heat-visible"></span>
        <span id="heat-rendered" class="heat-status-right"></span>
      </div>
    </div>
    <script>
      const entries = __RIBBON_ENTRIES__;
      const thumbMap = __RIBBON_IMAGES__;
      const thumbPort = __THUMB_PORT__;
      const syncRatio = __SYNC_RATIO__;
      const baseHeatCellW = __BASE_HEAT_CELL_WIDTH__;
      const showFirstOnly = __SHOW_FIRST_ONLY__;
      const root = document.getElementById("pair-ribbon");
      const pairTrack = document.getElementById("pair-track");
      const pairLayer = document.getElementById("pair-layer");
      const heatScroll = document.getElementById("heat-scroll");
      const heatStrip = document.getElementById("heat-strip");
      const heatVisibleLabel = document.getElementById("heat-visible");
      const heatRenderedLabel = document.getElementById("heat-rendered");
      const CARD_W = 430;
      const CARD_GAP = 18;
      const CARD_STRIDE = CARD_W + CARD_GAP;
      const OVERSCAN = 8;

      const getThumbBase = () => {
        try {
          const parentLoc = (window.parent && window.parent.location) ? window.parent.location : window.location;
          const proto = (parentLoc.protocol === "http:" || parentLoc.protocol === "https:")
            ? parentLoc.protocol
            : "http:";
          const host = parentLoc.hostname || "127.0.0.1";
          return `${proto}//${host}:${thumbPort}`;
        } catch (e) {
          return `http://127.0.0.1:${thumbPort}`;
        }
      };
      const thumbBase = getThumbBase();

      const esc = (s) => String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replace(/"/g, "&quot;")
        .replaceAll("'", "&#39;");

      const thumbUrl = (name) => {
        if (!name) return null;
        return `${thumbBase}/${encodeURIComponent(name)}`;
      };

      const imgOrMissing = (thumbName, alt) => {
        const src = thumbUrl(thumbName);
        if (src) return `<img src="${src}" alt="${esc(alt)}" loading="lazy">`;
        return `<div class="img-missing">missing</div>`;
      };

      const heatCells = [];
      for (const e of entries) {
        const hc = document.createElement("div");
        hc.className = e.drop ? "heat-cell drop" : "heat-cell";
        hc.style.background = e.motion_color;
        hc.title = showFirstOnly
          ? `Pair #${e.pair_idx} | ${e.frame1} -> next frame1 ${e.next_frame1} | motion=${e.dist}`
          : `Pair #${e.pair_idx} | motion=${e.dist}`;
        heatStrip.appendChild(hc);
        heatCells.push(hc);
      }
      const heatWindow = document.createElement("div");
      heatWindow.className = "heat-window";
      heatStrip.appendChild(heatWindow);

      const totalWidth = Math.max(1, entries.length * CARD_STRIDE);
      pairTrack.style.width = `${totalWidth}px`;

      const layoutHeatStrip = () => {
        if (entries.length === 0) {
          heatStrip.style.width = "1px";
          return;
        }
        const desiredHeatContentWidth = Math.max(
          entries.length * (baseHeatCellW + 1),
          ((root.scrollWidth - root.clientWidth) / syncRatio) + heatScroll.clientWidth
        );
        const effectiveCellW = Math.max(
          baseHeatCellW,
          (desiredHeatContentWidth - Math.max(0, entries.length - 1)) / entries.length
        );
        for (const cell of heatCells) {
          cell.style.flex = `0 0 ${effectiveCellW}px`;
        }
        const contentW = Math.max(
          1,
          entries.length * effectiveCellW + Math.max(0, entries.length - 1)
        );
        heatStrip.style.width = `${contentW}px`;
      };

      let renderedFirst = 0;
      let renderedLast = -1;

      const renderVisibleCards = () => {
        const viewportLeft = root.scrollLeft;
        const viewportRight = viewportLeft + root.clientWidth;
        const first = Math.max(0, Math.floor(viewportLeft / CARD_STRIDE) - OVERSCAN);
        const last = Math.min(
          entries.length,
          Math.ceil(viewportRight / CARD_STRIDE) + OVERSCAN
        );
        renderedFirst = first;
        renderedLast = Math.max(first, last - 1);

        pairLayer.innerHTML = "";
        const frag = document.createDocumentFragment();
        for (let i = first; i < last; i++) {
          const e = entries[i];
          const card = document.createElement("div");
          card.className = e.drop ? "pair-card pair-card-drop" : "pair-card";
          card.style.left = `${i * CARD_STRIDE}px`;
          card.style.width = `${CARD_W}px`;

          const fillWidth = Math.round((Number(e.motion_norm) || 0) * 100);
          if (showFirstOnly) {
            card.innerHTML = `
              <div class="pair-meta">Pair #${e.pair_idx}</div>
              <div class="pair-frames pair-frames-single">
                <div class="frame-cell">
                  ${imgOrMissing(thumbMap[e.frame1], e.frame1)}
                  <div class="frame-name">${esc(e.frame1)}</div>
                  <span class="clear-tag" style="background:${e.c1_color};">${e.c1_label} (${e.c1})</span>
                </div>
              </div>
              <div class="link-line">
                <div class="link-title">stats target:
                  <span class="link-target">${esc(e.next_frame1)}</span>
                </div>
              </div>
              <div class="motion-row">
                <span class="motion-arrow" style="color:${e.motion_color};">→</span>
                <div class="motion-track">
                  <div class="motion-fill" style="width:${fillWidth}%; background:${e.motion_color};"></div>
                </div>
                <span class="motion-chip" style="background:${e.motion_color};">${e.dist}</span>
                <span class="motion-level">${e.motion_level}</span>
              </div>
              <div class="sum-line">clearness_sum(frame1 + next frame1)=${e.clearness_sum}</div>
            `;
          } else {
            card.innerHTML = `
              <div class="pair-meta">Pair #${e.pair_idx}</div>
              <div class="pair-frames">
                <div class="frame-cell">
                  ${imgOrMissing(thumbMap[e.frame1], e.frame1)}
                  <div class="frame-name">${esc(e.frame1)}</div>
                  <span class="clear-tag" style="background:${e.c1_color};">${e.c1_label} (${e.c1})</span>
                </div>
                <div class="frame-cell">
                  ${imgOrMissing(thumbMap[e.frame2], e.frame2)}
                  <div class="frame-name">${esc(e.frame2)}</div>
                  <span class="clear-tag" style="background:${e.c2_color};">${e.c2_label} (${e.c2})</span>
                </div>
              </div>
              <div class="motion-row">
                <span class="motion-arrow" style="color:${e.motion_color};">→</span>
                <div class="motion-track">
                  <div class="motion-fill" style="width:${fillWidth}%; background:${e.motion_color};"></div>
                </div>
                <span class="motion-chip" style="background:${e.motion_color};">${e.dist}</span>
                <span class="motion-level">${e.motion_level}</span>
              </div>
              <div class="sum-line">clearness_sum=${e.clearness_sum}</div>
            `;
          }
          frag.appendChild(card);
        }
        pairLayer.appendChild(frag);
      };

      const updateHeatViewport = () => {
        if (entries.length === 0) return;
        const rootScrollable = Math.max(1, root.scrollWidth - root.clientWidth);
        const heatContentW = Math.max(1, heatStrip.scrollWidth);
        const markerW = Math.max(8, (root.clientWidth / Math.max(1, root.scrollWidth)) * heatContentW);
        const markerLeft = (root.scrollLeft / rootScrollable) * Math.max(0, heatContentW - markerW);
        heatWindow.style.width = `${markerW}px`;
        heatWindow.style.left = `${markerLeft}px`;

        const visFirstIdx = Math.max(0, Math.floor(root.scrollLeft / CARD_STRIDE));
        const visLastIdx = Math.max(
          visFirstIdx,
          Math.min(entries.length - 1, Math.ceil((root.scrollLeft + root.clientWidth) / CARD_STRIDE) - 1)
        );
        heatVisibleLabel.textContent = `Visible pairs: #${entries[visFirstIdx].pair_idx}..#${entries[visLastIdx].pair_idx}`;

        if (renderedLast >= renderedFirst) {
          heatRenderedLabel.textContent = `Rendered cards: #${entries[renderedFirst].pair_idx}..#${entries[renderedLast].pair_idx}`;
        } else {
          heatRenderedLabel.textContent = "";
        }
      };

      if (entries.length > 0) {
        const first = entries[0].pair_idx;
        const last = entries[entries.length - 1].pair_idx;
        const mid = entries[Math.floor(entries.length / 2)].pair_idx;
        document.getElementById("heat-start").textContent = `Pair #${first}`;
        document.getElementById("heat-mid").textContent = `Pair #${mid}`;
        document.getElementById("heat-end").textContent = `Pair #${last}`;
      }

      layoutHeatStrip();
      renderVisibleCards();
      updateHeatViewport();
      heatScroll.scrollLeft = root.scrollLeft / syncRatio;

      let syncing = false;
      root.addEventListener("scroll", () => {
        renderVisibleCards();
        if (syncing) return;
        syncing = true;
        heatScroll.scrollLeft = root.scrollLeft / syncRatio;
        syncing = false;
        updateHeatViewport();
      });
      heatScroll.addEventListener("scroll", () => {
        if (syncing) return;
        syncing = true;
        root.scrollLeft = heatScroll.scrollLeft * syncRatio;
        syncing = false;
        renderVisibleCards();
        updateHeatViewport();
      });

      window.addEventListener("resize", () => {
        layoutHeatStrip();
        renderVisibleCards();
        updateHeatViewport();
      });
    </script>
    """
    html_block = html_block.replace("__RIBBON_ENTRIES__", entries_json)
    html_block = html_block.replace("__RIBBON_IMAGES__", image_json)
    html_block = html_block.replace("__THUMB_PORT__", str(int(thumb_server_port)))
    html_block = html_block.replace("__BASE_HEAT_CELL_WIDTH__", str(base_heat_cell_width))
    html_block = html_block.replace("__SYNC_RATIO__", str(sync_ratio))
    html_block = html_block.replace("__RIBBON_TITLE__", ribbon_title)
    html_block = html_block.replace("__SHOW_FIRST_ONLY__", "true" if show_first_frame_only else "false")
    components.html(html_block, height=height, scrolling=False)


def render_sampled_frame_ribbon(
    entries: List[dict],
    thumb_name_by_frame_key: Dict[str, Optional[str]],
    thumb_server_port: int,
    title: str,
    height: int = 430,
) -> None:
    if not entries:
        st.info("No sampled frames to display.")
        return

    entries_json = json.dumps(entries, ensure_ascii=True)
    image_json = json.dumps(thumb_name_by_frame_key, ensure_ascii=True)
    html_block = """
    <style>
      .sf-root {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 8px;
        background: #ffffff;
      }
      .sf-title {
        font-size: 12px;
        color: #374151;
        margin: 2px 2px 8px 2px;
      }
      .sf-scroll {
        overflow-x: auto;
        overflow-y: hidden;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #f8fafc;
        padding: 8px;
      }
      .sf-track {
        display: flex;
        gap: 12px;
        width: max-content;
      }
      .sf-card {
        width: 230px;
        border: 1px solid #d1d5db;
        border-radius: 10px;
        padding: 8px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
      }
      .sf-image, .sf-missing {
        display: block;
        width: 100%;
        height: 220px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        object-fit: contain;
        background: #f3f4f6;
      }
      .sf-missing {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6b7280;
        font-size: 12px;
      }
      .sf-name {
        margin-top: 6px;
        font-size: 11px;
        color: #374151;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .sf-meta {
        margin-top: 6px;
        font-size: 11px;
        color: #4b5563;
        line-height: 1.45;
      }
      .sf-chip {
        margin-top: 6px;
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        color: #ffffff;
        background: #111827;
      }
    </style>
    <div class="sf-root">
      <div class="sf-title">__SAMPLED_TITLE__</div>
      <div class="sf-scroll">
        <div id="sf-track" class="sf-track"></div>
      </div>
    </div>
    <script>
      const entries = __SAMPLED_ENTRIES__;
      const thumbMap = __SAMPLED_IMAGES__;
      const thumbPort = __THUMB_PORT__;
      const track = document.getElementById("sf-track");

      const getThumbBase = () => {
        try {
          const parentLoc = (window.parent && window.parent.location) ? window.parent.location : window.location;
          const proto = (parentLoc.protocol === "http:" || parentLoc.protocol === "https:")
            ? parentLoc.protocol
            : "http:";
          const host = parentLoc.hostname || "127.0.0.1";
          return `${proto}//${host}:${thumbPort}`;
        } catch (e) {
          return `http://127.0.0.1:${thumbPort}`;
        }
      };
      const thumbBase = getThumbBase();

      const esc = (s) => String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replace(/"/g, "&quot;")
        .replaceAll("'", "&#39;");

      const thumbUrl = (name) => {
        if (!name) return null;
        return `${thumbBase}/${encodeURIComponent(name)}`;
      };

      for (const e of entries) {
        const card = document.createElement("div");
        card.className = "sf-card";
        const thumbName = thumbMap[e.frame1];
        const src = thumbUrl(thumbName);
        const imageHtml = src
          ? `<img class="sf-image" src="${src}" alt="${esc(e.frame1)}" loading="lazy">`
          : `<div class="sf-missing">missing</div>`;
        card.innerHTML = `
          ${imageHtml}
          <div class="sf-name">${esc(e.frame1)}</div>
          <span class="sf-chip">pair #${e.pair_idx}</span>
          <div class="sf-meta">
            bucket #${e.motion_bucket} | frames=${e.bucket_frames}<br>
            bucket_motion=${e.bucket_motion} | cum_motion=${e.cum_motion}<br>
            sharpness=${e.sharpness}
          </div>
        `;
        track.appendChild(card);
      }
    </script>
    """
    html_block = html_block.replace("__SAMPLED_TITLE__", title)
    html_block = html_block.replace("__SAMPLED_ENTRIES__", entries_json)
    html_block = html_block.replace("__SAMPLED_IMAGES__", image_json)
    html_block = html_block.replace("__THUMB_PORT__", str(int(thumb_server_port)))
    components.html(html_block, height=height, scrolling=False)


def render_motion_heatmap_timeline(
    df: pd.DataFrame, dist_col: str, motion_lo: float, motion_hi: float, cell_width: int = 3, height: int = 180
) -> None:
    if df.empty:
        st.info("No timeline entries available.")
        return

    timeline_data: List[dict] = []
    for row in df.itertuples(index=False):
        dist_value = _safe_float(getattr(row, dist_col))
        norm = _motion_norm(dist_value, motion_lo, motion_hi)
        timeline_data.append(
            {
                "pair_idx": int(row.pair_idx),
                "color": _motion_color(norm),
                "drop": bool(getattr(row, "drop", False)),
            }
        )

    data_json = json.dumps(timeline_data, ensure_ascii=True)
    html_block = """
    <style>
      .hm-root {
        border: 1px solid #d1d5db;
        border-radius: 10px;
        background: #ffffff;
        padding: 10px;
      }
      .hm-label {
        font-size: 12px;
        color: #374151;
        margin-bottom: 8px;
      }
      .hm-scroll {
        overflow-x: auto;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
        padding: 4px 4px 2px 4px;
        background: #f8fafc;
      }
      .hm-canvas {
        display: block;
      }
      .hm-legend {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 8px;
        font-size: 11px;
        color: #4b5563;
      }
      .hm-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
      }
      .hm-axis {
        margin-top: 6px;
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 8px;
        align-items: center;
        color: #6b7280;
        font-size: 11px;
      }
      .hm-axis-mid {
        text-align: center;
      }
    </style>
    <div id="hm-root" class="hm-root">
      <div class="hm-label">Motion heatmap timeline (green = low motion, red = high motion).</div>
      <div class="hm-scroll">
        <canvas id="hm-canvas" class="hm-canvas"></canvas>
      </div>
      <div class="hm-legend">
        <span><span class="hm-dot" style="background:hsl(120,82%,43%);"></span> low motion</span>
        <span><span class="hm-dot" style="background:hsl(60,82%,43%);"></span> medium</span>
        <span><span class="hm-dot" style="background:hsl(0,82%,43%);"></span> high motion</span>
        <span><span class="hm-dot" style="background:#111827;"></span> outlined = drop</span>
      </div>
      <div class="hm-axis">
        <span id="hm-start"></span>
        <span id="hm-mid" class="hm-axis-mid"></span>
        <span id="hm-end"></span>
      </div>
    </div>
    <script>
      const data = __TIMELINE_DATA__;
      const canvas = document.getElementById("hm-canvas");
      const ctx = canvas.getContext("2d");
      const cellW = __CELL_WIDTH__;
      const cellH = 30;
      const canvasW = Math.max(1, data.length * cellW);

      canvas.width = canvasW;
      canvas.height = cellH;

      for (let i = 0; i < data.length; i++) {
        const d = data[i];
        const x = i * cellW;
        ctx.fillStyle = d.color;
        ctx.fillRect(x, 0, cellW, cellH);
        if (d.drop) {
          ctx.strokeStyle = "#111827";
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, 0.5, Math.max(1, cellW - 1), cellH - 1);
        }
      }

      if (data.length > 0) {
        const first = data[0].pair_idx;
        const last = data[data.length - 1].pair_idx;
        const mid = data[Math.floor(data.length / 2)].pair_idx;
        document.getElementById("hm-start").textContent = `Pair #${first}`;
        document.getElementById("hm-mid").textContent = `Pair #${mid}`;
        document.getElementById("hm-end").textContent = `Pair #${last}`;
      }
    </script>
    """
    html_block = html_block.replace("__TIMELINE_DATA__", data_json)
    html_block = html_block.replace("__CELL_WIDTH__", str(max(1, int(cell_width))))
    components.html(html_block, height=height, scrolling=False)


def build_timeline(df: pd.DataFrame, dist_col: str) -> None:
    st.subheader("Timeline")
    has_clearness = "clearness_sum" in df.columns and (
        pd.to_numeric(df["clearness_sum"], errors="coerce").notna().any()
    )
    if HAS_PLOTLY:
        if has_clearness:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.09,
                subplot_titles=("Motion", "Blur/Clearness"),
            )
        else:
            fig = make_subplots(
                rows=1,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.09,
                subplot_titles=("Motion",),
            )
        fig.add_trace(
            go.Scatter(
                x=df["pair_idx"],
                y=df[dist_col],
                mode="lines",
                name=dist_col,
                line={"width": 2},
            ),
            row=1, col=1,
        )
        if "median_flow_magnitude" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["pair_idx"],
                    y=df["median_flow_magnitude"],
                    mode="lines",
                    name="median_flow_magnitude",
                    line={"dash": "dot"},
                ),
                row=1, col=1,
            )
        if "drop" in df.columns:
            dropped = df[df["drop"]]
            if not dropped.empty:
                fig.add_trace(
                    go.Scatter(
                        x=dropped["pair_idx"],
                        y=dropped[dist_col],
                        mode="markers",
                        name="drop=True",
                        marker={"color": "red", "symbol": "x", "size": 8},
                    ),
                    row=1, col=1,
                )
        if has_clearness:
            fig.add_trace(
                go.Scatter(
                    x=df["pair_idx"],
                    y=df["clearness_sum"],
                    mode="lines",
                    name="clearness_sum",
                    line={"width": 2},
                ),
                row=2, col=1,
            )
            fig.update_xaxes(title_text="Pair Index", row=2, col=1)
            fig.update_yaxes(title_text="Clearness", row=2, col=1)
        else:
            fig.update_xaxes(title_text="Pair Index", row=1, col=1)
            fig.add_annotation(
                text="No clearness columns provided",
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.98,
                showarrow=False,
                font={"size": 11, "color": "#6b7280"},
            )
        fig.update_yaxes(title_text="Motion", row=1, col=1)
        fig.update_layout(height=560, margin={"l": 20, "r": 20, "t": 40, "b": 20})
        st.plotly_chart(fig, width="stretch")
    else:
        motion_cols = [dist_col]
        if "median_flow_magnitude" in df.columns:
            motion_cols.append("median_flow_magnitude")
        st.line_chart(df.set_index("pair_idx")[motion_cols])
        if has_clearness:
            st.line_chart(df.set_index("pair_idx")[["clearness_sum"]])
        else:
            st.caption("No clearness columns provided.")
        if "drop" in df.columns:
            st.caption(f"Drop pairs: {int(df['drop'].sum())}")


def main() -> None:
    st.set_page_config(page_title="Optical Flow Dashboard", layout="wide")
    st.title("Optical Flow Frame-Sampling Dashboard")
    st.caption(
        "Inspect consecutive frame pairs, estimated motion, and blur/clearness to spot outliers."
    )

    with st.sidebar:
        st.header("Inputs")
        cpu_hint = os.cpu_count() or 8
        default_workers = min(16, max(4, cpu_hint))
        df_path = st.text_input("Optical flow dataframe", str(DEFAULT_DF_PATH))
        blur_path = st.text_input("Blur dataframe (optional)", str(DEFAULT_BLUR_PATH))
        frames_dir = st.text_input("Frames directory (optional)", "")
        use_ffill_dist = st.checkbox("Use forward-filled dist for dropped rows", value=True)
        image_workers = int(
            st.slider("Image loading workers (1 = single-thread)", 1, 64, default_workers, 1)
        )
        ribbon_all_pairs = st.checkbox("Ribbon: include all pairs in selected window", value=True)
        ribbon_pairs_limit = int(
            st.number_input("Ribbon pair limit (used when checkbox above is off)", min_value=10, max_value=3000, value=300)
        )
        ribbon_thumb_width = int(
            st.slider("Ribbon thumbnail width (px)", min_value=100, max_value=260, value=170, step=10)
        )
        heatmap_cell_width = int(
            st.slider("Heatmap cell width (px)", min_value=1, max_value=8, value=2, step=1)
        )
        scroll_sync_ratio = int(
            st.slider("Scroll sync ratio (ribbon:px to heatmap:px)", min_value=1, max_value=200, value=50, step=1)
        )
        ribbon_first_frame_only = st.checkbox(
            "Ribbon: show only frame1 (stats use next frame1)",
            value=False,
        )
        show_only_drop = st.checkbox("Only show drop=True in gallery", value=False)
        st.header("Cumulative Motion Sampler")
        show_sampled_ribbon = st.checkbox("Show sampled-frames ribbon", value=True)
        cumulative_motion_threshold = float(
            st.number_input(
                "Cumulative motion threshold",
                min_value=0.1,
                value=60.0,
                step=1.0,
            )
        )
        sampled_keep_sharpness_quantile = float(
            st.slider(
                "Sampler sharpness keep fraction (1.0 = no prefilter)",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.01,
            )
        )
        sampler_upside_thresh = float(
            st.slider(
                "Sampler sharpness upside threshold (IQR units)",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.05,
            )
        )
        sampled_output_dir = st.text_input(
            "Sampler output folder (for copy button)",
            "",
        )

    try:
        df_raw = load_table(df_path)
    except Exception as exc:
        st.error(f"Failed to load optical-flow dataframe: {exc}")
        st.stop()

    blur_df: Optional[pd.DataFrame] = None
    blur_path_clean = blur_path.strip()
    if blur_path_clean:
        try:
            blur_df = load_table(blur_path_clean)
        except Exception:
            st.warning("Could not load blur dataframe. Continuing without it.")

    try:
        df = ensure_expected_columns(df_raw, blur_df)
    except Exception as exc:
        st.error(f"Dataframe validation failed: {exc}")
        st.stop()

    dist_col = "dist_for_timeline"
    df[dist_col] = df["dist"].astype(float)
    if use_ffill_dist and "drop" in df.columns:
        df[dist_col] = df[dist_col].where(~df["drop"]).ffill()

    frame_lookup = build_frame_lookup(df, frames_dir, blur_df)
    missing_paths = sum(1 for v in frame_lookup.values() if v is None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs", len(df))
    c2.metric("Drop=True", int(df["drop"].sum()))
    c3.metric("Missing Frame Paths", missing_paths)
    has_frame2_input = "frame2" in df_raw.columns
    has_clearness_f1 = any(
        c in df_raw.columns for c in ("clearness_frame1", "frame1_clearness", "clearness", "sharpness")
    )
    has_clearness_f2 = "clearness_frame2" in df_raw.columns or "frame2_clearness" in df_raw.columns
    has_clearness_inputs = bool(has_clearness_f1 and (has_clearness_f2 or not has_frame2_input))
    c4.metric("Clearness In Input", "yes" if has_clearness_inputs else "no")

    with st.expander("Columns", expanded=False):
        st.write(df.columns.tolist())
    with st.expander("Input API", expanded=False):
        required_str = ", ".join(f"`{c}`" for c in REQUIRED_INPUT_COLUMNS)
        st.markdown(
            f"Required columns: {required_str}.\n"
            "Aliases: `name` is accepted as `frame1`.\n"
            "Optional columns: `frame2`, `clearness_frame1`/`frame1_clearness`, "
            "`clearness`/`sharpness` (used as `frame1_clearness`), "
            "`clearness_frame2`/`frame2_clearness`, `drop`, and any additional metrics."
        )

    thumb_server_port = get_thumb_server_port()
    build_timeline(df, dist_col=dist_col)

    idx_min = int(df["pair_idx"].min())
    idx_max = int(df["pair_idx"].max())
    idx_range = st.slider(
        "Pair index window",
        min_value=idx_min,
        max_value=idx_max,
        value=(idx_min, idx_max),
    )

    window_df = df[(df["pair_idx"] >= idx_range[0]) & (df["pair_idx"] <= idx_range[1])]
    window_df = window_df.sort_values("pair_idx").reset_index(drop=True)
    if window_df.empty:
        st.info("No rows in the selected pair-index window.")
        st.stop()

    keep_fraction = float(np.clip(sampled_keep_sharpness_quantile, 0.1, 1.0))
    threshold_available, sharpness_threshold = _compute_sharpness_prefilter_threshold(
        sharpness_values=window_df["frame1_clearness"],
        keep_fraction=keep_fraction,
    )

    build_distribution_histograms(
        df=window_df,
        dist_col=dist_col,
        sharpness_col="frame1_clearness",
        keep_fraction=keep_fraction,
        threshold_available=threshold_available,
        sharpness_threshold=sharpness_threshold,
    )

    st.subheader("Horizontal Scrollable Pair Ribbon")

    selected_df = window_df.copy()
    if show_only_drop:
        selected_df = selected_df[selected_df["drop"]]
    if not ribbon_all_pairs:
        selected_df = selected_df.head(ribbon_pairs_limit)
    selected_df = selected_df.sort_values("pair_idx").reset_index(drop=True)

    if "frame2" not in selected_df.columns and not ribbon_first_frame_only:
        st.warning("`frame2` is missing in input; switching ribbon to frame1-only mode.")
        ribbon_first_frame_only = True

    motion_lo, motion_hi = _motion_bounds(df[dist_col])
    clear_q1, clear_q2 = _clearness_thresholds(df)

    can_render_pair_ribbon = True
    if selected_df.empty:
        st.info("No rows to show in the pair ribbon with current filters.")
        can_render_pair_ribbon = False
    if can_render_pair_ribbon and ribbon_first_frame_only and len(selected_df) < 2:
        st.info("Frame1-only pair ribbon needs at least 2 rows in the selected subset.")
        can_render_pair_ribbon = False

    if can_render_pair_ribbon:
        with st.spinner("Preparing ribbon thumbnails..."):
            ribbon_entries, ribbon_image_map = _build_pair_entries(
                df=selected_df,
                frame_lookup=frame_lookup,
                dist_col=dist_col,
                motion_lo=motion_lo,
                motion_hi=motion_hi,
                clear_q1=clear_q1,
                clear_q2=clear_q2,
                thumb_width=ribbon_thumb_width,
                image_workers=image_workers,
                first_frame_only=ribbon_first_frame_only,
            )
        if ribbon_entries:
            render_horizontal_pair_ribbon(
                ribbon_entries,
                ribbon_image_map,
                thumb_server_port=thumb_server_port,
                heatmap_cell_width=heatmap_cell_width,
                scroll_sync_ratio=scroll_sync_ratio,
                show_first_frame_only=ribbon_first_frame_only,
                height=830,
            )
        else:
            st.info("No rows to show after applying first-frame gallery mapping.")

    if show_sampled_ribbon:
        st.subheader("Cumulative-Motion Sampled Frames Ribbon")
        sampler_df = window_df.copy()
        sampler_df["sampler_dist"] = pd.to_numeric(sampler_df["dist"], errors="coerce")
        sampler_df["sampler_sharpness"] = pd.to_numeric(sampler_df["frame1_clearness"], errors="coerce")

        quantile_filter_applied = False
        if threshold_available and np.isfinite(sharpness_threshold):
            sampler_df = sampler_df[sampler_df["sampler_sharpness"] > sharpness_threshold]
            sampler_df = sampler_df.reset_index(drop=True)
            quantile_filter_applied = True

        if sampler_df.empty:
            st.info("No frames left for the sampler after applying the sharpness prefilter.")
        else:
            sampled_df = sample_frames_by_cumulative_motion(
                df=sampler_df,
                motion_threshold=cumulative_motion_threshold,
                dist_col="sampler_dist",
                sharpness_col="sampler_sharpness",
                upside_thresh=sampler_upside_thresh,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Frames In Window", len(window_df))
            m2.metric("Sampled Frames", len(sampled_df))
            reduction = 100.0 * (1.0 - (len(sampled_df) / max(1, len(window_df))))
            m3.metric("Reduction", f"{reduction:.1f}%")

            if quantile_filter_applied:
                st.caption(
                    "Sharpness prefilter active: "
                    f"top {keep_fraction:.2f} kept (threshold > {_fmt(sharpness_threshold, 2)})."
                )
            elif keep_fraction < 1.0:
                st.caption(
                    "Sharpness prefilter skipped because frame1 sharpness is unavailable in the selected window."
                )

            output_clean = sampled_output_dir.strip()
            output_resolved = _resolve_path(output_clean) if output_clean else None
            if output_resolved is not None:
                st.caption(f"Sampler copy destination: `{output_resolved}`")
            else:
                st.caption("Set `Sampler output folder` in the sidebar to enable copying.")

            copy_clicked = st.button(
                "Copy sampled frames",
                key="copy_sampled_frames_button",
                disabled=not output_clean,
            )
            if copy_clicked:
                try:
                    with st.spinner("Copying sampled frames..."):
                        copy_stats = copy_sampled_frames_to_output(
                            sampled_df=sampled_df,
                            frame_lookup=frame_lookup,
                            output_dir_str=output_clean,
                        )
                    st.success(
                        f"Copied {copy_stats['copied']} sampled frames to `{output_resolved}`."
                    )
                    if copy_stats["overwritten"] > 0:
                        st.warning(
                            f"{copy_stats['overwritten']} destination files were overwritten."
                        )
                    if copy_stats["missing"] > 0:
                        st.warning(
                            f"{copy_stats['missing']} sampled frames were skipped (source path missing)."
                        )
                    if copy_stats["failed"] > 0:
                        st.error(
                            f"{copy_stats['failed']} sampled frames failed to copy."
                        )
                except Exception as exc:
                    st.error(f"Copy operation failed: {exc}")

            with st.spinner("Preparing sampled-frame thumbnails..."):
                sampled_entries, sampled_image_map = _build_sampled_frame_entries(
                    sampled_df=sampled_df,
                    frame_lookup=frame_lookup,
                    thumb_width=ribbon_thumb_width,
                    image_workers=image_workers,
                )

            render_sampled_frame_ribbon(
                sampled_entries,
                sampled_image_map,
                thumb_server_port=thumb_server_port,
                title=(
                    "Frames sampled with cumulative motion threshold "
                    f"{cumulative_motion_threshold:.3f}"
                ),
                height=460,
            )


if __name__ == "__main__":
    if not _has_streamlit_context():
        print(
            "Run this dashboard with Streamlit, e.g.:\n"
            "  uv run --with streamlit --with plotly "
            "streamlit run preprocessing/optical_flow_dashboard/app.py"
        )
        raise SystemExit(1)
    main()
