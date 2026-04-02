# Optical Flow Dashboard

Simple Streamlit dashboard to inspect frame-pair optical-flow results from
`optical_flow_vis.ipynb` / `sample_frames_optical_flow.py`.

## What it shows

- Motion/blur plots (`dist`, `median_flow_magnitude`, `clearness_sum`)
- Horizontal scrollable ribbon of consecutive frame pairs:
  - color-coded motion indicator with value and low/medium/high tag
  - clearness tags per frame (`clear`, `medium`, `blurry`) with exact value
  - outlier highlight for `drop=True`
  - optional `frame1-only` gallery mode where cards show only `frame1` and stats
    are mapped to the next row's `frame1`
- Bottom motion heat timeline:
  - color-coded from low motion (green) to high motion (red)
  - fixed full-range heatmap strip (no hover interaction)
  - scroll sync ratio is configurable (default `50:1`)
- ECC-style cumulative motion sampler (from `sample_frames_ecc.py` logic):
  - configurable cumulative motion threshold per sampled frame
  - optional sharpness quantile prefilter and upside-threshold selection rule
  - separate sampled-frames ribbon to inspect sampled frame choices
- Multi-threaded thumbnail generation for faster ribbon/timeline rendering
  (configure worker count in sidebar)
- Ribbon defaults to all pairs in the selected index window
- Virtualized card rendering keeps the UI responsive for long sequences
- Taller ribbon cards for easier visual inspection

## Input API

Minimal required columns:

- `frame1`: path to first frame in the pair (absolute path recommended)
- `dist`: float motion/distance score per pair

Accepted aliases:

- `name` (used as `frame1`)

Optional columns:

- `frame2`: path to second frame in the pair (required only for two-frame ribbon cards)
- `clearness_frame1` or `frame1_clearness` (or `clearness` / `sharpness`, used as `frame1_clearness`)
- `clearness_frame2` or `frame2_clearness`
- `drop` (boolean outlier flag)
- any additional metrics (for your own experiments)

If clearness columns are missing, the app still works; clearness UI elements are
shown as unknown. You can also provide a blur dataframe with columns `name` and
`clearness` for fallback mapping.

## Run

From the `preprocessing` folder:

```bash
pip install -r optical_flow_dashboard/requirements.txt
streamlit run optical_flow_dashboard/app.py
```

From the outer uv project root (`/home/roman/ba/envgs`):

```bash
uv run --with streamlit --with plotly streamlit run preprocessing/optical_flow_dashboard/app.py
```

By default it tries:

- `distances_vw_rain_all_frames.pkl`
- `blur_score_vw_rain_all_frames.pkl`

You can override paths in the sidebar and optionally set a frames directory if
`frame1`/`frame2` are basenames only.
