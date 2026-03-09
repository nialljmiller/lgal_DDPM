#!/usr/bin/env python3
"""
analyse_eagle_embeddings.py
===========================
Comprehensive diagnostic and exploratory visualisation for the Golden Eagle
SimCLR embeddings produced by eagle_simclr.py.

Expects as primary input the {step:08d}_embeddings.csv written by
eagle_simclr.py's Trainer.infer(), which has columns:
    bird_id, window_start, h_0 … h_H-1, z_0 … z_Z-1

Usage (all plots):
    python analyse_eagle_embeddings.py \
        --logdir  logs/simclr_run1 \
        --catalogue golden_eai_outputs/bird_catalogue_enhanced.csv

Usage (skip slow plots):
    python analyse_eagle_embeddings.py \
        --logdir  logs/simclr_run1 \
        --catalogue golden_eai_outputs/bird_catalogue_enhanced.csv \
        --skip cluster_profiles,anomaly

Usage (only specific plots):
    python analyse_eagle_embeddings.py \
        --logdir  logs/simclr_run1 \
        --catalogue golden_eai_outputs/bird_catalogue_enhanced.csv \
        --only global_umap,cluster_umap

Optional flags:
    --step          INT     use this checkpoint step (default: latest)
    --outdir        PATH    where to save plots (default: logdir/analysis)
    --n_clusters    INT     number of clusters for K-means (default: 8)
    --n_plot_birds  INT     how many birds to include in per-bird panels (default: 16)
    --umap_sample   INT     subsample this many windows for UMAP (default: 20000)
    --space         STR     which embedding to use: "z" or "h" (default: "z")
    --skip          LIST    comma-separated plot IDs to skip
    --only          LIST    comma-separated plot IDs to run exclusively

=== ADDING NEW PLOTS ===
1. Write a function with signature:
       def plot_my_new_plot(ctx: PlotContext, outdir: Path) -> None:
   where PlotContext is the dataclass defined below.
2. Register it at the bottom of this file in PLOT_REGISTRY:
       "my_new_plot": plot_my_new_plot,

That is it.  The runner will call it automatically unless --skip is used.

=== PLOT IDs ===
    loss              training loss curve
    global_pca        PCA of embedding space — multi-colour panels
    global_umap       UMAP of embedding space — multi-colour panels
    cluster_umap      UMAP coloured by K-means cluster
    cluster_profiles  violin plots of catalogue features per cluster
    cluster_heatmap   bird × cluster membership heatmap
    bird_timelines    per-bird cluster-assignment timeline
    bird_profiles     per-bird cluster-fraction bar charts
    pop_diversity     per-bird spread / variance in latent space
    pop_sex           male vs female comparison in UMAP
    anomaly           isolation-forest outlier windows in UMAP
    temporal_doy      seasonal (day-of-year) shift in embedding centroid
    temporal_diel     diel (hour-of-day) pattern in embedding centroid
    individual_bird   deep-dive panels for each of N_PLOT_BIRDS birds
"""

import argparse
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ── Thread caps — must be set before numpy/sklearn are imported ──────────────
# The GH200 nodes expose many cores; OpenBLAS crashes if it tries to use them
# all.  Cap at 16 which is safe for any current OpenBLAS build.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "16")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import circmean

warnings.filterwarnings("ignore")

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] sklearn not found — PCA / clustering / anomaly plots disabled.")

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not found — UMAP plots will be skipped.")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False   # fall back to K-means silently

# ── aesthetic config ─────────────────────────────────────────────────────────
PALETTE_BIRDS    = "tab20"
PALETTE_CLUSTERS = "Set1"
PALETTE_CONT     = "viridis"
PALETTE_DIV      = "RdBu_r"
SEASON_COLOURS   = {
    "Winter":  "#3a86ff",
    "Spring":  "#8ecae6",
    "Summer":  "#ffb703",
    "Autumn":  "#e63946",
}
S_DOT    = 3       # scatter point size
ALPHA    = 0.45    # scatter alpha
DPI      = 150


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def _find_latest_embedding_csv(logdir: Path) -> Optional[Path]:
    candidates = sorted(logdir.glob("*_embeddings.csv"))
    return candidates[-1] if candidates else None


def load_embeddings(logdir: Path, step: Optional[int] = None) -> pd.DataFrame:
    """Load the {step:08d}_embeddings.csv from logdir."""
    if step is not None:
        p = logdir / f"{step:08d}_embeddings.csv"
        if not p.exists():
            raise FileNotFoundError(f"Embeddings not found: {p}")
    else:
        p = _find_latest_embedding_csv(logdir)
        if p is None:
            raise FileNotFoundError(f"No *_embeddings.csv found in {logdir}")
    print(f"Loading embeddings: {p.name}")
    df = pd.read_csv(p, low_memory=False)
    print(f"  {len(df):,} windows  ·  {len(df.columns)} columns")
    return df


def load_catalogue(cat_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if cat_path is None or not Path(cat_path).exists():
        print("[warn] No bird catalogue provided — per-bird colour coding disabled.")
        return None
    df = pd.read_csv(cat_path, low_memory=False)
    print(f"Bird catalogue: {len(df)} birds  ·  {len(df.columns)} columns")
    return df


def load_loss(logdir: Path) -> Optional[pd.DataFrame]:
    p = logdir / "loss.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


def prepare_meta(emb_df: pd.DataFrame, cat_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Attach temporal features (DOY, hour, month, season) and catalogue columns to
    the window-level embeddings DataFrame.  Returns a lightweight metadata frame
    (no h_* / z_* columns).
    """
    meta = emb_df[["bird_id", "window_start"]].copy()

    # Parse timestamps
    meta["window_start"] = pd.to_datetime(meta["window_start"], utc=True, errors="coerce")
    meta["doy"]    = meta["window_start"].dt.dayofyear
    meta["hour"]   = meta["window_start"].dt.hour
    meta["month"]  = meta["window_start"].dt.month
    meta["year"]   = meta["window_start"].dt.year

    def _season(m):
        if m in (12, 1, 2):
            return "Winter"
        elif m in (3, 4, 5):
            return "Spring"
        elif m in (6, 7, 8):
            return "Summer"
        else:
            return "Autumn"

    meta["season"] = meta["month"].map(_season)

    # Join catalogue
    if cat_df is not None:
        cat_small = cat_df.copy()

        # Determine the join key.  bird_catalogue_enhanced.csv has both
        # bird_id (the safe-filename version) and individual-local-identifier.
        # The embeddings use individual-local-identifier as bird_id.
        # Strategy: prefer individual-local-identifier as the join key; if the
        # catalogue already has bird_id AND individual-local-identifier, drop the
        # catalogue's bird_id first to avoid the duplicate-column error.
        if "individual-local-identifier" in cat_small.columns:
            # Drop catalogue's own bird_id if present to avoid collision
            if "bird_id" in cat_small.columns:
                cat_small = cat_small.drop(columns=["bird_id"])
            cat_small = cat_small.rename(
                columns={"individual-local-identifier": "bird_id"})

        if "bird_id" in cat_small.columns:
            # Keep a useful subset of catalogue columns to avoid explosion
            keep_cols = ["bird_id", "animal-sex", "duration_days",
                         "bird_frac_stationary_windows",
                         "bird_frac_active_windows",
                         "bird_frac_fast_windows",
                         "bird_speed_mean_mean",
                         "bird_alt_mean_mean",
                         "bird_tortuosity_mean",
                         "bird_path_efficiency_mean",
                         "bird_net_displacement_km_mean",
                         "bird_total_path_km",
                         "bird_doy_first", "bird_doy_last",
                         "mean_lat", "mean_lon"]
            keep_cols = [c for c in keep_cols if c in cat_small.columns]
            # Deduplicate catalogue rows on bird_id before merging
            cat_small = cat_small[keep_cols].drop_duplicates(subset=["bird_id"])
            meta = meta.merge(cat_small, on="bird_id", how="left")

    return meta


def extract_arrays(emb_df: pd.DataFrame, space: str = "z"):
    """
    Extract the embedding matrix (z_* or h_* columns) as a float32 array.
    Returns (arr, col_names).
    """
    prefix = space + "_"
    cols = sorted(
        [c for c in emb_df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()],
        key=lambda c: int(c[len(prefix):])
    )
    if not cols:
        raise ValueError(f"No columns matching '{prefix}*' found in embeddings.")
    arr = emb_df[cols].values.astype(np.float32)
    print(f"Embedding space '{space}': {arr.shape[1]} dims  ·  {arr.shape[0]:,} windows")
    return arr, cols


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DIMENSIONALITY REDUCTION & CLUSTERING  (computed once, cached in context)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlotContext:
    """Everything a plot function might need — computed once, passed everywhere."""
    emb_df:    pd.DataFrame
    meta:      pd.DataFrame
    arr:       np.ndarray           # full embedding matrix (N, D)
    arr_sub:   np.ndarray           # subsampled for 2D reductions (M, D)
    sub_idx:   np.ndarray           # indices into emb_df / meta for arr_sub
    pca2:      Optional[np.ndarray] = None   # (M, 2) PCA coords
    umap2:     Optional[np.ndarray] = None   # (M, 2) UMAP coords
    cluster_labels: Optional[np.ndarray] = None   # (N,) cluster for every window
    cluster_labels_sub: Optional[np.ndarray] = None  # (M,) cluster subset
    loss_df:   Optional[pd.DataFrame] = None
    n_clusters: int = 8
    step:       int = 0
    space:      str = "z"
    cat_df:     Optional[pd.DataFrame] = None
    win_cat:    Optional[pd.DataFrame] = None   # window_catalogue joined to meta indices
    # convenience: unique birds in display order
    birds:      List[str] = field(default_factory=list)
    bird_colour: Dict[str, int] = field(default_factory=dict)


def build_context(
    logdir:      Path,
    cat_path:    Optional[Path],
    step:        Optional[int],
    space:       str,
    n_clusters:  int,
    umap_sample: int,
    win_cat_path: Optional[Path] = None,
) -> PlotContext:
    emb_df   = load_embeddings(logdir, step)
    cat_df   = load_catalogue(cat_path)
    meta     = prepare_meta(emb_df, cat_df)
    loss_df  = load_loss(logdir)
    arr, _   = extract_arrays(emb_df, space)

    # Resolve step
    resolved_step = step or 0
    if step is None:
        csv = _find_latest_embedding_csv(logdir)
        if csv:
            try:
                resolved_step = int(csv.stem.split("_")[0])
            except ValueError:
                pass

    # Subsample for 2D reductions
    N = len(arr)
    if N > umap_sample:
        rng    = np.random.default_rng(42)
        sub_idx = rng.choice(N, umap_sample, replace=False)
        sub_idx = np.sort(sub_idx)
    else:
        sub_idx = np.arange(N)
    arr_sub = arr[sub_idx]

    pca2 = None
    if HAS_SKLEARN:
        print(f"Computing PCA on {len(arr_sub):,} windows …")
        pca2 = PCA(n_components=2, random_state=42).fit_transform(arr_sub)

    umap2 = None
    if HAS_UMAP:
        print(f"Computing UMAP on {len(arr_sub):,} windows …")
        umap2 = UMAP(n_components=2, random_state=42,
                     n_neighbors=30, min_dist=0.05).fit_transform(arr_sub)

    # Clustering on the full set
    cluster_labels = None
    if HAS_SKLEARN:
        print(f"K-means clustering (k={n_clusters}) on {N:,} windows …")
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(arr).astype(int)

    cluster_labels_sub = cluster_labels[sub_idx] if cluster_labels is not None else None

    # Bird colour mapping
    birds = sorted(meta["bird_id"].unique())
    bird_colour = {b: i for i, b in enumerate(birds)}

    # Load and join window catalogue if provided
    win_cat = None
    if win_cat_path is not None and Path(win_cat_path).exists():
        print(f"Loading window catalogue: {win_cat_path.name}")
        wc = pd.read_csv(win_cat_path, low_memory=False)
        wc["window_start"] = pd.to_datetime(wc["window_start"], utc=True, errors="coerce")
        # Normalise bird_id column name
        if "bird_id" not in wc.columns and "individual-local-identifier" in wc.columns:
            wc = wc.rename(columns={"individual-local-identifier": "bird_id"})
        print(f"  {len(wc):,} window rows  ·  {len(wc.columns)} columns")
        win_cat = wc
    else:
        if win_cat_path is not None:
            print(f"[warn] window catalogue not found: {win_cat_path}")

    return PlotContext(
        emb_df=emb_df, meta=meta, arr=arr, arr_sub=arr_sub, sub_idx=sub_idx,
        pca2=pca2, umap2=umap2,
        cluster_labels=cluster_labels, cluster_labels_sub=cluster_labels_sub,
        loss_df=loss_df, n_clusters=n_clusters,
        step=resolved_step, space=space, cat_df=cat_df,
        birds=birds, bird_colour=bird_colour,
        win_cat=win_cat,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SHARED PLOT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: Path, tight=True):
    if tight:
        plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def _scatter(ax, x, y, c=None, cmap=PALETTE_CONT, s=S_DOT, alpha=ALPHA,
             vmin=None, vmax=None, title="", cb_label="", categorical=False):
    """Unified scatter helper. Returns (scatter_artist, colorbar_or_None)."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)

    if c is None:
        sc = ax.scatter(x, y, s=s, alpha=alpha, color="steelblue", rasterized=True)
        return sc, None

    c = np.asarray(c)
    if categorical:
        cats = np.unique(c[~pd.isna(c)])
        cmap_obj = cm.get_cmap(cmap, len(cats))
        handles = []
        for i, cat in enumerate(cats):
            mask = c == cat
            ax.scatter(x[mask], y[mask], s=s, alpha=alpha,
                       color=cmap_obj(i), label=str(cat), rasterized=True)
            handles.append(Patch(color=cmap_obj(i), label=str(cat)))
        ax.legend(handles=handles, fontsize=5, markerscale=2,
                  loc="lower right", framealpha=0.6)
        return None, None
    else:
        valid = ~np.isnan(c)
        sc = ax.scatter(x[valid], y[valid], c=c[valid], cmap=cmap,
                        s=s, alpha=alpha,
                        vmin=vmin or np.nanpercentile(c, 2),
                        vmax=vmax or np.nanpercentile(c, 98),
                        rasterized=True)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
        cb.ax.tick_params(labelsize=6)
        if cb_label:
            cb.set_label(cb_label, fontsize=6)
        return sc, cb


def _bird_colour_array(meta_sub, bird_colour):
    return np.array([bird_colour.get(b, 0) for b in meta_sub["bird_id"]])


def _cluster_cmap(n):
    return cm.get_cmap(PALETTE_CLUSTERS, n)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PLOT FUNCTIONS
#     Each takes (ctx: PlotContext, outdir: Path) and returns None.
#     Register new plots in PLOT_REGISTRY at the bottom of the file.
# ═══════════════════════════════════════════════════════════════════════════════

# ── 4.1  Training loss ───────────────────────────────────────────────────────

def plot_loss(ctx: PlotContext, outdir: Path) -> None:
    """SimCLR training loss curve with smoothing."""
    df = ctx.loss_df
    if df is None:
        print("  [skip] loss.csv not found.")
        return

    # Accept either "loss" or "Loss" column
    df.columns = [c.lower() for c in df.columns]
    if "loss" not in df.columns or "step" not in df.columns:
        print("  [skip] loss.csv missing expected columns.")
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["step"], df["loss"], lw=0.4, color="grey", alpha=0.4)
    window = max(1, len(df) // 100)
    smooth = df["loss"].rolling(window, center=True).mean()
    ax.plot(df["step"], smooth, lw=1.8, color="crimson", label="smoothed")
    ax.set_xlabel("Training step")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title(f"SimCLR training loss  (step {ctx.step:,})")
    ax.legend(fontsize=8)
    _save(fig, outdir / "loss_curve.png")


# ── 4.2  Global PCA ──────────────────────────────────────────────────────────

def plot_global_pca(ctx: PlotContext, outdir: Path) -> None:
    """3×2 grid of PCA coloured by different properties."""
    if ctx.pca2 is None:
        print("  [skip] PCA not available.")
        return
    _global_2d_grid(ctx, ctx.pca2, "PCA", outdir, "global_pca.png")


# ── 4.3  Global UMAP ─────────────────────────────────────────────────────────

def plot_global_umap(ctx: PlotContext, outdir: Path) -> None:
    """3×2 grid of UMAP coloured by different properties."""
    if ctx.umap2 is None:
        print("  [skip] UMAP not available.")
        return
    _global_2d_grid(ctx, ctx.umap2, "UMAP", outdir, "global_umap.png")


def _global_2d_grid(ctx, coords2d, tag, outdir, fname):
    """Shared logic for PCA/UMAP multi-panel coloured scatter."""
    x, y       = coords2d[:, 0], coords2d[:, 1]
    meta_sub   = ctx.meta.iloc[ctx.sub_idx].reset_index(drop=True)
    n_birds    = len(ctx.birds)
    bird_cmap  = cm.get_cmap(PALETTE_BIRDS, n_birds)

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{tag}  ·  step {ctx.step:,}  ·  {len(x):,} windows  "
                 f"({ctx.space}-space)", fontsize=11)

    # Panel 0: coloured by bird_id
    c_bird = _bird_colour_array(meta_sub, ctx.bird_colour)
    ax = axs[0, 0]
    ax.scatter(x, y, c=c_bird, cmap=PALETTE_BIRDS, s=S_DOT, alpha=ALPHA,
               vmin=0, vmax=n_birds, rasterized=True)
    ax.set_title("by bird", fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    # Panel 1: DOY
    _scatter(axs[0, 1], x, y, meta_sub["doy"].values,
             cmap="hsv", title="day of year", cb_label="DOY")

    # Panel 2: hour of day
    _scatter(axs[0, 2], x, y, meta_sub["hour"].values,
             cmap="twilight_shifted", title="hour of day", cb_label="hour")

    # Panel 3: season (categorical)
    _scatter(axs[1, 0], x, y, meta_sub["season"].values,
             cmap="Set2", title="season", categorical=True)

    # Panel 4: speed (if available)
    if "bird_speed_mean_mean" in meta_sub.columns:
        _scatter(axs[1, 1], x, y, meta_sub["bird_speed_mean_mean"].values,
                 cmap=PALETTE_CONT, title="bird mean speed", cb_label="m/s")
    else:
        axs[1, 1].axis("off")

    # Panel 5: tortuosity (if available)
    if "bird_tortuosity_mean" in meta_sub.columns:
        _scatter(axs[1, 2], x, y, meta_sub["bird_tortuosity_mean"].values,
                 cmap="plasma", title="bird tortuosity", cb_label="tortuosity")
    elif "animal-sex" in meta_sub.columns:
        _scatter(axs[1, 2], x, y, meta_sub["animal-sex"].values,
                 cmap="Set1", title="sex", categorical=True)
    else:
        axs[1, 2].axis("off")

    _save(fig, outdir / fname)


# ── 4.4  Cluster UMAP ────────────────────────────────────────────────────────

def plot_cluster_umap(ctx: PlotContext, outdir: Path) -> None:
    """UMAP (or PCA) coloured by cluster assignment."""
    coords = ctx.umap2 if ctx.umap2 is not None else ctx.pca2
    tag    = "UMAP" if ctx.umap2 is not None else "PCA"
    if coords is None or ctx.cluster_labels_sub is None:
        print("  [skip] No 2D coords or cluster labels.")
        return

    x, y  = coords[:, 0], coords[:, 1]
    k     = ctx.n_clusters
    cmap  = _cluster_cmap(k)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"K-means clusters (k={k})  ·  {tag}  ·  step {ctx.step:,}", fontsize=11)

    # Left: coloured by cluster
    for ci in range(k):
        mask = ctx.cluster_labels_sub == ci
        axs[0].scatter(x[mask], y[mask], s=S_DOT, alpha=ALPHA,
                       color=cmap(ci), label=f"C{ci}", rasterized=True)
    axs[0].legend(fontsize=6, markerscale=3, loc="lower right")
    axs[0].set_title("cluster assignment"); axs[0].set_xticks([]); axs[0].set_yticks([])

    # Right: cluster density (counts per cluster)
    full_counts = np.bincount(ctx.cluster_labels, minlength=k)
    bars = axs[1].bar(range(k), full_counts,
                      color=[cmap(i) for i in range(k)], edgecolor="white")
    axs[1].set_xlabel("Cluster"); axs[1].set_ylabel("Window count")
    axs[1].set_title("Cluster sizes (all windows)")
    axs[1].set_xticks(range(k)); axs[1].set_xticklabels([f"C{i}" for i in range(k)])
    for bar, cnt in zip(bars, full_counts):
        axs[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{cnt:,}", ha="center", va="bottom", fontsize=7)

    _save(fig, outdir / "cluster_umap.png")


# ── 4.5  Cluster profiles ────────────────────────────────────────────────────

def plot_cluster_profiles(ctx: PlotContext, outdir: Path) -> None:
    """
    For each cluster: distribution of key catalogue features.
    Uses violin plots.  One page per feature set.
    """
    if ctx.cluster_labels is None or ctx.cat_df is None:
        print("  [skip] Need cluster labels and bird catalogue.")
        return

    # Attach cluster labels to meta (full set)
    meta = ctx.meta.copy()
    meta["cluster"] = ctx.cluster_labels
    k = ctx.n_clusters
    cmap = _cluster_cmap(k)

    feature_groups = {
        "movement": [
            "bird_speed_mean_mean", "bird_tortuosity_mean",
            "bird_path_efficiency_mean", "bird_net_displacement_km_mean",
            "bird_alt_mean_mean",
        ],
        "activity": [
            "bird_frac_stationary_windows", "bird_frac_active_windows",
            "bird_frac_fast_windows",
        ],
        "temporal": ["doy", "hour"],
    }

    for group_name, features in feature_groups.items():
        features = [f for f in features if f in meta.columns]
        if not features:
            continue

        ncols = min(3, len(features))
        nrows = int(np.ceil(len(features) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axs = np.array(axs).flatten()
        fig.suptitle(f"Feature distributions per cluster — {group_name}", fontsize=11)

        for i, feat in enumerate(features):
            ax = axs[i]
            data_per_cluster = []
            for ci in range(k):
                vals = meta.loc[meta["cluster"] == ci, feat].dropna().values
                data_per_cluster.append(vals)

            parts = ax.violinplot(
                [d for d in data_per_cluster if len(d) > 1],
                positions=range(k),
                showmedians=True, showextrema=False,
            )
            for pi, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(cmap(pi))
                pc.set_alpha(0.65)

            ax.set_xticks(range(k))
            ax.set_xticklabels([f"C{i}" for i in range(k)], fontsize=7)
            ax.set_title(feat.replace("bird_", "").replace("_", " "), fontsize=8)

        for j in range(len(features), len(axs)):
            axs[j].axis("off")

        _save(fig, outdir / f"cluster_profiles_{group_name}.png")


# ── 4.6  Cluster × bird heatmap ──────────────────────────────────────────────

def plot_cluster_heatmap(ctx: PlotContext, outdir: Path) -> None:
    """
    Heatmap: rows = birds, cols = clusters.
    Cell = fraction of that bird's windows in each cluster.
    """
    if ctx.cluster_labels is None:
        print("  [skip] No cluster labels.")
        return

    meta = ctx.meta.copy()
    meta["cluster"] = ctx.cluster_labels
    k = ctx.n_clusters

    birds = sorted(meta["bird_id"].unique())
    mat = np.zeros((len(birds), k), dtype=float)
    for i, bird in enumerate(birds):
        rows = meta[meta["bird_id"] == bird]
        counts = np.bincount(rows["cluster"].values, minlength=k)
        total = counts.sum()
        if total > 0:
            mat[i] = counts / total

    fig, ax = plt.subplots(figsize=(max(8, k * 1.2), max(10, len(birds) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction of windows", fraction=0.03)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"C{i}" for i in range(k)], fontsize=8)
    ax.set_yticks(range(len(birds))); ax.set_yticklabels(birds, fontsize=6)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Bird ID")
    ax.set_title(f"Bird × cluster membership  (k={k}, step {ctx.step:,})")

    # Annotate cells with fractions ≥ 0.3
    for i in range(len(birds)):
        for j in range(k):
            if mat[i, j] >= 0.30:
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=5, color="black" if mat[i, j] < 0.7 else "white")

    _save(fig, outdir / "cluster_heatmap.png")


# ── 4.7  Per-bird cluster timelines ──────────────────────────────────────────

def plot_bird_timelines(ctx: PlotContext, outdir: Path,
                        n_plot_birds: int = 16) -> None:
    """
    For a sample of birds: horizontal timeline coloured by cluster assignment.
    Shows how each bird's behavioural state changes over time.
    """
    if ctx.cluster_labels is None:
        print("  [skip] No cluster labels.")
        return

    meta = ctx.meta.copy()
    meta["cluster"] = ctx.cluster_labels
    k = ctx.n_clusters
    cmap = _cluster_cmap(k)

    # Pick birds with most windows
    counts = meta.groupby("bird_id").size().sort_values(ascending=False)
    plot_birds = counts.index[:n_plot_birds].tolist()

    fig, axs = plt.subplots(len(plot_birds), 1,
                             figsize=(16, max(6, len(plot_birds) * 0.55)),
                             sharex=False)
    if len(plot_birds) == 1:
        axs = [axs]
    fig.suptitle(f"Per-bird cluster timelines  (k={k})  ·  step {ctx.step:,}",
                 fontsize=11)

    for ax, bird in zip(axs, plot_birds):
        rows = meta[meta["bird_id"] == bird].sort_values("window_start")
        t    = (rows["window_start"] - rows["window_start"].min()).dt.total_seconds() / 86400
        for ci in range(k):
            mask = rows["cluster"].values == ci
            ax.scatter(t.values[mask], np.zeros(mask.sum()),
                       c=[cmap(ci)], s=4, alpha=0.7, marker="|", rasterized=True)
        ax.set_yticks([])
        ax.set_ylabel(bird, fontsize=5, rotation=0, ha="right", va="center")
        ax.spines[["top", "right", "left"]].set_visible(False)

    axs[-1].set_xlabel("Days from first fix")

    # Legend
    handles = [Patch(color=cmap(ci), label=f"C{ci}") for ci in range(k)]
    fig.legend(handles=handles, loc="lower center", ncol=k,
               fontsize=7, framealpha=0.7, bbox_to_anchor=(0.5, 0.0))
    _save(fig, outdir / "bird_timelines.png")


# ── 4.8  Per-bird cluster fraction bar charts ─────────────────────────────────

def plot_bird_profiles(ctx: PlotContext, outdir: Path,
                       n_plot_birds: int = 16) -> None:
    """
    For a sample of birds: stacked bar showing fraction of windows per cluster.
    """
    if ctx.cluster_labels is None:
        print("  [skip] No cluster labels.")
        return

    meta = ctx.meta.copy()
    meta["cluster"] = ctx.cluster_labels
    k = ctx.n_clusters
    cmap = _cluster_cmap(k)

    counts = meta.groupby("bird_id").size().sort_values(ascending=False)
    plot_birds = counts.index[:n_plot_birds].tolist()

    # Build fraction matrix
    mat = []
    for bird in plot_birds:
        rows = meta[meta["bird_id"] == bird]
        c = np.bincount(rows["cluster"].values, minlength=k) / max(1, len(rows))
        mat.append(c)
    mat = np.array(mat)   # (n_birds, k)

    fig, ax = plt.subplots(figsize=(max(8, n_plot_birds * 0.65), 5))
    bottom = np.zeros(len(plot_birds))
    for ci in range(k):
        ax.bar(range(len(plot_birds)), mat[:, ci], bottom=bottom,
               color=cmap(ci), label=f"C{ci}", edgecolor="none", width=0.9)
        bottom += mat[:, ci]

    ax.set_xticks(range(len(plot_birds)))
    ax.set_xticklabels(plot_birds, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of windows")
    ax.set_title(f"Cluster profile per bird  (k={k})  ·  step {ctx.step:,}")
    ax.legend(loc="upper right", fontsize=7, ncol=max(1, k // 2))
    _save(fig, outdir / "bird_profiles.png")


# ── 4.9  Population diversity in latent space ─────────────────────────────────

def plot_pop_diversity(ctx: PlotContext, outdir: Path) -> None:
    """
    For each bird: mean and std of their embedding coordinates.
    Shows which birds are most 'variable' vs 'stereotyped' in latent space.
    """
    meta = ctx.meta.copy()

    # Compute per-bird mean embedding and std (summarise with PCA first if high-dim)
    per_bird_mean = {}
    per_bird_std  = {}
    for bird in ctx.birds:
        idx = meta.index[meta["bird_id"] == bird].tolist()
        if not idx:
            continue
        vecs = ctx.arr[idx]  # (n_windows_bird, D)
        per_bird_mean[bird] = vecs.mean(0)
        per_bird_std[bird]  = vecs.std(0).mean()   # scalar: mean std across dims

    birds_sorted = sorted(per_bird_std, key=per_bird_std.get)
    stds = np.array([per_bird_std[b] for b in birds_sorted])

    # Colour by sex if available
    sex_map = {}
    if ctx.cat_df is not None and "animal-sex" in ctx.cat_df.columns:
        col = "individual-local-identifier" if "individual-local-identifier" in ctx.cat_df else None
        if col:
            for _, row in ctx.cat_df.iterrows():
                sex_map[row[col]] = row.get("animal-sex", "?")

    colours = []
    for b in birds_sorted:
        sx = sex_map.get(b, "?")
        if sx in ("m", "M", "male"):
            colours.append("#3a86ff")
        elif sx in ("f", "F", "female"):
            colours.append("#e63946")
        else:
            colours.append("#999999")

    fig, ax = plt.subplots(figsize=(max(10, len(birds_sorted) * 0.4), 5))
    bars = ax.bar(range(len(birds_sorted)), stds, color=colours, edgecolor="none")
    ax.set_xticks(range(len(birds_sorted)))
    ax.set_xticklabels(birds_sorted, rotation=90, fontsize=6)
    ax.set_ylabel(f"Mean std in {ctx.space}-space\n(higher = more variable behaviour)")
    ax.set_title(f"Per-bird latent variability  ·  step {ctx.step:,}")

    handles = [Patch(color="#3a86ff", label="male"),
               Patch(color="#e63946", label="female"),
               Patch(color="#999999", label="unknown")]
    ax.legend(handles=handles, fontsize=8)
    _save(fig, outdir / "pop_diversity.png")


# ── 4.10  Sex comparison in UMAP ─────────────────────────────────────────────

def plot_pop_sex(ctx: PlotContext, outdir: Path) -> None:
    """
    UMAP (or PCA) split by sex.  Two panels: male and female.
    """
    coords = ctx.umap2 if ctx.umap2 is not None else ctx.pca2
    tag    = "UMAP" if ctx.umap2 is not None else "PCA"
    if coords is None:
        print("  [skip] No 2D reduction available.")
        return

    meta_sub = ctx.meta.iloc[ctx.sub_idx].reset_index(drop=True)
    if "animal-sex" not in meta_sub.columns:
        print("  [skip] Sex column not in meta — check catalogue join.")
        return

    x, y   = coords[:, 0], coords[:, 1]
    sex    = meta_sub["animal-sex"].fillna("?").str.lower().values

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sex comparison in {tag}  ·  step {ctx.step:,}", fontsize=11)

    # All
    axs[0].scatter(x, y, s=S_DOT, alpha=0.3, color="grey", rasterized=True)
    axs[0].set_title("all birds"); axs[0].set_xticks([]); axs[0].set_yticks([])

    for ax, sx, col, label in [
        (axs[1], "m",  "#3a86ff", "male"),
        (axs[2], "f",  "#e63946", "female"),
    ]:
        mask = sex == sx
        ax.scatter(x[~mask], y[~mask], s=S_DOT, alpha=0.15, color="lightgrey", rasterized=True)
        ax.scatter(x[mask],  y[mask],  s=S_DOT + 1, alpha=0.6, color=col, rasterized=True)
        ax.set_title(f"{label}  (n={mask.sum():,})")
        ax.set_xticks([]); ax.set_yticks([])

    _save(fig, outdir / "pop_sex.png")


# ── 4.11  Anomaly detection ───────────────────────────────────────────────────

def plot_anomaly(ctx: PlotContext, outdir: Path) -> None:
    """
    Isolation Forest outlier windows highlighted in UMAP / PCA.
    Also shows which birds have the highest outlier rates.
    """
    if not HAS_SKLEARN:
        print("  [skip] sklearn required.")
        return
    coords = ctx.umap2 if ctx.umap2 is not None else ctx.pca2
    tag    = "UMAP" if ctx.umap2 is not None else "PCA"
    if coords is None:
        print("  [skip] No 2D reduction available.")
        return

    print("  Running Isolation Forest …")
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=1)
    is_outlier_full = iso.fit_predict(ctx.arr) == -1   # True = outlier

    meta = ctx.meta.copy()
    meta["is_outlier"] = is_outlier_full

    meta_sub   = ctx.meta.iloc[ctx.sub_idx].reset_index(drop=True)
    outlier_sub = is_outlier_full[ctx.sub_idx]

    x, y = coords[:, 0], coords[:, 1]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Anomaly detection (IsolationForest)  ·  step {ctx.step:,}", fontsize=11)

    # Left: scatter coloured normal / outlier
    axs[0].scatter(x[~outlier_sub], y[~outlier_sub],
                   s=S_DOT, alpha=0.3, color="steelblue",
                   label="normal", rasterized=True)
    axs[0].scatter(x[outlier_sub], y[outlier_sub],
                   s=S_DOT + 2, alpha=0.8, color="crimson",
                   label=f"outlier ({outlier_sub.sum():,})", rasterized=True)
    axs[0].legend(fontsize=8)
    axs[0].set_title(f"Outliers in {tag}")
    axs[0].set_xticks([]); axs[0].set_yticks([])

    # Right: outlier rate per bird
    outlier_rate = meta.groupby("bird_id")["is_outlier"].mean().sort_values()
    colours = ["#e63946" if r > outlier_rate.median() + outlier_rate.std()
               else "#3a86ff" for r in outlier_rate.values]
    axs[1].barh(range(len(outlier_rate)), outlier_rate.values * 100,
                color=colours, edgecolor="none", height=0.8)
    axs[1].set_yticks(range(len(outlier_rate)))
    axs[1].set_yticklabels(outlier_rate.index, fontsize=6)
    axs[1].set_xlabel("Outlier rate (%)")
    axs[1].set_title("Per-bird outlier rate")
    axs[1].axvline(outlier_rate.median() * 100, color="grey",
                   linestyle="--", lw=1, label="median")
    axs[1].legend(fontsize=8)

    _save(fig, outdir / "anomaly.png")


# ── 4.12  Seasonal (DOY) shift in latent centroid ────────────────────────────

def plot_temporal_doy(ctx: PlotContext, outdir: Path) -> None:
    """
    Compresses embedding to 2D (PCA), then plots mean 2D position per
    day-of-year bin.  Reveals seasonal movement through latent space.
    """
    if not HAS_SKLEARN:
        print("  [skip] sklearn required for PCA.")
        return

    meta = ctx.meta.copy()
    meta["doy_bin"] = (meta["doy"] // 10) * 10   # 10-day bins

    pca_full = PCA(n_components=2, random_state=42).fit_transform(ctx.arr)

    meta["pc1"] = pca_full[:, 0]
    meta["pc2"] = pca_full[:, 1]

    grouped = meta.groupby("doy_bin")[["pc1", "pc2"]].mean().reset_index()
    grouped = grouped.dropna()

    doy_vals = grouped["doy_bin"].values
    norm_doy = (doy_vals - doy_vals.min()) / (doy_vals.max() - doy_vals.min() + 1e-8)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Seasonal trajectory in latent space  ·  step {ctx.step:,}", fontsize=11)

    # Left: 2D PC space with colour = DOY
    sc = axs[0].scatter(grouped["pc1"], grouped["pc2"],
                        c=doy_vals, cmap="hsv", s=60, zorder=5)
    # Connect dots
    axs[0].plot(grouped["pc1"], grouped["pc2"],
                color="grey", lw=0.8, alpha=0.5, zorder=4)
    plt.colorbar(sc, ax=axs[0], label="Day of year")
    axs[0].set_xlabel("PC1"); axs[0].set_ylabel("PC2")
    axs[0].set_title("Mean PC position per 10-day DOY bin")

    # Right: PC1 and PC2 vs DOY time series
    axs[1].plot(doy_vals, grouped["pc1"], label="PC1", color="steelblue")
    axs[1].plot(doy_vals, grouped["pc2"], label="PC2", color="crimson")
    axs[1].set_xlabel("Day of year")
    axs[1].set_ylabel("Mean PCA coord")
    axs[1].set_title("PC trajectories over the year")
    axs[1].legend(fontsize=8)

    # Mark seasons
    for start, label, col in [(1, "W", "#3a86ff"), (91, "Sp", "#8ecae6"),
                               (181, "Su", "#ffb703"), (271, "Au", "#e63946")]:
        axs[1].axvspan(start, start + 90, alpha=0.07, color=col)

    _save(fig, outdir / "temporal_doy.png")


# ── 4.13  Diel (hour-of-day) pattern ─────────────────────────────────────────

def plot_temporal_diel(ctx: PlotContext, outdir: Path) -> None:
    """
    Mean embedding per hour of day projected to 2D.
    Reveals activity rhythms encoded in the latent space.
    """
    if not HAS_SKLEARN:
        print("  [skip] sklearn required.")
        return

    meta = ctx.meta.copy()

    pca_full = PCA(n_components=2, random_state=42).fit_transform(ctx.arr)
    meta["pc1"] = pca_full[:, 0]
    meta["pc2"] = pca_full[:, 1]

    grouped = meta.groupby("hour")[["pc1", "pc2"]].mean().reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Diel latent pattern (hour of day)  ·  step {ctx.step:,}", fontsize=11)

    # Left: PC space looped through hours
    sc = axs[0].scatter(grouped["pc1"], grouped["pc2"],
                        c=grouped["hour"], cmap="twilight_shifted", s=80, zorder=5)
    axs[0].plot(list(grouped["pc1"]) + [grouped["pc1"].iloc[0]],
                list(grouped["pc2"]) + [grouped["pc2"].iloc[0]],
                color="grey", lw=0.8, alpha=0.6, zorder=4)
    # Label some hours
    for _, row in grouped[grouped["hour"] % 4 == 0].iterrows():
        axs[0].annotate(f'{int(row["hour"]):02d}h',
                        (row["pc1"], row["pc2"]), fontsize=7, ha="center")
    plt.colorbar(sc, ax=axs[0], label="Hour of day")
    axs[0].set_xlabel("PC1"); axs[0].set_ylabel("PC2")
    axs[0].set_title("Mean PC position per hour (cyclic)")

    # Right: PC1 and PC2 as a time series
    axs[1].fill_between(grouped["hour"], grouped["pc1"],
                        alpha=0.25, color="steelblue")
    axs[1].plot(grouped["hour"], grouped["pc1"], label="PC1", color="steelblue")
    ax2 = axs[1].twinx()
    ax2.fill_between(grouped["hour"], grouped["pc2"],
                     alpha=0.2, color="crimson")
    ax2.plot(grouped["hour"], grouped["pc2"], label="PC2", color="crimson")
    axs[1].set_xlabel("Hour of day (UTC)")
    axs[1].set_ylabel("PC1", color="steelblue")
    ax2.set_ylabel("PC2", color="crimson")
    axs[1].set_title("Diel rhythm of latent coordinates")
    axs[1].axvspan(19, 24, alpha=0.07, color="navy")
    axs[1].axvspan(0, 5, alpha=0.07, color="navy")
    axs[1].set_xticks(range(0, 24, 2))

    _save(fig, outdir / "temporal_diel.png")


# ── 4.14  Individual bird deep-dive ──────────────────────────────────────────

def plot_individual_bird(ctx: PlotContext, outdir: Path,
                         n_plot_birds: int = 12) -> None:
    """
    For each of N birds: a 2×2 panel showing
      [0,0] Their windows in UMAP, coloured by time
      [0,1] Their windows coloured by season
      [1,0] Cluster membership timeline
      [1,1] Cluster fraction bar + DOY range annotation
    One file per bird, saved in outdir/per_bird/.
    """
    coords = ctx.umap2 if ctx.umap2 is not None else ctx.pca2
    tag    = "UMAP" if ctx.umap2 is not None else "PCA"
    if coords is None:
        print("  [skip] No 2D reduction available.")
        return

    per_bird_dir = outdir / "per_bird"
    per_bird_dir.mkdir(exist_ok=True)

    meta    = ctx.meta.copy()
    meta_sub = meta.iloc[ctx.sub_idx].reset_index(drop=True)
    if ctx.cluster_labels is not None:
        meta["cluster"] = ctx.cluster_labels

    # Pick birds with most windows for the deep dive
    counts = meta.groupby("bird_id").size().sort_values(ascending=False)
    plot_birds = counts.index[:n_plot_birds].tolist()

    k    = ctx.n_clusters
    cmap = _cluster_cmap(k)

    x_all, y_all = coords[:, 0], coords[:, 1]

    for bird in plot_birds:
        # Windows in the subsample belonging to this bird
        bird_mask_sub = meta_sub["bird_id"] == bird
        x_bird = x_all[bird_mask_sub]
        y_bird = y_all[bird_mask_sub]

        bird_meta     = meta[meta["bird_id"] == bird].sort_values("window_start")
        bird_meta_sub = meta_sub[bird_mask_sub].copy()

        if len(x_bird) == 0:
            continue

        fig = plt.figure(figsize=(13, 9))
        fig.suptitle(f"Bird: {bird}  ·  {len(bird_meta):,} windows  ·  step {ctx.step:,}",
                     fontsize=11)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        axs = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

        # ── [0,0] Position in UMAP coloured by time ──
        ax = axs[0][0]
        ax.scatter(x_all, y_all, s=1, alpha=0.12, color="lightgrey", rasterized=True)
        if "doy" in bird_meta_sub.columns:
            sc = ax.scatter(x_bird, y_bird,
                            c=bird_meta_sub["doy"].values,
                            cmap="hsv", s=S_DOT + 2, alpha=0.75, zorder=5, rasterized=True)
            plt.colorbar(sc, ax=ax, label="DOY", fraction=0.04)
        else:
            ax.scatter(x_bird, y_bird, s=S_DOT + 2, alpha=0.75,
                       color="crimson", zorder=5, rasterized=True)
        ax.set_title(f"{tag} — coloured by DOY", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # ── [0,1] Position in UMAP coloured by season ──
        ax = axs[0][1]
        ax.scatter(x_all, y_all, s=1, alpha=0.12, color="lightgrey", rasterized=True)
        if "season" in bird_meta_sub.columns:
            season_vals = bird_meta_sub["season"].values
            for season, col in SEASON_COLOURS.items():
                m = season_vals == season
                ax.scatter(x_bird[m], y_bird[m],
                           s=S_DOT + 2, alpha=0.8, color=col,
                           label=season, zorder=5, rasterized=True)
            ax.legend(fontsize=6, markerscale=2)
        ax.set_title(f"{tag} — coloured by season", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # ── [1,0] Cluster timeline ──
        ax = axs[1][0]
        if "cluster" in bird_meta.columns:
            t = (bird_meta["window_start"] - bird_meta["window_start"].min()).dt.total_seconds() / 86400
            for ci in range(k):
                m = bird_meta["cluster"].values == ci
                ax.scatter(t.values[m], np.zeros(m.sum()),
                           c=[cmap(ci)], s=5, alpha=0.8, marker="|", rasterized=True)
            handles = [Patch(color=cmap(ci), label=f"C{ci}") for ci in range(k)]
            ax.legend(handles=handles, ncol=k, fontsize=5,
                      loc="upper right", framealpha=0.5)
            ax.set_yticks([])
            ax.set_xlabel("Days from first fix")
            ax.set_title("Cluster assignment over time", fontsize=8)
        else:
            ax.axis("off")

        # ── [1,1] Cluster fraction bar ──
        ax = axs[1][1]
        if "cluster" in bird_meta.columns:
            counts_cl = np.bincount(bird_meta["cluster"].values, minlength=k)
            frac   = counts_cl / counts_cl.sum()
            bars   = ax.bar(range(k), frac * 100,
                            color=[cmap(ci) for ci in range(k)], edgecolor="white")
            ax.set_xticks(range(k))
            ax.set_xticklabels([f"C{ci}" for ci in range(k)], fontsize=7)
            ax.set_ylabel("% of windows")
            ax.set_title("Cluster fraction", fontsize=8)

            # Annotate with counts
            for bar, cnt in zip(bars, counts_cl):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        str(cnt), ha="center", va="bottom", fontsize=6)

            # DOY annotation box
            if "doy" in bird_meta.columns:
                doy_first = bird_meta["doy"].min()
                doy_last  = bird_meta["doy"].max()
                ax.text(0.02, 0.97,
                        f"DOY {doy_first:.0f}–{doy_last:.0f}\n"
                        f"{len(bird_meta):,} windows\n"
                        f"Duration: {(bird_meta['window_start'].max() - bird_meta['window_start'].min()).days} days",
                        transform=ax.transAxes,
                        va="top", ha="left", fontsize=7,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
        else:
            ax.axis("off")

        _save(fig, per_bird_dir / f"bird_{bird}.png")


# ── 4.15  Comprehensive male vs female study ─────────────────────────────────

def plot_sex_study(ctx: PlotContext, outdir: Path) -> None:
    """
    Full male vs female comparison across five dimensions:
      A. UMAP / PCA positions — density and separation
      B. Feature distributions — violin plots for catalogue movement features
      C. Cluster usage — how much does each sex use each cluster?
      D. Seasonal (DOY) pattern — do they move through latent space differently?
      E. Diel pattern — do they differ by time of day?
    """
    coords = ctx.umap2 if ctx.umap2 is not None else ctx.pca2
    tag    = "UMAP" if ctx.umap2 is not None else "PCA"
    if coords is None:
        print("  [skip] No 2D reduction available.")
        return
    if "animal-sex" not in ctx.meta.columns:
        print("  [skip] animal-sex column not available — check catalogue join.")
        return

    meta     = ctx.meta.copy()
    meta_sub = meta.iloc[ctx.sub_idx].reset_index(drop=True)
    if ctx.cluster_labels is not None:
        meta["cluster"] = ctx.cluster_labels
    x, y = coords[:, 0], coords[:, 1]

    # Normalise sex labels to 'm' / 'f' / 'unknown'
    def _norm_sex(s):
        s = str(s).strip().lower()
        if s in ("m", "male"):   return "m"
        if s in ("f", "female"): return "f"
        return "unknown"

    meta["sex"]     = meta["animal-sex"].map(_norm_sex)
    meta_sub["sex"] = meta_sub["animal-sex"].map(_norm_sex)

    SEX_COL = {"m": "#3a86ff", "f": "#e63946", "unknown": "#aaaaaa"}
    SEX_LABEL = {"m": "Male", "f": "Female", "unknown": "Unknown"}

    # ── A. UMAP / PCA panel ──────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sex comparison — {tag}  ·  step {ctx.step:,}", fontsize=12)

    axs[0].scatter(x, y, s=S_DOT, alpha=0.2, color="lightgrey", rasterized=True)
    for sx, col in SEX_COL.items():
        m = meta_sub["sex"] == sx
        if m.sum() == 0:
            continue
        axs[0].scatter(x[m], y[m], s=S_DOT, alpha=0.55, color=col,
                       label=f"{SEX_LABEL[sx]} ({m.sum():,})", rasterized=True)
    axs[0].legend(fontsize=8); axs[0].set_title("All sexes"); axs[0].set_xticks([]); axs[0].set_yticks([])

    for ax, sx in [(axs[1], "m"), (axs[2], "f")]:
        mask = meta_sub["sex"] == sx
        ax.scatter(x[~mask], y[~mask], s=1, alpha=0.12, color="lightgrey", rasterized=True)
        ax.scatter(x[mask], y[mask], s=S_DOT + 1, alpha=0.65,
                   color=SEX_COL[sx], rasterized=True)
        ax.set_title(f"{SEX_LABEL[sx]}  (n={mask.sum():,} windows)")
        ax.set_xticks([]); ax.set_yticks([])

    _save(fig, outdir / "sex_umap.png")

    # ── B. Feature distribution violins ──────────────────────────────────────
    feat_cols = [
        "bird_speed_mean_mean", "bird_tortuosity_mean", "bird_path_efficiency_mean",
        "bird_net_displacement_km_mean", "bird_alt_mean_mean",
        "bird_frac_stationary_windows", "bird_frac_active_windows",
        "bird_frac_fast_windows", "duration_days",
    ]
    feat_cols = [c for c in feat_cols if c in meta.columns]

    if feat_cols:
        # One row per bird — use bird-level stats, not window-level
        bird_sex = (meta.groupby("bird_id")["sex"]
                    .first().reset_index().rename(columns={"sex": "bird_sex"}))
        bird_stats = (meta.groupby("bird_id")[feat_cols]
                      .first().reset_index())
        bird_df = bird_stats.merge(bird_sex, on="bird_id")

        ncols = 3
        nrows = int(np.ceil(len(feat_cols) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axs = np.array(axs).flatten()
        fig.suptitle("Feature distributions by sex  (bird-level)", fontsize=11)

        for i, feat in enumerate(feat_cols):
            ax = axs[i]
            groups = []
            labels = []
            colours = []
            for sx in ["m", "f"]:
                vals = bird_df.loc[bird_df["bird_sex"] == sx, feat].dropna().values
                if len(vals) > 0:
                    groups.append(vals)
                    labels.append(SEX_LABEL[sx])
                    colours.append(SEX_COL[sx])
            if not groups:
                ax.axis("off"); continue
            parts = ax.violinplot(groups, positions=range(len(groups)),
                                  showmedians=True, showextrema=False)
            for pc, col in zip(parts["bodies"], colours):
                pc.set_facecolor(col); pc.set_alpha(0.6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(feat.replace("bird_", "").replace("_", " "), fontsize=8)

            # Add individual points (n is small enough)
            for xi, (vals, col) in enumerate(zip(groups, colours)):
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
                ax.scatter(xi + jitter, vals, s=18, alpha=0.6, color=col, zorder=5)

        for j in range(len(feat_cols), len(axs)):
            axs[j].axis("off")
        _save(fig, outdir / "sex_features.png")

    # ── C. Cluster usage by sex ────────────────────────────────────────────
    if ctx.cluster_labels is not None:
        k    = ctx.n_clusters
        cmap = _cluster_cmap(k)

        sex_cluster = {}
        for sx in ["m", "f"]:
            rows = meta[meta["sex"] == sx]
            if len(rows) == 0:
                continue
            counts = np.bincount(rows["cluster"].values, minlength=k)
            sex_cluster[sx] = counts / counts.sum()

        if sex_cluster:
            fig, ax = plt.subplots(figsize=(max(8, k * 1.1), 5))
            w = 0.35
            xs = np.arange(k)
            for offset, (sx, fracs) in zip([-w/2, w/2], sex_cluster.items()):
                ax.bar(xs + offset, fracs * 100, width=w,
                       color=SEX_COL[sx], label=SEX_LABEL[sx],
                       alpha=0.8, edgecolor="white")
            ax.set_xticks(xs); ax.set_xticklabels([f"C{i}" for i in range(k)])
            ax.set_ylabel("% of windows"); ax.set_xlabel("Cluster")
            ax.set_title(f"Cluster usage by sex  (k={k})")
            ax.legend(fontsize=9)
            _save(fig, outdir / "sex_cluster_usage.png")

    # ── D. Seasonal pattern — mean PC1/PC2 per DOY bin per sex ───────────────
    if HAS_SKLEARN:
        from sklearn.decomposition import PCA as _PCA
        pca_full = _PCA(n_components=2, random_state=42).fit_transform(ctx.arr)
        meta["pc1"] = pca_full[:, 0]
        meta["pc2"] = pca_full[:, 1]
        meta["doy_bin"] = (meta["doy"] // 10) * 10

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Seasonal latent trajectory by sex", fontsize=11)

        for sx, col in [("m", "#3a86ff"), ("f", "#e63946")]:
            rows = meta[meta["sex"] == sx]
            if len(rows) < 10:
                continue
            g = rows.groupby("doy_bin")[["pc1", "pc2"]].mean().dropna()
            axs[0].plot(g["pc1"], g["pc2"], color=col, lw=1.5,
                        label=SEX_LABEL[sx], alpha=0.85)
            sc = axs[0].scatter(g["pc1"], g["pc2"],
                                c=g.index, cmap="hsv", s=40,
                                edgecolors=col, linewidths=0.8, zorder=5)
            axs[1].plot(g.index, g["pc1"], color=col, lw=1.5,
                        label=f"{SEX_LABEL[sx]} PC1", linestyle="-")
            axs[1].plot(g.index, g["pc2"], color=col, lw=1.5,
                        label=f"{SEX_LABEL[sx]} PC2", linestyle="--", alpha=0.6)

        axs[0].set_xlabel("PC1"); axs[0].set_ylabel("PC2")
        axs[0].set_title("Mean PC position per 10-day DOY bin")
        axs[0].legend(fontsize=8)
        axs[1].set_xlabel("Day of year"); axs[1].set_ylabel("Mean PC coord")
        axs[1].set_title("PC1 & PC2 through year by sex")
        axs[1].legend(fontsize=7, ncol=2)
        _save(fig, outdir / "sex_seasonal.png")

    # ── E. Diel pattern ────────────────────────────────────────────────────
    if HAS_SKLEARN and "pc1" in meta.columns:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Diel latent pattern by sex", fontsize=11)

        for sx, col in [("m", "#3a86ff"), ("f", "#e63946")]:
            rows = meta[meta["sex"] == sx]
            if len(rows) < 10:
                continue
            g = rows.groupby("hour")[["pc1", "pc2"]].mean()
            axs[0].plot(list(g["pc1"]) + [g["pc1"].iloc[0]],
                        list(g["pc2"]) + [g["pc2"].iloc[0]],
                        color=col, lw=1.5, label=SEX_LABEL[sx])
            axs[0].scatter(g["pc1"], g["pc2"], c=g.index,
                           cmap="twilight_shifted", s=50,
                           edgecolors=col, linewidths=0.8, zorder=5)
            axs[1].plot(g.index, g["pc1"], color=col, lw=1.5, linestyle="-",
                        label=f"{SEX_LABEL[sx]} PC1")
            axs[1].plot(g.index, g["pc2"], color=col, lw=1.5, linestyle="--",
                        alpha=0.6, label=f"{SEX_LABEL[sx]} PC2")

        axs[0].set_xlabel("PC1"); axs[0].set_ylabel("PC2")
        axs[0].set_title("Mean PC position per hour (cyclic)")
        axs[0].legend(fontsize=8)
        axs[1].set_xlabel("Hour (UTC)"); axs[1].set_ylabel("Mean PC coord")
        axs[1].set_title("Diel PC rhythm by sex")
        axs[1].set_xticks(range(0, 24, 2))
        axs[1].legend(fontsize=7, ncol=2)
        _save(fig, outdir / "sex_diel.png")


# ── 4.16  Behaviour inference ─────────────────────────────────────────────────

# Heuristic thresholds — edit these first if results look off
BEHAVIOUR_THRESHOLDS = {
    # name:         (condition_fn, priority)
    # Lower priority number = checked first
    "diving":       1,   # fastest vertical descent, very high speed burst
    "thermal":      2,   # circling upward
    "gliding":      3,   # descending, straight, low effort
    "flying":       4,   # fast directed translocation
    "hunting":      5,   # tortuous, variable, moderate speed
    "stationary":   6,   # near-zero movement
    "unknown":      99,
}

BEHAVIOUR_COLOURS = {
    "diving":    "#e63946",
    "thermal":   "#ffb703",
    "gliding":   "#8ecae6",
    "flying":    "#023e8a",
    "hunting":   "#6a994e",
    "stationary":"#aaaaaa",
    "unknown":   "#dddddd",
}

BEHAVIOUR_ORDER = ["stationary", "flying", "gliding", "thermal", "hunting", "diving", "unknown"]


def _classify_window(row) -> str:
    """
    Rule-based behavioural label for a single window-catalogue row.
    Uses speed, vertical rate, heading concentration, tortuosity, path efficiency.
    Returns a label string.
    """
    def get(col, default=np.nan):
        v = row.get(col, default)
        return default if pd.isna(v) else float(v)

    speed       = get("speed_mean")
    speed_max   = get("speed_max")
    vrate       = get("vrate_mean")
    vdesc       = get("vrate_max_descent")   # most negative value = fast descent
    vclimb      = get("vrate_max_climb")
    frac_climb  = get("frac_climbing")
    frac_desc   = get("frac_descending")
    frac_level  = get("frac_level")
    hconc       = get("heading_concentration")  # 1=directed, 0=circling
    tortuous    = get("tortuosity")
    path_eff    = get("path_efficiency")
    turn_mean   = get("turn_mean_deg")
    frac_sharp  = get("frac_sharp_turns")
    frac_str    = get("frac_straight")
    frac_low    = get("frac_low_speed")

    # ── Stationary / walking ──────────────────────────────────────────────
    # Very low speed AND barely any displacement
    if (not np.isnan(speed) and speed < 1.0) and \
       (np.isnan(frac_low) or frac_low > 0.7):
        return "stationary"

    # ── Diving ───────────────────────────────────────────────────────────
    # Rapid descent AND high speed  (e.g. stoop)
    if (not np.isnan(vdesc) and vdesc < -3.0) and \
       (not np.isnan(speed_max) and speed_max > 15.0):
        return "diving"

    # ── Thermal soaring ───────────────────────────────────────────────────
    # Net climbing, circling (low heading concentration), low path efficiency
    if (not np.isnan(frac_climb) and frac_climb > 0.35) and \
       (not np.isnan(hconc) and hconc < 0.55) and \
       (np.isnan(path_eff) or path_eff < 0.5):
        return "thermal"

    # ── Gliding ──────────────────────────────────────────────────────────
    # Net descending, relatively straight, moderate speed
    if (not np.isnan(frac_desc) and frac_desc > 0.40) and \
       (not np.isnan(frac_str) and frac_str > 0.35) and \
       (not np.isnan(speed) and speed > 4.0):
        return "gliding"

    # ── Directed flying / commuting ───────────────────────────────────────
    # Fast, straight, high heading concentration, efficient path
    if (not np.isnan(speed) and speed > 7.0) and \
       (not np.isnan(hconc) and hconc > 0.65) and \
       (np.isnan(path_eff) or path_eff > 0.55):
        return "flying"

    # ── Hunting / searching ───────────────────────────────────────────────
    # Moderate speed, tortuous, many turns, low path efficiency
    if (not np.isnan(speed) and 1.5 < speed < 10.0) and \
       (not np.isnan(tortuous) and tortuous > 2.5) and \
       (not np.isnan(frac_sharp) and frac_sharp > 0.15):
        return "hunting"

    return "unknown"


def _load_window_features_for_behaviour(ctx: PlotContext) -> Optional[pd.DataFrame]:
    """
    Joins the window catalogue to the embedding meta on (bird_id, window_start).
    Returns a DataFrame with behaviour labels attached, aligned to ctx.meta index.
    Returns None if win_cat is not available.
    """
    if ctx.win_cat is None:
        return None

    wc = ctx.win_cat.copy()
    wc["window_start"] = pd.to_datetime(wc["window_start"], utc=True, errors="coerce")

    meta = ctx.meta[["bird_id", "window_start"]].copy()
    meta["window_start"] = pd.to_datetime(meta["window_start"], utc=True, errors="coerce")
    meta = meta.reset_index(drop=True)
    meta["_meta_idx"] = meta.index

    merged = meta.merge(wc, on=["bird_id", "window_start"], how="left")

    # Apply heuristic classifier to each row
    print("  Classifying behaviour for each window …")
    merged["behaviour"] = merged.apply(_classify_window, axis=1)

    return merged


def plot_behaviour_inference(ctx: PlotContext, outdir: Path) -> None:
    """
    Heuristic behavioural state inference using window-catalogue movement features.
    Requires --window_catalogue to be provided.

    Produces:
      A. UMAP coloured by behaviour label — shows where each behaviour sits
      B. Behaviour fraction bar per bird — each bird's behavioural fingerprint
      C. Behaviour timeline per bird (sample) — when does each behaviour occur?
      D. Feature validation violins — confirms heuristics are sensible
      E. Seasonal and diel behaviour fractions — when does each behaviour peak?
      F. Behaviour × cluster confusion matrix — do clusters map to behaviours?
    """
    wf = _load_window_features_for_behaviour(ctx)
    if wf is None:
        print("  [skip] window_catalogue not provided — "
              "run with --window_catalogue golden_eai_outputs/window_catalogue_1h.csv")
        return

    coords = ctx.umap2 if ctx.umap2 is not None else ctx.pca2
    tag    = "UMAP" if ctx.umap2 is not None else "PCA"

    beh_all = wf["behaviour"].values   # (N,) aligned to ctx.meta / ctx.arr
    beh_sub = beh_all[ctx.sub_idx]

    behav_dir = outdir / "behaviour"
    behav_dir.mkdir(exist_ok=True)

    # ── A. UMAP coloured by behaviour ────────────────────────────────────────
    if coords is not None:
        x, y = coords[:, 0], coords[:, 1]
        fig, axs = plt.subplots(2, 4, figsize=(20, 9))
        axs = axs.flatten()
        fig.suptitle(f"Behavioural states in {tag}  ·  step {ctx.step:,}", fontsize=12)

        # Panel 0: all behaviours
        ax = axs[0]
        for beh in BEHAVIOUR_ORDER:
            m = beh_sub == beh
            if m.sum() == 0: continue
            ax.scatter(x[m], y[m], s=S_DOT, alpha=0.55,
                       color=BEHAVIOUR_COLOURS[beh], label=beh, rasterized=True)
        ax.legend(fontsize=6, markerscale=3, loc="lower right")
        ax.set_title("All behaviours"); ax.set_xticks([]); ax.set_yticks([])

        # One panel per behaviour
        for i, beh in enumerate(BEHAVIOUR_ORDER):
            ax = axs[i + 1]
            mask = beh_sub == beh
            ax.scatter(x[~mask], y[~mask], s=1, alpha=0.1, color="lightgrey", rasterized=True)
            ax.scatter(x[mask], y[mask], s=S_DOT + 1, alpha=0.7,
                       color=BEHAVIOUR_COLOURS[beh], rasterized=True)
            ax.set_title(f"{beh}  ({mask.sum():,})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        _save(fig, behav_dir / "behaviour_umap.png")

    # ── B. Per-bird behaviour fraction bars ───────────────────────────────────
    wf_birds = wf.copy()
    bird_counts = wf_birds.groupby("bird_id").size().sort_values(ascending=False)
    top_birds   = bird_counts.index[:20].tolist()

    mat = []
    for bird in top_birds:
        rows = wf_birds[wf_birds["bird_id"] == bird]
        counts = {beh: (rows["behaviour"] == beh).sum() for beh in BEHAVIOUR_ORDER}
        total  = max(1, len(rows))
        mat.append({beh: counts[beh] / total for beh in BEHAVIOUR_ORDER})
    mat_df = pd.DataFrame(mat, index=top_birds)

    fig, ax = plt.subplots(figsize=(max(10, len(top_birds) * 0.7), 5))
    bottom = np.zeros(len(top_birds))
    for beh in BEHAVIOUR_ORDER:
        vals = mat_df[beh].values
        ax.bar(range(len(top_birds)), vals * 100,
               bottom=bottom, color=BEHAVIOUR_COLOURS[beh],
               label=beh, edgecolor="none", width=0.9)
        bottom += vals
    ax.set_xticks(range(len(top_birds)))
    ax.set_xticklabels(top_birds, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of windows")
    ax.set_title(f"Behavioural profile per bird  ·  step {ctx.step:,}")
    ax.legend(loc="upper right", fontsize=7,
              handles=[Patch(color=BEHAVIOUR_COLOURS[b], label=b) for b in BEHAVIOUR_ORDER])
    _save(fig, behav_dir / "behaviour_per_bird.png")

    # ── C. Behaviour timeline (top 12 birds) ─────────────────────────────────
    plot_birds = top_birds[:12]
    fig, axs_list = plt.subplots(len(plot_birds), 1,
                                  figsize=(16, max(6, len(plot_birds) * 0.6)),
                                  sharex=False)
    if len(plot_birds) == 1:
        axs_list = [axs_list]
    fig.suptitle(f"Behaviour timelines  ·  step {ctx.step:,}", fontsize=11)

    for ax, bird in zip(axs_list, plot_birds):
        rows = wf_birds[wf_birds["bird_id"] == bird].sort_values("window_start")
        t    = (rows["window_start"] - rows["window_start"].min()).dt.total_seconds() / 86400
        for beh in BEHAVIOUR_ORDER:
            m = rows["behaviour"].values == beh
            ax.scatter(t.values[m], np.zeros(m.sum()),
                       c=[BEHAVIOUR_COLOURS[beh]], s=5, alpha=0.9,
                       marker="|", rasterized=True)
        ax.set_yticks([])
        ax.set_ylabel(bird, fontsize=5, rotation=0, ha="right", va="center")
        ax.spines[["top", "right", "left"]].set_visible(False)
    axs_list[-1].set_xlabel("Days from first fix")
    handles = [Patch(color=BEHAVIOUR_COLOURS[b], label=b) for b in BEHAVIOUR_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(BEHAVIOUR_ORDER),
               fontsize=7, framealpha=0.7, bbox_to_anchor=(0.5, 0.0))
    _save(fig, behav_dir / "behaviour_timelines.png")

    # ── D. Feature validation violins ─────────────────────────────────────────
    val_features = [
        ("speed_mean",        "Mean speed (m/s)"),
        ("vrate_mean",        "Mean vert rate (m/s)"),
        ("vrate_max_descent", "Max descent rate (m/s)"),
        ("heading_concentration", "Heading concentration"),
        ("tortuosity",        "Tortuosity"),
        ("path_efficiency",   "Path efficiency"),
        ("frac_climbing",     "Frac climbing"),
        ("frac_descending",   "Frac descending"),
        ("frac_sharp_turns",  "Frac sharp turns"),
    ]
    val_features = [(c, lbl) for c, lbl in val_features if c in wf.columns]

    if val_features:
        ncols = 3
        nrows = int(np.ceil(len(val_features) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axs = np.array(axs).flatten()
        fig.suptitle("Feature distributions per inferred behaviour  (validation)", fontsize=11)

        for i, (col, lbl) in enumerate(val_features):
            ax = axs[i]
            data   = [wf.loc[wf["behaviour"] == beh, col].dropna().values
                      for beh in BEHAVIOUR_ORDER]
            colors = [BEHAVIOUR_COLOURS[beh] for beh in BEHAVIOUR_ORDER]
            non_empty = [(d, c) for d, c in zip(data, colors) if len(d) > 1]
            if not non_empty: ax.axis("off"); continue
            vals, cols = zip(*non_empty)
            labels = [BEHAVIOUR_ORDER[i] for i, d in enumerate(data) if len(d) > 1]
            parts = ax.violinplot(vals, positions=range(len(vals)),
                                  showmedians=True, showextrema=False)
            for pc, col_ in zip(parts["bodies"], cols):
                pc.set_facecolor(col_); pc.set_alpha(0.65)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
            ax.set_title(lbl, fontsize=8)
        for j in range(len(val_features), len(axs)):
            axs[j].axis("off")
        _save(fig, behav_dir / "behaviour_feature_validation.png")

    # ── E. Seasonal and diel fractions ────────────────────────────────────────
    wf["doy"]  = pd.to_datetime(wf["window_start"], utc=True, errors="coerce").dt.dayofyear
    wf["hour"] = pd.to_datetime(wf["window_start"], utc=True, errors="coerce").dt.hour

    for time_col, bin_label, fname in [
        ("doy",  "Day of year", "behaviour_seasonal.png"),
        ("hour", "Hour (UTC)",  "behaviour_diel.png"),
    ]:
        if time_col not in wf.columns:
            continue
        wf["_bin"] = (wf[time_col] // (10 if time_col == "doy" else 1)) * \
                     (10 if time_col == "doy" else 1)
        pivot = (wf.groupby(["_bin", "behaviour"]).size()
                   .unstack(fill_value=0)
                   .reindex(columns=BEHAVIOUR_ORDER, fill_value=0))
        pivot_frac = pivot.div(pivot.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(12, 4))
        bottom = np.zeros(len(pivot_frac))
        for beh in BEHAVIOUR_ORDER:
            if beh not in pivot_frac.columns: continue
            ax.fill_between(pivot_frac.index, bottom,
                            bottom + pivot_frac[beh].values,
                            color=BEHAVIOUR_COLOURS[beh], alpha=0.82,
                            label=beh, step="mid")
            bottom = bottom + pivot_frac[beh].values
        ax.set_xlabel(bin_label); ax.set_ylabel("Fraction of windows")
        ax.set_title(f"Behaviour fractions over {bin_label.lower()}")
        ax.set_ylim(0, 1)
        ax.legend(handles=[Patch(color=BEHAVIOUR_COLOURS[b], label=b)
                            for b in BEHAVIOUR_ORDER],
                  loc="upper right", fontsize=7, ncol=2)
        _save(fig, behav_dir / fname)

    # ── F. Behaviour × cluster matrix ─────────────────────────────────────────
    if ctx.cluster_labels is not None:
        wf["cluster"] = ctx.cluster_labels
        k = ctx.n_clusters
        mat = np.zeros((len(BEHAVIOUR_ORDER), k), dtype=float)
        for bi, beh in enumerate(BEHAVIOUR_ORDER):
            rows = wf[wf["behaviour"] == beh]
            if len(rows) == 0: continue
            counts = np.bincount(rows["cluster"].astype(int).values, minlength=k)
            mat[bi] = counts / max(1, counts.sum())

        fig, ax = plt.subplots(figsize=(max(8, k * 1.2), 5))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=mat.max())
        plt.colorbar(im, ax=ax, label="Fraction of behaviour windows in cluster", fraction=0.03)
        ax.set_xticks(range(k)); ax.set_xticklabels([f"C{i}" for i in range(k)], fontsize=8)
        ax.set_yticks(range(len(BEHAVIOUR_ORDER)))
        ax.set_yticklabels(BEHAVIOUR_ORDER, fontsize=8)
        ax.set_xlabel("Cluster"); ax.set_title("Behaviour × cluster membership")
        for bi in range(len(BEHAVIOUR_ORDER)):
            for ci in range(k):
                v = mat[bi, ci]
                if v > 0.15:
                    ax.text(ci, bi, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if v < 0.6 else "white")
        _save(fig, behav_dir / "behaviour_cluster_matrix.png")

    print(f"  Behaviour counts: "
          + "  ".join(f"{b}:{(beh_all==b).sum():,}" for b in BEHAVIOUR_ORDER))


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PLOT REGISTRY
#     To add a plot: write a function, add it here.  Done.
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_REGISTRY: Dict[str, callable] = {
    "loss":             plot_loss,
    "global_pca":       plot_global_pca,
    "global_umap":      plot_global_umap,
    "cluster_umap":     plot_cluster_umap,
    "cluster_profiles": plot_cluster_profiles,
    "cluster_heatmap":  plot_cluster_heatmap,
    "bird_timelines":   plot_bird_timelines,
    "bird_profiles":    plot_bird_profiles,
    "pop_diversity":    plot_pop_diversity,
    "pop_sex":          plot_pop_sex,
    "anomaly":          plot_anomaly,
    "temporal_doy":     plot_temporal_doy,
    "temporal_diel":    plot_temporal_diel,
    "individual_bird":  plot_individual_bird,
    # ── new modules ──
    "sex_study":        plot_sex_study,
    "behaviour":        plot_behaviour_inference,
}


def run_plots(ctx: PlotContext, outdir: Path,
              skip: List[str], only: List[str],
              n_plot_birds: int) -> None:
    """Run all registered plots, honouring --skip and --only."""
    registry = PLOT_REGISTRY.copy()
    if only:
        registry = {k: v for k, v in registry.items() if k in only}
    if skip:
        registry = {k: v for k, v in registry.items() if k not in skip}

    print(f"\nRunning {len(registry)} plot(s): {', '.join(registry)}\n")

    for plot_id, func in registry.items():
        print(f"▶  {plot_id}")
        try:
            # Pass n_plot_birds to functions that accept it
            import inspect
            sig = inspect.signature(func)
            if "n_plot_birds" in sig.parameters:
                func(ctx, outdir, n_plot_birds=n_plot_birds)
            else:
                func(ctx, outdir)
        except Exception as e:
            print(f"   [ERROR] {plot_id} failed: {e}")
            import traceback; traceback.print_exc()

    print(f"\n✓  All plots written to: {outdir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive embedding analysis for Golden Eagle SimCLR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--logdir",      required=True,
                        help="Path to SimCLR logdir containing *_embeddings.csv")
    parser.add_argument("--catalogue",   default=None,
                        help="Path to bird_catalogue_enhanced.csv")
    parser.add_argument("--window_catalogue", default=None,
                        help="Path to window_catalogue_1h.csv (required for behaviour plot)")
    parser.add_argument("--outdir",      default=None,
                        help="Where to save plots (default: logdir/analysis)")
    parser.add_argument("--step",        type=int, default=None,
                        help="Use embeddings from this specific step")
    parser.add_argument("--n_clusters",  type=int, default=8)
    parser.add_argument("--n_plot_birds",type=int, default=16,
                        help="Birds to include in per-bird panels")
    parser.add_argument("--umap_sample", type=int, default=20_000,
                        help="Max windows to subsample for UMAP/PCA")
    parser.add_argument("--space",       default="z", choices=["z", "h"],
                        help="Embedding space to analyse (z=projection, h=representation)")
    parser.add_argument("--skip",        default="",
                        help="Comma-separated plot IDs to skip")
    parser.add_argument("--only",        default="",
                        help="Comma-separated plot IDs to run (overrides --skip)")
    args = parser.parse_args()

    logdir  = Path(args.logdir)
    outdir  = Path(args.outdir) if args.outdir else logdir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    print("=" * 60)
    print("  Golden Eagle SimCLR Embedding Analysis")
    print(f"  logdir:    {logdir}")
    print(f"  outdir:    {outdir}")
    print(f"  space:     {args.space}")
    print(f"  clusters:  {args.n_clusters}")
    print(f"  umap_n:    {args.umap_sample:,}")
    print("=" * 60 + "\n")

    print("Building analysis context …")
    ctx = build_context(
        logdir       = logdir,
        cat_path     = Path(args.catalogue) if args.catalogue else None,
        step         = args.step,
        space        = args.space,
        n_clusters   = args.n_clusters,
        umap_sample  = args.umap_sample,
        win_cat_path = Path(args.window_catalogue) if args.window_catalogue else None,
    )

    run_plots(ctx, outdir, skip=skip, only=only, n_plot_birds=args.n_plot_birds)


if __name__ == "__main__":
    main()
