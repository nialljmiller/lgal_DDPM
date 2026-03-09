#!/usr/bin/env python3
"""
plot_eagle_simclr.py

Standalone script to query a SimCLR logdir and generate diagnostic plots
from whatever has been saved so far.

Usage:
    python plot_eagle_simclr.py --logdir /cluster/ai4wy/project/ai4wy-eap/nmille39/GoldenEAI \
                                 --catalogue /path/to/bird_catalogue_enhanced.csv \
                                 [--outdir ./simclr_plots]
                                 [--step 9500]        # optional: use a specific checkpoint step
                                 [--no_umap]          # skip UMAP if slow / not installed
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Optional heavy imports                                                        #
# --------------------------------------------------------------------------- #
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] sklearn not found — PCA/tSNE plots will be skipped.")

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not found — UMAP plots will be skipped.")


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

BIRD_COLOUR_COLS = [
    # Movement
    "bird_speed_mean_mean",
    "bird_alt_mean_mean",
    "bird_total_path_km",
    "bird_net_displacement_km_mean",
    "bird_tortuosity_mean",
    "bird_path_efficiency_mean",
    "bird_frac_stationary_windows",
    "bird_frac_active_windows",
    "bird_frac_fast_windows",
    # Demography
    "duration_days",
    "animal-sex",
]


def _find_latest_step(logdir: Path):
    """Return the highest step number found from *_hs.npy files, or None."""
    npy_files = sorted(logdir.glob("*_hs.npy"))
    if not npy_files:
        return None
    # filenames like 00009500_hs.npy
    steps = []
    for f in npy_files:
        try:
            steps.append(int(f.stem.split("_")[0]))
        except ValueError:
            pass
    return max(steps) if steps else None


def _load_embeddings(logdir: Path, step: int | None):
    """
    Load hs and zs arrays.  Prefers the step-specific .npy files.
    Falls back to hs_labels.csv / zs_labels.csv (always-overwritten latest).
    Returns (hs, zs, labels) where labels is a 1-D int array (may be None).
    """
    hs, zs, labels = None, None, None

    if step is not None:
        hs_npy = logdir / f"{step:08d}_hs.npy"
        zs_npy = logdir / f"{step:08d}_zs.npy"
        lbl_npy = logdir / f"{step:08d}_labels.npy"
        if hs_npy.exists():
            hs = np.load(str(hs_npy))
            print(f"  Loaded hs from {hs_npy.name}  shape={hs.shape}")
        if zs_npy.exists():
            zs = np.load(str(zs_npy))
            print(f"  Loaded zs from {zs_npy.name}  shape={zs.shape}")
        if lbl_npy.exists():
            labels = np.load(str(lbl_npy))
            print(f"  Loaded labels from {lbl_npy.name}  n={len(labels)}")

    # Fallback to latest CSV
    if hs is None:
        csv_path = logdir / "hs_labels.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            labels = df["label"].values.astype(float)
            hs = df.drop(columns=["label"]).values.astype("float32")
            print(f"  Loaded hs from hs_labels.csv  shape={hs.shape}")
    if zs is None:
        csv_path = logdir / "zs_labels.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            labels = df["label"].values.astype(float)
            zs = df.drop(columns=["label"]).values.astype("float32")
            print(f"  Loaded zs from zs_labels.csv  shape={zs.shape}")

    return hs, zs, labels


def _load_3d_csv(logdir: Path, step: int | None):
    """Try to load a pre-computed hzs_3D.csv."""
    if step is not None:
        p = logdir / f"{step:08d}_hzs_3D.csv"
        if p.exists():
            return pd.read_csv(p)
    # fallback: latest
    candidates = sorted(logdir.glob("*_hzs_3D.csv"))
    if candidates:
        return pd.read_csv(candidates[-1])
    return None


def _load_output_info(logdir: Path):
    p = logdir / "output_info.csv"
    if not p.exists():
        p2 = logdir / "loss.txt"
        if p2.exists():
            df = pd.read_csv(p2, header=None, names=["Step", "Loss"])
            return df
        return None
    df = pd.read_csv(p, header=None,
                     names=["Step", "Loss", "Min_Z0", "Min_Z1",
                             "Min_H0", "Min_H1", "Max_H0", "Max_H1"])
    return df


def _scatter2d(ax, x, y, c=None, cmap="viridis", s=4, alpha=0.5,
               title="", xlabel="dim 1", ylabel="dim 2", label_cb=None):
    if c is not None and not pd.api.types.is_numeric_dtype(np.array(c)):
        # categorical
        cats = np.unique(c)
        cmap_cat = cm.get_cmap("tab10", len(cats))
        for i, cat in enumerate(cats):
            mask = np.array(c) == cat
            ax.scatter(x[mask], y[mask], s=s, alpha=alpha,
                       color=cmap_cat(i), label=str(cat))
        ax.legend(fontsize=6, markerscale=3)
    elif c is not None:
        sc = ax.scatter(x, y, c=c, cmap=cmap, s=s, alpha=alpha)
        cb = plt.colorbar(sc, ax=ax, pad=0.02)
        if label_cb:
            cb.set_label(label_cb, fontsize=7)
    else:
        ax.scatter(x, y, s=s, alpha=alpha, color="steelblue")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _add_bird_colour(labels, cat_df, col):
    """
    Given a label array (window-level float IDs) and the bird catalogue,
    attempt to map each window's label to a bird-level property.

    The labels in the SimCLR run are floats cast from bird indices or tag IDs.
    We try a few strategies to join.
    """
    if cat_df is None or col not in cat_df.columns:
        return None

    # Try direct join on tag-local-identifier
    if "tag-local-identifier" in cat_df.columns:
        mapping = cat_df.set_index("tag-local-identifier")[col]
        vals = pd.Series(labels.astype(int)).map(mapping)
        if vals.notna().sum() > len(labels) * 0.3:
            return vals.values

    # Try join on bird_id index
    if "bird_id" in cat_df.columns:
        mapping = cat_df.reset_index(drop=True)[col]
        idx = labels.astype(int) % len(cat_df)
        return mapping.iloc[idx].values

    return None


# --------------------------------------------------------------------------- #
# Plot functions                                                                #
# --------------------------------------------------------------------------- #

def plot_training_curves(df, outdir: Path):
    if df is None:
        print("  [skip] No output_info.csv or loss.txt found.")
        return

    has_full = "Min_Z0" in df.columns

    if has_full:
        fig, axs = plt.subplots(2, 3, figsize=(14, 7))
        fig.suptitle("SimCLR Training Diagnostics", fontsize=12)

        axs[0, 0].plot(df["Step"], df["Loss"], lw=0.8, color="crimson")
        axs[0, 0].set_title("Loss")

        axs[0, 1].plot(df["Step"], df["Min_Z0"], lw=0.8, label="Z0")
        axs[0, 1].plot(df["Step"], df["Min_Z1"], lw=0.8, label="Z1")
        axs[0, 1].set_title("Min of Z projections")
        axs[0, 1].legend(fontsize=8)

        axs[0, 2].plot(df["Step"], df["Min_H0"], lw=0.8, label="H0")
        axs[0, 2].plot(df["Step"], df["Min_H1"], lw=0.8, label="H1")
        axs[0, 2].set_title("Min of H hidden states")
        axs[0, 2].legend(fontsize=8)

        axs[1, 0].plot(df["Step"], df["Max_H0"], lw=0.8, label="H0")
        axs[1, 0].plot(df["Step"], df["Max_H1"], lw=0.8, label="H1")
        axs[1, 0].set_title("Max of H hidden states")
        axs[1, 0].legend(fontsize=8)

        axs[1, 1].plot(df["Min_Z0"], df["Min_Z1"], lw=0.5, alpha=0.6)
        axs[1, 1].set_title("Min Z0 vs Min Z1 (phase portrait)")
        axs[1, 1].set_xlabel("Min Z0")
        axs[1, 1].set_ylabel("Min Z1")

        # loss smoothed
        window = max(1, len(df) // 40)
        smooth = df["Loss"].rolling(window, center=True).mean()
        axs[1, 2].plot(df["Step"], df["Loss"], lw=0.4, color="grey", alpha=0.5)
        axs[1, 2].plot(df["Step"], smooth, lw=1.5, color="crimson", label=f"smooth w={window}")
        axs[1, 2].set_title("Loss (smoothed)")
        axs[1, 2].legend(fontsize=8)

        for ax in axs.flat:
            ax.set_xlabel("Step", fontsize=7)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("SimCLR Training Loss", fontsize=12)
        window = max(1, len(df) // 40)
        smooth = df["Loss"].rolling(window, center=True).mean()
        ax.plot(df["Step"], df["Loss"], lw=0.4, color="grey", alpha=0.5)
        ax.plot(df["Step"], smooth, lw=1.5, color="crimson")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

    plt.tight_layout()
    out = outdir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}")


def plot_embedding_grid(emb2d, labels, cat_df, tag, outdir: Path, step: int):
    """2×3 grid: raw | coloured by bird properties."""
    x, y = emb2d[:, 0], emb2d[:, 1]

    # Build colour columns list (only those available)
    cols_to_plot = []
    if cat_df is not None:
        for col in BIRD_COLOUR_COLS:
            c = _add_bird_colour(labels, cat_df, col)
            if c is not None:
                cols_to_plot.append((col, c))
    # pad with None if fewer than 5
    while len(cols_to_plot) < 5:
        cols_to_plot.append((None, None))

    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"{tag}  step={step}", fontsize=11)

    # panel 0: raw uncoloured
    _scatter2d(axs[0, 0], x, y, title="raw (no colour)")

    panels = [(axs[0, 1], axs[0, 2]), (axs[1, 0], axs[1, 1]), (axs[1, 2],)]
    flat_panels = [axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]

    for ax, (col, c) in zip(flat_panels, cols_to_plot):
        if col is None:
            ax.axis("off")
        else:
            _scatter2d(ax, x, y, c=c, title=col.replace("bird_", ""), label_cb=col)

    plt.tight_layout()
    fname = outdir / f"{step:08d}_{tag}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname.name}")


def run_reductions(hs, zs, labels, cat_df, step, outdir: Path, do_umap=True):
    """Run PCA, tSNE, UMAP on hs and zs, save grids."""
    if not HAS_SKLEARN:
        print("  [skip] sklearn not available.")
        return

    pairs = []
    if hs is not None:
        pairs.append(("hs", hs))
    if zs is not None:
        pairs.append(("zs", zs))

    for name, arr in pairs:
        print(f"\n  --- Reducing {name}  shape={arr.shape} ---")

        # PCA
        print("    PCA ... ", end="", flush=True)
        pca2 = PCA(n_components=2).fit_transform(arr)
        print("done")
        plot_embedding_grid(pca2, labels, cat_df, f"PCA_{name}", outdir, step)

        # tSNE
        n = min(len(arr), 5000)  # tSNE is slow for large N
        idx = np.random.choice(len(arr), n, replace=False) if len(arr) > n else np.arange(len(arr))
        print(f"    tSNE (n={n}) ... ", end="", flush=True)
        tsne2 = TSNE(n_components=2, learning_rate="auto", init="random",
                     perplexity=min(30, n // 4)).fit_transform(arr[idx])
        print("done")
        lbl_sub = labels[idx] if labels is not None else None
        plot_embedding_grid(tsne2, lbl_sub, cat_df, f"tSNE_{name}", outdir, step)

        # UMAP
        if do_umap and HAS_UMAP:
            print(f"    UMAP (n={len(arr)}) ... ", end="", flush=True)
            umap2 = UMAP(n_components=2, random_state=42).fit_transform(arr)
            print("done")
            plot_embedding_grid(umap2, labels, cat_df, f"UMAP_{name}", outdir, step)


def plot_precomputed_3d(df3d, cat_df, step, outdir: Path):
    """Plot 2-D projections from a pre-saved hzs_3D.csv."""
    if df3d is None:
        return
    labels = df3d["label"].values if "label" in df3d.columns else None

    for space, x_col, y_col in [
        ("Z_PCA",   "zpca1",  "zpca2"),
        ("Z_tSNE",  "ztsne1", "ztsne2"),
        ("Z_UMAP",  "zumap1", "zumap2"),
        ("H_PCA",   "hpca1",  "hpca2"),
        ("H_tSNE",  "htsne1", "htsne2"),
        ("H_UMAP",  "humap1", "humap2"),
    ]:
        if x_col not in df3d.columns:
            continue
        x, y = df3d[x_col].values, df3d[y_col].values
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{space}  step={step}", fontsize=10)
        _scatter2d(axs[0], x, y, title="raw")
        # try two bird props
        for ax, col in zip(axs[1:], ["bird_speed_mean_mean", "bird_tortuosity_mean"]):
            c = _add_bird_colour(labels, cat_df, col) if labels is not None and cat_df is not None else None
            _scatter2d(ax, x, y, c=c, title=col.replace("bird_", "") if c is not None else col)
        plt.tight_layout()
        fname = outdir / f"{step:08d}_{space}_from3d.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved: {fname.name}")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate SimCLR diagnostic plots from logdir.")
    parser.add_argument("--logdir", required=True, help="Path to the SimCLR logdir")
    parser.add_argument("--catalogue", default=None,
                        help="Path to bird_catalogue_enhanced.csv (for colour coding)")
    parser.add_argument("--outdir", default=None,
                        help="Where to save plots (default: logdir/plots)")
    parser.add_argument("--step", type=int, default=None,
                        help="Use a specific checkpoint step (default: latest)")
    parser.add_argument("--no_umap", action="store_true",
                        help="Skip UMAP (useful if it is too slow)")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"[error] logdir not found: {logdir}")
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else logdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {outdir}")

    # ------------------------------------------------------------------ #
    # Resolve step                                                         #
    # ------------------------------------------------------------------ #
    step = args.step
    if step is None:
        step = _find_latest_step(logdir)
    if step is not None:
        print(f"Using step: {step}")
    else:
        print("No step-specific checkpoint found — using latest CSVs only.")
        step = 0

    # ------------------------------------------------------------------ #
    # Load bird catalogue                                                   #
    # ------------------------------------------------------------------ #
    cat_df = None
    if args.catalogue and Path(args.catalogue).exists():
        cat_df = pd.read_csv(args.catalogue)
        print(f"Bird catalogue loaded: {len(cat_df)} birds, {len(cat_df.columns)} columns")
    else:
        print("[warn] No bird catalogue provided or found — embedding plots will be uncoloured.")

    # ------------------------------------------------------------------ #
    # Training curves                                                       #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Training curves")
    info_df = _load_output_info(logdir)
    if info_df is not None:
        print(f"  output_info has {len(info_df)} rows, up to step {info_df['Step'].max()}")
    plot_training_curves(info_df, outdir)

    # ------------------------------------------------------------------ #
    # Load embeddings                                                       #
    # ------------------------------------------------------------------ #
    print("\n[2/4] Loading embeddings")
    hs, zs, labels = _load_embeddings(logdir, step if step != 0 else None)

    if hs is None and zs is None:
        print("  [warn] No embedding arrays found. Nothing to reduce/plot.")
    else:
        # ---------------------------------------------------------------- #
        # Dimensionality reduction & scatter plots                          #
        # ---------------------------------------------------------------- #
        print("\n[3/4] Dimensionality reduction + scatter plots")
        run_reductions(hs, zs, labels, cat_df, step, outdir,
                       do_umap=not args.no_umap)

    # ------------------------------------------------------------------ #
    # Pre-computed 3D CSV (if available)                                   #
    # ------------------------------------------------------------------ #
    print("\n[4/4] Pre-computed 3D projections")
    df3d = _load_3d_csv(logdir, step if step != 0 else None)
    if df3d is not None:
        print(f"  Found hzs_3D.csv with {len(df3d)} rows")
        plot_precomputed_3d(df3d, cat_df, step, outdir)
    else:
        print("  No hzs_3D.csv found — skipping.")

    print(f"\nDone. All plots written to: {outdir}")


if __name__ == "__main__":
    main()
