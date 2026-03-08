#!/usr/bin/env python3
"""
build_gp_features.py

GPU-accelerated Gaussian Process feature extraction for golden eagle movement windows.
Reads per-bird CSVs and a window_catalogue CSV, fits batched GPs per window per channel,
and writes GP-derived features (hyperparameters, LML, predictive variance) per window.

Multi-node / multi-GPU:
    Reads RANK and WORLD_SIZE from environment (set by torchrun or SLURM).
    Each rank processes its own slice of birds. Outputs are rank-sharded CSVs.
    After all ranks complete, run with --merge to join everything back.

Usage (single GPU):
    python build_gp_features.py \\
        --per_bird_dir golden_eai_outputs/per_bird_csv \\
        --window_catalogue golden_eai_outputs/window_catalogue_1h.csv \\
        --outdir golden_eai_outputs

Usage (multi-node via torchrun, e.g. 20 nodes x 2 GPUs):
    torchrun --nnodes=20 --nproc_per_node=2 \\
        --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29500 \\
        build_gp_features.py \\
        --per_bird_dir golden_eai_outputs/per_bird_csv \\
        --window_catalogue golden_eai_outputs/window_catalogue_1h.csv \\
        --outdir golden_eai_outputs

Usage (merge shards after all ranks finish):
    python build_gp_features.py --merge --outdir golden_eai_outputs \\
        --window_catalogue golden_eai_outputs/window_catalogue_1h.csv

Kernels fitted per channel [speed, altitude, activity]:
    1. RBF (Squared Exponential)        → lengthscale, outputscale, noise, lml
    2. Matern 5/2                        → lengthscale, outputscale, noise, lml
    3. Periodic                          → period, lengthscale, outputscale, noise, lml
    4. RBF × Periodic (Quasi-Periodic)  → rbf_ls, per_ls, period, outputscale, noise, lml

The quasi-periodic kernel is the most biologically interesting: it captures a smooth
envelope (long-range temporal structure) modulated by a periodic cycle (diurnal rhythm).
Its hyperparameters directly characterise how strongly periodic vs how diffuse the
bird's movement rhythm is within each window.

All times are normalised to [0,1] within each window before GP fitting.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Distributed setup — reads torchrun / SLURM env vars
# ---------------------------------------------------------------------------

def _safe_filename(value: str) -> str:
    """Mirror of build_bird_catalogue.safe_filename — must match exactly."""
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        text = "UNKNOWN_BIRD"
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def get_rank_world():
    """Return (rank, world_size, local_rank) from environment."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, world_size)))
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# GP models (GPyTorch)
# ---------------------------------------------------------------------------

def _import_gpytorch():
    try:
        import torch
        import gpytorch
        return torch, gpytorch
    except ImportError:
        print("[FATAL] gpytorch not found. Install: pip install gpytorch torch")
        sys.exit(1)


class _RBFModel:
    """Thin wrapper — actual model constructed at fit time."""
    name = "rbf"
    kernel_params = ["lengthscale", "outputscale", "noise"]
    has_period = False


class _MaternModel:
    name = "matern52"
    kernel_params = ["lengthscale", "outputscale", "noise"]
    has_period = False


class _PeriodicModel:
    name = "periodic"
    kernel_params = ["period", "lengthscale", "outputscale", "noise"]
    has_period = True


class _QuasiPeriodicModel:
    name = "quasiperiodic"
    kernel_params = ["rbf_lengthscale", "per_lengthscale", "period", "outputscale", "noise"]
    has_period = True


KERNEL_SPECS = [_RBFModel, _MaternModel, _PeriodicModel, _QuasiPeriodicModel]


def _build_gpytorch_model(kernel_name, train_x, train_y, batch_shape, torch, gpytorch):
    """Construct a batched ExactGP model for the given kernel type."""

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
            if kernel_name == "rbf":
                base = gpytorch.kernels.RBFKernel(batch_shape=batch_shape)
            elif kernel_name == "matern52":
                base = gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=batch_shape)
            elif kernel_name == "periodic":
                base = gpytorch.kernels.PeriodicKernel(batch_shape=batch_shape)
            elif kernel_name == "quasiperiodic":
                rbf_k = gpytorch.kernels.RBFKernel(batch_shape=batch_shape)
                per_k = gpytorch.kernels.PeriodicKernel(batch_shape=batch_shape)
                base = rbf_k * per_k
            else:
                raise ValueError(kernel_name)
            self.covar_module = gpytorch.kernels.ScaleKernel(base, batch_shape=batch_shape)

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)
    model = _GP(train_x, train_y, likelihood)
    return model, likelihood


def _extract_hyperparams(kernel_name, model, likelihood, torch):
    """
    Extract named hyperparameters from a trained GP model.
    Returns a dict of {param_name: numpy_array_shape[B]}.
    """
    params = {}
    with torch.no_grad():
        params["noise"] = likelihood.noise.squeeze(-1).cpu().numpy()
        params["outputscale"] = model.covar_module.outputscale.cpu().numpy()

        base = model.covar_module.base_kernel

        if kernel_name in ("rbf", "matern52"):
            params["lengthscale"] = base.lengthscale.squeeze(-1).cpu().numpy()

        elif kernel_name == "periodic":
            params["period"] = base.period_length.squeeze(-1).cpu().numpy()
            params["lengthscale"] = base.lengthscale.squeeze(-1).cpu().numpy()

        elif kernel_name == "quasiperiodic":
            # base is a ProductKernel of RBF * Periodic
            rbf_k, per_k = base.kernels[0], base.kernels[1]
            params["rbf_lengthscale"] = rbf_k.lengthscale.squeeze(-1).cpu().numpy()
            params["per_lengthscale"] = per_k.lengthscale.squeeze(-1).cpu().numpy()
            params["period"] = per_k.period_length.squeeze(-1).cpu().numpy()

    return params


def fit_gp_batch(
    t_batch,      # np.ndarray shape [B, N], time in [0,1]
    y_batch,      # np.ndarray shape [B, N], normalised signal
    kernel_name,  # str
    n_iter,       # int
    device,       # torch.device
    torch, gpytorch,
    lr=0.1,
):
    """
    Fit B independent GPs (same kernel) simultaneously using GPyTorch batched mode.

    Parameters
    ----------
    t_batch : [B, N]  — input times (already normalised to [0,1])
    y_batch : [B, N]  — target values (standardised per-window)

    Returns
    -------
    dict  {param_name: np.ndarray[B]}   plus  "lml": np.ndarray[B]
    """
    B, N = t_batch.shape
    batch_shape = torch.Size([B])

    train_x = torch.tensor(t_batch, dtype=torch.float32).to(device)   # [B, N]
    train_y = torch.tensor(y_batch, dtype=torch.float32).to(device)   # [B, N]

    try:
        model, likelihood = _build_gpytorch_model(
            kernel_name, train_x, train_y, batch_shape, torch, gpytorch
        )
        model = model.to(device)
        likelihood = likelihood.to(device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(n_iter):
            optimizer.zero_grad()
            out = model(train_x)
            loss = -mll(out, train_y).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        model.eval()
        likelihood.eval()

        # Log marginal likelihood per window
        with torch.no_grad():
            out = model(train_x)
            lml_vals = mll(out, train_y).detach().cpu().numpy()   # [B]

        hp = _extract_hyperparams(kernel_name, model, likelihood, torch)
        hp["lml"] = lml_vals

        return hp

    except Exception as e:
        # Return NaN arrays if anything fails (numerical issues are common
        # for degenerate windows)
        nan_arr = np.full(B, np.nan)
        hp = {"lml": nan_arr, "noise": nan_arr, "outputscale": nan_arr}
        if kernel_name in ("rbf", "matern52"):
            hp["lengthscale"] = nan_arr
        elif kernel_name == "periodic":
            hp.update({"period": nan_arr, "lengthscale": nan_arr})
        elif kernel_name == "quasiperiodic":
            hp.update({"rbf_lengthscale": nan_arr, "per_lengthscale": nan_arr, "period": nan_arr})
        return hp


# ---------------------------------------------------------------------------
# Window data extraction
# ---------------------------------------------------------------------------

CHANNELS = {
    "speed":    "ground-speed",
    "altitude": "height-above-ellipsoid",
    "activity": "eobs:activity",
}


def extract_window_series(bird_df: pd.DataFrame, window_start: pd.Timestamp, window_hours: int = 1):
    """
    Extract multivariate time series for one window from a pre-sorted bird DataFrame.

    Returns
    -------
    t_norm : np.ndarray [N]   — times normalised to [0,1] within window
    channels : dict {name: np.ndarray [N]}  — valid data per channel (may be empty)
    n_valid : int
    """
    window_end = window_start + pd.Timedelta(hours=window_hours)
    mask = (bird_df["_t"] >= window_start) & (bird_df["_t"] < window_end)
    w = bird_df[mask].sort_values("_t")

    if len(w) < 3:
        return None, {}, 0

    t_abs = w["_t"].values.astype("int64") / 1e9   # unix seconds
    t_min, t_max = t_abs[0], t_abs[-1]
    duration = t_max - t_min
    if duration < 60:  # less than 1 minute span — degenerate
        return None, {}, 0

    t_norm = (t_abs - t_min) / duration   # [0, 1]

    channels = {}
    for ch_name, col in CHANNELS.items():
        if col in w.columns:
            vals = pd.to_numeric(w[col], errors="coerce").values
            finite = np.isfinite(vals)
            if finite.sum() >= 3:
                channels[ch_name] = (t_norm[finite], vals[finite])

    return t_norm, channels, len(w)


def normalise_channel(y: np.ndarray):
    """Zero-mean, unit-variance normalisation. Returns (y_norm, mean, std)."""
    mu = np.nanmean(y)
    sigma = np.nanstd(y)
    if sigma < 1e-9:
        sigma = 1.0
    return (y - mu) / sigma, mu, sigma


# ---------------------------------------------------------------------------
# Per-bird GP feature extraction
# ---------------------------------------------------------------------------

def process_bird_gp(
    bird_csv: Path,
    window_starts_for_bird: list,   # list of pd.Timestamp
    n_iter: int,
    batch_size: int,
    device,
    torch, gpytorch,
    window_hours: int = 1,
):
    """
    For one bird, load raw CSV, extract window series, fit all GPs, return feature rows.

    Returns
    -------
    list of dicts, one per window, keyed by bird_id + window_start + all GP features.
    """
    # Load bird CSV
    try:
        df = pd.read_csv(bird_csv, low_memory=False)
    except Exception as e:
        print(f"[warn] Could not read {bird_csv.name}: {e}")
        return []

    if "timestamp" not in df.columns:
        return []

    df["_t"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)

    if len(df) < 3:
        return []

    bird_id = str(df["individual-local-identifier"].dropna().iloc[0]) \
        if "individual-local-identifier" in df.columns and df["individual-local-identifier"].notna().any() \
        else bird_csv.stem

    # Pre-extract all window series for this bird
    window_data = {}   # window_start -> {ch_name: (t, y)}
    for ws in window_starts_for_bird:
        _, channels, n_valid = extract_window_series(df, ws, window_hours)
        if n_valid >= 3:
            window_data[ws] = channels

    if not window_data:
        return []

    # For each channel and kernel, batch-fit GPs
    # Organise: ch_name -> kernel_name -> list of (window_start, t, y_norm)
    channel_kernel_batches = {}
    for ch_name in CHANNELS:
        ch_windows = [(ws, *window_data[ws][ch_name])
                      for ws in window_data if ch_name in window_data[ws]]
        if not ch_windows:
            continue
        channel_kernel_batches[ch_name] = ch_windows

    # Collect per-window results
    window_results = {ws: {"bird_id": bird_id, "window_start": ws}
                      for ws in window_data}

    for ch_name, ch_windows in channel_kernel_batches.items():
        for kernel_spec in KERNEL_SPECS:
            kname = kernel_spec.name
            prefix = f"gp_{ch_name}_{kname}"

            # Process in mini-batches (group by N for batched GP)
            # Group windows by observation count for valid batching
            by_n = {}
            for ws, t, y in ch_windows:
                n = len(t)
                by_n.setdefault(n, []).append((ws, t, y))

            for n_pts, group in by_n.items():
                # Split into batches of at most batch_size
                for i in range(0, len(group), batch_size):
                    mini = group[i:i + batch_size]
                    ws_list = [m[0] for m in mini]
                    B = len(mini)

                    t_batch = np.stack([m[1] for m in mini])   # [B, N]
                    y_raw = np.stack([m[2] for m in mini])      # [B, N]

                    # Normalise each window independently
                    y_batch = np.zeros_like(y_raw)
                    for b in range(B):
                        y_batch[b], _, _ = normalise_channel(y_raw[b])

                    # Fit batch
                    hp = fit_gp_batch(
                        t_batch, y_batch,
                        kname, n_iter, device,
                        torch, gpytorch,
                    )

                    # Scatter results back
                    for b, ws in enumerate(ws_list):
                        for param_name, arr in hp.items():
                            col = f"{prefix}_{param_name}"
                            window_results[ws][col] = float(arr[b]) if b < len(arr) else np.nan

    return list(window_results.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU-batched GP feature extraction for eagle windows.")
    parser.add_argument("--per_bird_dir", default="golden_eai_outputs/per_bird_csv")
    parser.add_argument("--window_catalogue", default="golden_eai_outputs/window_catalogue_1h.csv")
    parser.add_argument("--outdir", default="golden_eai_outputs")
    parser.add_argument("--window_hours", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=150,
                        help="Adam iterations for GP hyperparameter optimisation")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of windows processed simultaneously per GPU (batched GP)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-rank GP feature shards and join to window_catalogue")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- MERGE MODE ----
    if args.merge:
        print("Merging GP feature shards...")
        shard_files = sorted(outdir.glob("gp_features_rank*.csv"))
        if not shard_files:
            print("[error] No shard files found.")
            sys.exit(1)

        shards = [pd.read_csv(f, low_memory=False) for f in shard_files]
        gp_df = pd.concat(shards, ignore_index=True)
        print(f"  Total GP feature rows: {len(gp_df)}")

        gp_out = outdir / "gp_features.csv"
        gp_df.to_csv(gp_out, index=False)
        print(f"  Wrote: {gp_out}")

        if Path(args.window_catalogue).exists():
            wcat = pd.read_csv(args.window_catalogue, low_memory=False)
            wcat["bird_id"] = wcat["bird_id"].astype(str)
            wcat["window_start"] = wcat["window_start"].astype(str)
            gp_df["bird_id"] = gp_df["bird_id"].astype(str)
            gp_df["window_start"] = gp_df["window_start"].astype(str)
            merged = wcat.merge(gp_df, on=["bird_id", "window_start"], how="left")
            enhanced_out = outdir / "window_catalogue_with_gp.csv"
            merged.to_csv(enhanced_out, index=False)
            print(f"  Joined to window_catalogue → {enhanced_out}")
            print(f"  Final rows: {len(merged)}, columns: {len(merged.columns)}")
        return

    # ---- FEATURE EXTRACTION MODE ----
    torch, gpytorch = _import_gpytorch()

    rank, world_size, local_rank = get_rank_world()

    # GPU device selection
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_id = local_rank % n_gpus
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[rank {rank}/{world_size}] Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print(f"[rank {rank}/{world_size}] No GPU found — using CPU (will be slow)")

    # Load window catalogue to know which windows exist per bird
    wcat_path = Path(args.window_catalogue)
    if not wcat_path.exists():
        print(f"[error] window_catalogue not found: {wcat_path}")
        print("  Run build_window_catalogue.py first.")
        sys.exit(1)

    print(f"[rank {rank}] Loading window catalogue: {wcat_path}")
    wcat = pd.read_csv(wcat_path, low_memory=False)
    wcat["window_start"] = pd.to_datetime(wcat["window_start"], errors="coerce", utc=True)
    wcat["bird_id"] = wcat["bird_id"].astype(str)

    # Group windows by bird — build TWO lookups:
    #   raw_key  -> window list  (for passing to process_bird_gp as bird_id)
    #   safe_key -> raw_key      (for matching against CSV filename stems)
    windows_by_bird = {
        bid: list(grp["window_start"])
        for bid, grp in wcat.groupby("bird_id")
    }
    safe_to_raw = {_safe_filename(k): k for k in windows_by_bird}

    # Diagnostic: show first few keys so mismatches are obvious in the log
    print(f"[rank {rank}] Example catalogue bird_ids (raw):  {list(windows_by_bird.keys())[:3]}")
    print(f"[rank {rank}] Example catalogue bird_ids (safe): {list(safe_to_raw.keys())[:3]}")

    # Per-bird CSVs
    per_bird_dir = Path(args.per_bird_dir)
    bird_files = sorted(per_bird_dir.glob("*.csv"))
    print(f"[rank {rank}] Found {len(bird_files)} bird files")
    print(f"[rank {rank}] Example CSV stems: {[f.stem for f in bird_files[:3]]}")

    # Distribute birds across ranks
    my_birds = [f for i, f in enumerate(bird_files) if i % world_size == rank]
    print(f"[rank {rank}] Processing {len(my_birds)} birds")

    n_matched = 0
    n_skipped = 0
    all_rows = []

    for bird_csv in tqdm(my_birds, desc=f"rank {rank}", position=rank, leave=True):
        stem = bird_csv.stem

        # Exact match: CSV stem was produced by safe_filename(individual-local-identifier)
        raw_key = safe_to_raw.get(stem)

        if raw_key is None:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"[rank {rank}][no match] stem='{stem}' not found in catalogue")
            continue

        n_matched += 1
        window_starts = windows_by_bird[raw_key]

        try:
            rows = process_bird_gp(
                bird_csv,
                window_starts,
                n_iter=args.n_iter,
                batch_size=args.batch_size,
                device=device,
                torch=torch,
                gpytorch=gpytorch,
                window_hours=args.window_hours,
            )
        except Exception as e:
            print(f"[rank {rank}][error] {bird_csv.name}: {e}")
            continue

        all_rows.extend(rows)

        # Flush periodically to avoid memory blow-up
        if len(all_rows) > 5_000:
            _flush(all_rows, rank, outdir)
            all_rows = []

    if all_rows:
        _flush(all_rows, rank, outdir)

    print(f"[rank {rank}] Done. matched={n_matched} skipped={n_skipped}")


def _flush(rows: list, rank: int, outdir: Path):
    """Append rows to rank-specific output shard."""
    df = pd.DataFrame(rows)
    out_path = outdir / f"gp_features_rank{rank:04d}.csv"
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=write_header, index=False)
    print(f"  [rank {rank}] Flushed {len(rows)} rows → {out_path.name}")


if __name__ == "__main__":
    main()