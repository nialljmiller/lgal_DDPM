#!/usr/bin/env python3
"""
eagle_simclr.py

SimCLR contrastive representation learning on golden eagle 1-hour movement windows.

Input channels (7 dims per GPS fix):
    ground_speed, delta_lon, delta_lat, activity,
    sin(heading), cos(heading), altitude

Encoder:  Bidirectional GRU  →  h  (hidden_dim*2, pre-projection representation)
Projector: MLP  →  z  (64-dim, what NT-Xent loss trains on)

At inference, both h and z are written to disk alongside bird_id and window_start
so they join directly to window_catalogue_with_gp.csv.

The encoder backbone is isolated behind a clean interface — swap BACKBONE = "transformer"
to use a small Transformer encoder instead with no other changes required.

Training: single node, both GH200s via nn.DataParallel.
Note: DataParallel + PackedSequence has a known PyTorch bug (batch_sizes must stay on CPU).
      We avoid it by using padded tensors + sequence length masking instead of PackedSequence.

Usage (train):
    python eagle_simclr.py --train \\
        --per_bird_dir golden_eai_outputs/per_bird_csv \\
        --logdir logs/simclr_run1

Usage (infer — write embeddings):
    python eagle_simclr.py \\
        --per_bird_dir golden_eai_outputs/per_bird_csv \\
        --logdir logs/simclr_run1 \\
        --milestone 50000

Usage (resume training):
    python eagle_simclr.py --train \\
        --per_bird_dir golden_eai_outputs/per_bird_csv \\
        --logdir logs/simclr_run1 \\
        --milestone 50000
"""

import os
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.decomposition import PCA
from umap import UMAP

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

BACKBONE = "gru"          # "gru" | "transformer"  — only GRU is implemented;
                           # swap this string and add TransformerBackbone below

INPUT_DIM  = 7             # speed, dlon, dlat, activity, sin_hdg, cos_hdg, alt
HIDDEN_DIM = 128           # GRU hidden units per direction (total h = 256)
N_LAYERS   = 4             # GRU layers
OUT_DIM    = 64            # projected z dimension
DROP_PROB  = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(value: str) -> str:
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


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def check_logdir(logdir: Path):
    if logdir.exists():
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        logdir.rename(f"{logdir}_{now}")
    logdir.mkdir(parents=True)


# ---------------------------------------------------------------------------
# Data — window extraction from per-bird CSVs
# ---------------------------------------------------------------------------

def extract_window_sequence(df: pd.DataFrame, window_start: pd.Timestamp,
                             window_hours: int = 1) -> np.ndarray | None:
    """
    Extract a [T, 7] float32 array for one 1-hour window from a sorted bird DataFrame.

    Channels: speed, delta_lon, delta_lat, activity, sin_heading, cos_heading, altitude
    Returns None if fewer than 3 valid fixes.
    """
    window_end = window_start + pd.Timedelta(hours=window_hours)
    mask = (df["_t"] >= window_start) & (df["_t"] < window_end)
    w = df[mask].sort_values("_t")

    if len(w) < 3:
        return None

    def col(name, default=0.0):
        if name in w.columns:
            v = pd.to_numeric(w[name], errors="coerce").values.astype(np.float32)
        else:
            v = np.full(len(w), default, dtype=np.float32)
        v = np.nan_to_num(v, nan=default)
        return v

    speed    = col("ground-speed",          0.0)
    lon      = col("location-long",         0.0)
    lat      = col("location-lat",          0.0)
    activity = col("eobs:activity",         0.0)
    heading  = col("heading",               0.0)
    alt      = col("height-above-ellipsoid",0.0)

    # Relative displacement (delta, not absolute coords)
    dlon = np.diff(lon, prepend=lon[0])
    dlat = np.diff(lat, prepend=lat[0])

    # Circular heading encoding
    sin_h = np.sin(np.radians(heading)).astype(np.float32)
    cos_h = np.cos(np.radians(heading)).astype(np.float32)

    # Per-window standardisation — each channel zero-mean unit-variance
    def norm(x):
        mu, sigma = x.mean(), x.std()
        return (x - mu) / (sigma + 1e-8)

    seq = np.stack([
        norm(speed),
        norm(dlon),
        norm(dlat),
        norm(activity),
        sin_h,          # already bounded [-1,1], skip norm
        cos_h,
        norm(alt),
    ], axis=1)          # [T, 7]

    return seq.astype(np.float32)


class EagleWindowDataset(data.Dataset):
    """
    Loads all 1-hour windows from per-bird CSVs.

    __getitem__ returns TWO independently augmented views of the same window
    (the SimCLR positive pair), plus metadata for joining back to catalogue.
    """

    def __init__(self, per_bird_dir: Path, augmenter, window_hours: int = 1):
        self.augmenter    = augmenter
        self.window_hours = window_hours

        # Index of (bird_id, window_start, sequence [T,7])
        self.windows = []
        self._build_index(per_bird_dir)
        print(f"Dataset: {len(self.windows)} windows from {per_bird_dir}")

    def _build_index(self, per_bird_dir: Path):
        bird_files = sorted(per_bird_dir.glob("*.csv"))
        for bird_csv in tqdm(bird_files, desc="Indexing windows"):
            try:
                df = pd.read_csv(bird_csv, low_memory=False)
            except Exception:
                continue

            if "timestamp" not in df.columns:
                continue

            df["_t"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)

            if len(df) < 3:
                continue

            bird_id = str(df["individual-local-identifier"].dropna().iloc[0]) \
                if "individual-local-identifier" in df.columns \
                and df["individual-local-identifier"].notna().any() \
                else bird_csv.stem

            # Floor timestamps to window boundaries
            df["_window"] = df["_t"].dt.floor(f"{self.window_hours}h")

            for ws, _ in df.groupby("_window", sort=True):
                seq = extract_window_sequence(df, ws, self.window_hours)
                if seq is not None and len(seq) >= 3:
                    self.windows.append((bird_id, ws, seq))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        bird_id, window_start, seq = self.windows[idx]
        view0 = self.augmenter(seq.copy())
        view1 = self.augmenter(seq.copy())
        return view0, view1, bird_id, str(window_start)


def collate_eagle(batch):
    """
    Pad variable-length sequences to the longest in the batch.
    Returns padded tensors + lengths (no PackedSequence — avoids DataParallel bug).
    """
    v0s, v1s, bird_ids, window_starts = zip(*batch)

    def pad(seqs):
        lengths = [len(s) for s in seqs]
        T_max   = max(lengths)
        C       = seqs[0].shape[1]
        out     = np.zeros((len(seqs), T_max, C), dtype=np.float32)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        return torch.tensor(out), torch.tensor(lengths, dtype=torch.long)

    v0_padded, v0_lens = pad(v0s)
    v1_padded, v1_lens = pad(v1s)

    return v0_padded, v0_lens, v1_padded, v1_lens, list(bird_ids), list(window_starts)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

class Augmenter:
    """
    Applies a random subset of augmentations to a [T, C] numpy sequence.

    Augmentations:
        time_jitter    — shuffle a small fraction of fix ordering
        random_crop    — take a contiguous subset (>= min_frac of fixes)
        channel_dropout — zero out an entire channel
        gaussian_noise  — add N(0, noise_std) to all channels
    """

    def __init__(
        self,
        time_jitter_frac: float = 0.05,
        crop_min_frac:    float = 0.6,
        channel_drop_p:   float = 0.2,
        noise_std:        float = 0.05,
        p_each:           float = 0.5,
    ):
        self.time_jitter_frac = time_jitter_frac
        self.crop_min_frac    = crop_min_frac
        self.channel_drop_p   = channel_drop_p
        self.noise_std        = noise_std
        self.p_each           = p_each

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        T, C = seq.shape

        # 1. Gaussian noise
        if np.random.rand() < self.p_each:
            seq = seq + np.random.randn(*seq.shape).astype(np.float32) * self.noise_std

        # 2. Random crop
        if np.random.rand() < self.p_each and T >= 4:
            min_len = max(3, int(T * self.crop_min_frac))
            crop_len = np.random.randint(min_len, T + 1)
            start    = np.random.randint(0, T - crop_len + 1)
            seq      = seq[start : start + crop_len]
            T        = len(seq)

        # 3. Time jitter — swap a small fraction of adjacent fixes
        if np.random.rand() < self.p_each and T >= 4:
            n_swaps = max(1, int(T * self.time_jitter_frac))
            for _ in range(n_swaps):
                i = np.random.randint(0, T - 1)
                seq[[i, i + 1]] = seq[[i + 1, i]]

        # 4. Channel dropout
        if np.random.rand() < self.p_each:
            ch = np.random.randint(0, C)
            seq[:, ch] = 0.0

        return seq


# ---------------------------------------------------------------------------
# Encoder backbone  (GRU — swap here for Transformer)
# ---------------------------------------------------------------------------

class GRUBackbone(nn.Module):
    """
    Bidirectional GRU encoder.
    Input:  x [B, T, input_dim],  lengths [B]
    Output: h [B, hidden_dim*2]  (last-layer fwd+bwd hidden states concatenated)
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, drop_prob: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers,
            bidirectional=True,
            dropout=drop_prob if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Pack to ignore padding
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        # h: [n_layers*2, B, hidden_dim]
        # Take last layer, concat fwd + bwd
        fwd = h[-2]   # [B, hidden_dim]
        bwd = h[-1]   # [B, hidden_dim]
        return torch.cat([fwd, bwd], dim=1)   # [B, hidden_dim*2]


class TransformerBackbone(nn.Module):
    """
    Stub — implement when ready to swap.
    Must match GRUBackbone interface: forward(x, lengths) -> h [B, d_model]
    """
    def __init__(self, input_dim, hidden_dim, n_layers, drop_prob):
        super().__init__()
        raise NotImplementedError("TransformerBackbone not yet implemented. Set BACKBONE='gru'.")


def build_backbone(backbone: str, input_dim, hidden_dim, n_layers, drop_prob) -> nn.Module:
    if backbone == "gru":
        return GRUBackbone(input_dim, hidden_dim, n_layers, drop_prob)
    elif backbone == "transformer":
        return TransformerBackbone(input_dim, hidden_dim, n_layers, drop_prob)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP with BatchNorm, maps h → z.
    Kept separate from backbone so we use h (not z) as the final representation.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# SimCLR model (backbone + projector)
# ---------------------------------------------------------------------------

class SimCLR(nn.Module):
    def __init__(self, backbone: nn.Module, projector: nn.Module):
        super().__init__()
        self.backbone  = backbone
        self.projector = projector

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        h = self.backbone(x, lengths)     # [B, hidden_dim*2]
        z = self.projector(h)             # [B, out_dim]
        return z, h


# ---------------------------------------------------------------------------
# NT-Xent loss (from Chen et al. 2020, adapted from contrastive_curves.py)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau       = tau
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sim_fn    = nn.CosineSimilarity(dim=2)

    def forward(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        B = z0.shape[0]
        z0 = F.normalize(z0, dim=1)
        z1 = F.normalize(z1, dim=1)

        zs  = torch.cat([z0, z1], dim=0)               # [2B, D]
        sim = self.sim_fn(zs.unsqueeze(0), zs.unsqueeze(1))  # [2B, 2B]

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_ij = torch.diag(sim,  B)
        pos_ji = torch.diag(sim, -B)
        positives = torch.cat([pos_ij, pos_ji]).reshape(2 * B, 1)

        # Mask out self-similarity and positive pairs
        mask = torch.ones(2 * B, 2 * B, dtype=torch.bool, device=z0.device)
        mask.fill_diagonal_(False)
        for i in range(B):
            mask[i, i + B] = False
            mask[i + B, i] = False

        negatives = sim[mask].reshape(2 * B, -1)

        logits = torch.cat([positives, negatives], dim=1) / self.tau
        labels = torch.zeros(2 * B, dtype=torch.long, device=z0.device)

        loss = self.criterion(logits, labels) / (2 * B)
        return loss


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_embeddings(hs: np.ndarray, logdir: Path, step: int, bird_ids: list):
    """PCA and UMAP of h vectors, coloured by bird."""
    unique_birds  = sorted(set(bird_ids))
    bird_to_idx   = {b: i for i, b in enumerate(unique_birds)}
    colour_idx    = np.array([bird_to_idx[b] for b in bird_ids])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Step {step:08d}")

    # PCA
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(hs)
    axes[0].scatter(pca_emb[:, 0], pca_emb[:, 1], c=colour_idx,
                    cmap="tab20", alpha=0.4, s=2)
    axes[0].set_title(f"PCA  (var: {pca.explained_variance_ratio_.sum():.2f})")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # UMAP
    try:
        umap_emb = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(hs)
        axes[1].scatter(umap_emb[:, 0], umap_emb[:, 1], c=colour_idx,
                        cmap="tab20", alpha=0.4, s=2)
        axes[1].set_title("UMAP")
        axes[1].set_xticks([]); axes[1].set_yticks([])
    except Exception as e:
        axes[1].set_title(f"UMAP failed: {e}")

    plt.tight_layout()
    plt.savefig(logdir / f"{step:08d}_embeddings.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model:          SimCLR,
        per_bird_dir:   Path,
        logdir:         Path,
        batch_size:     int   = 2048,
        lr:             float = 3e-4,
        tau:            float = 0.07,
        train_num_steps:int   = 200_000,
        save_every:     int   = 5_000,
        infer_every:    int   = 10_000,
        num_workers:    int   = 8,
        window_hours:   int   = 1,
    ):
        self.logdir          = logdir
        self.batch_size      = batch_size
        self.train_num_steps = train_num_steps
        self.save_every      = save_every
        self.infer_every     = infer_every
        self.step            = 0

        # DataParallel — wraps whole SimCLR, works because we use padded tensors
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
        self.model.to(DEVICE)

        self.criterion = NTXentLoss(tau=tau).to(DEVICE)
        self.opt       = Adam(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=train_num_steps, eta_min=1e-6)

        augmenter = Augmenter()
        self.ds   = EagleWindowDataset(per_bird_dir, augmenter, window_hours)
        self.dl   = cycle(data.DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_eagle,
            pin_memory=True,
            drop_last=True,       # NT-Xent needs consistent B
        ))

        self.loss_log = logdir / "loss.csv"
        with open(self.loss_log, "w") as f:
            f.write("step,loss\n")

    # ------------------------------------------------------------------
    def save(self, step: int):
        m = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            "step":       step,
            "model":      m.state_dict(),
            "opt":        self.opt.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
        }, self.logdir / f"{step:08d}_checkpoint.pt")

    def load(self, step: int):
        ckpt = torch.load(self.logdir / f"{step:08d}_checkpoint.pt", map_location=DEVICE)
        m = self.model.module if hasattr(self.model, "module") else self.model
        m.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        print(f"Loaded checkpoint at step {self.step}")

    # ------------------------------------------------------------------
    def train(self):
        print(f"Training for {self.train_num_steps} steps, batch_size={self.batch_size}")
        self.model.train()

        while self.step < self.train_num_steps:
            v0, v0_lens, v1, v1_lens, bird_ids, _ = next(self.dl)

            v0 = v0.to(DEVICE); v0_lens = v0_lens.to(DEVICE)
            v1 = v1.to(DEVICE); v1_lens = v1_lens.to(DEVICE)

            z0, _ = self.model(v0, v0_lens)
            z1, _ = self.model(v1, v1_lens)

            loss = self.criterion(z0, z1)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            self.opt.zero_grad()
            self.scheduler.step()

            if self.step % 10 == 0:
                print(f"step {self.step:08d}  loss {loss.item():.4f}  "
                      f"lr {self.scheduler.get_last_lr()[0]:.2e}")

            with open(self.loss_log, "a") as f:
                f.write(f"{self.step},{loss.item():.6f}\n")

            if self.step > 0 and self.step % self.save_every == 0:
                self.save(self.step)

            if self.step > 0 and self.step % self.infer_every == 0:
                self.infer()
                self.model.train()

            self.step += 1

        self.save(self.step)
        print("Training complete.")

    # ------------------------------------------------------------------

    
    @torch.no_grad()
    def infer(self):
        print(f"[step {self.step}] Running inference on {len(self.ds)} windows...")
        self.model.eval()
    
        m = self.model.module if hasattr(self.model, "module") else self.model
    
        loader = data.DataLoader(
            self.ds,
            batch_size=512,
            shuffle=False,
            num_workers=0,        # data already in RAM — workers just add fork overhead
            collate_fn=collate_eagle,
            pin_memory=False,
        )
    
        all_h, all_z, all_birds, all_windows = [], [], [], []
    
        for v0, v0_lens, _, _, bird_ids, window_starts in tqdm(loader, desc="infer", leave=False):
            v0      = v0.to(DEVICE)
            v0_lens = v0_lens.to(DEVICE)
            z, h    = m(v0, v0_lens)
            all_h.append(h.cpu().numpy())
            all_z.append(z.cpu().numpy())
            all_birds.extend(bird_ids)
            all_windows.extend(window_starts)
    
        hs = np.concatenate(all_h, axis=0)
        zs = np.concatenate(all_z, axis=0)

        H, Z = hs.shape[1], zs.shape[1]
        h_cols = {f"h_{i}": hs[:, i] for i in range(H)}
        z_cols = {f"z_{i}": zs[:, i] for i in range(Z)}

        df = pd.DataFrame({
            "bird_id":      all_birds,
            "window_start": all_windows,
            **h_cols,
            **z_cols,
        })

        out_csv = self.logdir / f"{self.step:08d}_embeddings.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Wrote embeddings → {out_csv}  ({len(df)} rows)")
    
        # Subsample for plotting — UMAP on 144k points hangs
        plot_n = min(10_000, len(hs))
        idx    = np.random.choice(len(hs), plot_n, replace=False)
        plot_embeddings(hs[idx], self.logdir, self.step, [all_birds[i] for i in idx])
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eagle SimCLR contrastive learning")
    parser.add_argument("--per_bird_dir",   default="golden_eai_outputs/per_bird_csv")
    parser.add_argument("--logdir",         default="logs/simclr")
    parser.add_argument("--train",          action="store_true", default=False)
    parser.add_argument("--milestone",      type=int, default=0,
                        help="Resume from this checkpoint step (0 = fresh start)")
    parser.add_argument("--batch_size",     type=int,   default=2048)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--tau",            type=float, default=0.07)
    parser.add_argument("--hidden_dim",     type=int,   default=128)
    parser.add_argument("--n_layers",       type=int,   default=4)
    parser.add_argument("--out_dim",        type=int,   default=64)
    parser.add_argument("--drop_prob",      type=float, default=0.2)
    parser.add_argument("--train_steps",    type=int,   default=200_000)
    parser.add_argument("--save_every",     type=int,   default=5_000)
    parser.add_argument("--infer_every",    type=int,   default=10_000)
    parser.add_argument("--num_workers",    type=int,   default=8)
    parser.add_argument("--window_hours",   type=int,   default=1)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if args.milestone == 0 and args.train:
        check_logdir(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Sanity
    print(f"torch {torch.__version__}")
    print(f"CUDA  {torch.cuda.is_available()},  {torch.cuda.device_count()} GPU(s)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Build model
    backbone  = build_backbone(BACKBONE, INPUT_DIM, args.hidden_dim, args.n_layers, args.drop_prob)
    projector = ProjectionHead(args.hidden_dim * 2, args.out_dim)
    model     = SimCLR(backbone, projector)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SimCLR params: {n_params:,}  backbone={BACKBONE}  h={args.hidden_dim*2}  z={args.out_dim}")

    trainer = Trainer(
        model         = model,
        per_bird_dir  = Path(args.per_bird_dir),
        logdir        = logdir,
        batch_size    = args.batch_size,
        lr            = args.lr,
        tau           = args.tau,
        train_num_steps = args.train_steps,
        save_every    = args.save_every,
        infer_every   = args.infer_every,
        num_workers   = args.num_workers,
        window_hours  = args.window_hours,
    )

    if args.milestone != 0:
        trainer.load(args.milestone)

    if args.train:
        # Save hyperparams
        with open(logdir / "hyperparameters.txt", "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")
        trainer.train()
    else:
        print("Inference mode")
        trainer.infer()


if __name__ == "__main__":
    main()
