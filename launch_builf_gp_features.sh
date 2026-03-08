#!/bin/bash
#SBATCH --account=ai4wy-eap
#SBATCH --partition=ai4wy
#SBATCH --job-name=eagle_gp_features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2              # adjust gpu type if not h100
#SBATCH --cpus-per-task=64
#SBATCH --mem=0                         # take all memory on the node
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "JOB START  : $(date)"
echo "HOST       : $(hostname)"
echo "WORKDIR    : $(pwd)"
echo "SLURM JOB  : ${SLURM_JOB_ID}"
echo "NODES      : ${SLURM_JOB_NUM_NODES}"
echo "NTASKS     : ${SLURM_NTASKS}"
echo "GPUS       : ${SLURM_GPUS:-unset}"
echo "CUDA VIS   : ${CUDA_VISIBLE_DEVICES:-unset}"

# --- Activate conda env
if [ -f "$HOME/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
    conda activate ai4wy
fi
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
conda activate ai4wy

# --- Sanity checks
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('ngpu', torch.cuda.device_count())" || true
python -c "import gpytorch; print('gpytorch', gpytorch.__version__)" || true

# ---- Paths
PER_BIRD_DIR="golden_eai_outputs/per_bird_csv"
WINDOW_CATALOGUE="golden_eai_outputs/window_catalogue_1h.csv"
OUTDIR="golden_eai_outputs"

# ---- Run across all nodes/tasks via srun
# SLURM sets SLURM_PROCID (== RANK) and SLURM_NTASKS (== WORLD_SIZE) automatically.
# LOCAL_RANK is derived from SLURM_LOCALID so each task selects its own GPU.
srun python build_gp_features.py \
    --per_bird_dir  "${PER_BIRD_DIR}" \
    --window_catalogue "${WINDOW_CATALOGUE}" \
    --outdir "${OUTDIR}" \
    --window_hours 1 \
    --n_iter 150 \
    --batch_size 512

# ---- Merge shards (runs once on the head node after srun completes)
echo "Merging shards..."
python build_gp_features.py \
    --merge \
    --window_catalogue "${WINDOW_CATALOGUE}" \
    --outdir "${OUTDIR}"

echo "JOB END    : $(date)"
