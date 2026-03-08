#!/bin/bash
#SBATCH --account=ai4wy-eap
#SBATCH --partition=ai4wy
#SBATCH --job-name=eagle_SimCLR
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


if [ -f "$HOME/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
    conda activate ai4wy
fi
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
conda activate ai4wy

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('ngpu', torch.cuda.device_count())" || true

PER_BIRD_DIR="golden_eai_outputs/per_bird_csv"
LOGDIR="logs/simclr_$(date +%Y%m%d_%H%M%S)"
MILESTONE=0       # set to a step number to resume

python eagle_simclr.py \
    --train \
    --per_bird_dir   "${PER_BIRD_DIR}" \
    --logdir         "${LOGDIR}" \
    --milestone      "${MILESTONE}" \
    --batch_size     2048 \
    --lr             3e-4 \
    --tau            0.07 \
    --hidden_dim     128 \
    --n_layers       4 \
    --out_dim        64 \
    --train_steps    200000 \
    --save_every     5000 \
    --infer_every    10000 \
    --num_workers    8 \
    --window_hours   1

echo "JOB END    : $(date)"
