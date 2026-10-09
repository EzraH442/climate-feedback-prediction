#!/bin/bash
#SBATCH --job-name=sob-lambda
#SBATCH --account=rrg-yihuang-ad
#SBATCH --time=3:00:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ezra.huang@mail.mcgill.ca
#SBATCH --array=0-24

set -euo pipefail

CONFIG="configs/experiments/fal/sobolev_lambda.yaml"
CHECKPOINT_ROOT="./sobolev-ablation/checkpoints"
LAMBDAS=(0.25 0.5 1 2 4)

train_one() {
    local lambda="$1"
    local seed="$2"
    local name="2011-2014_3,6,9,12_sob_fal_${lambda}_clearsky"

    python train.py \
        --config_file "${CONFIG}" \
        --no-resume \
        --seed "${seed}" \
        "train.sobolev_alpha=${lambda}" \
        "train.name=${name}" \
        "train.checkpoint_dir=${CHECKPOINT_ROOT}/${name}"
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export COMET_API_KEY="${COMET_API_KEY:-MoBGkV7uhoNarGzMpipBaZYsJ}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module purge
    module load StdEnv/2023
    module load python/3.11
    module load cuda/12.2
    module load mpi4py/4.1.0
    module load scipy-stack
    module load httpproxy

    source ~/venv/bin/activate
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    task_id="${SLURM_ARRAY_TASK_ID}"
    lambda_index=$((task_id / 5))
    seed=$((task_id % 5 + 1))
    train_one "${LAMBDAS[$lambda_index]}" "${seed}"
else
    for lambda in "${LAMBDAS[@]}"; do
        for seed in 1 2 3 4 5; do
            train_one "${lambda}" "${seed}"
        done
    done
fi
