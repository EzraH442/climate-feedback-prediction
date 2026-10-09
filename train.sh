#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --account=rrg-yihuang-ad
#SBATCH --time=3:00:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ezra.huang@mail.mcgill.ca

set -euo pipefail

CONFIG_FILE="${1:-}"
SEED="${2:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
    echo "Usage: $0 <config-file>"
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export COMET_API_KEY=MoBGkV7uhoNarGzMpipBaZYsJ

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load mpi4py/4.1.0
module load scipy-stack
module load httpproxy

source ~/venv/bin/activate

if [[ -n "${SEED}" ]]; then
    python train.py --config_file "${CONFIG_FILE}" --no-resume --seed "${SEED}"
else
    python train.py --config_file "${CONFIG_FILE}" --no-resume
fi
