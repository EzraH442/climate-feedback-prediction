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
#SBATCH --array=0-3

ARGS=(
"configs/model/fal/2011-2014_3,6,9,12_baseline.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml"
)

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export COMET_API_KEY=MoBGkV7uhoNarGzMpipBaZYsJ

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load mpi4py/4.1.0
module load scipy-stack
module load httpproxy

source ~/.venv/bin/activate

config_file=${ARGS[$SLURM_ARRAY_TASK_ID]}
python train.py --config_file "${config_file}" --no-resume
