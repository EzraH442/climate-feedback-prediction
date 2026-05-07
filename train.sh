#!/bin/bash
#SBATCH --job-name=baseline-large-training
#SBATCH --account=rrg-yihuang-ad
#SBATCH --time=13:00:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ezra.huang@mail.mcgill.ca

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load mpi4py/4.1.0
module load scipy-stack
module load httpproxy

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python train.py
