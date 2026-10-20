#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=rrg-yihuang-ad
#SBATCH --time=01:00:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
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

export PROJ_DATA="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/proj/9.2.0/share/proj"
source ~/venv/bin/activate

# python train.py
