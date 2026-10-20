#!/bin/bash
#SBATCH --job-name=var-ablate
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
#SBATCH --array=0-3

set -euo pipefail

CONFIG="configs/experiments/fal/variable_ablation.yaml"
CHECKPOINT_ROOT="./variable-ablation/checkpoints"

ARGS=(
"train.name=variable_ablation_ozone_ecod train.checkpoint_dir=${CHECKPOINT_ROOT}/variable_ablation_ozone_ecod"
"dataset.input_vars=[fal,hcc,mcc,lcc,sp,tciw,tclw,tcwv,tisr,ecod,ecod_fal] model.input_dim=11 train.name=variable_ablation_noozone_ecod train.checkpoint_dir=${CHECKPOINT_ROOT}/variable_ablation_noozone_ecod"
"dataset.input_vars=[fal,hcc,mcc,lcc,sp,tciw,tclw,tcwv,tco3,tisr] model.input_dim=10 train.name=variable_ablation_ozone_noecod train.checkpoint_dir=${CHECKPOINT_ROOT}/variable_ablation_ozone_noecod"
"dataset.input_vars=[fal,hcc,mcc,lcc,sp,tciw,tclw,tcwv,tisr] model.input_dim=9 train.name=variable_ablation_noozone_noecod train.checkpoint_dir=${CHECKPOINT_ROOT}/variable_ablation_noozone_noecod"
)

train_one() {
    local overrides="$1"
    local override_args=()
    read -r -a override_args <<< "${overrides}"
    python train.py --config_file "${CONFIG}" --no-resume "${override_args[@]}"
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export COMET_API_KEY="${COMET_API_KEY:-MoBGkV7uhoNarGzMpipBaZYsJ}"
export PROJ_DATA="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/proj/9.2.0/share/proj"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module purge
    module load StdEnv/2023
    module load python/3.11
    module load cuda/12.2
    module load mpi4py/4.1.0
    module load scipy-stack
    module load httpproxy

    source ~/venv/bin/activate

    train_one "${ARGS[$SLURM_ARRAY_TASK_ID]}"
else
    for overrides in "${ARGS[@]}"; do
        train_one "${overrides}"
    done
fi
