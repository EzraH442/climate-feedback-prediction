#!/bin/bash
#SBATCH --job-name=baseline-opt
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
#SBATCH --array=0-19

set -euo pipefail

CONFIG="configs/experiments/fal/baseline_optimization.yaml"
CHECKPOINT_ROOT="./baseline-optimization/checkpoints"
HIDDEN_SIZES=(30 50)
MONTH_KEYS=("3,6,9,12" "1-12")

ARGS=()
for month_key in "${MONTH_KEYS[@]}"; do
    for hidden_size in "${HIDDEN_SIZES[@]}"; do
        if [[ "${month_key}" == "3,6,9,12" ]]; then
            name="baseline_opt_2011-2014_3,6,9,12_hidden-${hidden_size}"
            ARGS+=(
                "model.hidden_dim_sizes=[${hidden_size}] dataset.months=[3,6,9,12] dataset.era5.path=./data/fal/era5_processed_mar_june_sep_dec_2011_2014 preprocess.params_dir=preprocess_states/fal/2011-2014_3,6,9,12 train.name=${name} train.checkpoint_dir=${CHECKPOINT_ROOT}/${name}"
            )
        else
            name="baseline_opt_2011-2014_1-12_hidden-${hidden_size}"
            ARGS+=(
                "model.hidden_dim_sizes=[${hidden_size}] dataset.months=[1,2,3,4,5,6,7,8,9,10,11,12] dataset.era5.path=./data/fal/era5_processed preprocess.params_dir=preprocess_states/fal/2011-2014_1-12 train.name=${name} train.checkpoint_dir=${CHECKPOINT_ROOT}/${name}"
            )
        fi
    done
done

train_one() {
    local overrides="$1"
    local seed="$2"
    local override_args=()
    read -r -a override_args <<< "${overrides}"
    python train.py \
        --config_file "${CONFIG}" \
        --no-resume \
        --seed "${seed}" \
        "${override_args[@]}"
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

    source ~/.venv/bin/activate

    task_id="${SLURM_ARRAY_TASK_ID}"
    config_index=$((task_id / 5))
    seed=$((task_id % 5 + 1))
    train_one "${ARGS[$config_index]}" "${seed}"
else
    for overrides in "${ARGS[@]}"; do
        for seed in 1 2 3 4 5; do
            train_one "${overrides}" "${seed}"
        done
    done
fi
