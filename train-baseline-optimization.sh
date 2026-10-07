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
#SBATCH --array=0-9

set -euo pipefail

ROOT="baseline-optimization"
CONFIG_DIR="${ROOT}/configs"
CHECKPOINT_DIR="${ROOT}/checkpoints"
mkdir -p "${CONFIG_DIR}" "${CHECKPOINT_DIR}"

HIDDEN_SIZES=(11 15 20 30 50)
MONTH_KEYS=("3,6,9,12" "1-12")

write_config() {
    local hidden_size="$1"
    local month_key="$2"
    local months data_path preprocess_dir name

    if [[ "${month_key}" == "3,6,9,12" ]]; then
        months="  months: [3, 6, 9, 12]"
        data_path="./data/fal/era5_processed_mar_june_sep_dec_2011_2014"
        preprocess_dir="preprocess_states/fal/2011-2014_3,6,9,12"
        name="baseline_opt_2011-2014_3,6,9,12_hidden-${hidden_size}"
    else
        months="  months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"
        data_path="./data/fal/era5_processed"
        preprocess_dir="preprocess_states/fal/2011-2014_1-12"
        name="baseline_opt_2011-2014_1-12_hidden-${hidden_size}"
    fi

    local config_path="${CONFIG_DIR}/${name}.yaml"
    cat > "${config_path}" <<EOF
base: ../../configs/base/model/common.yaml

dataset:
  input_vars:
    - fal
    - hcc
    - mcc
    - lcc
    - sp
    - tciw
    - tclw
    - tcwv
    - tco3
    - tisr
    - ecod
    - ecod_fal
${months}
  era5:
    path: '${data_path}'
  kernels:
    raw_path: './data/fal/kernels'
    path: './data/fal/kernels_processed'
  train_years: [2011, 2012, 2013, 2014]
  val_years: [2015]
  test_years: [2015]

preprocess:
  params_dir: '${preprocess_dir}'
  ecod:
    enabled: True
    method: 'true'

model:
  input_dim: 12
  hidden_dim_sizes: [${hidden_size}]

train:
  name: '${name}'
  checkpoint_dir: './${CHECKPOINT_DIR}/${name}'
  sobolev: False
  sobolev_vars: []
EOF
    echo "${config_path}"
}

CONFIGS=()
for month_key in "${MONTH_KEYS[@]}"; do
    for hidden_size in "${HIDDEN_SIZES[@]}"; do
        CONFIGS+=("$(write_config "${hidden_size}" "${month_key}")")
    done
done

if [[ "${1:-}" == "--generate-only" ]]; then
    printf '%s\n' "${CONFIGS[@]}"
    exit 0
fi

train_one() {
    local config_file="$1"
    python train.py --config_file "${config_file}" --no-resume
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

    virtualenv --no-download "${SLURM_TMPDIR}/env"
    source "${SLURM_TMPDIR}/env/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index -r requirements.txt

    train_one "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
else
    for config_file in "${CONFIGS[@]}"; do
        train_one "${config_file}"
    done
fi
