#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/model/fal/2011-2014_3,6,9,12_baseline.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_fast-ecod.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal_fast-ecod.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal_noozone.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_noozone.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_noozone.yaml
  configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_clearsky_noozone.yaml
)

EVAL_ARGS=""
CLOSURE_ARGS="--skip_cross --skip_input_anomaly_plots"
for config in "${CONFIGS[@]}"; do
  echo "==> eval.py ${config}"
  python eval.py --config_file "${config}" ${EVAL_ARGS:-}

  #echo "==> eval_closure.py ${config}"
  #python eval_closure.py --config_file "${config}" ${CLOSURE_ARGS:-}
done
