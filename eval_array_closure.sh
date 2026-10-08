#!/bin/bash

ARGS=(
"configs/model/fal/2011-2014_3,6,9,12_baseline.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_noozone.yaml"
)
ARGS+=(
"configs/model/fal/2011-2014_3,6,9,12_baseline_true-tcc-ecod.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_true-tcc-ecod.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_true-tcc-ecod.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_true-tcc-ecod.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_noozone_true-tcc-ecod.yaml"
)

ARGS+=(
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_fast-ecod.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_fast-ecod.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_noozone.yaml"
)
ARGS+=(
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_noozone.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_fast-ecod_noozone.yaml"
"configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_clearsky_noozone.yaml"
)

for config_file in "${ARGS[@]}"; do
    #python experiments/standard_eval/main.py --config_file "${config_file}"
    python experiments/radiative_closure_timeseries/main.py --config_file "${config_file}" --skip_input_anomaly_plots --skip_cross
done
