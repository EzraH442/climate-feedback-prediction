#!/usr/bin/env bash


set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <era5_directory>"
    echo "Example: $0 data/era5"
    exit 1
fi

DATA_ROOT="$1"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Error: '$DATA_ROOT' is not a directory"
    exit 1
fi

shopt -s nullglob nocaseglob

for f in "$DATA_ROOT"/*.grib; do
    out="${f%.*}.nc"

    if [[ -f "$out" ]]; then
        echo "Skipping (exists): $out"
        continue
    fi

    echo "Converting: $f -> $out"

    grib_to_netcdf \
        -T \
        -o "$out" \
        "$f"
done