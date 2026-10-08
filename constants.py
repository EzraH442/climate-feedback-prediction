from pathlib import Path

CLOSURE_NORTH_BOUNDARY = 75
SHAKIROVA_NORTH_BOUNDARY = 70

ALBEDO_KERNEL_PATH = Path("data/ERA5_kernels/ERA5_kernel_alb_TOA.nc")
WATER_VAPOR_KERNEL_PATH = Path(
    "data/ERA5_kernels/layer_specified_ta_wv_kernel/ERA5_kernel_wv_sw_nodp_TOA.nc"
)
QT_PATH = Path("data/era5/era5_plev_qt_monthly_downscaled.nc")
NEURIPS_QT_PATH = Path("data/era5/era5_plev_qt_monthly_downscaled_neurips_1990_2020.nc")

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
RESIDUAL_SOURCES = ["nn", "k", "nn_allcross"]
KERNEL_YEARS = set(range(2011, 2016))

TABLE_MODEL_SPECS = [
    ("NN", Path("configs/model/fal/2011-2014_3,6,9,12_baseline.yaml"), None),
    (
        "NN + clear-sky",
        Path("configs/model/fal/2011-2014_3,6,9,12_baseline_clearsky.yaml"),
        None,
    ),
    (
        "NN + clear-sky + Sob.",
        Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml"),
        None,
    ),
]
SEED_AGGREGATED_MODELS = ["NN", "NN + clear-sky", "NN + clear-sky + Sob."]

ABLATION_SPECS = [
    ("kernel", None, None, "kernel"),
    (
        "NN",
        Path("configs/model/fal/2011-2014_3,6,9,12_baseline.yaml"),
        None,
        "nn",
    ),
    (
        "NN+clear-sky",
        Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml"),
        Path("checkpoints/fal/2011-2014_3,6,9,12_baseline_clearsky/best_model.pt"),
        "nn",
    ),
    (
        "NN+clear-sky+Sob.",
        Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml"),
        Path("checkpoints/fal/2011-2014_3,6,9,12_sob_fal_clearsky/best_model.pt"),
        "nn",
    ),
]

SOBOLEV_LAMBDA_SPECS = [
    (
        "lambda=0.25",
        Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_0.25_clearsky.yaml"),
    ),
    (
        "lambda=0.5",
        Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_0.5_clearsky.yaml"),
    ),
    ("lambda=1", Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_1_clearsky.yaml")),
    ("lambda=2", Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_2_clearsky.yaml")),
    ("lambda=4", Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_4_clearsky.yaml")),
]
