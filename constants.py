from pathlib import Path

# fmt: off
CLOSURE_NORTH_BOUNDARY   = 75
SHAKIROVA_NORTH_BOUNDARY = 70

ALBEDO_KERNEL_PATH      = Path("data/ERA5_kernels/ERA5_kernel_alb_TOA.nc")
WATER_VAPOR_KERNEL_PATH = Path("data/ERA5_kernels/layer_specified_ta_wv_kernel/ERA5_kernel_wv_sw_nodp_TOA.nc" )
QT_PATH                 = Path("data/era5/era5_plev_qt_monthly_downscaled.nc")

MONTH_NAMES = [
    "Jan", "Feb", "Mar",
    "Apr", "May", "Jun",
    "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec",
]
