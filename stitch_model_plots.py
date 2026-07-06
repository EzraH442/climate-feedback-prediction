import argparse
import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


MODELS = [
    ("baseline", Path("checkpoints/fal/2011-2014_3,6,9,12_baseline")),
    ("sob_fal", Path("checkpoints/fal/2011-2014_3,6,9,12_sob_fal")),
    ("sob_fal_clear", Path("checkpoints/fal/2011-2014_3,6,9,12_sob_fal_clearsky")),
    ("sob_fal_tcwv", Path("checkpoints/fal/2011-2014_3,6,9,12_sob_fal_tcwv")),
    ("sob_fal_noozone", Path("checkpoints/fal/2011-2014_3,6,9,12_sob_fal_noozone")),
    ("sob_fal_clear_noozone", Path("checkpoints/fal/2011-2014_3,6,9,12_sob_fal_clearsky_noozone")),
    ("sob_fal_tcwv_noozone", Path("checkpoints/fal/2011-2014_3,6,9,12_sob_fal_tcwv_noozone")),
    ("sob_fal_tcwv_clear_noozone", Path("checkpoints/fal/2011-2014_3,6,9,12_sob_fal_tcwv_clearsky_noozone")),
]

PLOTS = [
    "all/tsr_test/era5.png",
    "all/tsr_test/nn.png",
    "all/tsr_test/mbe.png",
    "all/tsr_test/rmse.png",
    "all/tsr_contour_test/fal/2015-09/fal_ecod.png",
    "all/kernel_contour_test/ecod/2015-09/ecod_fal.png",
    *[
        f"{sky}/kernel_date_test/{kernel}/{date}/{plot}.png"
        for kernel in ("fal", "tcwv")
        for date in ("2013-09", "2015-09", "2015-12")
        for sky, plots in (
            ("all", ("nn_grad", "nn_grad_np", "rrtm", "rrtm_np", "nn_grad-rrtm", "nn_grad-rrtm_np")),
            ("clr", ("nn_grad", "nn_grad_np", "rrtm", "rrtm_np", "nn_grad-rrtm", "nn_grad-rrtm_np")),
        )
        for plot in plots
    ],
    *[
        f"clr/second_order_test/{kernel}/2013-09_minus_2012-09/{plot}.png"
        for kernel in ("fal", "tcwv")
        for plot in ("nn_grad_np", "rrtm_np", "nn_grad-rrtm_np")
    ],
    "closure_test/2012_09/delta_fal.png",
    "closure_test/2012_09/delta_tcwv.png",
    "closure_test/2012_09/delta_ecod.png",
    "closure_test/2012_09/delta_tsr.png",
    "closure_test/2012_09/all/dR_a.png",
    "closure_test/2012_09/all/dR_c.png",
    "closure_test/2012_09/all/dR_q.png",
    "closure_test/2012_09/all/dR_sum.png",
    "closure_test/2012_09/all/dR_res.png",
    "closure_test/2012_09/all/k_dR_sum.png",
    "closure_test/2012_09/all/k_dR_res.png",
    "closure_test/2012_09/all/cross_dR_sum.png",
    "closure_test/2012_09/all/cross_dR_res.png",
    "closure_test/2012_09/clr/dR_a,clr.png",
    "closure_test/2012_09/clr/dR_q,clr.png",
    "closure_test/2012_09/clr/dR_sum,clr.png",
    "closure_test/2012_09/clr/dR_res,clr.png",
    "closure_test/2012_09/clr/k_dR_sum,clr.png",
    "closure_test/2012_09/clr/k_dR_res,clr.png",
    "closure_test/all/timeseries_net.png",
    "closure_test/all/timeseries_rmse.png",
    "closure_test/all/timeseries_mbe.png",
    "closure_test/all/timeseries_sum_cross.png",
    "closure_test/all/boxplot_residuals_by_year_month_global.png",
    "closure_test/clr/timeseries_net.png",
    "closure_test/clr/timeseries_rmse.png",
    "closure_test/clr/timeseries_mbe.png",
    "closure_test/clr/boxplot_residuals_by_year_month_global.png",
]


def latest_figures_dir(model_dir: Path) -> Path:
    figures = model_dir / "figures"
    epochs = [p for p in figures.iterdir() if p.is_dir() and p.name.isdigit()]
    if not epochs:
        raise FileNotFoundError(f"No numeric figures directory under {figures}")
    return max(epochs, key=lambda p: int(p.name))


def stitch(plot_path: str, output: Path, cols: int, skip_missing: bool):
    images = []
    for label, model_dir in MODELS:
        try:
            path = latest_figures_dir(model_dir) / plot_path
        except FileNotFoundError:
            path = None
        if path is not None and not path.exists():
            path = None
        if skip_missing and path is None:
            continue
        images.append((label, path))
    if not images:
        return

    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (label, img_path) in zip(axes, images):
        if img_path is not None:
            ax.imshow(mpimg.imread(img_path))
        ax.set_title(label if img_path is not None else f"{label}\nmissing")
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plot_path", nargs="?", help="Path relative to each model's latest figures dir.")
    parser.add_argument("-o", "--output", default="stitched")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.plot_path:
        stitch(args.plot_path, output, args.cols, args.skip_missing)
        return

    output.mkdir(parents=True, exist_ok=True)
    for plot_path in PLOTS:
        output_name = plot_path.replace("/", "__")
        stitch(plot_path, output / output_name, args.cols, args.skip_missing)


if __name__ == "__main__":
    main()
