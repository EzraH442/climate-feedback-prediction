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
    "global_tsr_era5.png",
    "global_tsr_nn.png",
    "global_tsr_mbe.png",
    "global_tsr_rmse.png",
    "kern_all_rrtm_2015-09.png",
    "kern_clr_rrtm_2015-09.png",
    "kern_all_nn_grad_2015-09.png",
    "kern_clr_nn_grad_2015-09.png",
    "kern_all_nn_grad-rrtm_2015-09.png",
    "kern_clr_nn_grad-rrtm_2015-09.png",
    "kern_all_rrtm_2015-12.png",
    "kern_clr_rrtm_2015-12.png",
    "kern_all_nn_grad-rrtm_2015-12.png",
    "kern_clr_nn_grad-rrtm_2015-12.png",
    "delta_k_rrtm_np.png",
    "delta_k_nn_grad-rrtm_north_pole.png",
    "tsr_contour_fal_ecod.png",
    "closure_test/2012_09/delta_fal.png",
    "closure_test/2012_09/delta_tcwv.png",
    "closure_test/2012_09/delta_ecod.png",
    "closure_test/2012_09/delta_tsr.png",
    "closure_test/2012_09/all/a.png",
    "closure_test/2012_09/all/c.png",
    "closure_test/2012_09/all/q.png",
    "closure_test/2012_09/all/dR_sum.png",
    "closure_test/2012_09/all/dR_res.png",
    "closure_test/2012_09/all/k_dR_sum.png",
    "closure_test/2012_09/all/k_dR_res.png",
    "closure_test/2012_09/all/cross_dR_sum.png",
    "closure_test/2012_09/all/cross_dR_res.png",
    "closure_test/2012_09/clr/a.png",
    "closure_test/2012_09/clr/q.png",
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
        path = latest_figures_dir(model_dir) / plot_path
        if not path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(path)
        images.append((label, path))

    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = [axes] if len(images) == 1 else axes.ravel()

    for ax, (label, img_path) in zip(axes, images):
        ax.imshow(mpimg.imread(img_path))
        ax.set_title(label)
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
