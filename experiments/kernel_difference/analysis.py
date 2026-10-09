from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point

from utils import global_mean


def as_dataarray(data, lon, lat):
    if hasattr(data, "dims"):
        return data
    return xr.DataArray(
        data,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )


def cyclic_data(data, lon):
    values = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
    return add_cyclic_point(values, coord=np.asarray(lon), axis=-1)


def mean_text(data, lon, lat):
    return f"{float(global_mean(as_dataarray(data, lon, lat))):.2f} W m$^{{-2}}$ %$^{{-1}}$"


def residual_text(data, lon, lat):
    field = as_dataarray(data, lon, lat)
    mbe = float(global_mean(field))
    rmse = float(np.sqrt(global_mean(field**2)))
    return f"MBE: {mbe:.2f} W m$^{{-2}}$ %$^{{-1}}$, RMSE: {rmse:.2f} W m$^{{-2}}$ %$^{{-1}}$"


def plot_text(ax, text):
    ax.text(
        1.02,
        1,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )


def plot_north_contours(
    ax, data, lon, lat, title, vmin, vmax, step, cmap="RdBu_r", title_padding=None
):
    plot_data, plot_lon = cyclic_data(data, lon)
    im = ax.contourf(
        plot_lon,
        lat,
        plot_data,
        levels=np.arange(vmin, vmax + step, step),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines()
    ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"rotation": 0}
    gl.ylabel_style = {"rotation": 0}
    ax.set_title(title, loc="left", pad=title_padding)
    return im


def plot_kernel_difference(fields, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(
        3,
        3,
        figsize=(12, 12),
        subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=0)},
        layout="constrained",
        dpi=100,
    )
    fig.get_layout_engine().set(rect=[0, 0, 0.92, 1])

    kernel_vmin, kernel_vmax, kernel_step = -2, 0, 0.1
    diff_vmin, diff_vmax, diff_step = -0.6, 0.6, 0.05
    base_date = fields["base_date"]
    perturbed_date = fields["perturbed_date"]
    lon = fields["lon"]
    lat = fields["lat"]

    text = [
        [
            mean_text(fields["rrtm_perturbed"], lon, lat),
            mean_text(fields["rrtm_base"], lon, lat),
            mean_text(fields["rrtm_diff"], lon, lat),
        ],
        [
            mean_text(fields["nn_perturbed"], lon, lat),
            mean_text(fields["nn_base"], lon, lat),
            mean_text(fields["nn_diff"], lon, lat),
        ],
        [None, None, residual_text(fields["bias"], lon, lat)],
    ]
    ims = [
        plot_north_contours(
            ax[0][0],
            fields["rrtm_perturbed"],
            lon,
            lat,
            rf"$\bf{{(a)}}$ RRTM {perturbed_date}",
            kernel_vmin,
            kernel_vmax,
            kernel_step,
            "Blues_r",
        ),
        plot_north_contours(
            ax[0][1],
            fields["rrtm_base"],
            lon,
            lat,
            rf"$\bf{{(b)}}$ RRTM {base_date}",
            kernel_vmin,
            kernel_vmax,
            kernel_step,
            "Blues_r",
        ),
        plot_north_contours(
            ax[0][2],
            fields["rrtm_diff"],
            lon,
            lat,
            r"$\bf{{(c)}}$ (a) - (b)",
            diff_vmin,
            diff_vmax,
            diff_step,
        ),
        plot_north_contours(
            ax[1][0],
            fields["nn_perturbed"],
            lon,
            lat,
            rf"$\bf{{(d)}}$ NN {perturbed_date}",
            kernel_vmin,
            kernel_vmax,
            kernel_step,
            "Blues_r",
        ),
        plot_north_contours(
            ax[1][1],
            fields["nn_base"],
            lon,
            lat,
            rf"$\bf{{(e)}}$ NN {base_date}",
            kernel_vmin,
            kernel_vmax,
            kernel_step,
            "Blues_r",
        ),
        plot_north_contours(
            ax[1][2],
            fields["nn_diff"],
            lon,
            lat,
            r"$\bf{{(f)}}$ (d) - (e)",
            diff_vmin,
            diff_vmax,
            diff_step,
        ),
    ]
    ax[2][0].set_axis_off()
    ax[2][1].set_axis_off()
    ims.append(
        plot_north_contours(
            ax[2][2],
            fields["bias"],
            lon,
            lat,
            r"$\bf{{(g)}}$ (f) - (c)",
            diff_vmin,
            diff_vmax,
            diff_step,
            title_padding=20,
        )
    )

    for i, row in enumerate(ax):
        for j, axis in enumerate(row):
            if text[i][j] is not None:
                plot_text(axis, text[i][j])

    fig.canvas.draw()
    left0 = ax[0][0].get_position()
    left1 = ax[1][0].get_position()
    right0 = ax[0][1].get_position()
    right1 = ax[1][1].get_position()
    x0 = min(left0.x0, left1.x0)
    x1 = max(right0.x1, right1.x1)
    y0 = min(left0.y0, left1.y0) - 0.055
    kernel_cax = fig.add_axes([x0, y0, x1 - x0, 0.018])
    kernel_cax.set_in_layout(False)
    fig.colorbar(
        ims[1], cax=kernel_cax, orientation="horizontal", label="W m$^{-2}$ %$^{-1}$"
    )

    top = ax[0][2].get_position()
    mid = ax[1][2].get_position()
    bottom = ax[2][2].get_position()
    y0 = min(top.y0, mid.y0, bottom.y0)
    y1 = max(top.y1, mid.y1, bottom.y1)
    diff_cax = fig.add_axes([top.x1 + 0.012, y0, 0.018, y1 - y0])
    diff_cax.set_in_layout(False)
    fig.colorbar(ims[-1], cax=diff_cax, label="W m$^{-2}$ %$^{-1}$")

    fig.set_layout_engine("none")
    fig.savefig(output_dir / "kernel_difference.png")
    plt.close(fig)


def plot_hybrid_kernel_difference(fields, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=0)},
        layout="constrained",
    )

    im_fal = plot_north_contours(
        ax[0][0],
        fields["delta_fal"],
        fields["delta_fal"].longitude,
        fields["delta_fal"].latitude,
        r"$\bf{{(a)}}$ $\Delta$ albedo",
        -0.8,
        0.8,
        0.1,
    )
    im_nn_fal = plot_north_contours(
        ax[0][1],
        fields["hybrid_albedo"],
        fields["lon"],
        fields["lat"],
        r"$\bf{{(b)}}$ $\Delta K_a$",
        -0.6,
        0.6,
        0.05,
    )
    im_ecod = plot_north_contours(
        ax[1][0],
        fields["delta_ecod"],
        fields["delta_ecod"].longitude,
        fields["delta_ecod"].latitude,
        r"$\bf{{(c)}}$ $\Delta$ ecod",
        -1,
        1,
        0.1,
    )
    plot_north_contours(
        ax[1][1],
        fields["hybrid_cloud"],
        fields["lon"],
        fields["lat"],
        r"$\bf{{(d)}}$ $\Delta K_c$",
        -0.6,
        0.6,
        0.05,
    )

    cbar_kwargs = {
        "orientation": "vertical",
        "location": "right",
        "fraction": 0.05,
        "pad": 0.04,
        "aspect": 25,
    }
    fig.colorbar(im_fal, ax=ax[0][0], label="", **cbar_kwargs)
    fig.colorbar(im_ecod, ax=ax[1][0], label="", **cbar_kwargs)
    fig.colorbar(
        im_nn_fal, ax=[ax[0][1], ax[1][1]], label="W m$^{-2}$ %$^{-1}$", **cbar_kwargs
    )
    fig.savefig(output_dir / "hybrid_kernel_difference.png")
    plt.close(fig)
