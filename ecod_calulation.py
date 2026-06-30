"""
compute_ecod_era5.py

Compute "effective cloud optical depth" (ECOD, tau_vis) from ERA5 monthly
data, following the liquid-water (Slingo 1989) and ice-water (Fu 1996)
parameterizations used by Zelinka et al. (2011) to translate cloud water
path into visible-band optical depth.

    tau_liq = LWP * (a_i + b_i / r_e)                      [Slingo 1989, Eq.1]
    tau_ice = IWP * (a0 + a1 / D_ge)                        [Fu 1996, Eq.3.9a]
    D_ge    = (8 / (3*sqrt(3))) * r_e                        [inverted Eq.3.12]

Band coefficients are taken for the "visible" band (0.44-0.48 um, band 6 in
Table 1 / band index 6 in Table 3a of Slingo/Fu, as referenced in the
Zelinka slides), matching the ~0.6 micron ISCCP convention.

ECOD is then defined as the liquid+ice optical depth, optionally weighted
by liquid/ice cloud fraction if you want a cloud-fraction-masked diagnostic
rather than an in-cloud value.

Two input modes are supported:

1. PROFILE MODE (preferred, physically correct):
   ERA5 model/pressure-level specific cloud liquid/ice water content
   (clwc, ciwc, kg/kg) + geopotential or pressure levels, integrated
   vertically (Eq. 4 of Slingo) to get LWP/IWP, then converted to tau.

2. COLUMN MODE (fallback, single-layer approximation):
   ERA5 single-level vertically-integrated cloud water variables
   (tclw, tciw, kg/m^2) treated as a single "overcast" layer, directly
   converted to LWP/IWP (just a unit conversion) and then to tau. This
   matches the bulk LWP -> tau relationship but loses any vertical
   structure information (CTP-resolved optical depth bins as in ISCCP).

Usage:
    python compute_ecod_era5.py --mode column --file era5_monthly.nc
    python compute_ecod_era5.py --mode profile --file era5_monthly_levels.nc
"""

import numpy as np
import xarray as xr

# Slingo (1989) Table 1 coefficients, liquid water, band 6 (0.44-0.48 um)
SLINGO_a = 2.698e-2  # m^2 g^-1
SLINGO_b = 1.315  # um m^2 g^-1
LIQUID_RE = 10.0  # assumed liquid effective radius (um), per Zelinka et al. (2012)

# Fu (1996) Table 3a coefficients, ice water, band 9 (0.57-0.64 um)
FU_A0 = 0.982244e-4  # um^-1
FU_A1 = 0.250875e1  # um m^2 g^-1
ICE_RE = 30.0  # assumed ice effective radius (um), per Zelinka et al. (2012)


def Dge_from_ice_re(r_e_um=ICE_RE):
    """
    Fu (1996) Eq. 3.12:

        r_e = (3*sqrt(3)/8) * D_ge

    """
    return (8.0 / (3.0 * np.sqrt(3.0))) * r_e_um


def ecod_from_water_path(
    liquid_water_path, ice_water_path, liquid_re=LIQUID_RE, ice_re=ICE_RE
):
    """
    Convert liquid/ice water path (g/m^2) to cloud optical depth (unitless)

    Parameters
    ----------
    liquid_water_path, ice_water_path : array-like
        Liquid / ice water path in g m^-2.
    liquid_re, ice_re : float
        Liquid / ice water effective radius (um).

    Returns
    -------
    ecod : arrays, same shape as input
    """

    """
    Slingo (1989) Eq. 1:

        \tau_i = LWP * (a_i + b_i / r_e)

    """
    tau_liq = liquid_water_path * (SLINGO_a + SLINGO_b / liquid_re)

    """
    Fu (1996) Eq. 3.9a:

        \beta = IWC * (a0 + a1 / D_ge)

    """
    tau_ice = ice_water_path * (FU_A0 + FU_A1 / Dge_from_ice_re(ice_re))
    return tau_liq + tau_ice


def water_path_from_water_content(wc, p, g=9.80665):
    """
    Convert specific cloud water content (kg/kg) to column water path (g/m^2).

    Parameters
    ----------
    wc : xr.DataArray
        Specific liquid or ice cloud water content
    p : xr.DataArray
        Pressure (Pa) on the same level_dim, monotonic.

    Returns
    -------
    xr.DataArray
        Water path in g/m^2 (column-integrated), with level_dim reduced.
    """
    p_sorted = p.sortby("level")
    wc_sorted = wc.sortby("level")

    dp = xr.apply_ufunc(
        np.gradient,
        p_sorted,
        input_core_dims=[["level"]],
        output_core_dims=[["level"]],
        vectorize=True,
    )
    dp = np.abs(dp)

    wp = (wc_sorted * dp / g).sum(dim="level") * 1000.0
    return wp


def ecod_from_profiles(ciwc, clwc, level, liquid_re=LIQUID_RE, ice_re=ICE_RE):
    """
    Compute ECOD from ERA5 monthly profile cloud liquid/ice water content
    (kg/kg) by vertically integrating to LWP/IWP (Eq. 4), then applying the Slingo/Fu tau parameterizations.

    Parameters
    ----------
    ciwc, clwc : xarray.DataArray
        Specific cloud ice/liquid water content (kg/kg), dims include level.
    level : xarray.DataArray
        Pressure (Pa) on the same level dim, monotonic.
    liquid_re, ice_re : float
        Assumed effective radii. (um)

    Returns
    -------
    xarray.DataArray
        ECOD (unitless), same shape as ciwc/clwc with level dim reduced.
    """
    lwp = water_path_from_water_content(clwc, level)
    iwp = water_path_from_water_content(ciwc, level)

    ecod = ecod_from_water_path(lwp, iwp, liquid_re=liquid_re, ice_re=ice_re)
    return ecod
