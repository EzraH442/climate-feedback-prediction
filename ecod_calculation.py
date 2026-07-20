import numpy as np
import xarray as xr

# Slingo (1989) Table 1 coefficients, liquid water, band 9 (0.57-0.64 um)
SLINGO_a = 2.381-2  # m^2 g^-1
SLINGO_b = 1.317  # um m^2 g^-1
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
    return (8.0 / (3.0 * np.sqrt(3.0))) * r_e_um # 46.1880215352 x 10^-6


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
    #print("c1:", (SLINGO_a + SLINGO_b / liquid_re))

    """
    Fu (1996) Eq. 3.9a:

        \beta = IWC * (a0 + a1 / D_ge)

    """
    tau_ice = ice_water_path * (FU_A0 + FU_A1 / Dge_from_ice_re(ice_re))
    #print("c2:", (FU_A0 + FU_A1 / Dge_from_ice_re(ice_re)))

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
