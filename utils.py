def compute_cloud_optical_depth(
    tclw,  # total column liquid water
    tciw,  # total column ice water
    tcc,  # total cloud cover (0–1)
    re_liquid=10e-6,  # liquid effective radius (m)
    re_ice=30e-6,  # ice effective radius (m)
    rho_water=1000.0,  # water density (kg/m^3)
    rho_ice=917.0,  # ice density (kg/m^3)
):

    tau_l = 1.5 * tclw / (rho_water * re_liquid)
    tau_i = 1.5 * tciw / (rho_ice * re_ice)

    tau_total = tau_l + tau_i
    tau_effective = tau_total * tcc

    return tau_effective
