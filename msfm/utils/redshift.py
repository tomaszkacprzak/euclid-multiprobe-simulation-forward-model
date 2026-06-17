# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created April 2023
Author: Arne Thomsen
"""

import numpy as np

from msfm.utils import files


def get_tomo_amplitudes(
    amplitude, exponent, tomo_z, tomo_nz, z0, truncate_nz=True, z_min_quantile=0.05, z_max_quantile=0.95
):
    """Parametrization of an effective per redshift bin evolution like in equastions (8) and (9) in DeepLSS
    https://arxiv.org/pdf/2203.09616.pdf. This is suitable both for the intrinsic alignment amplitude and the linear
    galaxy bias.

    Args:
        amplitude (float): Overall amplitude parameter.
        exponent (float): Parameter that determines the strenght of the redshift dependence.
        tomo_z (list): Per redshift bin z value of the distribution.
        tomo_nz (list): Per redshift bin n(z) value of the distribution.
        z0 (float): Pivot redshift.

    Returns:
        list: Per redshift bin amplitude.
    """

    tomo_amplitudes = []
    for z, nz in zip(tomo_z, tomo_nz):
        if truncate_nz:
            cdf = np.cumsum(nz / np.sum(nz))
            z_min = np.interp(z_min_quantile, cdf, z)
            z_max = np.interp(z_max_quantile, cdf, z)
            nz = np.where((z >= z_min) & (z <= z_max), nz, 0.0)

        integrand = nz * ((1 + z) / (1 + z0)) ** exponent
        integral = np.sum(integrand) / np.sum(nz)

        tomo_amplitudes.append(amplitude * integral)

    return np.array(tomo_amplitudes).astype(np.float32)


def get_tomo_amplitudes_according_to_config(conf, amplitude, exponent, sample="metacal"):
    tomo_z, tomo_nz = files.load_redshift_distributions(sample, conf)

    if sample == "metacal":
        truncate_nz = conf["analysis"]["modelling"]["lensing"]["nla"]["truncate_nz"]
        z_min_quantile = conf["analysis"]["modelling"]["lensing"]["nla"]["z_min_quantile"]
        z_max_quantile = conf["analysis"]["modelling"]["lensing"]["nla"]["z_max_quantile"]
    else:
        truncate_nz = False
        z_min_quantile = 0.0
        z_max_quantile = 1.0

    return get_tomo_amplitudes(
        amplitude,
        exponent,
        tomo_z,
        tomo_nz,
        z0=conf["survey"][sample]["z0"],
        truncate_nz=truncate_nz,
        z_min_quantile=z_min_quantile,
        z_max_quantile=z_max_quantile,
    )


def get_tomo_amplitudes_vectorized(
    amplitude,
    exponent,
    tomo_z,
    tomo_nz,
    z0,
    truncate_nz: bool = True,
    z_min_quantile: float = 0.05,
    z_max_quantile: float = 0.95,
):
    """Vectorized version of :func:`get_tomo_amplitudes`.

    Provides identical functionality (including optional truncation of the n(z) by quantiles) while
    supporting batch evaluation over multiple (amplitude, exponent) parameter pairs.

    Parameters
    ----------
    amplitude : float or array-like, shape (B,)
        Overall amplitude parameter(s). If scalar and ``exponent`` is an array it is broadcast; vice versa.
    exponent : float or array-like, shape (B,)
        Redshift evolution exponent parameter(s).
    tomo_z : sequence of arrays
        Per tomographic bin redshift grid arrays (length = n_bins).
    tomo_nz : sequence of arrays
        Per tomographic bin n(z) arrays matching ``tomo_z`` shapes.
    z0 : float
        Pivot redshift.
    truncate_nz : bool, default False
        If True, zero out parts of each n(z) outside the quantile interval before integration.
    z_min_quantile : float, default 0.05
        Lower CDF quantile (fraction, not percent) used when ``truncate_nz`` is True.
    z_max_quantile : float, default 0.95
        Upper CDF quantile (fraction, not percent) used when ``truncate_nz`` is True.

    Returns
    -------
    np.ndarray
        Shape (n_bins,) if a single (amplitude, exponent) pair is supplied, else (B, n_bins).
    """

    # Convert tomographic lists to arrays (n_bins, n_z)
    tomo_z_array = np.asarray(tomo_z, dtype=float)
    tomo_nz_array = np.asarray(tomo_nz, dtype=float)

    # Optional truncation per bin replicating the scalar implementation
    if truncate_nz and not (z_min_quantile <= 0.0 and z_max_quantile >= 1.0):
        truncated = []
        for z, nz in zip(tomo_z_array, tomo_nz_array):
            cdf = np.cumsum(nz / np.sum(nz))
            z_min = np.interp(z_min_quantile, cdf, z)
            z_max = np.interp(z_max_quantile, cdf, z)
            truncated.append(np.where((z >= z_min) & (z <= z_max), nz, 0.0))
        tomo_nz_array = np.stack(truncated, axis=0)

    # Prepare parameter arrays with broadcasting rules similar to numpy ufuncs
    amp_arr = np.atleast_1d(amplitude).astype(float)
    exp_arr = np.atleast_1d(exponent).astype(float)

    if amp_arr.size == 1 and exp_arr.size > 1:
        amp_arr = np.full(exp_arr.shape, amp_arr[0])
    if exp_arr.size == 1 and amp_arr.size > 1:
        exp_arr = np.full(amp_arr.shape, exp_arr[0])
    if amp_arr.size != exp_arr.size:
        raise ValueError(
            "amplitude and exponent must be broadcastable to the same length (or be scalars); got %d vs %d"
            % (amp_arr.size, exp_arr.size)
        )

    batch = amp_arr.size
    # Add batch axis for computation: (B, n_bins, n_z)
    z_grid = tomo_z_array[None, :, :]
    nz_grid = tomo_nz_array[None, :, :]

    # Compute integrals per batch item
    exp_grid = exp_arr[:, None, None]
    integrand = nz_grid * ((1.0 + z_grid) / (1.0 + z0)) ** exp_grid
    integrals = np.sum(integrand, axis=2) / np.sum(nz_grid, axis=2)
    amplitudes = amp_arr[:, None] * integrals  # (B, n_bins)

    result = amplitudes.astype(np.float32)
    if batch == 1:
        return result[0]
    return result


def get_tomo_amplitudes_according_to_config_vectorized(
    conf,
    amplitude,
    exponent,
    sample: str = "WL",
    truncate_nz: bool = True,
    z_min_quantile: float = 0.05,
    z_max_quantile: float = 0.95,
):
    tomo_z, tomo_nz = files.load_redshift_distributions(sample, conf)
    z0 = conf["survey"][sample]["z0"]
    return get_tomo_amplitudes_vectorized(
        amplitude,
        exponent,
        tomo_z,
        tomo_nz,
        z0,
        truncate_nz=truncate_nz,
        z_min_quantile=z_min_quantile,
        z_max_quantile=z_max_quantile,
    )
