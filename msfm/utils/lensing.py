"""
Created on October 2022
Author: Arne Thomsen

Tools to handle the scale cuts, kaiser-squires transformation and multiplicative and additive shear biases.
"""

import numpy as np

from msfm.utils import files, logger, scales, imports
from numba import njit, prange


hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


def get_kaiser_squires_factors(l_max):
    """Factors for a spherical Kaiser Squires transformation
    from eq. (11) in https://academic.oup.com/mnras/article/505/3/4626/6287258
    """
    l = hp.Alm.getlm(l_max)[0]

    kappa2gamma_fac = np.where(
        np.logical_and(l != 1, l != 0),
        -np.sqrt(((l + 2.0) * (l - 1)) / ((l + 1) * l)),
        0,
    )
    gamma2kappa_fac = np.where(
        np.logical_and(l != 1, l != 0),
        1 / kappa2gamma_fac,
        0,
    )
    l_mask_fac = np.where(np.logical_and(l != 1, l != 0), 1.0, 0.0)

    return kappa2gamma_fac, gamma2kappa_fac, l_mask_fac


def get_m_bias_distribution(conf=None):
    """Return a SciPy distribution from which the (shear) multiplicative bias can be sampled.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        scipy.stats._multivariate.multivariate_normal_frozen: Multiplicative bias distribution.
    """
    conf = files.load_config(conf)

    from scipy.stats import multivariate_normal
    m_bias_dist = multivariate_normal(
        mean=conf["survey"]["WL"]["shear_bias"]["multiplicative"]["mu"],
        cov=np.diag(conf["survey"]["WL"]["shear_bias"]["multiplicative"]["sigma"])**2,
    )

    m_bias_dist.sample = m_bias_dist.rvs # compatibility alias for callers expecting a sample method

    return m_bias_dist


def mode_removal(
    gamma1_patch,
    gamma2_patch,
    gamma2kappa_fac,
    n_side,
    hp_datapath=None,
    # deprecated
    apply_smoothing=False,
    l_min=None,
    l_max=None,
    make_grf=False,
    np_seed=None,
):
    """Takes in survey patches of gamma maps and puts out survey patches of kappa maps that only contain E-modes

    Args:
        gamma1_patch (np.ndarray): Array of size n_pix, but only the survey patch is populated
        gamma2_patch (np.ndarray): Same
        gamma2kappa_fac (np.ndarray): Kaiser squires conversion factors
        n_side (int): Resolution of the map
        apply_smoothing (bool, optional): Whether to apply smoothing to the kappa map. This is included here because
            the alm coefficients are already computed anyways for the mode removal. Defaults to False.
        l_min (int, optional): Minimal ell, this removes the large scales if smoothing is applied. Defaults to None.
        l_max (int, optional): Maximal ell, this smoothes the small scales if smoothing is applied. Defaults to None.
        make_grf (bool, optional): Whether to degrade the map to a Gaussian random field instead of a smoothed map.
            Defaults to False.
        hp_datapath (str, optional): Path to a healpy pixel weights file. Defaults to None.

    Returns:
        np.ndarray: Array of size n_pix, but only the survey patch is populated
    """
    # gamma: map -> alm
    _, gamma_alm_E, gamma_alm_B = hp.map2alm(
        [np.zeros_like(gamma1_patch), gamma1_patch, gamma2_patch],
        pol=True,
        use_pixel_weights=True,
        datapath=hp_datapath,
    )
    # gamma -> kappa
    kappa_alm = gamma_alm_E * gamma2kappa_fac

    # kappa: alm -> map
    if apply_smoothing:
        LOGGER.warning(f"Double check what you're doing, smoothing within the mode removal has been deprecated")
        if make_grf:
            kappa_patch = scales.alm_to_grf_map(kappa_alm, l_min, l_max, n_side, np_seed)
        else:
            kappa_patch = scales.alm_to_smoothed_map(kappa_alm, n_side, l_min, l_max, nest=False)
    else:
        kappa_patch = hp.alm2map(kappa_alm, n_side, pol=False).astype(np.float32)

    return kappa_patch


def _as_torch_generator(rng=None, seed=None):
    """Create or reuse a CPU PyTorch generator for deterministic random draws."""
    import torch

    if isinstance(rng, torch.Generator):
        return rng

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    elif isinstance(rng, np.random.Generator):
        generator.manual_seed(int(rng.integers(0, np.iinfo(np.int64).max)))
    else:
        generator.seed()
    return generator


def _safe_divide_torch(numerator, denominator):
    """Divide tensors only where the denominator is nonzero, returning zero otherwise."""
    import torch

    output = torch.zeros_like(numerator)
    mask = denominator != 0
    return torch.where(mask, numerator / torch.where(mask, denominator, torch.ones_like(denominator)), output)


def noise_gen(counts, gamma_abs, weights, n_noise_per_signal, rng=None, seed=None):
    """Generates shape noise from galaxy counts and empirical catalog values.

    Args:
        counts (np.ndarray): Array of shape ``(len(base_patch_pix),)`` that contains the galaxy count per pixel.
        gamma_abs (np.ndarray): Absolute shear ``|e|`` samples from the catalog.
        weights (np.ndarray): Catalog weights corresponding to ``gamma_abs``.
        n_noise_per_signal (int): Number of noise realizations to create; this dimension is included for vectorization.
        rng (torch.Generator or np.random.Generator, optional): Random generator used for deterministic sampling.
        seed (int, optional): Seed used when ``rng`` is not a PyTorch generator.

    Returns:
        np.ndarray: Arrays of shape ``(len(base_patch_pix), n_noise_per_signal)`` containing the two gamma components.
    """

    import torch

    counts = np.asarray(counts, dtype=np.int64)
    n_pix_patch = counts.shape[0]
    n_gals_patch = int(counts.sum())

    if n_gals_patch == 0:
        zeros = np.zeros((n_pix_patch, n_noise_per_signal), dtype=np.float32)
        return zeros.copy(), zeros.copy()

    generator = _as_torch_generator(rng=rng, seed=seed)

    seg_ids = torch.as_tensor(np.repeat(np.arange(n_pix_patch, dtype=np.int64), counts), dtype=torch.long)
    gamma_abs_t = torch.as_tensor(np.asarray(gamma_abs), dtype=torch.float32)
    weights_t = torch.as_tensor(np.asarray(weights), dtype=torch.float32)

    cat_idx = torch.randint(
        low=0,
        high=gamma_abs_t.numel(),
        size=(n_gals_patch, n_noise_per_signal),
        generator=generator,
        dtype=torch.long,
    )
    phase_samples = torch.rand((n_gals_patch, n_noise_per_signal), generator=generator, dtype=torch.float32) * (2 * np.pi)

    gamma_samples = gamma_abs_t[cat_idx]
    w_samples = weights_t[cat_idx]
    sum_per_pix = torch.zeros((n_pix_patch, n_noise_per_signal, 3), dtype=torch.float32)
    weighted_gamma_samples = torch.stack(
        [torch.cos(phase_samples) * gamma_samples * w_samples,
         torch.sin(phase_samples) * gamma_samples * w_samples,
         w_samples],
        dim=-1,
    )
    sum_per_pix.index_add_(0, seg_ids, weighted_gamma_samples)

    denom = sum_per_pix[..., 2:3]
    gamma_per_pix = _safe_divide_torch(sum_per_pix[..., :2], denom)

    return gamma_per_pix[..., 0].numpy(), gamma_per_pix[..., 1].numpy()


def noise_gen_in_place(gamma_abs, w, pix, base_patch_pix, n_pix, n_noise_per_signal, rng=None, seed=None):
    """Generates shape noise by rotating galaxies from the catalog in-place.

    Args:
        gamma_abs (np.ndarray): Absolute shear |e| for each catalog galaxy.
        w (np.ndarray): Weight for each catalog galaxy.
        pix (np.ndarray): Pixel index for each catalog galaxy in the full sky map.
        base_patch_pix (np.ndarray): The pixels that make up the current footprint cutout.
        n_pix (int): Total number of pixels in the healpy map.
        n_noise_per_signal (int): Number of noise realizations.
        rng (torch.Generator or np.random.Generator, optional): Random generator used for deterministic sampling.
        seed (int, optional): Seed used when ``rng`` is not a PyTorch generator.

    Returns:
        np.ndarray: Arrays of shape ``(len(base_patch_pix), n_noise_per_signal)`` containing the two gamma components.
    """
    import torch

    generator = _as_torch_generator(rng=rng, seed=seed)

    gamma_abs_t = torch.as_tensor(np.asarray(gamma_abs), dtype=torch.float32)
    w_t = torch.as_tensor(np.asarray(w), dtype=torch.float32)
    pix_t = torch.as_tensor(np.asarray(pix), dtype=torch.long)
    base_patch_pix_t = torch.as_tensor(np.asarray(base_patch_pix), dtype=torch.long)

    n_gals = gamma_abs_t.numel()
    phase_samples = torch.rand((n_gals, n_noise_per_signal), generator=generator, dtype=torch.float32) * (2 * np.pi)

    w_samples = w_t[:, None]
    weighted_g1 = torch.cos(phase_samples) * gamma_abs_t[:, None] * w_samples
    weighted_g2 = torch.sin(phase_samples) * gamma_abs_t[:, None] * w_samples
    w_expanded = w_samples.expand(-1, n_noise_per_signal)

    sum_g1 = torch.zeros((n_pix, n_noise_per_signal), dtype=torch.float32)
    sum_g2 = torch.zeros((n_pix, n_noise_per_signal), dtype=torch.float32)
    sum_w = torch.zeros((n_pix, n_noise_per_signal), dtype=torch.float32)
    sum_g1.index_add_(0, pix_t, weighted_g1)
    sum_g2.index_add_(0, pix_t, weighted_g2)
    sum_w.index_add_(0, pix_t, w_expanded)

    gamma1_per_pix = _safe_divide_torch(sum_g1, sum_w)
    gamma2_per_pix = _safe_divide_torch(sum_g2, sum_w)

    return gamma1_per_pix[base_patch_pix_t].numpy(), gamma2_per_pix[base_patch_pix_t].numpy()


def noise_gen_numba(counts, gamma_abs, weights, n_noise_per_signal, rng):
    """
    Python wrapper: normalizes input types before calling the JIT function.
    """

    counts = np.ascontiguousarray(counts, dtype=np.int64)
    gamma_abs = np.ascontiguousarray(gamma_abs, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)

    # return noise_gen_numba_impl(counts, gamma_abs, weights, int(n_noise_per_signal))
    return noise_gen_numba_parallel(counts, gamma_abs, weights, int(n_noise_per_signal), rng)

@njit(cache=True)
def noise_gen_numba_impl(counts, gamma_abs, weights, n_noise_per_signal):
    """
    JIT-compiled implementation.

    Samples the empirical catalog WITH replacement.
    """
    n_pix = len(counts)
    n_cat = len(gamma_abs)

    gamma1_out = np.zeros((n_pix, n_noise_per_signal), dtype=np.float32)
    gamma2_out = np.zeros((n_pix, n_noise_per_signal), dtype=np.float32)

    w_sum = np.empty(n_noise_per_signal, dtype=np.float32)

    two_pi = np.float32(2.0 * np.pi)

    for pix in range(n_pix):
        n_gals = counts[pix]

        if n_gals <= 0:
            continue

        for j in range(n_noise_per_signal):
            gamma1_out[pix, j] = 0.0
            gamma2_out[pix, j] = 0.0
            w_sum[j] = 0.0

        for _ in range(n_gals):
            for j in range(n_noise_per_signal):
                # Empirical sampling with replacement
                cat_idx = np.random.randint(0, n_cat)

                gamma = gamma_abs[cat_idx]
                w = weights[cat_idx]

                phase = two_pi * np.float32(np.random.random())

                gamma1_out[pix, j] += np.cos(phase) * gamma * w
                gamma2_out[pix, j] += np.sin(phase) * gamma * w
                w_sum[j] += w

        for j in range(n_noise_per_signal):
            if w_sum[j] != 0.0:
                gamma1_out[pix, j] /= w_sum[j]
                gamma2_out[pix, j] /= w_sum[j]
            else:
                gamma1_out[pix, j] = 0.0
                gamma2_out[pix, j] = 0.0

    return gamma1_out, gamma2_out


@njit(parallel=True, cache=True)
def noise_gen_numba_parallel(counts, gamma_abs, weights, n_noise_per_signal, rng):
    """
    Parallel NumPy/Numba version.

    Parallelization is over pixels. Each thread handles independent pixels,
    so there are no shared-output race conditions.

    Sampling from the empirical catalog is WITH replacement.
    """
    n_pix = counts.shape[0]
    n_cat = gamma_abs.shape[0]

    g1_out = np.zeros((n_pix, n_noise_per_signal), dtype=np.float32)
    g2_out = np.zeros((n_pix, n_noise_per_signal), dtype=np.float32)

    two_pi = 2.0 * np.pi

    for pix in prange(n_pix):
        n_gals = counts[pix]

        if n_gals == 0:
            continue

        # Thread-local because it is allocated inside the prange loop.
        w_sum = np.zeros(n_noise_per_signal, dtype=np.float64)

        for _ in range(n_gals):
            for j in range(n_noise_per_signal):
                idx = rng.integers(0, n_cat)

                gamma = gamma_abs[idx]
                w = weights[idx]

                phase = two_pi * rng.random()

                g1_out[pix, j] += np.float32(np.cos(phase) * gamma * w)
                g2_out[pix, j] += np.float32(np.sin(phase) * gamma * w)
                w_sum[j] += w

        for j in range(n_noise_per_signal):
            if w_sum[j] != 0.0:
                inv_w = 1.0 / w_sum[j]
                g1_out[pix, j] = np.float32(g1_out[pix, j] * inv_w)
                g2_out[pix, j] = np.float32(g2_out[pix, j] * inv_w)
            else:
                g1_out[pix, j] = 0.0
                g2_out[pix, j] = 0.0

    return g1_out, g2_out


@njit(cache=True)
def _build_patch_lookup(base_patch_pix, n_pix):
    """
    Build a lookup table from full-sky pixel id -> patch row index.

    Pixels not in the patch get -1.
    """
    patch_lookup = np.full(n_pix, -1, dtype=np.int64)

    for i in range(base_patch_pix.shape[0]):
        patch_lookup[base_patch_pix[i]] = i

    return patch_lookup

@njit(parallel=True, cache=True)
def noise_gen_in_place_numba_parallel_core(
    gamma_abs,
    weights,
    pix,
    patch_lookup,
    n_noise_per_signal,
    rng,
):
    """
    Numba core for in-place-style noise generation.

    This computes only the requested patch pixels, not the full n_pix map.

    Args:
        gamma_abs: float array, shape (n_gals,)
        weights: float array, shape (n_gals,)
        pix: int array, shape (n_gals,)
            Full-sky pixel index for each catalog galaxy.
        patch_lookup: int array, shape (n_pix,)
            Maps full-sky pixel id to patch row index, or -1 if outside patch.
        n_noise_per_signal: int

    Returns:
        gamma1_patch, gamma2_patch:
            float32 arrays, shape (len(base_patch_pix), n_noise_per_signal)
    """
    n_gals = gamma_abs.shape[0]
    n_patch = 0

    # Infer patch length from lookup.
    # This assumes patch_lookup contains exactly 0, ..., n_patch - 1.
    for i in range(patch_lookup.shape[0]):
        if patch_lookup[i] + 1 > n_patch:
            n_patch = patch_lookup[i] + 1

    gamma1_patch = np.zeros((n_patch, n_noise_per_signal), dtype=np.float32)
    gamma2_patch = np.zeros((n_patch, n_noise_per_signal), dtype=np.float32)

    two_pi = 2.0 * np.pi

    # Parallelize over noise realization columns.
    # Each thread owns one column j, so no two threads write the same output cell.
    for j in prange(n_noise_per_signal):
        # Temporary weight sum for this one noise realization.
        # Shape is only (n_patch,), not (n_pix,) or (n_gals, n_noise).
        sum_w = np.zeros(n_patch, dtype=np.float64)

        for gal in range(n_gals):
            patch_idx = patch_lookup[pix[gal]]

            # Skip galaxies outside the requested patch.
            if patch_idx < 0:
                continue

            gamma = gamma_abs[gal]
            w = weights[gal]

            phase = two_pi * rng.random()

            gamma1_patch[patch_idx, j] += np.float32(np.cos(phase) * gamma * w)
            gamma2_patch[patch_idx, j] += np.float32(np.sin(phase) * gamma * w)
            sum_w[patch_idx] += w

        for p in range(n_patch):
            if sum_w[p] != 0.0:
                inv_w = 1.0 / sum_w[p]
                gamma1_patch[p, j] = np.float32(gamma1_patch[p, j] * inv_w)
                gamma2_patch[p, j] = np.float32(gamma2_patch[p, j] * inv_w)
            else:
                gamma1_patch[p, j] = 0.0
                gamma2_patch[p, j] = 0.0

    return gamma1_patch, gamma2_patch


def noise_gen_in_place_numba(
    gamma_abs,
    w,
    pix,
    base_patch_pix,
    n_pix,
    n_noise_per_signal,
    rng,
):

    LOGGER.warning("noise_gen_in_place_numba_parallel: This code is not tested yet")

    gamma_abs = np.asarray(gamma_abs, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pix = np.asarray(pix, dtype=np.int64)
    base_patch_pix = np.asarray(base_patch_pix, dtype=np.int64)

    patch_lookup = _build_patch_lookup(base_patch_pix, n_pix)

    return noise_gen_in_place_numba_parallel_core(
        gamma_abs,
        w,
        pix,
        patch_lookup,
        n_noise_per_signal,
        rng,
    )

def kappa_to_gamma(kappa_full_sky, hp_datapath, kappa2gamma_fac, n_side):

    # kappa -> gamma (full sky)
    kappa_alm = hp.map2alm(
        kappa_full_sky,
        use_pixel_weights=True,
        datapath=hp_datapath,
    )

    gamma_alm = kappa_alm * kappa2gamma_fac
    dummy_alm = np.zeros_like(gamma_alm)
    _, gamma1_full, gamma2_full = hp.alm2map(
        [dummy_alm, gamma_alm, dummy_alm], nside=n_side
    )

    return gamma1_full, gamma2_full