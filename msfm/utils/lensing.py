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
    """Return a tensorflow probability distribution from which the (shear) multiplicative bias can be sampled.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        tfp.distribution: Multiplicative bias.s
    """
    conf = files.load_config(conf)

    # import tensorflow_probability as tfp

    # m_bias_dist = tfp.distributions.MultivariateNormalDiag(
    #     loc=conf["survey"]["WL"]["shear_bias"]["multiplicative"]["mu"],
    #     scale_diag=conf["survey"]["WL"]["shear_bias"]["multiplicative"]["sigma"],
    # )
    from scipy.stats import multivariate_normal
    m_bias_dist = multivariate_normal(
        mean=conf["survey"]["WL"]["shear_bias"]["multiplicative"]["mu"],
        cov=np.diag(conf["survey"]["WL"]["shear_bias"]["multiplicative"]["sigma"])**2,
    )

    m_bias_dist.sample = m_bias_dist.rvs # for compatibility with tensorflow_probability

    return m_bias_dist

def get_Emode(method, gamma1_patch, gamma2_patch, gamma2kappa_fac, n_side, hp_datapath):

    if method == "mode_removal":

        kappa_patch = mode_removal(gamma1_patch, gamma2_patch, gamma2kappa_fac, n_side,
                                   apply_smoothing=False,
                                   hp_datapath=hp_datapath
                                   )
    elif method == "real_space":

        raise NotImplementedError("Real space mode removal is not implemented yet")


    return kappa_patch


        

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


# making this a tf.function doesn't speed things up because the seg_ids are always different
def noise_gen(counts, cat_dist, n_noise_per_signal):
    """Generates shape noise from a map of galaxy counts and joint distribution of absolute shear values and their
    weights.

    Args:
        counts (np.ndarray): Array of shape (len(base_patch_pix),) that contains the galaxy count per pixel
        cat_dist (tfp.distributions): Distribution with samples of length 2 that contains the absolute magnitudes and
            weights
        n_noise_per_signal (int): Number of noise realizations to create, this dimension is included for vectorization

    Returns:
        np.ndarray: Arrays of shape (len(base_patch_pix, n_noise_per_signal) containing the two gamma components
    """

    import tensorflow as tf

    # indices to sum over all of the galaxies in the individual pixels
    seg_ids = []
    for id, n_gals in enumerate(counts):
        seg_ids.extend(n_gals * [id])

    # make a tensor, this is important for performance
    seg_ids = tf.constant(seg_ids, dtype=tf.int32)

    # total number of galaxies in the patch
    n_gals_patch = len(seg_ids)

    # shape (n_gals_patch, n_noise_per_signal, 2)
    cat_samples = cat_dist.sample(sample_shape=(n_gals_patch, n_noise_per_signal))
    # shape (n_gals_patch, n_noise_per_signal)
    phase_samples = tf.random.uniform(
        shape=(
            n_gals_patch,
            n_noise_per_signal,
        ),
        minval=0,
        maxval=2 * np.pi,
    )

    # shape (n_gals_patch, n_noise_per_signal)
    g1_samples = tf.math.cos(phase_samples) * cat_samples[..., 0]
    g2_samples = tf.math.sin(phase_samples) * cat_samples[..., 0]
    w_samples = cat_samples[..., 1]

    # shape (n_gals_patch, n_noise_per_signal, 3)
    weighted_gamma_samples = tf.stack([g1_samples * w_samples, g2_samples * w_samples, w_samples], axis=-1)

    # len(base_patch_pix), unless the final pixels of the patch don't contain galaxies. Then, it's smaller
    sum_per_pix = tf.math.segment_sum(weighted_gamma_samples, seg_ids)

    # normalize with weights, set 0/0 equal to 0 instead of nan
    gamma_per_pix = tf.math.divide_no_nan(sum_per_pix[..., :2], tf.expand_dims(sum_per_pix[..., 2], axis=-1))

    # The condition means that the final pixel contains zero galaxies. Then, its index is not included in the seg_ids
    # (multiplication with zero) and because it's the last, tensorflow has no way of knowing that it should still take
    # the segmented_sum over this index, which evaluates to zero. The while loop allows more than one of the last
    # pixels to be zero.
    n_final_zero_pix = 0
    while counts[-(n_final_zero_pix + 1)] == 0:
        n_final_zero_pix += 1

    if n_final_zero_pix > 0:
        # There is no galaxy in the final pixels, so the shape noise there is equal to zero
        zero_pix = tf.zeros((n_final_zero_pix, n_noise_per_signal, 2), dtype=tf.float32)
        gamma_per_pix = tf.concat((gamma_per_pix, zero_pix), axis=0)

    # shape (len(base_patch_pix), n_noise_per_signal)
    return gamma_per_pix[..., 0].numpy(), gamma_per_pix[..., 1].numpy()


def noise_gen_in_place(gamma_abs, w, pix, base_patch_pix, n_pix, n_noise_per_signal):
    """Generates shape noise by rotating galaxies from the catalog in-place.

    Args:
        gamma_abs (np.ndarray or tf.Tensor): Absolute shear |e| for each catalog galaxy
        w (np.ndarray or tf.Tensor): Weight for each catalog galaxy
        pix (np.ndarray or tf.Tensor): Pixel index for each catalog galaxy in the full sky map
        base_patch_pix (np.ndarray): The pixels that make up the current footprint cutout
        n_pix (int): Total number of pixels in the healpy map
        n_noise_per_signal (int): Number of noise realizations

    Returns:
        np.ndarray: Arrays of shape (len(base_patch_pix), n_noise_per_signal) containing the two gamma components
    """
    import tensorflow as tf

    # Place operations on CPU to avoid GPU OOM on shared login nodes where GPU memory is highly restricted
    with tf.device("/CPU:0"):
        pix = tf.cast(pix, tf.int32)
        n_gals = tf.shape(gamma_abs)[0]

        # shape (n_gals, n_noise_per_signal)
        phase_samples = tf.random.uniform(
            shape=(
                n_gals,
                n_noise_per_signal,
            ),
            minval=0,
            maxval=2 * np.pi,
        )

        g1_samples = tf.math.cos(phase_samples) * tf.expand_dims(gamma_abs, axis=1)
        g2_samples = tf.math.sin(phase_samples) * tf.expand_dims(gamma_abs, axis=1)
        w_samples = tf.expand_dims(w, axis=1)

        weighted_g1 = g1_samples * w_samples
        weighted_g2 = g2_samples * w_samples

        sum_g1 = tf.math.unsorted_segment_sum(weighted_g1, pix, num_segments=n_pix)
        sum_g2 = tf.math.unsorted_segment_sum(weighted_g2, pix, num_segments=n_pix)
        sum_w = tf.math.unsorted_segment_sum(w_samples, pix, num_segments=n_pix)

        gamma1_per_pix = tf.math.divide_no_nan(sum_g1, sum_w)
        gamma2_per_pix = tf.math.divide_no_nan(sum_g2, sum_w)

        gamma1_patch = tf.gather(gamma1_per_pix, base_patch_pix)
        gamma2_patch = tf.gather(gamma2_per_pix, base_patch_pix)

    return gamma1_patch.numpy(), gamma2_patch.numpy()


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


import numpy as np
from numba import njit, prange


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
    )