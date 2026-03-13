"""
Created on October 2022
Author: Arne Thomsen

Tools to handle the scale cuts, kaiser-squires transformation and multiplicative and additive shear biases.
"""

import numpy as np

from msfm.utils import files, logger, scales, imports

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

    import tensorflow_probability as tfp

    m_bias_dist = tfp.distributions.MultivariateNormalDiag(
        loc=conf["survey"]["metacal"]["shear_bias"]["multiplicative"]["mu"],
        scale_diag=conf["survey"]["metacal"]["shear_bias"]["multiplicative"]["sigma"],
    )

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
