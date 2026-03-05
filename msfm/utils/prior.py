# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Based off https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/probability.py
by Janis Fluri
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import fsolve
from scipy.stats import norm

from msfm.utils import files, parameters, logger, parameters

LOGGER = logger.get_logger(__file__)


def in_grid_prior(cosmos, conf=None, params=None):
    """Determines whether the elements of the given array of cosmological parameters are contained within the analysis
    prior. This is needed to build a vectorized log posterior.

    Args:
        cosmos (np.ndarray): A 2D array of cosmological parameters with shape (n_cosmos, n_params), where n_params
            has to be in the right ordering (as defined in the config) and n_theta corresponds to n_cosmos.
        conf (str, dict, optional): Config to use, can be either a string to the config.yaml file, the dictionary
            obtained by reading such a file or None, where the default config within the repo is used. Defaults to
            None.
        params (list, optional): List of strings containing "Om", "s8", "Ob", "H0", "ns" and "w0" in the same order as
            within the cosmos array.

    Raises:
        ValueError: If an incompatible type is passed to the conf argument

    Returns:
        in_prior: A 1D boolean array of the shape (n_cosmos,) that specifies whether the values in params are
        contained within the prior.
    """
    conf = files.load_config(conf)
    params = parameters.get_parameters(params, conf)

    # make the params 2d
    cosmos = np.atleast_2d(cosmos)

    prior_intervals = parameters.get_prior_intervals(params, conf)

    # check if we are in the prior intervals
    in_prior = np.all(np.logical_and(prior_intervals[:, 0] <= cosmos, cosmos <= prior_intervals[:, 1]), axis=1)

    # simplex in the Om - s8 plane
    try:
        i_Om = params.index("Om")
        i_s8 = params.index("s8")
    except ValueError:
        LOGGER.debug(f"The hull prior is only checked when Om and s8 are included as parameters")
    else:
        hull = Delaunay(conf["analysis"]["grid"]["priors"]["Om_s8_border_points"])

        # check if we are in the hull, shape (n_cosmos,)
        in_hull = hull.find_simplex(cosmos[:, [i_Om, i_s8]]) >= 0

        # what is False will stay false irrespective of the rhs
        in_prior[in_prior] = in_hull[in_prior]

    # w0 threshold
    try:
        i_Om = params.index("Om")
        i_w0 = params.index("w0")
    except ValueError:
        LOGGER.debug(f"The w0 threshold is only checked if Om and w0 are included as parameters")
    else:
        # check if we are above the w0 threshold (same as get_min_w0 with margin = 0.01)
        in_prior[in_prior] = 1.0 / (cosmos[in_prior, i_Om] - 1.0) + 0.01 <= cosmos[in_prior, i_w0]

    return in_prior


def log_posterior(cosmos, log_probs, conf=None, params=None, gaussian_kwargs=None):
    """Vectorized version of the log posterior to be used in the MCMC runs, for example with emcee.

    Args:
        cosmos (np.ndarray): A 2D array of cosmological parameters with shape (n_cosmos, n_params), where n_params
            has to be in the right ordering (as defined in the config) and n_theta corresponds to n_cosmos.
        log_probs (np.ndarray): Log probabilities associated with the parameters of shape (n_cosmos, 1) or (n_cosmos,).
            These are output values of the Gaussian Process emulator for example.
        conf (str, dict, optional): Config to use, can be either a string to the config.yaml file, the dictionary
            obtained by reading such a file or None, where the default config within the repo is used. Defaults to
            None.

        Example usage:
            log_posterior = lambda X: prior.log_posterior(X, predictor(X), params=params, conf=conf)
            from emcee import EnsembleSampler
            sampler = EnsembleSampler(nwalkers, ndim, log_posterior, vectorize=True)

    Returns:
        np.ndarray: The log posterior values obtained by restricting the emulator's predictions to the prior range.
    """
    # - infinity if outside the pior range, given input probability otherwise
    log_post = np.where(in_grid_prior(cosmos, conf, params), np.squeeze(log_probs), -np.inf)

    if gaussian_kwargs is not None:
        log_post += gaussian_prior(cosmos, conf, params, **gaussian_kwargs)

    return log_post


def get_min_w0(Om, margin=0.01):
    """Calculates the minimum possible w0 value given an Om value. The minimum w0 value is calculated with a formula
    from the concept creator and ensures that the "w0 phantom crossing" occurs after z = 0.

    Args:
        Om (float): Omega matter value
        margin (float, optional): Margin to add to the minimum value. Defaults to 0.01.

    Returns:
        float: The minimum w0
    """
    f = lambda w: 1.0 - ((Om - 1.0) / Om * (1.0 + w)) ** (1.0 / (3.0 * w))
    w0 = fsolve(f, -1.05)[0]
    return w0 + margin


def gaussian_prior(cosmos, conf=None, params=None, std_fac=0.01, params_unaffected=["Om", "s8"]):
    """For debugging purposes only! This has no physical meaing"""
    conf = files.load_config(conf)
    params = parameters.get_parameters(params, conf)
    fiducials = parameters.get_fiducials(params, conf)
    prior_size = np.squeeze(np.diff(parameters.get_prior_intervals(params, conf)))

    log_prior = np.zeros(cosmos.shape[0])
    for i, param in enumerate(params):
        if not param in params_unaffected:
            log_prior += norm(loc=fiducials[i], scale=std_fac * prior_size[i]).logpdf(cosmos[:, i])

    return log_prior


def generate_randoms(params=None, conf=None, n_draws=100_000, output_S8=False, invert=False):
    params = ["s8" if p == "S8" else p for p in params]

    rands = np.random.uniform(
        low=[parameters.get_prior_intervals([param], conf)[0][0] for param in params],
        high=[parameters.get_prior_intervals([param], conf)[0][1] for param in params],
        size=(n_draws, len(params)),
    )
    prior_mask = in_grid_prior(rands, params=params, conf=conf)
    if invert:
        prior_mask = ~prior_mask
    rands_in_prior = rands[prior_mask]
    LOGGER.info(f"Generated {len(rands_in_prior)} randoms in prior")

    if output_S8:
        i_s8 = params.index("s8")
        i_Om = params.index("Om")
        rands_in_prior[:, i_s8] = rands_in_prior[:, i_s8] * np.sqrt(rands_in_prior[:, i_Om] / 0.3)

    return rands_in_prior


def assess_prior_boundary(
    samples, params, conf=None, subspace_params=None, epsilon=0.05, press_threshold=0.1, blinded=False
):
    """Assess whether MCMC posterior samples are pressing against prior boundaries.

    For each active constraint, computes a normalised distance per sample (0 = on the
    boundary, 1 = at the opposite edge) and logs a summary table with frac_near, p5,
    and median. Samples with distance < epsilon are counted as "near boundary"; rows
    with frac_near > 10 % are flagged as PRESSING.

    Three constraint types are handled:
        - Box walls: one entry per (param, lower/upper) pair, normalised by prior width.
        - Om-s8 hull: Euclidean distance to the nearest facet of the Om_s8_border_points
          convex hull, normalised by sqrt(Om_width * s8_width). Only checked when both
          "Om" and "s8" are in subspace_params.
        - w0-Om threshold: signed gap w0 - (1/(Om-1) + 0.01), normalised by w0 prior
          width. Only checked when both "Om" and "w0" are in subspace_params.

    Args:
        samples (np.ndarray): Posterior samples of shape (n_samples, n_full_params),
            assumed to lie inside the prior (i.e. post burn-in chain).
        params (list): Parameter names corresponding to the columns of samples.
        conf (str, dict, optional): Config. Defaults to None.
        subspace_params (list, optional): Subset of params to assess. Defaults to all
            of params.
        epsilon (float): Normalised distance threshold for "near boundary". Defaults to
            0.05 (5 % of the relevant prior scale).
        press_threshold (float): frac_near value above which a constraint is flagged as
            PRESSING in the log table. Defaults to 0.10.
        blinded (bool): If True, suppresses the detailed table and return value. Only
            logs a single boolean indicating whether any boundary is being pressed.
            Defaults to False.

    Returns:
        dict or None: When blinded=False, returns a dict with constraint identifiers as
            keys (e.g. "Om_lower", "Om_s8_hull"), each mapping to a dict with keys
            "frac_near", "p5", and "median". When blinded=True, returns None.
    """

    conf = files.load_config(conf)
    samples = np.atleast_2d(samples)
    if subspace_params is None:
        subspace_params = list(params)

    LOGGER.info(f"Assessing prior boundaries in {subspace_params} for chain in {params}")

    results = {}

    # box constraints – one entry per (param × boundary) pair
    prior_intervals = parameters.get_prior_intervals(subspace_params, conf)
    for i, param in enumerate(subspace_params):
        j = params.index(param)
        col = samples[:, j]
        width = prior_intervals[i, 1] - prior_intervals[i, 0]
        dist_lower = (col - prior_intervals[i, 0]) / width
        dist_upper = (prior_intervals[i, 1] - col) / width
        for label, dist in [(f"{param}_lower", dist_lower), (f"{param}_upper", dist_upper)]:
            results[label] = {
                "frac_near": float(np.mean(dist < epsilon)),
                "p5": float(np.percentile(dist, 5)),
                "median": float(np.median(dist)),
            }

    # Om-s8 convex-hull boundary
    if "Om" in subspace_params and "s8" in subspace_params:
        border_pts = np.array(conf["analysis"]["grid"]["priors"]["Om_s8_border_points"])
        i_Om = params.index("Om")
        i_s8 = params.index("s8")
        pts_2d = samples[:, [i_Om, i_s8]]

        hull_cv = ConvexHull(border_pts)
        # equations[:, :2] are unit-norm facet normals; equations[:, 2] is the offset.
        # For interior points: A @ x + b < 0, so -(A @ x + b) > 0 is the distance.
        signed = -(hull_cv.equations[:, :2] @ pts_2d.T + hull_cv.equations[:, 2:3])
        dist_hull = np.min(signed, axis=0)  # distance to the nearest facet, shape (n_samples,)

        # normalise by geometric mean of the Om and s8 prior widths
        Om_intervals = parameters.get_prior_intervals(["Om"], conf)
        s8_intervals = parameters.get_prior_intervals(["s8"], conf)
        norm_scale = np.sqrt((Om_intervals[0, 1] - Om_intervals[0, 0]) * (s8_intervals[0, 1] - s8_intervals[0, 0]))
        dist_hull_norm = dist_hull / norm_scale

        results["Om_s8_hull"] = {
            "frac_near": float(np.mean(dist_hull_norm < epsilon)),
            "p5": float(np.percentile(dist_hull_norm, 5)),
            "median": float(np.median(dist_hull_norm)),
        }

    # w0-Om threshold:  w0 >= 1 / (Om - 1) + 0.01
    if "Om" in subspace_params and "w0" in subspace_params:
        i_Om = params.index("Om")
        i_w0 = params.index("w0")
        Om_col = samples[:, i_Om]
        w0_col = samples[:, i_w0]
        w0_intervals = parameters.get_prior_intervals(["w0"], conf)
        w0_width = w0_intervals[0, 1] - w0_intervals[0, 0]
        threshold = 1.0 / (Om_col - 1.0) + 0.01
        dist_threshold = (w0_col - threshold) / w0_width

        results["w0_Om_threshold"] = {
            "frac_near": float(np.mean(dist_threshold < epsilon)),
            "p5": float(np.percentile(dist_threshold, 5)),
            "median": float(np.median(dist_threshold)),
        }

    any_pressing = any(s["frac_near"] > press_threshold for s in results.values())

    if blinded:
        LOGGER.info(f"Prior boundary assessment (blinded)  any_pressing={any_pressing}")
        return any_pressing

    LOGGER.info(f"Prior boundary assessment  (n_samples={len(samples)}, epsilon={epsilon:.2f})")
    LOGGER.info(f"  {'Constraint':<24}  {'frac_near':>10}  {'p5':>8}  {'median':>8}")
    LOGGER.info("  " + "-" * 56)
    for key, stats in results.items():
        flag = "  <-- PRESSING" if stats["frac_near"] > press_threshold else ""
        LOGGER.info(f"  {key:<24}  {stats['frac_near']:>10.3f}  {stats['p5']:>8.3f}  {stats['median']:>8.3f}{flag}")

    return results
