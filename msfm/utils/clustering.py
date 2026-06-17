"""
Created on May 2023
Author: Arne Thomsen

Contains functions that are specific to galaxy clustering. Since the healpy alm transform only takes either a single
or three maps (polarized case), these functions are not vectorized accross the example dimension.
"""

import os
import numpy as np

from sobol_seq import i4_sobol

from msfm.utils import files, imports, parameters
import tensorflow as tf
import torch

hp = imports.import_healpy()


def galaxy_density_to_count(
    ng_bar,
    # linear
    dg,
    bg,
    # quadratic
    qdg=None,
    qbg=None,
    # magnification
    mg=None,
    cg=None,
    # modeling
    systematics_map=None,
    # format
    mask=None,
    backend='tensorflow',
):
    """Transform a galaxy density to a galaxy count map, according to the constants defined in the config file.
    Negative values are clipped and the maps tranformed to conserve the total number of galaxies like in DeepLSS.

    Args:
        ng_bar (np.ndarray): Average number of galaxies per pixel (optionally per tomographic bin).
        dg (Union[np.ndarray, tf.Tensor, torch.Tensor]): Galaxy density contrast map or datavector. Optionally per tomographic bin
            in the last array dimension.
        bg (np.ndarray): Effective linear galaxy biasing parameter (optionally per tomographic bin).
        qdg (np.ndarray, optional): Squared galaxy density contrast map (optionally per tomographic bin).
        qbg (np.ndarray, optional): Effective quadratic galaxy biasing parameter (optionally per tomographic bin).
        systematics_map (bool): Whether to multiply with the maglim systematics map. Defaults to False.
        stochasticity (float, optional): Raises a NotImplementedError if not None. Defaults to None.


    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        ng: Galaxy number count map.
    """

    # linear bias
    ng = 1 + bg * dg

    # quadratic bias
    if (qbg is not None) and (qdg is not None):
        ng += qbg * qdg

    ng *= ng_bar

    # map-level magnification bias as derived by Laura
    if (mg is not None) and (cg is not None):
        ng *= 1 + cg * mg

    # transform like in DeepLSS Appendix E and https://github.com/tomaszkacprzak/deep_lss/blob/3c145cf8fe04c4e5f952dca984c5ce7e163b8753/deep_lss/lss_astrophysics_model_batch.py#L609
    # this ensures that all of the values are positive, while the total number of galaxies is conserved
    if backend == 'numpy':
        import numpy as np
        ng_clip = np.clip(ng, a_min=0, a_max=None, dtype=np.float32)
        ng = ng_clip * np.sum(ng) / np.sum(ng_clip)
    
    elif backend == 'tensorflow':
        import tensorflow as tf
        ng_clip = tf.clip_by_value(ng, clip_value_min=0, clip_value_max=1e5)
        ng = ng_clip * tf.reduce_sum(ng) / tf.reduce_sum(ng_clip)
    
    elif backend == 'torch':
        import torch
        ng_clip = torch.clamp(ng, min=0, max=1e5)
        ng = ng_clip * torch.sum(ng) / torch.sum(ng_clip)

    else:
        raise ValueError(f"Unsupported type {type(dg)} for dg")

    if systematics_map is not None:
        # mask zeros, this is expecially important for the padded data vectors
        ng[systematics_map != 0.0] /= systematics_map[systematics_map != 0.0]

    # mask the footprint within the padded data vector
    if mask is not None:
        ng *= mask

    return ng


def galaxy_count_to_noise(ng, n_noise, np_seed=None):
    """
    Draw Poisson noise according to the given map of galaxy counts.

    Args:
        ng (Union[np.ndarray, tf.Tensor]): Galaxy number count map or datavector. Optionally per tomographic bin.
        n_noise (int): Number of noise realizations to draw.
        np_seed (int, optional): Seed for the numpy random number generator. Defaults to None.

    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        poisson_noise: Pure (e.g. the input galaxy count map has been subtracted) Poisson noise consistent with the
            input.
    """

    if isinstance(ng, np.ndarray):
        rng = np.random.default_rng(np_seed)

        try:
            # draw noise, poisson realizations along axis
            noisy_ngs = rng.poisson(np.repeat(ng[np.newaxis, :], n_noise, axis=0), size=None).astype(np.float32)
        except ValueError:
            out_dir = os.getcwd()
            np.save(os.path.join(out_dir, f"ng_seed={np_seed}.npy"), ng)
            print(f"Saved ng to {out_dir}")
            print("nan count", np.sum(np.isnan(ng)))
            print("neg count", np.sum(ng < 0))

        # shape (n_noise, n_pix) is broadcast along the first axis
        poisson_noise = noisy_ngs - ng

    # elif isinstance(ng, tf.Tensor):
    #     raise NotImplementedError

    return poisson_noise


def extend_sobol_sequence_by_stochasticity(conf, full_sky_map, simset, i_sobol, rng):
    """decorrelate the galaxy density contrast from the galaxy number"""

    if simset == "grid":
        # extend the Sobol sequence
        cosmo_params = conf["analysis"]["params"]["cosmo"].copy()
        if conf["analysis"]["modelling"]["baryonified"]:
            cosmo_params += conf["analysis"]["params"]["bary"]
        sobol_params = cosmo_params + conf["analysis"]["params"]["bg"]["stochasticity"]

        sobol_priors = parameters.get_prior_intervals(sobol_params, conf=conf)
        sobol_point, _ = i4_sobol(sobol_priors.shape[0], i_sobol)
        sobol_point = sobol_point * np.squeeze(np.diff(sobol_priors)) + sobol_priors[:, 0]
        sobol_point = sobol_point.astype(np.float32)
        rg = sobol_point[-1]
    elif simset == "fiducial":
        rg = parameters.get_fiducials(["rg"], conf=conf)[0]

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    alm = hp.map2alm(full_sky_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)
    # empirical formula from (12) in DeepLSS https://arxiv.org/abs/2203.09616
    random_phases = (1 - rg) ** (2 / 3) * rng.uniform(-np.pi, np.pi, alm.shape[0])
    stochastic_alm = np.exp(1j * random_phases) * alm

    return hp.alm2map(stochastic_alm, nside=conf["analysis"]["n_side"], pol=False)
