# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created May 2024
Author: Arne Thomsen

Utilities to forward model (mock) observations to be consistent with the CosmoGrid maps.
"""

import os, h5py, pickle
import numpy as np
from msfm.utils import (
    files,
    logger,
    lensing,
    imports,
    scales,
    maps,
    power_spectra,
    filenames,
    redshift,
    clustering,
    lensing,
)
from typing import Union

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


def forward_model_observation_map(
    wl_gamma_map: np.ndarray = None,
    gc_count_map: np.ndarray = None,
    conf: Union[str, dict] = None,
    apply_norm: bool = True,
    with_padding: bool = True,
    nest_in: bool = True,
    apply_maglim_sys_map: bool = False,
):
    """Take a (mock) observation and apply the same transformations to it as within the CosmoGrid pipeline, such that
    everything (masking, mode removal, normalization, ...) is consistent.

    Args:
        wl_gamma (np.ndarray, optional): The weak lensing shear map of shape (n_pix,n_z_metacal,2), where the first
            axis corresponds to a full sky map at the correct n_side and the last axis contains the gamma_1 and gamma_2
            components. Note that the input footprint has to be rotated to the correct position on the sky. Defaults
            to None.
        gc_count (np.ndarray, optional): The galaxy clustering galaxy number count map of shape (n_pix,n_z_maglim),
            like for wl_gamma. Defaults to None.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        apply_norm (bool, optional): Whether to rescale the maps to approximate unit range. Defaults to True.
        with_padding (bool, optional): Whether to include the padding of the data vectors (the healpy DeepSphere
            networks) need this. Defaults to True.
        nest (bool, optional): Whether the full sky input maps wl_gamma and gc_count are in nested (or ring if false)
            ordering. Defaults to True.
    """
    assert wl_gamma_map is not None or gc_count_map is not None, "Either wl_gamma or gc_count must be provided."

    conf = files.load_config(conf)

    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    n_z_metacal = len(conf["survey"]["metacal"]["z_bins"])
    n_z_maglim = len(conf["survey"]["maglim"]["z_bins"])

    data_vec_pix, patches_pix_dict, corresponding_pix_dict, _ = files.load_pixel_file(conf)
    data_vec_len = len(data_vec_pix)

    masks_dict = files.get_tomo_masks(conf, nest_out=nest_in)
    masks_metacal = masks_dict["metacal"]
    masks_maglim = masks_dict["maglim"]

    dv_masks_dict = files.get_tomo_dv_masks(conf)
    dv_masks_metacal = dv_masks_dict["metacal"]
    dv_masks_maglim = dv_masks_dict["maglim"]

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    if wl_gamma_map is not None:
        assert wl_gamma_map.shape == (
            n_pix,
            n_z_metacal,
            2,
        ), f"Expected shape {(n_pix, n_z_metacal, 2)}, got {wl_gamma_map.shape}"

        LOGGER.info(f"Forward modeling the weak lensing map")

        wl_gamma_map *= masks_metacal[:, :, np.newaxis]

        # the input to the mode removal must always be in RING ordering
        if nest_in:
            wl_gamma_map[..., 0] = maps.tomographic_reorder(wl_gamma_map[..., 0], n2r=True)
            wl_gamma_map[..., 1] = maps.tomographic_reorder(wl_gamma_map[..., 1], n2r=True)
        _, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(l_max=3 * n_side - 1)

        wl_kappa_dv = np.zeros((data_vec_len, n_z_metacal), dtype=np.float32)
        for i in range(n_z_metacal):
            # full sky (but only partially occupied)
            wl_kappa_map = lensing.mode_removal(
                wl_gamma_map[:, i, 0], wl_gamma_map[:, i, 1], gamma2kappa_fac, n_side, hp_datapath=hp_datapath
            )

            # full sky (but only footprint occupied) -> padded data vector
            wl_kappa_dv[:, i] = maps.map_to_data_vec(
                hp_map=wl_kappa_map,
                data_vec_len=data_vec_len,
                corresponding_pix=corresponding_pix_dict["metacal"][i],
                cutout_pix=patches_pix_dict["metacal"][i][0],
                remove_mean=True,
            )

        wl_kappa_dv *= dv_masks_metacal
        wl_kappa_dv, wl_alms = scales.data_vector_to_smoothed_data_vector(
            wl_kappa_dv,
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=conf["analysis"]["scale_cuts"]["lensing"]["l_min"],
            l_max=conf["analysis"]["scale_cuts"]["lensing"]["l_max"],
            theta_fwhm=conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            arcmin=True,
            mask=dv_masks_metacal,
            hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
            conf=conf,
        )
        wl_kappa_dv *= dv_masks_metacal

        if apply_norm:
            wl_kappa_dv = wl_kappa_dv / conf["analysis"]["normalization"]["lensing"]

    if gc_count_map is not None:
        assert gc_count_map.shape == (
            n_pix,
            n_z_maglim,
        ), f"Expected shape {(n_pix, n_z_maglim)}, got {gc_count_map.shape}"

        LOGGER.info(f"Forward modeling the galaxy clustering map")

        gc_count_map *= masks_maglim

        # the input to map_to_data_vec must always be in RING ordering
        if nest_in:
            gc_count_map = maps.tomographic_reorder(gc_count_map, n2r=True)

        gc_count_dv = np.zeros((data_vec_len, n_z_maglim), dtype=np.float32)
        for i in range(n_z_maglim):
            # full sky (but only footprint occupied) -> padded data vector
            gc_count_dv[:, i] = maps.map_to_data_vec(
                hp_map=gc_count_map[:, i],
                data_vec_len=data_vec_len,
                corresponding_pix=corresponding_pix_dict["maglim"][i],
                cutout_pix=patches_pix_dict["maglim"][i][0],
            )

        if apply_maglim_sys_map:
            LOGGER.warning("Applying maglim systematics map")
            gc_count_dv *= files.get_clustering_systematics(conf, pixel_type="data_vector")

        gc_count_dv *= dv_masks_maglim
        gc_count_dv, gc_alms = scales.data_vector_to_smoothed_data_vector(
            gc_count_dv,
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            l_max=conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
            theta_fwhm=conf["analysis"]["scale_cuts"]["clustering"]["theta_fwhm"],
            arcmin=True,
            mask=dv_masks_maglim,
            hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
            conf=conf,
        )
        gc_count_dv *= dv_masks_maglim

        if apply_norm:
            gc_count_dv = gc_count_dv / conf["analysis"]["normalization"]["clustering"]

    if wl_gamma_map is not None and gc_count_map is not None:
        observation = np.concatenate([wl_kappa_dv, gc_count_dv], axis=-1)
        observation_cls = power_spectra.get_cls(np.concatenate([wl_alms, gc_alms], axis=-1))
    elif wl_gamma_map is not None:
        observation = wl_kappa_dv
        observation_cls = power_spectra.get_cls(wl_alms)
    elif gc_count_map is not None:
        observation = gc_count_dv
        observation_cls = power_spectra.get_cls(gc_alms)
    else:
        raise ValueError("At least one of wl_gamma or gc_count must be provided.")

    if with_padding:
        return observation, observation_cls, data_vec_pix
    else:
        # only keep indices that are in all (per tomographic bin and galaxy sample) masks
        mask_total = np.prod(np.concatenate([dv_masks_metacal, dv_masks_maglim], axis=-1), axis=-1)
        mask_total = mask_total.astype(bool)
        footprint_pix = data_vec_pix[mask_total]

        observation_full_sky = np.zeros((n_pix, observation.shape[-1]), dtype=observation.dtype)
        observation_full_sky[data_vec_pix] = observation
        observation = observation_full_sky[footprint_pix]

        return observation, observation_cls, footprint_pix


def forward_model_cosmogrid(
    map_dir,
    conf=None,
    noisy=True,
    noise_only=False,
    i_patch=0,
    # lensing
    with_lensing=True,
    tomo_Aia=None,
    bta=None,
    tomo_bg_metacal=None,
    i_sobol=None,
    shear_biasing=False,
    reduced_shear=False,
    # clustering
    with_clustering=True,
    tomo_bg=None,
    tomo_qbg=None,
    tomo_cg=None,
    survey_sys=False,
    noise_seed=12,
):
    """Take a full-sky CosmoGrid maps as they are projected with UFalcon and transform them into fiducial probe maps
    that are in the same format as the (synthetic) observations like a gamma map for weak lensing, no smoothing, etc.
    The steps here are the same what is implemented in the run_grid/fiducial_postprocessing.py files. The later steps
    of that pipeline are implemented in forward_model_observation_map.
    This function is for example useful for the benchmark simulations.

    Args:
        map_dir (str): The directory where the full-sky CosmoGrid map is stored in the .h5 format.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        with_lensing (bool, optional): Whether to include the weak lensing map. Defaults to True.
        with_clustering (bool, optional): Whether to include the galaxy clustering map. Defaults to True.
        noisy (bool, optional): Whether to generate shape and poisson noise or return noiseless maps. Defaults to
            False.

    Returns:
        (np.ndarray, np.ndarray): Weak lensing and galaxy clustering full-sky maps of shape (n_pix, n_z, 2) and
            (n_pix, n_z) respectively.
    """

    conf = files.load_config(conf)
    if noise_only:
        assert noisy, "If noise_only is true, noisy must also be true."

    # constants
    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    data_vec_pix, patches_pix_dict, _, gamma2_signs = files.load_pixel_file(conf)

    map_file = filenames.get_filename_full_maps(map_dir, with_bary=conf["analysis"]["modelling"]["baryonified"])
    LOGGER.info(f"Loading the full-sky map from {map_file}")
    with h5py.File(map_file, "r") as f:
        if with_lensing:
            LOGGER.info(f"Starting with the weak lensing map")
            LOGGER.timer.start("weak_lensing")

            metacal_mask = files.get_tomo_dv_masks(conf)["metacal"]
            kappa2gamma_fac, _, _ = lensing.get_kaiser_squires_factors(3 * n_side - 1)
            metacal_bins = conf["survey"]["metacal"]["z_bins"]

            file_dir = os.path.dirname(__file__)
            repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
            hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

            extended_nla = conf["analysis"]["modelling"]["lensing"]["extended_nla"] if bta is None else True

            kg = []
            ia = []
            ds = []
            dg = []
            for z_bin in metacal_bins:
                kg.append(hp.ud_grade(f[f"map/kg/{z_bin}"], n_side))
                ia.append(hp.ud_grade(f[f"map/ia/{z_bin}"], n_side))
                if extended_nla:
                    full_sky_ia = hp.ud_grade(f[f"map/ia/{z_bin}"], n_side)
                    full_sky_dg = hp.ud_grade(f[f"map/dg/{z_bin}"], n_side)
                    ds.append(
                        (full_sky_ia - np.mean(full_sky_ia))
                        * ((full_sky_dg - np.mean(full_sky_dg)) / np.mean(full_sky_dg))
                    )
                if noisy:
                    dg.append(hp.ud_grade(f[f"map/dg/{z_bin}"], n_side))
            kg = np.stack(kg, axis=-1)
            ia = np.stack(ia, axis=-1)
            if extended_nla:
                ds = np.stack(ds, axis=-1)
            if noisy:
                dg = np.stack(dg, axis=-1)

            # create the noiseless fiducial map

            if tomo_Aia is None:
                Aia = conf["analysis"]["fiducial"]["Aia"]
                n_Aia = conf["analysis"]["fiducial"]["n_Aia"]
                tomo_Aia = redshift.get_tomo_amplitudes_according_to_config(conf, Aia, n_Aia, "metacal")
                LOGGER.info(f"Using tomo_Aia={tomo_Aia} from the config")
            else:
                LOGGER.info(f"Using tomo_Aia={tomo_Aia} from the function call")

            if bta is None and extended_nla:
                bta = conf["analysis"]["fiducial"]["bta"]
                LOGGER.info(f"Using bta={bta} from the config")
            else:
                LOGGER.info(f"Using bta={bta} from the function call")

            if extended_nla:
                wl_kappa_map = kg + tomo_Aia * (ia + bta * ds)
                LOGGER.info("Using delta-NLA")
            else:
                wl_kappa_map = kg + tomo_Aia * ia
                LOGGER.info("Using standard NLA")

            if shear_biasing:
                m_bias_dist = lensing.get_m_bias_distribution(conf)
                m_bias = m_bias_dist.sample()
                wl_kappa_map *= 1.0 + m_bias

            if noisy:
                sc_mode = conf["analysis"]["modelling"]["lensing"]["source_clustering"]

                if tomo_bg_metacal is not None:
                    LOGGER.info(
                        f"Using tomo_bg_metacal={tomo_bg_metacal} from the function call, setting source_clustering to 'fixed'"
                    )
                    sc_mode = "fixed"

                if sc_mode in ["fixed", "prior"]:
                    if tomo_bg_metacal is None:
                        if i_sobol is not None:
                            tomo_bg_metacal = files.read_metacal_bias(f"cosmo_{i_sobol:06}", conf=conf)
                            LOGGER.info(f"Using tomo_bg_metacal={tomo_bg_metacal} from the Sobol index {i_sobol}")
                        else:
                            raise ValueError(
                                "Either tomo_bg_metacal or i_sobol must be provided to generate the shape noise for fixed source clustering"
                            )

                    tomo_n_gal = np.array(conf["survey"]["metacal"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
                    dg = (dg - np.mean(dg, axis=0)) / np.mean(dg, axis=0)
                    counts_map = clustering.galaxy_density_to_count(
                        tomo_n_gal, dg, tomo_bg_metacal, systematics_map=None
                    ).astype(int)
                elif sc_mode == "rotate":
                    LOGGER.info("Rotating galaxies in place for shape noise")
                else:
                    raise ValueError(f"Unknown source clustering mode {sc_mode}")

                tomo_gamma_cat = files.load_noise_file(conf)

            gamma1 = []
            gamma2 = []
            for i_z in range(wl_kappa_map.shape[-1]):
                patch_pix = patches_pix_dict["metacal"][i_z][0]
                cutout_patch_pix = patches_pix_dict["metacal"][i_z][i_patch]

                kappa_full = wl_kappa_map[:, i_z]

                # kappa -> gamma (full sky)
                kappa_alm = hp.map2alm(
                    kappa_full,
                    use_pixel_weights=True,
                    datapath=hp_datapath,
                )

                gamma_alm = kappa_alm * kappa2gamma_fac
                _, gamma1_full, gamma2_full = hp.alm2map(
                    [np.zeros_like(gamma_alm), gamma_alm, np.zeros_like(gamma_alm)], nside=n_side
                )

                if reduced_shear:
                    gamma1_full /= 1 - kappa_full
                    gamma2_full /= 1 - kappa_full

                if noisy:
                    import tensorflow as tf
                    import tensorflow_probability as tfp

                    tf.random.set_seed(noise_seed)

                    with tf.device("/CPU:0"):
                        gamma_cat = tomo_gamma_cat[i_z]
                        gamma_abs = tf.math.abs(gamma_cat[:, 0] + 1j * gamma_cat[:, 1])
                        w = gamma_cat[:, 2]

                        if sc_mode in ["fixed", "prior"]:
                            counts = counts_map[cutout_patch_pix, i_z]

                            # create joint distribution, as this is faster than random indexing
                            cat_dist = tfp.distributions.Empirical(
                                samples=tf.stack([gamma_abs, w], axis=-1), event_ndims=1
                            )

                            gamma1_noise, gamma2_noise = lensing.noise_gen(counts, cat_dist, n_noise_per_signal=1)
                            gamma1_noise = gamma1_noise[:, 0]
                            gamma2_noise = gamma2_noise[:, 0]
                        else:
                            pix_cat = gamma_cat[:, 3]
                            gamma1_noise, gamma2_noise = lensing.noise_gen_in_place(
                                gamma_abs, w, pix_cat, patch_pix, n_pix, n_noise_per_signal=1
                            )
                            gamma1_noise = gamma1_noise[:, 0]
                            gamma2_noise = gamma2_noise[:, 0]
                else:
                    gamma1_noise = 0
                    gamma2_noise = 0

                gamma1_patch = np.zeros(n_pix, dtype=np.float32)
                gamma2_patch = np.zeros(n_pix, dtype=np.float32)

                if noise_only:
                    gamma1_patch[patch_pix] = gamma1_noise
                    gamma2_patch[patch_pix] = gamma2_noise
                else:
                    gamma1_patch[patch_pix] = gamma1_full[cutout_patch_pix] + gamma1_noise
                    gamma2_patch[patch_pix] = gamma2_full[cutout_patch_pix] + gamma2_noise

                gamma2_patch *= gamma2_signs[i_patch]

                gamma1.append(gamma1_patch)
                gamma2.append(gamma2_patch)

            gamma1 = np.stack(gamma1, axis=-1)
            gamma2 = np.stack(gamma2, axis=-1)

            wl_gamma_patch = np.stack([gamma1, gamma2], axis=-1)
            LOGGER.info(f"Finished weak lensing after {LOGGER.timer.elapsed('weak_lensing')}")
        else:
            wl_gamma_patch = None

        if with_clustering:
            LOGGER.info(f"Starting with the galaxy clustering map")
            LOGGER.timer.start("galaxy_clustering")

            maglim_bins = conf["survey"]["maglim"]["z_bins"]
            tomo_n_gal_maglim = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

            # NOTE this assumes that the patches are the same for all tomographic bins, which is currently the case
            i_z_pix = 0
            patch_pix = patches_pix_dict["maglim"][i_z_pix][0]
            cutout_patch_pix = patches_pix_dict["maglim"][i_z_pix][i_patch]
            maglim_mask = files.get_tomo_dv_masks(conf)["maglim"]

            # full sky map
            dg = []
            for z_bin in maglim_bins:
                dg.append(hp.ud_grade(f[f"map/dg/{z_bin}"], n_side))
            dg = np.stack(dg, axis=-1)

            # cut out the footprint
            dg_patch = np.zeros_like(dg)
            dg_patch[patch_pix] = dg[cutout_patch_pix]

            # subtract and divide by mean (within the patch and tomographic bin)
            dg_mean = np.mean(dg_patch[patch_pix], axis=0)
            dg_patch[patch_pix] = (dg_patch[patch_pix] - dg_mean) / dg_mean

            dg_patch = maps.tomographic_reorder(dg_patch, r2n=True)
            dg_dv = dg_patch[data_vec_pix]

            # density contrast to count
            if tomo_bg is None:
                if conf["analysis"]["modelling"]["clustering"]["power_law_biasing"]:
                    bg = conf["analysis"]["fiducial"]["bg"]
                    n_bg = conf["analysis"]["fiducial"]["n_bg"]
                    tomo_bg = redshift.get_tomo_amplitudes_according_to_config(conf, bg, n_bg, "maglim")
                elif conf["analysis"]["modelling"]["clustering"]["per_bin_biasing"]:
                    tomo_bg = np.array(
                        [conf["analysis"]["fiducial"][f"bg{i}"] for i in range(1, len(maglim_bins) + 1)]
                    )
                LOGGER.info(f"Using tomo_bg={tomo_bg} from the config")
            else:
                LOGGER.info(f"Using tomo_bg={tomo_bg} from the function call")

            if tomo_qbg is None:
                if conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]:
                    if conf["analysis"]["modelling"]["clustering"]["power_law_biasing"]:
                        bg = conf["analysis"]["fiducial"]["qbg"]
                        n_bg = conf["analysis"]["fiducial"]["n_qbg"]
                        tomo_qbg = redshift.get_tomo_amplitudes_according_to_config(conf, bg, n_bg, "maglim")
                    elif conf["analysis"]["modelling"]["clustering"]["per_bin_biasing"]:
                        tomo_qbg = np.array(
                            [conf["analysis"]["fiducial"][f"qbg{i}"] for i in range(1, len(maglim_bins) + 1)]
                        )
                    LOGGER.info(f"Using tomo_qbg={tomo_qbg} from the config")
                    qdg_dv = dg_dv**2 * np.sign(dg_dv)
                else:
                    tomo_qbg = None
                    qdg_dv = None
                    LOGGER.info("No quadratic biasing")
            else:
                LOGGER.info(f"Using tomo_qbg={tomo_qbg} from the function call")
                qdg_dv = dg_dv**2 * np.sign(dg_dv)

            # TODO
            if tomo_cg is not None:
                LOGGER.warning(f"!!!EXPERIMENTAL!!! Using tomo_cg={tomo_cg} from the function call")

                perm_index = int(map_dir[-4:])

                mg = []
                for i, z_bin in enumerate(maglim_bins):
                    mg_file = f"/pscratch/sd/a/athomsen/laura/DES_maps_260128/mag_map_bin_{i+1}_run_{perm_index}.npy"
                    mg.append(np.load(mg_file))
                mg = np.stack(mg, axis=-1)

                mg_patch = np.zeros_like(mg)
                mg_patch[patch_pix] = mg[cutout_patch_pix]
                mg_patch = maps.tomographic_reorder(mg_patch, r2n=True)
                mg_dv = mg_patch[data_vec_pix]

            else:
                mg_dv = None
                tomo_cg = None

            if survey_sys:
                LOGGER.info("Including the maglim survey systematics map in the forward model")
                systematics_map = files.get_clustering_systematics(conf, pixel_type="data_vector")

            gc_count_dv = clustering.galaxy_density_to_count(
                tomo_n_gal_maglim,
                dg_dv,
                tomo_bg,
                qdg_dv,
                tomo_qbg,
                mg_dv,
                tomo_cg,
                systematics_map=systematics_map if survey_sys else None,
                mask=maglim_mask,
            )

            if noisy:
                gc_noise_dv = clustering.galaxy_count_to_noise(gc_count_dv, n_noise=1, np_seed=noise_seed)[0]
                if noise_only:
                    gc_count_dv = gc_noise_dv
                else:
                    gc_count_dv += gc_noise_dv

            gc_count_patch = np.zeros((n_pix, gc_count_dv.shape[-1]))
            gc_count_patch[data_vec_pix] = gc_count_dv
            gc_count_patch = maps.tomographic_reorder(gc_count_patch, n2r=True)

            LOGGER.info(f"Finished galaxy clustering after {LOGGER.timer.elapsed('galaxy_clustering')}")
        else:
            gc_count_patch = None

        return wl_gamma_patch, gc_count_patch


def make_shape_noise_map(wl_counts_map, conf, source_clustering="fixed", noise_seed=12):
    import tensorflow as tf
    import tensorflow_probability as tfp

    tf.random.set_seed(noise_seed)

    # constants
    n_pix = conf["analysis"]["n_pix"]
    _, patches_pix_dict, _, _ = files.load_pixel_file(conf)

    tomo_gamma_cat = files.load_noise_file(conf)

    gamma1 = []
    gamma2 = []
    for i in range(wl_counts_map.shape[-1]):
        patch_pix = patches_pix_dict["metacal"][i][0]

        with tf.device("/CPU:0"):
            counts = wl_counts_map[patch_pix, i]

            gamma_abs = tf.math.abs(tomo_gamma_cat[i][:, 0] + 1j * tomo_gamma_cat[i][:, 1])
            w = tomo_gamma_cat[i][:, 2]

            if source_clustering in ["fixed", "prior"]:
                # create joint distribution, as this is faster than random indexing
                cat_dist = tfp.distributions.Empirical(samples=tf.stack([gamma_abs, w], axis=-1), event_ndims=1)

                gamma1_noise, gamma2_noise = lensing.noise_gen(counts, cat_dist, n_noise_per_signal=1)
            elif source_clustering == "rotate":
                pix_cat = tomo_gamma_cat[i][:, 3]
                gamma1_noise, gamma2_noise = lensing.noise_gen_in_place(gamma_abs, w, pix_cat, patch_pix, n_pix, 1)
            else:
                raise ValueError(f"Unknown source clustering mode {source_clustering}")

            # only take the first noise realization
            gamma1_noise = gamma1_noise[:, 0]
            gamma2_noise = gamma2_noise[:, 0]

        gamma1_patch = np.zeros(n_pix, dtype=np.float32)
        gamma1_patch[patch_pix] = gamma1_noise

        gamma2_patch = np.zeros(n_pix, dtype=np.float32)
        gamma2_patch[patch_pix] = gamma2_noise

        gamma1.append(gamma1_patch)
        gamma2.append(gamma2_patch)

    gamma1 = np.stack(gamma1, axis=-1)
    gamma2 = np.stack(gamma2, axis=-1)

    return np.stack([gamma1, gamma2], axis=-1)
