# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen

These utils used to be in run_data_vectors.py, but were moved here to facilitate the CosmoGridV1.1 all-in-one
processing where no intermediate .h5 files are stored.

TODO the function argument orders in this file aren't consistent, this should be fixed at some point
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os, time, h5py, copy_guardian, pickle
from msfm.utils import logger, filenames, imports, lensing, clustering, maps, input_output, files

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)

# fiducial ############################################################################################################


def postprocess_fiducial_permutations(args, conf, cosmo_dir_in, i_perm, pixel_file, noise_file):
    LOGGER.info(f"Starting simulation permutation {i_perm:04d}")
    LOGGER.timer.start("permutation")

    full_maps_file = _get_full_sky_perm(args, conf, cosmo_dir_in, i_perm)
    rng = np.random.default_rng()

    is_fiducial = "cosmo_fiducial" in cosmo_dir_in

    store_lensing = conf["analysis"]["modelling"]["lensing"]["store"]
    store_clustering = conf["analysis"]["modelling"]["clustering"]["store"]
    samples = []
    if store_lensing:
        samples.append("metacal")
    if store_clustering:
        samples.append("maglim")

    # output container, one for each example
    data_vec_container = _set_up_per_example_dv_container(conf, pixel_file, is_fiducial)
    for sample in samples:
        LOGGER.timer.start("sample")
        LOGGER.info(f"Starting with sample {sample}")

        # sample specific
        in_map_types = conf["survey"][sample]["map_types"]["input"]
        out_map_types = conf["survey"][sample]["map_types"]["output"]
        z_bins = conf["survey"][sample]["z_bins"]

        for in_map_type, out_map_type in zip(in_map_types, out_map_types):
            # some fiducial perturbations are skipped for lensing
            if (not is_fiducial) and (sample == "metacal") and (in_map_type == "dg"):
                LOGGER.info(f"Skipping input map type {in_map_type} for this perturbation")
                continue

            LOGGER.info(f"Starting with map type {in_map_type} -> {out_map_type}")
            LOGGER.timer.start("map_type")

            for i_z, z_bin in enumerate(z_bins):
                full_sky_bin = _read_full_sky_bin(conf, full_maps_file, in_map_type, z_bin)

                if sample == "metacal":
                    data_vecs = postprocess_metacal_bin(
                        conf,
                        full_sky_bin,
                        in_map_type,
                        out_map_type,
                        i_z,
                        "fiducial",
                        pixel_file,
                        noise_file,
                        full_maps_file,
                        bgs_key="fiducial",
                    )
                elif sample == "maglim":
                    data_vecs = postprocess_maglim_bin(
                        conf, full_sky_bin, in_map_type, out_map_type, i_z, "fiducial", pixel_file, rng=rng
                    )

                # collect the different permutations along the first axis
                data_vec_container[out_map_type][..., i_z] = data_vecs

            LOGGER.info(f"Done with map type {out_map_type} after {LOGGER.timer.elapsed('map_type')}")
        LOGGER.info(f"Done with sample {sample} after {LOGGER.timer.elapsed('sample')}")
    LOGGER.info(f"Done with permutation {i_perm:04d} after {LOGGER.timer.elapsed('permutation')}")

    return data_vec_container


def _set_up_per_example_dv_container(conf, pixel_file, is_fiducial):
    n_patches = conf["analysis"]["n_patches"]
    n_noise_per_signal = conf["analysis"]["fiducial"]["n_noise_per_signal"]
    data_vec_len = len(pixel_file[0])

    store_lensing = conf["analysis"]["modelling"]["lensing"]["store"]
    store_clustering = conf["analysis"]["modelling"]["clustering"]["store"]

    out_map_types = []
    if store_lensing:
        out_map_types += conf["survey"]["metacal"]["map_types"]["output"]
    if store_clustering:
        out_map_types += conf["survey"]["maglim"]["map_types"]["output"]

    data_vec_container = {}
    for out_map_type in out_map_types:
        if out_map_type in ["kg", "ia", "ds"]:
            n_z_bins = len(conf["survey"]["metacal"]["z_bins"])
            dvs_shape = (n_patches, data_vec_len, n_z_bins)
        elif out_map_type == "sn":
            n_z_bins = len(conf["survey"]["metacal"]["z_bins"])
            if is_fiducial:
                dvs_shape = (n_patches, n_noise_per_signal, data_vec_len, n_z_bins)
            else:
                dvs_shape = None
        elif out_map_type == "dg":
            n_z_bins = len(conf["survey"]["maglim"]["z_bins"])
            dvs_shape = (n_patches, data_vec_len, n_z_bins)

        if dvs_shape is not None:
            data_vec_container[out_map_type] = np.zeros(dvs_shape, dtype=np.float32)

    return data_vec_container


# grid ################################################################################################################


def postprocess_grid_permutations(args, conf, cosmo_dir_in, pixel_file, noise_file, bsc_samples=None):
    # hard-coded with respect to the filenames
    i_sobol = int(cosmo_dir_in[-7:-1])
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    rng = np.random.default_rng()

    store_lensing = conf["analysis"]["modelling"]["lensing"]["store"]
    store_clustering = conf["analysis"]["modelling"]["clustering"]["store"]
    samples = []
    if store_lensing:
        samples.append("metacal")
    if store_clustering:
        samples.append("maglim")

    # output container, one for each cosmology
    data_vec_container = _set_up_per_cosmo_dv_container(conf, pixel_file)
    for i_perm in LOGGER.progressbar(range(n_perms_per_cosmo), desc="Looping through permutations\n", at_level="info"):
        LOGGER.info(f"Starting simulation permutation {i_perm:04d}")

        if args.debug and i_perm > 3:
            LOGGER.warning("Debug mode, aborting after 3 permutations")
            break

        full_maps_file = _get_full_sky_perm(args, conf, cosmo_dir_in, i_perm)

        for sample in samples:
            LOGGER.timer.start("sample")
            LOGGER.info(f"Starting with sample {sample}")

            # sample specific
            in_map_types = conf["survey"][sample]["map_types"]["input"]
            out_map_types = conf["survey"][sample]["map_types"]["output"]
            z_bins = conf["survey"][sample]["z_bins"]

            for in_map_type, out_map_type in zip(in_map_types, out_map_types):
                LOGGER.info(f"Starting with map type {in_map_type} -> {out_map_type}")
                LOGGER.timer.start("map_type")

                for i_z, z_bin in enumerate(z_bins):
                    full_sky_bin = _read_full_sky_bin(conf, full_maps_file, in_map_type, z_bin)

                    if sample == "metacal":
                        data_vecs = postprocess_metacal_bin(
                            conf,
                            full_sky_bin,
                            in_map_type,
                            out_map_type,
                            i_z,
                            "grid",
                            pixel_file,
                            noise_file,
                            full_maps_file,
                            bgs_key=f"cosmo_{i_sobol:06d}",
                            i_perm=i_perm,
                            bsc_samples=bsc_samples,
                        )
                    elif sample == "maglim":
                        data_vecs = postprocess_maglim_bin(
                            conf,
                            full_sky_bin,
                            in_map_type,
                            out_map_type,
                            i_z,
                            "grid",
                            pixel_file,
                            i_sobol=i_sobol,
                            rng=rng,
                        )

                    # collect the different permutations along the first axis
                    data_vec_container[out_map_type][
                        n_patches * i_perm : n_patches * (i_perm + 1), ..., i_z
                    ] = data_vecs

                LOGGER.info(f"Done with map type {out_map_type} after {LOGGER.timer.elapsed('map_type')}")
            LOGGER.info(f"Done with sample {sample} after {LOGGER.timer.elapsed('sample')}")

    return data_vec_container


def _set_up_per_cosmo_dv_container(conf, pixel_file):
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_signal = conf["analysis"]["grid"]["n_noise_per_signal"]
    data_vec_len = len(pixel_file[0])

    store_lensing = conf["analysis"]["modelling"]["lensing"]["store"]
    store_clustering = conf["analysis"]["modelling"]["clustering"]["store"]

    out_map_types = []
    if store_lensing:
        out_map_types += conf["survey"]["metacal"]["map_types"]["output"]
    if store_clustering:
        out_map_types += conf["survey"]["maglim"]["map_types"]["output"]

    data_vec_container = {}
    for out_map_type in out_map_types:
        if out_map_type in ["kg", "ia", "ds"]:
            n_z_bins = len(conf["survey"]["metacal"]["z_bins"])
            dvs_shape = (n_perms_per_cosmo * n_patches, data_vec_len, n_z_bins)
        elif out_map_type == "dg":
            n_z_bins = len(conf["survey"]["maglim"]["z_bins"])
            dvs_shape = (n_perms_per_cosmo * n_patches, data_vec_len, n_z_bins)
        elif out_map_type == "sn":
            n_z_bins = len(conf["survey"]["metacal"]["z_bins"])
            dvs_shape = (n_perms_per_cosmo * n_patches, n_noise_per_signal, data_vec_len, n_z_bins)

        data_vec_container[out_map_type] = np.zeros(dvs_shape, dtype=np.float32)

    return data_vec_container


# lensing #############################################################################################################


def postprocess_metacal_bin(
    conf,
    full_sky_map,
    in_map_type,
    out_map_type,
    i_z,
    simset,
    pixel_file,
    noise_file,
    full_maps_file,
    bgs_key,
    i_perm=None,
    bsc_samples=None,
):
    if in_map_type in ["kg", "ia"]:
        # shape (n_patches, data_vec_len)
        kappa_dvs = postprocess_lensing(full_sky_map, conf, pixel_file, i_z)
    elif in_map_type == "dg" and out_map_type == "sn":
        # shape (n_patches, n_noise_per_signal, data_vec_len)
        kappa_dvs = postprocess_shape_noise(
            full_sky_map, conf, simset, pixel_file, noise_file, i_z, bgs_key, i_perm, bsc_samples
        )
    elif in_map_type == "dg" and out_map_type == "ds":
        full_sky_ia = _read_full_sky_bin(conf, full_maps_file, "ia", conf["survey"]["metacal"]["z_bins"][i_z])
        full_sky_ds = (full_sky_ia - np.mean(full_sky_ia)) * (
            (full_sky_map - np.mean(full_sky_map)) / np.mean(full_sky_map)
        )
        # shape (n_patches, data_vec_len)
        kappa_dvs = postprocess_lensing(full_sky_ds, conf, pixel_file, i_z)
    else:
        raise ValueError(f"Unknown input map type {in_map_type} for metacal/weak lensing")

    return kappa_dvs


def postprocess_lensing(kappa_full_sky, conf, pixel_file, i_z):
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    n_patches = conf["analysis"]["n_patches"]

    # pixel file
    data_vec_pix, patches_pix_dict, corresponding_pix_dict, gamma2_signs = pixel_file
    patches_pix = patches_pix_dict["metacal"][i_z]
    corresponding_pix = corresponding_pix_dict["metacal"][i_z]
    data_vec_len = len(data_vec_pix)
    base_patch_pix = patches_pix[0]

    kappa2gamma_fac, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(3 * n_side - 1)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # kappa -> gamma (full sky)
    kappa_alm = hp.map2alm(
        kappa_full_sky,
        use_pixel_weights=True,
        datapath=hp_datapath,
    )

    gamma_alm = kappa_alm * kappa2gamma_fac
    _, gamma1_full, gamma2_full = hp.alm2map(
        [np.zeros_like(gamma_alm), gamma_alm, np.zeros_like(gamma_alm)], nside=n_side
    )

    kappa_dvs = np.zeros((n_patches, data_vec_len), dtype=np.float32)
    for i_patch, patch_pix in enumerate(patches_pix):
        # The 90° rots do NOT change the shear, however, the mirroring does,
        # therefore we have to swap sign of gamma2 for the last 2 patches!
        gamma2_sign = gamma2_signs[i_patch]
        LOGGER.debug(f"Using gamma2 sign {gamma2_sign} for patch index {i_patch}")

        gamma1_patch = np.zeros(n_pix, dtype=np.float32)
        gamma1_patch[base_patch_pix] = gamma1_full[patch_pix]

        gamma2_patch = np.zeros(n_pix, dtype=np.float32)
        gamma2_patch[base_patch_pix] = gamma2_full[patch_pix]

        # fix the sign
        gamma2_patch *= gamma2_sign

        # kappa_patch is a full sky map, but only the patch is occupied
        kappa_patch = lensing.mode_removal(
            gamma1_patch,
            gamma2_patch,
            gamma2kappa_fac,
            n_side,
            apply_smoothing=False,
            hp_datapath=hp_datapath,
        )

        # cut out padded data vector
        kappa_dv = maps.map_to_data_vec(
            hp_map=kappa_patch,
            data_vec_len=data_vec_len,
            corresponding_pix=corresponding_pix,
            cutout_pix=base_patch_pix,
            remove_mean=True,
        )

        kappa_dvs[i_patch] = kappa_dv

    # shape (n_patches, data_vec_len)
    return kappa_dvs


def postprocess_shape_noise(
    delta_full_sky, conf, simset, pixel_file, noise_file, i_z, bgs_key, i_perm=None, bsc_samples=None
):
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    n_patches = conf["analysis"]["n_patches"]
    n_noise_per_signal = conf["analysis"][simset]["n_noise_per_signal"]

    # pixel file
    data_vec_pix, patches_pix_dict, corresponding_pix_dict, _ = pixel_file
    patches_pix = patches_pix_dict["metacal"][i_z]
    corresponding_pix = corresponding_pix_dict["metacal"][i_z]
    data_vec_len = len(data_vec_pix)
    base_patch_pix = patches_pix[0]

    # noise file
    tomo_gamma_cat = noise_file
    gamma_cat = tomo_gamma_cat[i_z]

    # metacal clustering
    sc_mode = conf["analysis"]["modelling"]["lensing"]["source_clustering"]

    if sc_mode == "fixed":
        tomo_bias = files.read_metacal_bias(bgs_key, conf)
        bias = tomo_bias[i_z]
    elif sc_mode == "prior":
        bias = None  # will be sampled per patch
    elif sc_mode == "rotate":
        bias = None
    else:
        raise ValueError(f"Unknown source clustering mode {sc_mode}")

    tomo_n_gal = np.array(conf["survey"]["metacal"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
    n_bar = tomo_n_gal[i_z]

    _, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(3 * n_side - 1)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # create joint distribution, as this is faster than random indexing
    gamma_abs = tf.math.abs(gamma_cat[:, 0] + 1j * gamma_cat[:, 1])
    w = gamma_cat[:, 2]

    if sc_mode in ["fixed", "prior"]:
        cat_dist = tfp.distributions.Empirical(samples=tf.stack([gamma_abs, w], axis=-1), event_ndims=1)

        # normalize to number density contrast
        delta_full_sky_norm = (delta_full_sky - np.mean(delta_full_sky)) / np.mean(delta_full_sky)

        if sc_mode == "fixed":
            counts_full = clustering.galaxy_density_to_count(
                n_bar, delta_full_sky_norm, bias, systematics_map=None
            ).astype(int)
            counts_full = np.random.poisson(counts_full).astype(int)
    else:
        LOGGER.warning("Rotating galaxies in place for shape noise")
        pix_cat = gamma_cat[:, 3]

    kappa_dvs = np.zeros((n_patches, n_noise_per_signal, data_vec_len), dtype=np.float32)
    for i_patch, patch_pix in enumerate(patches_pix):
        if sc_mode in ["fixed", "prior"]:
            if sc_mode == "prior" and bsc_samples is not None:
                bias_patch = bsc_samples[(i_perm * n_patches) + i_patch]
                delta_patch = delta_full_sky_norm[patch_pix]
                counts_patch = clustering.galaxy_density_to_count(
                    n_bar, delta_patch, bias_patch, systematics_map=None
                ).astype(int)
                counts_patch = np.random.poisson(counts_patch).astype(int)
                counts = counts_patch
            else:
                counts = counts_full[patch_pix]

            # vectorized sampling, shape (len(counts), n_noise_per_signal)
            gamma1, gamma2 = lensing.noise_gen(counts, cat_dist, n_noise_per_signal)
        else:
            gamma1, gamma2 = lensing.noise_gen_in_place(
                gamma_abs, w, pix_cat, base_patch_pix, n_pix, n_noise_per_signal
            )

        # not vectorized because of the healpy alm transform
        for i_noise in range(n_noise_per_signal):
            # full healpy map with zeros outside the footprint
            gamma1_patch = np.zeros(n_pix, dtype=np.float32)
            gamma1_patch[base_patch_pix] = gamma1[:, i_noise]

            gamma2_patch = np.zeros(n_pix, dtype=np.float32)
            gamma2_patch[base_patch_pix] = gamma2[:, i_noise]

            kappa_patch = lensing.mode_removal(
                gamma1_patch,
                gamma2_patch,
                gamma2kappa_fac,
                n_side,
                apply_smoothing=False,
                hp_datapath=hp_datapath,
            )

            # cut out padded data vector
            kappa_dv = maps.map_to_data_vec(
                hp_map=kappa_patch,
                data_vec_len=data_vec_len,
                corresponding_pix=corresponding_pix,
                cutout_pix=base_patch_pix,
                remove_mean=True,
            )

            kappa_dvs[i_patch, i_noise] = kappa_dv

    # shape (n_patches, n_noise_per_signal, data_vec_len)
    return kappa_dvs


# clustering ##########################################################################################################


def postprocess_maglim_bin(
    conf, full_sky_map, in_map_type, out_map_type, i_z, simset, pixel_file, i_sobol=None, rng=None
):
    if in_map_type in ["dg", "dg2"]:
        delta_dvs = postprocess_clustering(full_sky_map, conf, i_z, simset, pixel_file, "maglim", i_sobol, rng)
    else:
        raise ValueError(f"Unknown input map type {in_map_type} for maglim/galaxy clustering")

    return delta_dvs


def postprocess_clustering(
    delta_full_sky, conf, i_z, simset, pixel_file, galaxy_sample="maglim", i_sobol=None, rng=None
):
    n_pix = hp.nside2npix(conf["analysis"]["n_side"])
    n_patches = conf["analysis"]["n_patches"]

    # pixel file
    data_vec_pix, patches_pix_dict, corresponding_pix_dict, _ = pixel_file
    patches_pix = patches_pix_dict[galaxy_sample][i_z]
    corresponding_pix = corresponding_pix_dict[galaxy_sample][i_z]
    data_vec_len = len(data_vec_pix)
    base_patch_pix = patches_pix[0]

    # DeepLSS-style stochasticity has to be applied to the full-sky maps
    if conf["analysis"]["modelling"]["clustering"]["stochasticity"] and (i_sobol is not None) and (rng is not None):
        delta_full_sky = clustering.extend_sobol_sequence_by_stochasticity(conf, delta_full_sky, simset, i_sobol, rng)

    delta_dvs = np.zeros((n_patches, data_vec_len), dtype=np.float32)
    for i_patch, patch_pix in enumerate(patches_pix):
        # always populate the same patch
        delta_patch = np.zeros(n_pix, dtype=np.float32)
        delta_patch[base_patch_pix] = delta_full_sky[patch_pix]

        # cut out padded data vector
        delta_dv = maps.map_to_data_vec(
            delta_patch, data_vec_len, corresponding_pix, base_patch_pix, divide_by_mean=True
        )

        delta_dvs[i_patch] = delta_dv

    # shape (n_patches, data_vec_len)
    return delta_dvs


# shared utils ########################################################################################################


def _get_full_sky_perm(args, conf, cosmo_dir_in, i_perm):
    with_bary = conf["analysis"]["modelling"]["baryonified"]

    # prepare the full sky input file
    perm_dir_in = os.path.join(cosmo_dir_in, f"perm_{i_perm:04d}")
    full_maps_file = filenames.get_filename_full_maps(perm_dir_in, with_bary=with_bary, version=args.cosmogrid_version)

    if args.from_san:
        san_conf = conf["dirs"]["connections"]["san"]
        local_scratch_dir = os.environ["TMPDIR"]

        t0_rsync = time.time()
        with copy_guardian.BoundedSemaphore(san_conf["max_connections"], timeout=san_conf["timeout"]):
            connection = copy_guardian.Connection(
                host=san_conf["host"],
                user=san_conf["user"],
                private_key=san_conf["private_key"],
                port=san_conf["port"],
            )
            # overwrite the local scratch file within the loop iterations
            connection.rsync_from(full_maps_file, local_scratch_dir)

        tdelta_rsync = time.time() - t0_rsync
        LOGGER.info(f"Rsynced {full_maps_file} to {local_scratch_dir} after {tdelta_rsync:.2f}s")
        assert (
            tdelta_rsync > 1 or args.debug
        ), f"Rsync took only {tdelta_rsync:.2f}s, which indicates that nothing was actually transferred"
        full_maps_file = filenames.get_filename_full_maps(
            local_scratch_dir, with_bary=with_bary, version=args.cosmogrid_version
        )

    return full_maps_file


def _rsync_tfrecord_to_san(conf, tfr_file, san_dir_out):
    LOGGER.info("Copying the .tfrecord to the SAN")
    LOGGER.timer.start("copy_to_san")

    san_conf = conf["dirs"]["connections"]["san"]
    input_output.robust_makedirs(san_conf["user"] + "@" + san_conf["host"] + ":" + san_dir_out)
    san_file_out = os.path.join(san_dir_out, os.path.basename(tfr_file))
    with copy_guardian.BoundedSemaphore(san_conf["max_connections"], timeout=san_conf["timeout"]):
        connection = copy_guardian.Connection(
            host=san_conf["host"],
            user=san_conf["user"],
            private_key=san_conf["private_key"],
            port=san_conf["port"],
        )
        connection.rsync_to(tfr_file, san_file_out)

    LOGGER.info(f"Done copying {tfr_file} to the SAN after {LOGGER.timer.elapsed('copy_to_san')}")


def _read_full_sky_bin(conf, full_maps_file, in_map_type, z_bin):
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)

    # load the full sky maps
    LOGGER.timer.start("load_map")
    map_dir = f"map/{in_map_type}/{z_bin}"
    with h5py.File(full_maps_file, "r") as f:
        map_full = f[map_dir][:]

        # ud_grade if the stored map is at a different resolution than the analysis n_side
        if map_full.shape[0] != n_pix:
            map_full = hp.ud_grade(map_full, nside_out=n_side, order_in="RING", order_out="RING", pess=True)

    LOGGER.debug(f"Loaded {map_dir} from {full_maps_file} after {LOGGER.timer.elapsed('load_map')}")
    return map_full
