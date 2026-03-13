# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created September 2022
Author: Arne Thomsen

Transform the full sky weak lensing signal and intrinsic alignment maps into multiple survey footprint cut-outs,
both for the fiducial and the grid cosmology

Meant for Euler (CPU nodes, local scratch)

TODO clean this up as there is way too much nesting.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os, argparse, warnings, h5py, time, yaml, copy_guardian

from msfm.utils import files, lensing, logger, input_output, maps, cosmogrid, clustering, imports, filenames

hp = imports.import_healpy(parallel=True)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    args = setup(args)

    resources = {
        "main_memory": 2048,
        "main_time": 4,
        "main_n_cores": 4,
    }

    if args.from_san:
        # in MB. One projected_probes_maps_v11dmb.h5 should be around 1 GB
        resources["main_scratch"] = 2000
    else:
        resources["main_scratch"] = 0

    return resources


def setup(args):
    description = "Make weak lensing and intrinsic alignment datavectors from full sky maps"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument(
        "--simset", type=str, default="grid", choices=("grid", "fiducial"), help="set of simulations to use"
    )
    parser.add_argument(
        "--dir_in",
        type=str,
        default="/global/cfs/cdirs/des/cosmogrid/DESY3/grid",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/grid",
        help="output root dir of the simulations",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration yaml file",
    )
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=120,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    parser.add_argument(
        "--from_san",
        action="store_true",
        help="copy the CosmoGrid files from the SAN instead of accessing them locally",
    )
    parser.add_argument(
        "--cosmogrid_version", type=str, default="1.1", choices=["1.1", "1"], help="version of the input CosmoGrid"
    )
    parser.add_argument("--with_bary", action="store_true", help="whether to include the baryonification in the input")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")
    parser.add_argument("--store_counts", action="store_true", help="whether to store the metacal galaxy count maps")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    if args.from_san:
        LOGGER.warning(f"Copying the CosmoGrid directly from the SAN")

    args.config = os.path.abspath(args.config)

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("!!! debug mode !!!")

    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    # args.config is a string to the file
    conf = files.load_config(args.config)

    # save the config
    with open(os.path.join(args.dir_out, "config.yaml"), "w") as f:
        yaml.dump(conf, f)

    # general constants
    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]

    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"][args.simset]["n_perms_per_cosmo"]
    n_noise_per_signal = conf["analysis"][args.simset]["n_noise_per_signal"]
    LOGGER.info(f"Looping through {n_perms_per_cosmo} permutations per cosmological parameter set")
    LOGGER.info(f"Generating {n_noise_per_signal} noise realizations per signal realization")

    degrade_to_grf = conf["analysis"]["modelling"]["degrade_to_grf"]
    if degrade_to_grf:
        LOGGER.warning(f"Degrading the weak lensing maps to Gaussian Random Fields")

    # metacal, TODO this is only a placeholder bias
    tomo_bias_metacal = conf["survey"]["metacal"]["galaxy_bias"]
    tomo_n_gal_metacal = np.array(conf["survey"]["metacal"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

    # alm
    kappa2gamma_fac, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(3 * n_side - 1)

    # pixel file
    data_vec_pix, patches_pix_dict, corresponding_pix_dict, gamma2_signs = files.load_pixel_file(conf)
    data_vec_len = len(data_vec_pix)

    # noise file
    tomo_gamma_cat, _ = files.load_noise_file(conf)

    # set up general directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # set up CosmoGrid directories
    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, args.simset)
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]

    # remove baryon perturbations for the fiducial set
    if args.simset == "fiducial":
        if args.with_bary:
            cosmo_dirs = [cosmo_dir for cosmo_dir in cosmo_dirs]
            LOGGER.info(f"Using the baryonified inputs, then there's {len(cosmo_dirs) - 1} fiducial perturbations")
        else:
            cosmo_dirs = [cosmo_dir for cosmo_dir in cosmo_dirs if not "bary" in cosmo_dir]
            LOGGER.info(
                f"Using the dark matter only inputs, then there's {len(cosmo_dirs) - 1} fiducial perturbations"
            )

    cosmo_dirs_in = [os.path.join(args.dir_in, args.simset, cosmo_dir) for cosmo_dir in cosmo_dirs]
    cosmo_dirs_out = [os.path.join(args.dir_out, args.simset, cosmo_dir) for cosmo_dir in cosmo_dirs]

    n_cosmos = len(cosmo_dirs_in)
    LOGGER.info(f"Got simulation set {args.simset} of size {n_cosmos} with base path {args.dir_in}")

    # index corresponds to a cosmological parameter (either on the grid or for the fiducial perturbations) ############
    for index in indices:
        LOGGER.timer.start("index")

        cosmo_dir_in = cosmo_dirs_in[index]
        cosmo_dir_out = cosmo_dirs_out[index]
        if not os.path.isdir(cosmo_dir_out):
            input_output.robust_makedirs(cosmo_dir_out)
        data_vec_file = filenames.get_filename_data_vectors(cosmo_dir_out, with_bary=args.with_bary)
        LOGGER.info(f"Index {index} takes input from {cosmo_dir_in} and writes to {data_vec_file}")

        for i_perm in LOGGER.progressbar(range(n_perms_per_cosmo), desc="Loop over permutations\n", at_level="info"):
            if args.debug and i_perm > 5:
                LOGGER.warning("Debug mode, aborting after 5 permutations")
                break

            LOGGER.timer.start("permutation")
            LOGGER.info(f"Starting simulation permutation {i_perm:04d}")

            # prepare the full sky input file
            perm_dir_in = os.path.join(cosmo_dir_in, f"perm_{i_perm:04d}")
            full_maps_file = filenames.get_filename_full_maps(
                perm_dir_in, with_bary=args.with_bary, version=args.cosmogrid_version
            )

            if args.from_san:
                san_conf = conf["dirs"]["connections"]["san"]
                local_scratch_dir = os.environ["TMPDIR"]

                t0_rsync = time.time()
                with copy_guardian.BoundedSemaphore(san_conf["max_connections"], timeout=san_conf["timeout"]):
                    connection = copy_guardian.Connection(
                        host=san_conf["host"], user=san_conf["user"], private_key=san_conf["private_key"], port=22
                    )
                    # overwrite the local scratch file within the loop iterations
                    connection.rsync_from(full_maps_file, local_scratch_dir)

                tdelta_rsync = time.time() - t0_rsync
                LOGGER.info(f"Rsynced {full_maps_file} to {local_scratch_dir} after {tdelta_rsync:.2f}s")
                assert (
                    tdelta_rsync > 1
                ), f"Rsync took only {tdelta_rsync:.2f}s, which indicates that nothing was actually transferred"
                full_maps_file = filenames.get_filename_full_maps(
                    local_scratch_dir, with_bary=args.with_bary, version=args.cosmogrid_version
                )

            # output containers, one for each permutation, in NEST ordering with padding
            data_vec_container = {}

            for sample, probe in zip(["metacal", "maglim"], ["lensing", "clustering"]):
                LOGGER.info(f"Starting with sample {sample}")
                LOGGER.timer.start("sample")

                # constants
                z_bins = conf["survey"][sample]["z_bins"]
                n_z_bins = len(z_bins)

                # map types
                in_map_types = conf["survey"][sample]["map_types"]["input"]
                out_map_types = conf["survey"][sample]["map_types"]["output"]

                # pixel indices
                all_patches_pix = patches_pix_dict[sample]
                all_corresponding_pix = corresponding_pix_dict[sample]

                for in_map_type, out_map_type in zip(in_map_types, out_map_types):
                    # some fiducial perturbations are skipped for lensing
                    # if ("delta" in cosmo_dir_in) and (sample == "metacal") and (in_map_type in ["ia", "dg"]):
                    if ("delta" in cosmo_dir_in) and (sample == "metacal") and (in_map_type == "dg"):
                        LOGGER.info(f"Skipping input map type {in_map_type} for this perturbation")
                        continue

                    LOGGER.info(f"Starting with input map type {in_map_type}")
                    LOGGER.timer.start("map_type")

                    # array content of the output container
                    if out_map_type in ["kg", "ia", "dg"]:
                        dvs_shape = (n_patches, data_vec_len, n_z_bins)

                    elif out_map_type == "sn":
                        dvs_shape = (n_patches, n_noise_per_signal, data_vec_len, n_z_bins)

                        if args.store_counts:
                            data_vec_container["ct"] = np.zeros((n_patches, data_vec_len, n_z_bins), dtype=np.int16)

                    data_vec_container[out_map_type] = np.zeros(dvs_shape, dtype=np.float32)

                    for i_z, z_bin in enumerate(z_bins):
                        # load the full sky maps
                        map_dir = f"{in_map_type}/{z_bin}"
                        with h5py.File(full_maps_file, "r") as f:
                            map_full = f[map_dir][:]

                            # to convert from nside 512 to 1024
                            if map_full.shape[0] != n_pix:
                                map_full = hp.ud_grade(
                                    map_full, nside_out=n_side, order_in="RING", order_out="RING", pess=False
                                )

                        LOGGER.debug(f"Loaded {map_dir} from {full_maps_file}")

                        # lensing, metacal sample #####################################################################
                        if sample == "metacal":
                            # only consider this tomographic bin
                            patches_pix = all_patches_pix[i_z]
                            corresponding_pix = all_corresponding_pix[i_z]
                            base_patch_pix = patches_pix[0]

                            if in_map_type in ["kg", "ia"]:
                                kappa_full = map_full

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

                                for i_patch, patch_pix in enumerate(patches_pix):
                                    # The 90° rots do NOT change the shear, however, the mirroring does,
                                    # therefore we have to swap sign of gamma2 for the last 2 patches!
                                    gamma2_sign = gamma2_signs[i_patch]
                                    LOGGER.debug(f"Using gamma2 sign {gamma2_sign} for patch index {i_patch}")

                                    # TODO do each patch multiple times

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

                                    data_vec_container[out_map_type][i_patch, :, i_z] = kappa_dv

                            elif in_map_type == "dg":
                                delta_full = map_full

                                # only consider this tomographic bin
                                n_bar = tomo_n_gal_metacal[i_z]
                                bias = tomo_bias_metacal[i_z]
                                gamma_cat = tomo_gamma_cat[i_z]

                                # create joint distribution, as this is faster than random indexing
                                gamma_abs = tf.math.abs(gamma_cat[:, 0] + 1j * gamma_cat[:, 1])
                                w = gamma_cat[:, 2]
                                cat_dist = tfp.distributions.Empirical(
                                    samples=tf.stack([gamma_abs, w], axis=-1), event_ndims=1
                                )

                                # normalize to number density contrast
                                delta_full = (delta_full - np.mean(delta_full)) / np.mean(delta_full)

                                # number of galaxies per pixel
                                counts_full = clustering.galaxy_density_to_count(
                                    delta_full, n_bar, bias, conf=conf, systematics_map=None
                                ).astype(int)

                                for i_patch, patch_pix in enumerate(patches_pix):
                                    # not a full healpy map, just the patch with no zeros
                                    counts = counts_full[patch_pix]

                                    # vectorized sampling, shape (len(counts), n_noise_per_signal)
                                    gamma1, gamma2 = lensing.noise_gen(counts, cat_dist, n_noise_per_signal)

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

                                        data_vec_container[out_map_type][i_patch, i_noise, :, i_z] = kappa_dv

                                    if args.store_counts:
                                        # correct cut out procedure involves a full sky map
                                        counts_patch_map = np.zeros(n_pix, dtype=np.int16)
                                        counts_patch_map[base_patch_pix] = counts
                                        counts_dv = maps.map_to_data_vec(
                                            counts_patch_map, data_vec_len, corresponding_pix, base_patch_pix
                                        )
                                        data_vec_container["ct"][i_patch, :, i_z] = counts_dv

                        # clustering, maglim sample ###################################################################
                        elif sample == "maglim":
                            # here, the mask for the tomographic bins is shared
                            patches_pix = all_patches_pix
                            corresponding_pix = all_corresponding_pix
                            base_patch_pix = patches_pix[0]

                            delta_full = map_full

                            for i_patch, patch_pix in enumerate(patches_pix):
                                # always populate the same patch
                                delta_patch = np.zeros(n_pix, dtype=np.float32)
                                delta_patch[base_patch_pix] = delta_full[patch_pix]

                                # cut out padded data vector
                                delta_dv = maps.map_to_data_vec(
                                    delta_patch, data_vec_len, corresponding_pix, base_patch_pix, divide_by_mean=True
                                )

                                data_vec_container[out_map_type][i_patch, :, i_z] = delta_dv

                        else:
                            raise ValueError

                    LOGGER.info(f"Done with map type {out_map_type} after {LOGGER.timer.elapsed('map_type')}")

            # save the results
            _save_output_container(
                conf,
                data_vec_file,
                data_vec_container,
                i_perm,
                n_perms_per_cosmo,
                n_patches,
                n_noise_per_signal,
                data_vec_len,
            )

            LOGGER.info(f"Done with permutation {i_perm:04d} after {LOGGER.timer.elapsed('permutation')}")

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def _save_output_container(
    conf, filename, container, i_perm, n_perms_per_cosmo, n_patches, n_noise_per_signal, output_len
):
    """Saves an .h5 file collecting all results on the level of the cosmological parameters (so for different
    permutations/runs and patches)

    Args:
        filename (str): path to the output .h5 file
        output_container (dict): Dictionary of arrays of shape (n_patches, output_len, n_z_bins)
        i_perm (int): Index of the permutation
        n_perms_per_cosmo (int): Number of permutations/simulation runs per cosmology
        n_patches (int): Number of cut outs from the full sky
        output_len (int): Length of the (padded) data vector
        n_z_bins (int): Number of tomographic bins
    """

    with h5py.File(filename, "a") as f:
        for map_type in container.keys():
            # set the number of redshift bins
            if map_type in conf["survey"]["metacal"]["map_types"]["output"]:
                n_z_bins = len(conf["survey"]["metacal"]["z_bins"])
            elif map_type in conf["survey"]["maglim"]["map_types"]["output"]:
                n_z_bins = len(conf["survey"]["maglim"]["z_bins"])

            # there's multiple shape noise realizations
            if map_type == "sn":
                out_shape = (n_perms_per_cosmo * n_patches, n_noise_per_signal, output_len, n_z_bins)
            else:
                out_shape = (n_perms_per_cosmo * n_patches, output_len, n_z_bins)

            # create dataset for every parameter level directory, collecting the permutation levels
            try:
                f.create_dataset(name=map_type, shape=out_shape)
            except ValueError:
                LOGGER.debug(f"dataset {map_type} already exists in {filename}")

            f[map_type][n_patches * i_perm : n_patches * (i_perm + 1)] = container[map_type]

    LOGGER.info(f"Stored {filename}")
