# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen

DEPRECATED legacy TFRecord utility. Active preprocessing should write WebDataset `.tar` shards.

Convert the .h5 files containing the lensing and clustering maps to scrambled .tfrecord files suitable for training
with the delta loss at the fiducial cosmology and its perturbations.

Adapted from https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/apps/run_build_tfrecords.py 
by Tomasz Kacprzak
and 
https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/map_projection/tfr_generation/fiducial.py
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/data.py
by Janis Fluri
"""

import numpy as np
import os, argparse, warnings, h5py

from numpy.random import default_rng

from msfm.utils import (
    files,
    logger,
    input_output,
    cosmogrid,
    tfrecords,
    filenames,
    parameters,
    clustering,
    scales,
    imports,
    power_spectra,
)

hp = imports.import_healpy(parallel=False)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    return dict(main_memory=1024, main_time=4, main_scratch=0, main_n_cores=4, merge_memory=2048)


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
    parser.add_argument("--n_files", type=int, default=100, help="number of .tfrecord files to produce")
    parser.add_argument(
        "--dir_in",
        type=str,
        default=b"/global/cfs/cdirs/des/cosmogrid/DESY3/fiducial",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/fiducial",
        help="legacy output root dir of the .tfrecords; active preprocessing should write WebDataset .tar shards",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration yaml file",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default="",
        help="Optional suffix to be appended to the end of the filename, for example to distinguish different runs",
    )
    parser.add_argument("--np_seed", type=int, default=7, help="random seed to shuffle the patches")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    args.config = os.path.abspath(args.config)

    return args


def main(indices, args):
    tf = tfrecords.require_tensorflow()
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")

    conf = files.load_config(args.config)

    # setup up directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "fiducial")

    # constants
    n_side = conf["analysis"]["n_side"]

    quadratic_biasing = conf["analysis"]["modelling"]["quadratic_biasing"]
    if quadratic_biasing:
        LOGGER.warning(f"Using quadratic galaxy biasing")
    else:
        LOGGER.warning(f"Using linear galaxy biasing")

    degrade_to_grf = conf["analysis"]["modelling"]["degrade_to_grf"]
    if degrade_to_grf:
        LOGGER.warning(f"Degrading the galaxy clustering maps to Gaussian Random Fields")

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo
    n_noise_per_example = conf["analysis"]["fiducial"]["n_noise_per_example"]

    data_vec_pix, _, _, _ = files.load_pixel_file()

    def data_vector_smoothing(dv, l_min, theta_fwhm, np_seed):
        # Gaussian Random Field
        if degrade_to_grf:
            dv, alm = scales.data_vector_to_grf_data_vector(
                np_seed,
                dv,
                data_vec_pix=data_vec_pix,
                n_side=n_side,
                l_min=l_min,
                theta_fwhm=theta_fwhm,
                arcmin=True,
            )

        # standard smoothing with a Gaussian kernel
        else:
            dv, alm = scales.data_vector_to_smoothed_data_vector(
                dv,
                data_vec_pix=data_vec_pix,
                n_side=n_side,
                l_min=l_min,
                theta_fwhm=theta_fwhm,
                arcmin=True,
            )

        return dv, alm

    # lensing (intrinsic alignment)
    tomo_Aia_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("Aia", conf)
    metacal_mask = files.get_tomo_dv_masks(conf)["metacal"]

    def lensing_smoothing(kg, np_seed):
        kg, alm = data_vector_smoothing(
            kg,
            conf["analysis"]["scale_cuts"]["lensing"]["l_min"],
            conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            np_seed,
        )

        return kg, alm

    def lensing_transform(kg, ia, ia_label, is_true_fiducial=False, sn_realz=None, np_seed=None):
        assert bool(not is_true_fiducial) != bool(sn_realz is not None)

        # important not to use +=, since then the array is transformed in place
        kg = kg + tomo_Aia_perts_dict[ia_label] * ia
        kg *= metacal_mask

        # only smooth the shape noise and return the alms for the fiducial, not the perturbations
        if is_true_fiducial:
            assert sn_realz is not None, "sn_realz has to be provided if is_true_fiducial is True"

            smooth_sn_realz, alm_sn_realz = [], []
            for shape_noise in sn_realz:
                shape_noise *= metacal_mask

                smooth_sn, alm_sn = lensing_smoothing(shape_noise, np_seed)

                smooth_sn_realz.append(smooth_sn)
                alm_sn_realz.append(alm_sn)

            sn_realz = np.stack(smooth_sn_realz, axis=0)
            alm_sn_realz = np.stack(alm_sn_realz, axis=0)

            # noiseless
            kg, alm_kg = lensing_smoothing(kg, np_seed)

            return kg, sn_realz, alm_kg, alm_sn_realz

        else:
            kg, _ = lensing_smoothing(kg, np_seed)

            return kg

    # clustering (linear + optionally quadratic galaxy bias)
    tomo_n_gal_maglim = tf.constant(conf["survey"]["clustering"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
    tomo_bg_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("bg", conf)

    if quadratic_biasing:
        tomo_bg2_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("bg2", conf)

    if conf["analysis"]["modelling"]["maglim_survey_systematics_map"]:
        tomo_maglim_sys_dv = files.get_clustering_systematics(conf, pixel_type="data_vector")
    else:
        tomo_maglim_sys_dv = None

    maglim_mask = files.get_tomo_dv_masks(conf)["maglim"]

    def clustering_smoothing(dg, np_seed):
        dg, alm = data_vector_smoothing(
            dg,
            conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            conf["analysis"]["scale_cuts"]["clustering"]["theta_fwhm"],
            np_seed,
        )

        return dg, alm

    def clustering_counts(dg, bg_tomo, bg2_tomo=None):
        """To focus on the function arguments that are actually varying within clustering_transform"""

        galaxy_counts = clustering.galaxy_density_to_count(
            dg,
            tomo_n_gal_maglim,
            bg_tomo,
            bg2_tomo,
            conf=conf,
            systematics_map=tomo_maglim_sys_dv,
            stochasticity=conf["analysis"]["modelling"]["galaxy_stochasticity"],
            data_vec_pix=data_vec_pix,
            mask=maglim_mask,
            np_seed=None,
        )

        return galaxy_counts

    def clustering_transform(dg, bg_label, is_true_fiducial=False, np_seed=None):
        # quadratic bias
        if quadratic_biasing:
            if bg_label == "fiducial":
                dg = clustering_counts(dg, tomo_bg_perts_dict["fiducial"], tomo_bg2_perts_dict["fiducial"])
            elif "bg_" in bg_label:
                dg = clustering_counts(dg, tomo_bg_perts_dict[bg_label], tomo_bg2_perts_dict["fiducial"])
            elif "bg2_" in bg_label:
                dg = clustering_counts(dg, tomo_bg_perts_dict["fiducial"], tomo_bg2_perts_dict[bg_label])
            else:
                raise ValueError(f"Inconsistent bias label {bg_label}")

        # linear bias
        else:
            dg = clustering_counts(dg, tomo_bg_perts_dict[bg_label])

        # only draw the Poisson noise and return the alms for the fiducial, not the perturbations
        if is_true_fiducial:
            pn_realz = clustering.galaxy_count_to_noise(dg, n_noise_per_example, np_seed=np_seed + 1)

            smooth_pn_realz, alm_pn_realz = [], []
            for pn in pn_realz:
                pn *= maglim_mask

                smooth_pn, alm_smooth_pn = clustering_smoothing(pn, np_seed)

                smooth_pn_realz.append(smooth_pn)
                alm_pn_realz.append(alm_smooth_pn)

            pn_realz = np.stack(smooth_pn_realz, axis=0)
            alm_pn_realz = np.stack(alm_pn_realz, axis=0)

            # noiseless
            dg, alm_dg = clustering_smoothing(dg, np_seed)

            return dg, pn_realz, alm_dg, alm_pn_realz

        else:
            dg, _ = clustering_smoothing(dg, np_seed)

            return dg

    # set up the paths
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]

    # remove baryon perturbations for the fiducial set
    cosmo_dirs = [cosmo_dir for cosmo_dir in cosmo_dirs if not "bary" in cosmo_dir]

    cosmo_dirs_in = [os.path.join(args.dir_in, cosmo_dir) for cosmo_dir in cosmo_dirs]
    LOGGER.debug(cosmo_dirs_in)

    n_cosmos = len(cosmo_dirs_in)
    LOGGER.info(f"Got simulation set fiducial of size {n_cosmos} with base path {args.dir_in}")

    if n_examples_per_cosmo % args.n_files == 0:
        n_examples_per_file = n_examples_per_cosmo // args.n_files
    else:
        raise ValueError(
            f"The total number of examples per cosmology {n_examples_per_cosmo}"
            f" has to be divisible by the number of files {args.n_files}"
        )
    LOGGER.info(f"n_examples_per_file = {n_examples_per_file}")

    if args.debug:
        i_examples = np.arange(6)
        LOGGER.debug(f"Debug mode, using ordered indices {i_examples}")
    else:
        # shuffle the indices
        rng = default_rng(seed=args.np_seed)
        i_examples = rng.permutation(n_examples_per_cosmo)

    # CosmoGrid perturbations in cosmological parameters
    cosmo_pert_labels = [label.split("cosmo_")[1].replace("/", "") for label in cosmo_dirs_in]
    LOGGER.info(f"There's {len(cosmo_pert_labels)} cosmological labels = {cosmo_pert_labels}")

    # separate label lists for astrophysics perturbations
    ia_pert_labels = parameters.get_fiducial_perturbation_labels(conf["analysis"]["params"]["ia"])[1:]
    LOGGER.info(f"There's {len(ia_pert_labels)} intrinsic alignment labels = {ia_pert_labels}")

    bg_params = conf["analysis"]["params"]["bg"]["linear"]
    if quadratic_biasing:
        bg_params += conf["analysis"]["params"]["bg"]["quadratic"]
    bg_pert_labels = parameters.get_fiducial_perturbation_labels(bg_params)[1:]
    LOGGER.info(f"There's {len(bg_pert_labels)} linear galaxy clustering labels = {bg_pert_labels}")

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfr_file = filenames.get_filename_tfrecords(
            args.dir_out, tag=conf["survey"]["name"] + args.file_suffix, index=index, simset="fiducial"
        )
        LOGGER.info(f"Index {index} is writing to {tfr_file}")

        # used to index examples in the .h5 files
        js = index * n_examples_per_file
        je = (index + 1) * n_examples_per_file

        n_done = 0
        with tf.io.TFRecordWriter(tfr_file) as file_writer:
            for j in LOGGER.progressbar(range(js, je), at_level="info", desc="Storing DES examples\n", total=je - js):
                if args.debug:
                    if n_done > 5:
                        LOGGER.warning("Debug mode, aborting after 5 subindices")
                        break

                # get the indices to the example
                i_example = i_examples[j]
                LOGGER.info(f"j = {j} in range({js},{je}): i_example = {i_example}")

                # loop over the perturbations in the right order
                kg_perts = []
                dg_perts = []
                ia_perts = []
                bg_perts = []
                # (2 * n_cosmos + 1,) iterations
                for cosmo_dir_in in cosmo_dirs_in:
                    file_cosmo = filenames.get_filename_data_vectors(cosmo_dir_in, with_bary=False)
                    LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")

                    (kg, ia, dg) = _load_example(file_cosmo, i_example, ["kg", "ia", "dg"])

                    # astrophysics perturbations are calculated with respect to the fiducial cosmo params
                    if "cosmo_fiducial" in cosmo_dir_in:
                        # intrinsic alignment perturbations
                        for label in ia_pert_labels:
                            ia_perts.append(lensing_transform(kg, ia, ia_label=label, np_seed=j))

                        # galaxy clustering perturbations
                        for label in bg_pert_labels:
                            bg_perts.append(clustering_transform(dg, bg_label=label, np_seed=j))

                        # load the shape noise realization
                        (sn_realz,) = _load_example(file_cosmo, i_example, ["sn"])

                        # add the signal and ia maps and smooth everything
                        kg, sn_realz, alm_kg, alm_sn_realz = lensing_transform(
                            kg, ia, ia_label="fiducial", is_true_fiducial=True, sn_realz=sn_realz, np_seed=j
                        )

                        # convert dg to galaxy number and draw the poisson noise realization
                        dg, pn_realz, alm_dg, alm_pn_realz = clustering_transform(
                            dg, bg_label="fiducial", is_true_fiducial=True, np_seed=j
                        )

                        cls = power_spectra.run_tfrecords_alm_to_cl(alm_kg, alm_sn_realz, alm_dg, alm_pn_realz)

                    # cosmological perturbations
                    else:
                        kg = lensing_transform(kg, ia, ia_label="fiducial", np_seed=j)
                        dg = clustering_transform(dg, bg_label="fiducial", np_seed=j)

                    kg_perts.append(kg)
                    dg_perts.append(dg)

                # serialize the lists of tensors of shape (n_pix, n_z_bins)
                serialized = tfrecords.parse_forward_fiducial(
                    cosmo_pert_labels,
                    kg_perts,
                    dg_perts,
                    # lensing
                    ia_pert_labels,
                    ia_perts,
                    sn_realz,
                    # clustering
                    bg_pert_labels,
                    bg_perts,
                    pn_realz,
                    # power spectra
                    cls,
                    i_example,
                ).SerializeToString()

                # check correctness
                inv_data_vectors = tfrecords.parse_inverse_fiducial(
                    serialized, cosmo_pert_labels + ia_pert_labels + bg_pert_labels, range(n_noise_per_example)
                )
                inv_kg_perts = tf.stack(
                    [inv_data_vectors[f"kg_{pert_label}"] for pert_label in cosmo_pert_labels], axis=0
                )
                inv_ia_perts = tf.stack(
                    [inv_data_vectors[f"kg_{pert_label}"] for pert_label in ia_pert_labels], axis=0
                )
                inv_dg_perts = tf.stack(
                    [inv_data_vectors[f"dg_{pert_label}"] for pert_label in cosmo_pert_labels], axis=0
                )
                inv_bg_perts = tf.stack(
                    [inv_data_vectors[f"dg_{pert_label}"] for pert_label in bg_pert_labels], axis=0
                )

                assert np.allclose(inv_kg_perts, kg_perts)
                assert np.allclose(inv_ia_perts, ia_perts)
                assert np.allclose(inv_dg_perts, dg_perts)
                assert np.allclose(inv_bg_perts, bg_perts)
                for i_noise in range(n_noise_per_example):
                    assert np.allclose(inv_data_vectors[f"sn_{i_noise}"], sn_realz[i_noise])
                    assert np.allclose(inv_data_vectors[f"pn_{i_noise}"], pn_realz[i_noise])
                assert np.allclose(inv_data_vectors["i_example"], i_example)

                inv_cls = tfrecords.parse_inverse_fiducial_cls(serialized)

                assert np.allclose(inv_cls["cls"], cls)
                assert np.allclose(inv_cls["i_example"], i_example)

                LOGGER.debug("decoded successfully")

                file_writer.write(serialized)

                n_done += 1

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def merge(indices, args):
    args = setup(args)
    conf = files.load_config(args.config)

    # for proper bookkeeping
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples = n_patches * n_perms_per_cosmo

    tfr_pattern = filenames.get_filename_tfrecords(
        args.dir_out, tag=conf["survey"]["name"] + args.file_suffix, index=None, simset="fiducial", return_pattern=True
    )

    cls_dset = tf.data.Dataset.list_files(tfr_pattern)
    cls_dset = cls_dset.interleave(tf.data.TFRecordDataset, cycle_length=16, block_length=1)
    # the default arguments for parse_inverse_fiducial_cls are fine since we're not in graph mode
    cls_dset = cls_dset.map(tfrecords.parse_inverse_fiducial_cls)

    cls = []
    i_examples = []
    for example in LOGGER.progressbar(
        cls_dset, total=n_examples, desc="Looping through the .tfrecords", at_level="info"
    ):
        cls.append(example["cls"].numpy())
        i_examples.append(int(example["i_example"]))

    # noise realizations
    n_noise = example["cls"].numpy().shape[0]
    i_noise = np.arange(n_noise)
    i_noise = np.tile(i_noise, n_examples)

    # concatenate the different simulation runs and noise realizations along the same axis
    # cls.shape[0] = n_examples * n_noise
    cls = np.concatenate(cls, axis=0)

    # i_examples.shape[0] = n_examples
    i_examples = np.array(i_examples)
    i_examples = np.repeat(i_examples, n_noise, axis=0)

    # sort by example index
    i_sort = np.argsort(i_examples)
    cls = cls[i_sort, ...]
    i_examples = i_examples[i_sort]
    i_noise = i_noise[i_sort]

    # perform the binning (all examples at the same time)
    binned_cls, bin_edges = power_spectra.smooth_and_bin_cls(
        cls,
        l_mins_smoothing=conf["analysis"]["scale_cuts"]["lensing"]["l_min"]
        + conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
        l_maxs_smoothing=conf["analysis"]["scale_cuts"]["lensing"]["l_max"]
        + conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
        n_bins=conf["analysis"]["power_spectra"]["n_bins"],
        with_cross=True,
    )

    # separate folder on the same level as tfrecords
    if args.debug:
        out_dir = args.dir_out
    else:
        out_dir = os.path.join(args.dir_out, "../../cls")
    os.makedirs(out_dir, exist_ok=True)

    LOGGER.info(f"Saving the results in {out_dir}")
    with h5py.File(os.path.join(out_dir, "fiducial_cls.h5"), "w") as f:
        f.create_dataset("cls/raw", data=cls)
        f.create_dataset("cls/binned", data=binned_cls)
        f.create_dataset("cls/bin_edges", data=bin_edges)
        f.create_dataset("i_example", data=i_examples)
        f.create_dataset("i_noise", data=i_noise)

    LOGGER.info(f"Done with merging of the fiducial power spectra")


def _load_example(filename, i_example, map_labels):
    with h5py.File(filename, "r") as f:
        maps = []
        for map_label in map_labels:
            # shape (n_examples_per_cosmo, n_pix, n_z_bins) before the indexing
            maps.append(f[map_label][i_example, ...])

    return maps
