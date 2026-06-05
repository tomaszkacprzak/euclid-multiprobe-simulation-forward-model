# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen

Convert the .h5 files containing the lensing and clustering maps to ordered .tfrecord files suitable for evaluation
of summary statistics.

Adapted from https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/apps/run_build_tfrecords.py 
by Tomasz Kacprzak
and 
https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/map_projection/tfr_generation/fiducial.py
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/data.py
by Janis Fluri
"""

import numpy as np
import tensorflow as tf
import os, argparse, warnings, h5py

from sobol_seq import i4_sobol

from msfm.utils import (
    files,
    lensing,
    logger,
    input_output,
    cosmogrid,
    tfrecords,
    parameters,
    filenames,
    redshift,
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
    return dict(main_memory=1024, main_time=4, main_scratch=0, main_n_cores=4, merge_memory=16384, merge_n_cores=8)


def setup(args):
    description = "Make .tfrecord files from the survey footprints collected in the .h5 files."
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
        default=b"/global/cfs/cdirs/des/cosmogrid/DESY3/grid",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/grid",
        help="output root dir of the .tfrecords",
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
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    conf = files.load_config(args.config)

    # setup up directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")
    LOGGER.info(f"Loaded meta information")

    # constants
    n_side = conf["analysis"]["n_side"]
    # get all of the parameters, the only distinction in here is between the linear and quadratic bias
    params = parameters.get_parameters(conf=conf)
    sobol_priors = parameters.get_prior_intervals(params, conf=conf)
    z0 = conf["analysis"]["modelling"]["z0"]

    quadratic_biasing = conf["analysis"]["modelling"]["quadratic_biasing"]
    if quadratic_biasing:
        LOGGER.warning(f"Using quadratic galaxy biasing")
    else:
        LOGGER.warning(f"Using linear galaxy biasing")

    degrade_to_grf = conf["analysis"]["modelling"]["degrade_to_grf"]
    if degrade_to_grf:
        LOGGER.warning(f"Degrading the galaxy clustering maps to Gaussian Random Fields")

        if n_noise_per_example > 1:
            LOGGER.warning(f"For the GRF, the different noise realizations und up being identical")

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"]["grid"]["n_noise_per_example"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example

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
    tomo_z_metacal, tomo_nz_metacal = files.load_redshift_distributions("metacal", conf)
    m_bias_dist = lensing.get_m_bias_distribution(conf)
    metacal_mask = files.get_tomo_dv_masks(conf)["metacal"]

    def lensing_smoothing(kg, np_seed):
        kg, alm = data_vector_smoothing(
            kg,
            conf["analysis"]["scale_cuts"]["lensing"]["l_min"],
            conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            np_seed,
        )

        return kg, alm

    def lensing_transform(kg, ia, sn_realz, Aia, n_Aia, np_seed=None):
        # intrinsic alignment
        tomo_Aia = redshift.get_tomo_amplitudes(Aia, n_Aia, tomo_z_metacal, tomo_nz_metacal, z0)
        LOGGER.debug(f"Per z bin Aia = {tomo_Aia}")

        kg = kg + tomo_Aia * ia

        # fixing this in the .tfrecords simplifies reproducibility
        m_bias = m_bias_dist.sample()
        kg *= 1.0 + m_bias

        kg *= metacal_mask
        kg, alm_kg = lensing_smoothing(kg, np_seed)

        smooth_sn_realz, alm_sn_realz = [], []
        for shape_noise in sn_realz:
            shape_noise *= maglim_mask

            smooth_sn, alm_sn = lensing_smoothing(shape_noise, np_seed)

            smooth_sn_realz.append(smooth_sn)
            alm_sn_realz.append(alm_sn)

        sn_realz = np.stack(smooth_sn_realz, axis=0)
        alm_sn_realz = np.stack(alm_sn_realz, axis=0)

        return kg, sn_realz, alm_kg, alm_sn_realz

    # clustering (linear galaxy bias)
    tomo_z_maglim, tomo_nz_maglim = files.load_redshift_distributions("maglim", conf)
    tomo_n_gal_maglim = np.array(conf["survey"]["clustering"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

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

    def clustering_transform(dg, bg, n_bg, bg2=None, n_bg2=None, np_seed=None):
        assert (not quadratic_biasing and (bg2 is None) and (n_bg2 is None)) or (
            quadratic_biasing and (bg2 is not None) and (n_bg2 is not None)
        ), f"The galaxy biasing setup must be consistent"

        # the linear galaxy bias is needed in both cases
        tomo_bg = redshift.get_tomo_amplitudes(bg, n_bg, tomo_z_maglim, tomo_nz_maglim, z0)
        LOGGER.debug(f"Per z bin linear bg = {tomo_bg}")

        # quadratic bias
        if quadratic_biasing:
            tomo_bg2 = redshift.get_tomo_amplitudes(bg2, n_bg2, tomo_z_maglim, tomo_nz_maglim, z0)
            LOGGER.debug(f"Per z bin quadratic bg2 = {tomo_bg2}")

            dg = clustering.galaxy_density_to_count(
                dg,
                tomo_n_gal_maglim,
                tomo_bg,
                tomo_bg2,
                conf=conf,
                stochasticity=conf["analysis"]["modelling"]["galaxy_stochasticity"],
                data_vec_pix=data_vec_pix,
                systematics_map=tomo_maglim_sys_dv,
                mask=maglim_mask,
                np_seed=np_seed + 1,
            )

        # linear bias
        else:
            dg = clustering.galaxy_density_to_count(
                dg,
                tomo_n_gal_maglim,
                tomo_bg,
                conf=conf,
                stochasticity=conf["analysis"]["modelling"]["galaxy_stochasticity"],
                data_vec_pix=data_vec_pix,
                systematics_map=tomo_maglim_sys_dv,
                mask=maglim_mask,
                np_seed=np_seed + 1,
            )

        # draw noise, mask, smooth
        pn_realz = clustering.galaxy_count_to_noise(dg, n_noise_per_example, np_seed=np_seed + 2)

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

        # shapes (n_pix, n_z_maglim), (n_noise_per_example, n_pix, n_z_maglim)
        # (n_noise_per_example, )
        return dg, pn_realz, alm_dg, alm_pn_realz

    # set up the paths
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]

    cosmo_dirs_in = [os.path.join(args.dir_in, cosmo_dir) for cosmo_dir in cosmo_dirs]
    n_cosmos = len(cosmo_dirs_in)
    LOGGER.info(f"Got simulation set grid of size {n_cosmos} with base path {args.dir_in}")

    # configure file structure
    if n_cosmos % args.n_files == 0:
        n_cosmos_per_file = n_cosmos // args.n_files
        LOGGER.info(f"The number of files implies {n_cosmos_per_file} cosmological parameters per file")
    else:
        raise ValueError(
            f"The total number of cosmologies {n_cosmos} has to be divisible by the number of files {args.n_files}"
        )
    n_examples_per_file = n_examples_per_cosmo * n_cosmos_per_file

    LOGGER.info(
        f"In total, there are n_examples_per_cosmo * n_cosmos_per_file = {n_examples_per_cosmo} * {n_cosmos_per_file}"
        f" = {n_examples_per_file} examples per file"
    )

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfr_file = filenames.get_filename_tfrecords(
            args.dir_out, tag=conf["survey"]["name"] + args.file_suffix, index=index, simset="grid"
        )
        LOGGER.info(f"Index {index} is writing to {tfr_file}")

        # index for the cosmological parameters
        js = index * n_cosmos_per_file
        je = (index + 1) * n_cosmos_per_file
        LOGGER.info(f"And includes {cosmo_dirs[js : je]}")

        n_done = 0
        with tf.io.TFRecordWriter(tfr_file) as file_writer:
            # loop over the cosmological parameters
            for i_cosmo, cosmo_dir_in in LOGGER.progressbar(
                zip(range(js, je), cosmo_dirs_in[js:je]),
                at_level="info",
                desc="Looping through cosmologies",
                total=je - js,
            ):
                LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")

                if args.debug and n_done > 5:
                    LOGGER.warning("Debug mode, aborting after 5 subindices")
                    break

                # select the relevant cosmological parameters
                cosmo = [
                    cosmo_params_info[cosmo_param][i_cosmo] for cosmo_param in conf["analysis"]["params"]["cosmo"]
                ]
                cosmo = np.array(cosmo, dtype=np.float32)

                i_sobol = cosmo_params_info["sobol_index"][i_cosmo]

                # extend the Sobol sequence by astrophysical parameters
                sobol_point, _ = i4_sobol(sobol_priors.shape[0], i_sobol)
                sobol_params = sobol_point * np.squeeze(np.diff(sobol_priors)) + sobol_priors[:, 0]
                sobol_params = sobol_params.astype(np.float32)

                # add these to the label, the parameters are ordered as in sobol_priors
                Aia = sobol_params[6]
                n_Aia = sobol_params[7]
                bg = sobol_params[8]
                n_bg = sobol_params[9]
                cosmo = np.concatenate((cosmo, np.array([Aia, n_Aia, bg, n_bg])))

                # additional parameters for quadratic galaxy biasing
                if quadratic_biasing:
                    bg2 = sobol_params[10]
                    n_bg2 = sobol_params[11]
                    cosmo = np.concatenate((cosmo, np.array([bg2, n_bg2])))
                else:
                    bg2 = None
                    n_bg2 = None

                # redshift evolution, only calculate the integrals once here
                current_lensing_transform = lambda kg, ia, sn_realz, np_seed: lensing_transform(
                    kg, ia, sn_realz, Aia, n_Aia, np_seed
                )
                current_clustering_transform = lambda dg, np_seed: clustering_transform(
                    dg, bg, n_bg, bg2, n_bg2, np_seed
                )

                # verify that the Sobol sequences (stored and newly generated) are identical for the cosmo params
                assert np.allclose(sobol_params[0], cosmo[0], rtol=1e-3, atol=1e-5)  # Om
                assert np.allclose(sobol_params[1], cosmo[1], rtol=1e-3, atol=1e-5)  # s8
                assert np.allclose(sobol_params[2], cosmo[2], rtol=1e-3, atol=1e-3)  # Ob
                assert np.allclose(sobol_params[3], cosmo[3], rtol=1e-3, atol=1e-5)  # H0
                assert np.allclose(sobol_params[4], cosmo[4], rtol=1e-3, atol=1e-5)  # ns
                assert np.allclose(sobol_params[5], cosmo[5], rtol=1e-3, atol=1e-5)  # w0

                # load the .h5 files
                file_cosmo = filenames.get_filename_data_vectors(cosmo_dir_in, with_bary=False)

                kg_examples, ia_examples, sn_examples, dg_examples = _load_data_vecs(file_cosmo)

                # loop over the n_examples_per_cosmo
                for i_example, (kg, ia, sn_realz, dg) in LOGGER.progressbar(
                    enumerate(zip(kg_examples, ia_examples, sn_examples, dg_examples)),
                    at_level="info",
                    desc="Looping through the per cosmology examples",
                    total=n_examples_per_cosmo // n_noise_per_example,
                ):
                    # maps
                    kg, sn_realz, alm_kg, alm_sn_realz = current_lensing_transform(
                        kg, ia, sn_realz, np_seed=i_sobol + i_example
                    )
                    dg, pn_realz, alm_dg, alm_pn_realz = current_clustering_transform(dg, np_seed=i_sobol + i_example)

                    # power spectra
                    cls = power_spectra.run_tfrecords_alm_to_cl(alm_kg, alm_sn_realz, alm_dg, alm_pn_realz)

                    serialized = tfrecords.parse_forward_grid(
                        kg, sn_realz, dg, pn_realz, cls, cosmo, i_sobol, i_example
                    ).SerializeToString()

                    # check correctness
                    inv_maps = tfrecords.parse_inverse_grid(serialized, range(n_noise_per_example))

                    for i_noise in range(n_noise_per_example):
                        assert np.allclose(inv_maps[f"kg_{i_noise}"], kg + sn_realz[i_noise])
                        assert np.allclose(inv_maps[f"dg_{i_noise}"], dg + pn_realz[i_noise])
                    assert np.allclose(inv_maps["cosmo"], cosmo)
                    assert np.allclose(inv_maps["i_sobol"], i_sobol)
                    assert np.allclose(inv_maps["i_example"], i_example)

                    inv_cls = tfrecords.parse_inverse_grid_cls(serialized)

                    assert np.allclose(inv_cls["cls"], cls)
                    assert np.allclose(inv_cls["cosmo"], cosmo)
                    assert np.allclose(inv_cls["i_sobol"], i_sobol)
                    assert np.allclose(inv_cls["i_example"], i_example)

                    LOGGER.debug("decoded successfully")

                    file_writer.write(serialized)

                n_done += 1

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def merge(indices, args):
    args = setup(args)
    conf = files.load_config(args.config)

    n_cosmos = conf["analysis"]["grid"]["n_cosmos"]
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"]["grid"]["n_noise_per_example"]
    n_signal_per_cosmo = n_patches * n_perms_per_cosmo

    tfr_pattern = filenames.get_filename_tfrecords(
        args.dir_out, tag=conf["survey"]["name"] + args.file_suffix, index=None, simset="grid", return_pattern=True
    )

    cls_dset = tf.data.Dataset.list_files(tfr_pattern)
    # cycle_length = num_readers = 1 to not mix cosmologies
    cls_dset = cls_dset.interleave(tf.data.TFRecordDataset, cycle_length=1, block_length=1)
    # the default arguments for parse_inverse_fiducial_cls are fine since we're not in graph mode
    cls_dset = cls_dset.map(tfrecords.parse_inverse_grid_cls, num_parallel_calls=tf.data.AUTOTUNE)
    # every batch is a single cosmology
    cls_dset = cls_dset.batch(n_signal_per_cosmo)

    cls = []
    binned_cls = []
    bin_edges = []
    cosmos = []
    i_sobols = []
    i_examples = []
    i_noises = []
    for example in LOGGER.progressbar(
        cls_dset, total=n_cosmos, desc="Looping through the different cosmologies in the .tfrecords", at_level="info"
    ):
        cl = example["cls"].numpy()
        cosmo = example["cosmo"].numpy()
        i_sobol = example["i_sobol"].numpy()
        i_example = example["i_example"].numpy()

        # concatenate the noise realizations along the same axis as the examples
        cl = np.concatenate([cl[:, i, ...] for i in range(cl.shape[1])], axis=0)

        # perform the binning (all examples of a single cosmology at once)
        binned_cl, bin_edge = power_spectra.smooth_and_bin_cls(
            cl,
            l_mins_smoothing=conf["analysis"]["scale_cuts"]["lensing"]["l_min"]
            + conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            l_maxs_smoothing=conf["analysis"]["scale_cuts"]["lensing"]["l_max"]
            + conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
            n_bins=conf["analysis"]["power_spectra"]["n_bins"],
            with_cross=True,
        )

        # tiling has the same form as the above concatenation
        cosmo = np.tile(cosmo, (n_noise_per_example, 1))
        i_sobol = np.tile(i_sobol, n_noise_per_example)
        i_example = np.tile(i_example, n_noise_per_example)

        # noise is treated separately because it's along a separate dimension in the .tfrecords. This here is preserves
        # the order imposed above in power_spectrum = ...
        i_noise = np.arange(n_noise_per_example)
        i_noise = np.repeat(i_noise, n_signal_per_cosmo)

        cls.append(cl)
        binned_cls.append(binned_cl)
        bin_edges.append(bin_edge)
        cosmos.append(cosmo)
        i_sobols.append(i_sobol)
        i_examples.append(i_example)
        i_noises.append(i_noise)

    # results
    cls = np.stack(cls, axis=0)
    binned_cls = np.stack(binned_cls, axis=0)
    bin_edges = np.stack(bin_edges, axis=0)
    cosmos = np.stack(cosmos, axis=0)
    i_sobols = np.array(i_sobols)
    i_examples = np.array(i_examples)
    i_noises = np.array(i_noises)

    # separate folder on the same level as tfrecords
    if args.debug:
        out_dir = args.dir_out
    else:
        out_dir = os.path.join(args.dir_out, "../../cls")
    os.makedirs(out_dir, exist_ok=True)

    LOGGER.info(f"Saving the results in {out_dir}")
    with h5py.File(os.path.join(out_dir, "grid_cls.h5"), "w") as f:
        f.create_dataset("cls/raw", data=cls)
        f.create_dataset("cls/binned", data=binned_cls)
        f.create_dataset("cls/bin_edges", data=bin_edges)
        f.create_dataset("cosmo", data=cosmos)
        f.create_dataset("i_sobol", data=i_sobols)
        f.create_dataset("i_example", data=i_examples)
        f.create_dataset("i_noise", data=i_noises)

    LOGGER.info(f"Done with merging of the grid power spectra")


def _load_data_vecs(filename):
    with h5py.File(filename, "r") as f:
        # shape (n_examples_per_cosmo, n_pix, n_z_bins) before the indexing
        kg = f["kg"][:]
        ia = f["ia"][:]
        sn_realz = f["sn"][:]
        dg = f["dg"][:]

    LOGGER.debug(f"Successfully loaded the data vectors")
    return kg, ia, sn_realz, dg
