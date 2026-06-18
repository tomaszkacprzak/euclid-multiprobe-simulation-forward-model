# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2023
Author: Arne Thomsen

Evaluate the peaks from the WebDataset .tar shards produced by the forward model pipelines. Based off
https://github.com/des-science/y3-combined-peaks/blob/main/combined_peaks/weak_lensing/peaks/MPI_WL_peaks_fiducial.py
by Virginia Ajani and run_power_spectra.py from this repo.

Before running this, create the binning scheme with notebooks/peaks/binning.ipynb.
"""

import numpy as np
import os, argparse, warnings, h5py, glob, time

from msfm import fiducial_pipeline, grid_pipeline
from msfm.utils import files, logger, input_output, imports, maps, peak_statistics

hp = imports.import_healpy(parallel=False)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def get_tasks(args):
    """Returns a list of task indices to be executed by the workers. Each index corresponds to a single WebDataset .tar shard"""
    args = setup(args)

    # associate each file with a task
    shards = sorted(glob.glob(args.wds_pattern))
    indices = list(range(len(shards)))

    # only read three files in debug mode
    if args.debug:
        indices = indices[:3]

    LOGGER.warning(f"Found {len(shards)} files, running on the first {len(indices)} indices")

    return indices


def resources(args):
    args = setup(args)

    # memory: typically, the total memory consumption is around 2 GB per task
    # CPU: with 8 cores per task, the total CPU utilization is around 70%
    # time: this is meant for 2 smoothing scales

    if args.simset == "fiducial":
        resource_dict = dict(main_memory=512, main_time=4, main_scratch=0, main_n_cores=8)
    elif args.simset == "grid":
        # when there are 2500 WebDataset .tar shards, such that each only contains a single cosmology, the 4h timeframe fits easily
        resource_dict = dict(main_memory=512, main_time=4, main_scratch=0, main_n_cores=8)

    return resource_dict


def setup(args):
    description = "evaluate peak statistics from the input WebDataset pipelines"
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
        "--wds_pattern",
        type=str,
        required=True,
        help="input root dir of the WebDataset shards to construct the dataset",
    )
    parser.add_argument(
        "--simset", type=str, default="grid", choices=("grid", "fiducial"), help="set of simulations to use"
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/grid",
        help="output root dir for peak statistic .h5 files",
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
        default=60,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode, then only run on 5 indices")

    args, _ = parser.parse_known_args(args)

    if args.wds_pattern is None:
        if args.deprecated_tfr_pattern is None:
            parser.error("one of --wds_pattern/--pattern or deprecated --tfr_pattern is required")
        LOGGER.warning("--tfr_pattern is deprecated; use --wds_pattern or --pattern for WebDataset .tar shards")
        args.wds_pattern = args.deprecated_tfr_pattern
    elif args.deprecated_tfr_pattern is not None:
        parser.error("provide only one of --wds_pattern/--pattern or deprecated --tfr_pattern")

    delattr(args, "deprecated_tfr_pattern")

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    assert args.simset in args.wds_pattern

    return args


def main(indices, args):
    args = setup(args)
    shards = sorted(glob.glob(args.wds_pattern))

    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("!!! debug mode !!!")

    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    conf = files.load_config(args.config)

    # setup up directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    binning_file = os.path.join(repo_dir, conf["files"]["peak_binning"])

    # constants
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    n_z_bins = len(conf["survey"]["lensing"]["z_bins"]) + len(conf["survey"]["clustering"]["z_bins"])

    # peaks
    n_bins = conf["analysis"]["peak_statistics"]["n_bins"]
    theta_fwhm = conf["analysis"]["peak_statistics"]["theta_fwhm"]
    bins_centers, bins_edges, bins_fwhms = peak_statistics.get_peaks_bins(
        binning_file, n_z_bins=n_z_bins, with_cross=True
    )

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"][args.simset]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"][args.simset]["n_noise_per_example"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example

    def data_vector_to_peaks(data_vector, patch_pix):
        full_sky = np.full((n_pix, data_vector.shape[-1]), hp.UNSEEN)
        full_sky[patch_pix] = data_vector
        full_sky = maps.tomographic_reorder(full_sky, n2r=True)

        peaks = peak_statistics.get_peaks(
            full_sky,
            n_side=n_side,
            n_bins=n_bins,
            theta_fwhm=theta_fwhm,
            with_cross=True,
            bins_centers=bins_centers,
            bins_edges=bins_edges,
            bins_fwhms=bins_fwhms,
        )

        return peaks

    # index corresponds to a WebDataset .tar shard ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        shard = shards[index]
        LOGGER.info(f"Index {index} is reading from {shard}")

        if args.simset == "grid":
            pipe = grid_pipeline.GridPipeline(
                conf, with_lensing=True, with_clustering=True, with_padding=False, apply_norm=False
            )
            dset = pipe.get_dset(
                pattern=shard,
                local_batch_size="cosmo",
                noise_indices=n_noise_per_example,
                n_readers=1,
                n_prefetch=0,
            )
            dset = dset.as_numpy_iterator()

            # one cosmology each
            for data_vectors, cosmos, (i_sobols, i_examples, i_noises) in dset:
                assert n_examples_per_cosmo == data_vectors.shape[0] == cosmos.shape[0] == i_sobols.shape[0]
                assert np.all(i_sobols == i_sobols[0]), f"All i_sobols should be the same, but are {i_sobols}"
                assert np.all(cosmos == cosmos[0]), f"All cosmological parameters should be the same, but are {cosmos}"
                cosmo = cosmos[0]
                i_sobol = i_sobols[0]
                LOGGER.info(f"Processing the cosmology with i_sobol = {i_sobol}")

                # loop over the batch dimension
                peaks = []
                for data_vector in LOGGER.progressbar(
                    data_vectors, total=n_examples_per_cosmo, desc="Loop over examples", at_level="info"
                ):
                    peaks.append(data_vector_to_peaks(data_vector, pipe.patch_pix))

                # shape (n_examples_per_cosmo, n_scales, n_bins, n_z_cross)
                peaks = np.stack(peaks, axis=0)

                # save one .h5 file per cosmology, TODO could also save to local scratch instead
                with h5py.File(os.path.join(args.dir_out, f"grid_peaks_{i_sobol:06}.h5"), "w") as f:
                    f.create_dataset(name="peaks", data=peaks)
                    f.create_dataset(name="cosmo", data=cosmo)
                    f.create_dataset(name="i_sobol", data=i_sobol)
                    f.create_dataset(name="i_example", data=i_examples)
                    f.create_dataset(name="i_noise", data=i_noises)

        elif args.simset == "fiducial":
            pipe = fiducial_pipeline.FiducialPipeline(
                conf,
                params=[],
                with_lensing=True,
                with_clustering=True,
                with_padding=False,
                apply_norm=False,
                apply_m_bias=True,
                shape_noise_scale=1.0,
                poisson_noise_scale=1.0,
            )
            dset = pipe.get_dset(
                pattern=shard,
                local_batch_size=1,
                noise_indices=n_noise_per_example,
                n_readers=1,
                n_prefetch=0,
                is_eval=True,
            )
            dset = dset.as_numpy_iterator()

            peaks = []
            i_examples = []
            i_noises = []
            # loop over individual examples
            for data_vector, (i_example, i_noise) in LOGGER.progressbar(
                dset, total=n_examples_per_cosmo // len(shards), desc="Loop over examples", at_level="info"
            ):
                # get rid of the batch dimension
                i_examples.append(i_example[0])
                i_noises.append(i_noise[0])
                data_vector = np.squeeze(data_vector)

                peaks.append(data_vector_to_peaks(data_vector, pipe.patch_pix))

            # shape (n_examples, n_scales, n_bins, n_z_cross)
            peaks = np.stack(peaks, axis=0)
            i_examples = np.stack(i_examples, axis=0)
            i_noises = np.stack(i_noises, axis=0)

            # save one .h5 file per input WebDataset .tar shard
            with h5py.File(os.path.join(args.dir_out, f"fiducial_peaks_{index:06}.h5"), "w") as f:
                f.create_dataset(name="peaks", data=peaks)
                f.create_dataset(name="i_example", data=i_examples)
                f.create_dataset(name="i_noise", data=i_noises)

        else:
            raise ValueError(f"Unknown simset {args.simset}")

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def merge(indices, args):
    args = setup(args)
    LOGGER.info(f"Beginning with merge for {args.simset}")

    h5_pattern = os.path.join(args.dir_out, f"{args.simset}_peaks_??????.h5")
    h5_files = sorted(glob.glob(h5_pattern))
    n_files = len(h5_files)
    LOGGER.info(f"Found {n_files} files to merge")

    # determine the per cosmology shapes
    with h5py.File(h5_files[0], "r") as f:
        peaks_shape = f["peaks"].shape
        i_example_shape = f["i_example"].shape
        i_noise_shape = f["i_noise"].shape

        if args.simset == "grid":
            cosmo_shape = f["cosmo"].shape
            i_sobol_shape = f["i_sobol"].shape

    # open the combined file
    with h5py.File(os.path.join(args.dir_out, f"{args.simset}_peaks.h5"), "w") as f_combined:
        LOGGER.info(f"Created the combined {args.simset} .h5 file")

        if args.simset == "grid":
            # define the combined output shapes
            f_combined.create_dataset(name="peaks", shape=(n_files,) + peaks_shape)
            f_combined.create_dataset(name="i_example", shape=(n_files,) + i_example_shape)
            f_combined.create_dataset(name="i_noise", shape=(n_files,) + i_noise_shape)
            f_combined.create_dataset(name="cosmo", shape=(n_files,) + cosmo_shape)
            f_combined.create_dataset(name="i_sobol", shape=(n_files,) + i_sobol_shape)

            # loop over the per cosmology .h5 files
            for i, h5_file in LOGGER.progressbar(enumerate(h5_files), desc="loop over files", at_level="info"):
                with h5py.File(h5_file, "r") as f:
                    peaks = f["peaks"][:]
                    cosmo = f["cosmo"][:]
                    i_sobol = f["i_sobol"][()]
                    i_example = f["i_example"][()]
                    i_noise = f["i_noise"][()]
                os.remove(h5_file)

                # write to the combined .h5 file
                f_combined["peaks"][i] = peaks
                f_combined["i_example"][i] = i_example
                f_combined["i_noise"][i] = i_noise
                f_combined["i_sobol"][i] = i_sobol
                f_combined["cosmo"][i] = cosmo

        elif args.simset == "fiducial":
            # define the combined output shapes
            f_combined.create_dataset(name="peaks", shape=(n_files * peaks_shape[0],) + peaks_shape[1:])
            f_combined.create_dataset(name="i_example", shape=(n_files * i_example_shape[0],) + i_example_shape[1:])
            f_combined.create_dataset(name="i_noise", shape=(n_files * i_noise_shape[0],) + i_noise_shape[1:])

            # loop over the per WebDataset .tar shard .h5 files
            for i, h5_file in LOGGER.progressbar(enumerate(h5_files), desc="loop over files", at_level="info"):
                with h5py.File(h5_file, "r") as f:
                    peaks = f["peaks"][:]
                    i_example = f["i_example"][()]
                    i_noise = f["i_noise"][()]
                os.remove(h5_file)

                # write to the combined .h5 file
                f_combined["peaks"][i * peaks_shape[0] : (i + 1) * peaks_shape[0]] = peaks
                f_combined["i_example"][i * i_example_shape[0] : (i + 1) * i_example_shape[0]] = i_example
                f_combined["i_noise"][i * i_noise_shape[0] : (i + 1) * i_noise_shape[0]] = i_noise

        else:
            raise ValueError(f"Unknown simset {args.simset}")

    LOGGER.info(f"Merged the per cosmology files into one and deleted them")
