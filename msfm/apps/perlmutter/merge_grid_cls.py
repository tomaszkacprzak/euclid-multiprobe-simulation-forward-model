# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created May 2024
Author: Arne Thomsen

Merge function from msfm/apps/run_grid_preprocessing.py since this only works if the .tfrecords stay on Euler,
not when they are directly stored on the SAN or Perlmutter. In that case, the merge has to be run on Perlmutter later,
like here.
"""


import argparse, os, h5py
import numpy as np
import tensorflow as tf

from msfm.utils import files, logger, filenames, tfrecords, power_spectra

LOGGER = logger.get_logger(__file__)


def setup(args):
    description = "Preprocess the CosmoGrid projections into forward-modeled survey footprints in .tfrecord files"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "--dir_out",
        type=str,
        required=True,
        help="output root dir of the forward-modeled survey footprints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration .yaml file",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default="",
        help="Optional suffix to be appended to the end of the filename, for example to distinguish different runs",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    # paths
    args.config = os.path.abspath(args.config)

    # compute
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    return args


def merge(indices, args):
    args = setup(args)
    conf = files.load_config(args.config)

    n_cosmos = conf["analysis"]["grid"]["n_cosmos"]
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_signal = conf["analysis"]["grid"]["n_noise_per_signal"]
    n_signal_per_cosmo = n_patches * n_perms_per_cosmo

    tfr_pattern = filenames.get_filename_tfrecords(
        args.dir_out,
        tag=conf["survey"]["name"] + args.file_suffix,
        with_bary=conf["analysis"]["modelling"]["baryonified"],
        index=None,
        simset="grid",
        return_pattern=True,
    )

    cls_dset = tf.data.Dataset.list_files(tfr_pattern)
    # flat_map to not mix cosmologies
    cls_dset = cls_dset.flat_map(tf.data.TFRecordDataset)
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
        i_signal = example["i_signal"].numpy()

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
        cosmo = np.tile(cosmo, (n_noise_per_signal, 1))
        i_sobol = np.tile(i_sobol, n_noise_per_signal)
        i_signal = np.tile(i_signal, n_noise_per_signal)

        # noise is treated separately because it's along a separate dimension in the .tfrecords. This here is preserves
        # the order imposed above in power_spectrum = ...
        i_noise = np.arange(n_noise_per_signal)
        i_noise = np.repeat(i_noise, n_signal_per_cosmo)

        cls.append(cl)
        binned_cls.append(binned_cl)
        bin_edges.append(bin_edge)
        cosmos.append(cosmo)
        i_sobols.append(i_sobol)
        i_examples.append(i_signal)
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
        f.create_dataset("i_signal", data=i_examples)
        f.create_dataset("i_noise", data=i_noises)

    LOGGER.info(f"Done with merging of the grid power spectra")


if __name__ == "__main__":
    merge([], None)
