# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created May 2024
Author: Arne Thomsen

Merge function from msfm/apps/run_fiducial_preprocessing.py since this only works if the .tfrecords stay on Euler,
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

    # for proper bookkeeping
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples = n_patches * n_perms_per_cosmo

    tfr_pattern = filenames.get_filename_tfrecords(
        args.dir_out,
        tag=conf["survey"]["name"] + args.file_suffix,
        index=None,
        simset="fiducial",
        with_bary=conf["analysis"]["modelling"]["baryonified"],
        return_pattern=True,
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
        i_examples.append(int(example["i_signal"]))

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
        f.create_dataset("i_signal", data=i_examples)
        f.create_dataset("i_noise", data=i_noise)

    LOGGER.info(f"Done with merging of the fiducial power spectra")


if __name__ == "__main__":
    merge([], None)
