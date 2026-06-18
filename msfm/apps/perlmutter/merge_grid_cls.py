# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created May 2024
Author: Arne Thomsen

<<<<<<< HEAD
Merge WebDataset power spectra into HDF5 on Perlmutter.
"""


import argparse, os, h5py, glob, itertools
=======
Merge function from msfm/apps/run_grid_preprocessing.py since this only works if the WebDataset tar shards stay on Euler,
not when they are directly stored on the SAN or Perlmutter. In that case, the merge has to be run on Perlmutter later,
like here.
"""

import argparse, os, h5py, glob
>>>>>>> torch-rewrite
import numpy as np
import webdataset as wds

from msfm.utils import files, logger, filenames, webdatasets, power_spectra

LOGGER = logger.get_logger(__file__)


def _batched(samples, batch_size):
    """Yield dictionaries of WebDataset examples stacked into cosmology batches."""
    iterator = iter(samples)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield {key: np.stack([sample[key].numpy() for sample in batch], axis=0) for key in batch[0]}


def setup(args):
<<<<<<< HEAD
    description = "Merge WebDataset power spectra into HDF5"
=======
    description = (
        "Preprocess the CosmoGrid projections into forward-modeled survey footprints in .tar WebDataset shards"
    )
>>>>>>> torch-rewrite
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

<<<<<<< HEAD
    webdataset_pattern = filenames.get_filename_webdataset(
=======
    wds_pattern = filenames.get_filename_webdataset(
>>>>>>> torch-rewrite
        args.dir_out,
        tag=conf["survey"]["name"] + args.file_suffix,
        with_bary=conf["analysis"]["modelling"]["baryonified"],
        index=None,
        simset="grid",
        return_pattern=True,
    )
<<<<<<< HEAD
    webdataset_files = sorted(glob.glob(webdataset_pattern))

    cls_samples = (webdatasets.decode_grid_cls_sample(sample) for sample in wds.WebDataset(webdataset_files, shardshuffle=False))
    cls_dset = _batched(cls_samples, n_signal_per_cosmo)
=======
    wds_files = sorted(glob.glob(wds_pattern))
    if not wds_files:
        raise FileNotFoundError(f"No WebDataset tar shards match pattern {wds_pattern!r}")

    def iter_cosmology_batches():
        batch = []
        for sample in wds.WebDataset(wds_files, shardshuffle=False):
            batch.append(webdatasets.decode_grid_cls_sample(sample))
            if len(batch) == n_signal_per_cosmo:
                yield {key: np.stack([example[key].numpy() for example in batch], axis=0) for key in batch[0]}
                batch = []
        if batch:
            raise ValueError(
                f"Found an incomplete cosmology batch with {len(batch)} samples; "
                f"expected {n_signal_per_cosmo} samples per cosmology"
            )
>>>>>>> torch-rewrite

    cls = []
    binned_cls = []
    bin_edges = []
    cosmos = []
    i_sobols = []
    i_examples = []
    i_noises = []
    for example in LOGGER.progressbar(
<<<<<<< HEAD
        cls_dset, total=n_cosmos, desc="Looping through the different cosmologies in the WebDataset shards", at_level="info"
=======
        iter_cosmology_batches(),
        total=n_cosmos,
        desc="Looping through the different cosmologies in the WebDataset tar shards",
        at_level="info",
>>>>>>> torch-rewrite
    ):
        cl = example["cls"]
        cosmo = example["cosmo"]
        i_sobol = example["i_sobol"]
        i_signal = example["i_signal"]

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

<<<<<<< HEAD
        # noise is treated separately because it's along a separate dimension in the WebDataset shards. This here preserves
=======
        # noise is treated separately because it's along a separate dimension in the WebDataset tar shards. This here is preserves
>>>>>>> torch-rewrite
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

<<<<<<< HEAD
    # separate folder on the same level as WebDataset shards
=======
    # separate folder on the same level as WebDataset tar shards
>>>>>>> torch-rewrite
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
