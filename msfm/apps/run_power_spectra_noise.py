# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics
"""
Created August 2024
Author: Arne Thomsen

Generate white noise maps from independent, pixel-wise Gaussian samples and compute their power spectra. This is needed
to make the power spectra consistent with the map-level summary statistics in terms of the scale cuts. Because of
the linearity of Gaussians, the noise can be drawn for a fixed standard deviation of one and rescaled later.
"""

import numpy as np
import os, argparse, warnings, h5py

from msfm.utils import files, imports, logger, power_spectra

hp = imports.import_healpy()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def get_tasks(args):
    """Returns a list of task indices to be executed by the workers"""
    args = setup(args)

    n_indices = int(np.ceil(args.n_noise / args.n_noise_per_index))
    indices = list(range(n_indices))

    if args.debug:
        indices = indices[:3]
        LOGGER.warning(f"Debug mode: running on the first {len(indices)} indices of {n_indices}")
    else:
        LOGGER.warning(f"There are {n_indices} indices")

    return indices


def resources(args):
    args = setup(args)

    if args.cluster == "perlmutter":
        # because of hyperthreading, there's a total of 256 threads per node
        resources = {
            "main_time": 2,
            "main_n_cores": 2,
            "main_memory": 1952,
            "main_scratch": 0,
            "merge_time": 1,
            "merge_n_cores": 8,
            "merge_memory": 1952,
            "merge_scratch": 0,
        }
    elif args.cluster == "euler":
        resources = {"main_time": 4, "main_memory": 4096, "main_n_cores": 4, "merge_memory": 4096, "merge_n_cores": 16}

    return resources


def setup(args):
    description = "Generate realizations of the power spectrum of white noise within the footprint."
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument("--n_noise", type=int, required=True, help="number of white noise samples to generate")
    parser.add_argument(
        "--n_noise_per_index", type=int, default=int(1e5), help="number of noise samples per task/index"
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration .yaml file",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="perlmutter",
        choices=("perlmutter", "euler"),
        help="the cluster to execute on",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument("--np_seed", default=12, type=int, help="numpy random seed")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    # paths
    args.config = os.path.abspath(args.config)

    return args


def main(indices, args):
    args = setup(args)

    conf = files.load_config(args.config)
    n_pix = hp.nside2npix(conf["analysis"]["n_side"])
    n_z = len(conf["survey"]["metacal"]["z_bins"] + conf["survey"]["maglim"]["z_bins"])

    n_bins = conf["analysis"]["power_spectra"]["n_bins"]

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    _, patches_pix_dict, _, _ = files.load_pixel_file(conf)
    # TODO this assumes that the mask is the same for the different galaxy samples and tomographic bins
    base_patch = patches_pix_dict["metacal"][0][0]

    # every index corresponds to one worker and produces n_noise_per_index samples
    for index in indices:
        rng = np.random.default_rng(args.np_seed + index)

        if args.debug:
            out_file = os.path.join(args.dir_out, f"debug/white_noise_{index:04}.h5")
        else:
            out_file = os.path.join(args.dir_out, f"white_noise_{index:04}.h5")

        with h5py.File(out_file, "w") as f:
            f.create_dataset(
                "cls/binned", shape=(args.n_noise_per_index, n_bins - 1, int(n_z * (n_z + 1) / 2)), dtype=np.float32
            )

            for i in LOGGER.progressbar(range(args.n_noise_per_index), desc=f"Generating noise for index {index}"):
                standard_samples = rng.standard_normal(size=(base_patch.size, n_z), dtype=np.float32)

                # ring ordering
                noise_map = np.zeros((n_pix, n_z), dtype=np.float32)
                noise_map[base_patch] = standard_samples

                noise_alms = power_spectra.get_alms(noise_map, nest=False, datapath=hp_datapath)
                noise_cls = power_spectra.get_cls(noise_alms, with_cross=True)

                binned_cls, _ = power_spectra.bin_according_to_config(noise_cls, conf)

                f["cls/binned"][i, :] = binned_cls

        yield index


def merge(indices, args):
    args = setup(args)

    # this one always exists
    in_file = os.path.join(args.dir_out, f"white_noise_{0:04}.h5")
    with h5py.File(in_file, "r") as f:
        n_noise_per_index, n_bins, n_cross_z = f["cls/binned"].shape

    out_file = os.path.join(args.dir_out, "white_noise.h5")
    with h5py.File(out_file, "w") as f:
        f.create_dataset("cls/binned", shape=(len(indices) * n_noise_per_index, n_bins, n_cross_z), dtype=np.float32)

        for index in LOGGER.progressbar(indices, desc="Merging white noise realizations"):
            in_file = os.path.join(args.dir_out, f"white_noise_{index:04}.h5")
            with h5py.File(in_file, "r") as g:
                f["cls/binned"][index * n_noise_per_index : (index + 1) * n_noise_per_index, :] = g["cls/binned"][:]

    LOGGER.info(f"Merged white noise realizations to {out_file}")

    # only remove the files after the above loop has terminated successfully
    for index in indices:
        in_file = os.path.join(args.dir_out, f"white_noise_{index:04}.h5")
        os.remove(in_file)
    LOGGER.info(f"Removed temporary files")
