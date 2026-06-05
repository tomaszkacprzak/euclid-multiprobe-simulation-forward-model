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

from msfm.utils import files, imports, logger, power_spectra, scales

hp = imports.import_healpy()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def get_tasks(args):
    """Returns a list of task indices to be executed by the workers. Each index corresponds to one cosmology"""
    args = setup(args)

    conf = files.load_config(args.config)
    n_cosmos = conf["analysis"]["grid"]["n_cosmos"]
    indices = list(range(n_cosmos))

    if args.debug:
        indices = indices[:3]

    LOGGER.warning(f"There are {n_cosmos} cosmologies/tasks; running on the first {len(indices)} indices")

    return indices


def resources(args):
    args = setup(args)

    if args.cluster == "perlmutter":
        # because of hyperthreading, there's a total of 256 threads per node
        resources = {
            "main_time": 1,
            "main_n_cores": 2,
            "main_memory": 1952,
            "merge_time": 1,
            "merge_n_cores": 8,
            "merge_memory": 1952,
        }
    elif args.cluster == "euler":
        resources = {"main_time": 4, "main_memory": 4096, "main_n_cores": 4, "merge_memory": 4096, "merge_n_cores": 16}

    return resources


def setup(args):
    description = "Generate realizations of the power spectrum of white noise within the footprint."
    parser = argparse.ArgumentParser(description=description, add_help=True)

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
    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    n_z = len(conf["survey"]["lensing"]["z_bins"] + conf["survey"]["clustering"]["z_bins"])

    l_max = 3 * n_side
    l_mins = conf["analysis"]["scale_cuts"]["lensing"]["l_min"] + conf["analysis"]["scale_cuts"]["clustering"]["l_min"]
    l_min = np.unique(np.array(l_mins))
    assert (
        l_min.size == 1
    ), f"l_min has size {l_min.size}, but should be 1 (the same largest scale for all redshift bins)"
    l_min = l_min[0]

    n_examples_per_cosmo = (
        conf["analysis"]["n_patches"]
        * conf["analysis"]["grid"]["n_perms_per_cosmo"]
        * conf["analysis"]["grid"]["n_noise_per_example"]
    )
    LOGGER.info(f"Generating {n_examples_per_cosmo} noise realizations per cosmology")

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    _, patches_pix_dict, _, _ = files.load_pixel_file(conf)
    # TODO this assumes that there is no difference between the galaxy samples and tomographic bins
    base_patch = patches_pix_dict["metacal"][0][0]

    # every index corresponds to one grid cosmology
    for index in indices:
        rng = np.random.default_rng(args.np_seed + index)

        out_file = os.path.join(args.dir_out, f"white_noise_{index:04}.h5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset(
                "cls/raw", shape=(n_examples_per_cosmo, l_max, int(n_z * (n_z + 1) / 2)), dtype=np.float32
            )

            for i in LOGGER.progressbar(range(n_examples_per_cosmo), desc=f"Generating noise for cosmology {index}"):
                standard_samples = rng.standard_normal(size=(base_patch.size, n_z), dtype=np.float32)

                # ring ordering
                noise_map = np.zeros((n_pix, n_z), dtype=np.float32)
                noise_map[base_patch] = standard_samples

                noise_alms = power_spectra.get_alms(noise_map, nest=False, datapath=hp_datapath)
                noise_cls = power_spectra.get_cls(noise_alms, with_cross=True)
                # because the maps in the .tfrecords get smoothed on the large scales too
                for j in range(noise_cls.shape[-1]):
                    noise_cls[..., j] = scales.cls_to_smoothed_cls(noise_cls[..., j], l_min=l_min)

                f["cls/raw"][i, :] = noise_cls

        yield index


def merge(indices, args):
    args = setup(args)

    conf = files.load_config(args.config)

    n_side = conf["analysis"]["n_side"]
    l_max = 3 * n_side
    n_examples_per_cosmo = (
        conf["analysis"]["n_patches"]
        * conf["analysis"]["grid"]["n_perms_per_cosmo"]
        * conf["analysis"]["grid"]["n_noise_per_example"]
    )
    n_z = len(conf["survey"]["lensing"]["z_bins"] + conf["survey"]["clustering"]["z_bins"])

    out_file = os.path.join(args.dir_out, "white_noise.h5")
    with h5py.File(out_file, "w") as f:
        f.create_dataset("cls/raw", shape=(len(indices), n_examples_per_cosmo, l_max, n_z), dtype=np.float32)

        for index in LOGGER.progressbar(indices, desc="Merging white noise realizations"):
            in_file = os.path.join(args.dir_out, f"white_noise_{index:04}.h5")
            with h5py.File(in_file, "r") as g:
                f["cls/raw"][index, :, :, :] = g["cls/raw"][:]
            os.remove(in_file)

    LOGGER.info(f"Merged white noise realizations to {out_file}")
