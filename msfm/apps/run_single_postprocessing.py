# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Like run_fiducial_postprocessing.py and run_grid_postprocessing.py, but for a single cosmology.

example usage:

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_008963 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_lensing --with_clustering \
    --suffix_out="_test" \
    --msfm_config=../../configs/v14/extended.yaml \
    --debug \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --jobname=postproc_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
"""

import numpy as np
import os, argparse, warnings, h5py, time, re

from msfm.utils import files, logger, input_output, imports, observation, parameters

hp = imports.import_healpy(parallel=False)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    args = setup(args)

    if args.cluster == "perlmutter":
        # because of hyperthreading, there's a total of 256 threads per node
        # the 8 cores don't speed things up much, but are included to increase the memory
        resources = {
            "main_time": 1,
            "main_n_cores": 8,
            "main_memory": 1952,
            "main_scratch": 0,
            "merge_time": 0.1,
            "merge_n_cores": 8,
            "merge_memory": 1952,
            "merge_scratch": 0,
        }
    elif args.cluster == "euler":
        resources = {"main_time": 4, "main_memory": 4096, "main_n_cores": 8, "merge_memory": 4096, "merge_n_cores": 16}

    return resources


def setup(args):
    description = "evaluate the power spectra from the input pipelines"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    # required setup
    parser.add_argument(
        "--dir_in",
        type=str,
        required=True,
        help="directory containing the CosmoGrid permutations for the given cosmology",
        # /global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_008963
        # /global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        required=True,
        help="directory to write the output files to, will be created if it does not exist",
        # /pscratch/sd/a/athomsen/v11desy3/v14/extended/obs
    )
    parser.add_argument(
        "--suffix_out",
        type=str,
        default="",
        help="suffix to append to the output files",
    )
    parser.add_argument(
        "--msfm_config",
        type=str,
        default="configs/config.yaml",
        help="msfm configuration yaml file for postprocessing of the CosmoGrid",
    )

    # input arguments to observation.forward_model_cosmogrid
    parser.add_argument(
        "--noiseless",
        action="store_true",
        help="whether to include shape and Poisson noise on top of the signal",
    )
    parser.add_argument(
        "--noise_only",
        action="store_true",
        help="whether to only include shape and Poisson noise",
    )
    parser.add_argument(
        "--with_lensing",
        action="store_true",
        help="whether to include lensing in the forward model",
    )
    parser.add_argument(
        "--tomo_Aia",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--bta",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--tomo_bg_metacal",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--reduced_shear",
        action="store_true",
        help="whether to use the reduced shear instead of shear in the forward model",
    )
    parser.add_argument(
        "--with_clustering",
        action="store_true",
        help="whether to include clustering in the forward model",
    )
    parser.add_argument(
        "--tomo_bg",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--tomo_qbg",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--tomo_cg",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--contaminate_survey_systematics",
        action="store_true",
        help="whether to include the maglim survey systematics map in the forward model",
    )

    # run
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="perlmutter",
        choices=("perlmutter", "euler"),
        help="the cluster to execute on",
    )
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=60,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    parser.add_argument(
        "--np_seed",
        type=int,
        default=12,
        help="seed for the numpy random number generator, used for the shape and Poisson noise",
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    return args


def main(indices, args):
    args = setup(args)
    msfm_conf = files.load_config(args.msfm_config)

    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("!!! debug mode !!!")

    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    # index corresponds to a CosmoGridV1 permutation ##################################################################
    for index in indices:
        perm_dir = os.path.join(args.dir_in, f"perm_{index:04}")

        # metacal bias logic
        sc_mode = msfm_conf["analysis"]["modelling"]["lensing"]["source_clustering"]
        if args.tomo_bg_metacal is not None:
            tomo_bg_metacal = args.tomo_bg_metacal
        elif sc_mode == "fixed":
            if "/grid/" in perm_dir:
                match = re.search(r"cosmo_(\d{6})", perm_dir)
                i_sobol = int(match.group(1))
                tomo_bg_metacal = files.read_metacal_bias(f"cosmo_{i_sobol:06}", conf=msfm_conf)
            elif "/fiducial/" in perm_dir or "benchmark" in perm_dir:
                tomo_bg_metacal = files.read_metacal_bias(f"fiducial", conf=msfm_conf)
            else:
                raise ValueError(f"Cannot determine metacal bias key from perm_dir={perm_dir!r}: expected '/grid/' or '/fiducial/'/'benchmark' in path")
        elif sc_mode == "prior":
            sc_prior = parameters.get_prior_intervals(["bsc"], conf=msfm_conf)
            bsc_samples = np.random.default_rng(seed=index).uniform(
                sc_prior[0, 0], sc_prior[0, 1], size=msfm_conf["analysis"]["n_patches"]
            )
            tomo_bg_metacal = None  # set per patch below
        else:  # rotate
            tomo_bg_metacal = None

        obs_maps = []
        obs_cls_raw = []
        for i_patch in LOGGER.progressbar(
            range(msfm_conf["analysis"]["n_patches"]), desc=f"loop through patches\n", at_level="info"
        ):
            if args.debug and i_patch > 0:
                LOGGER.warning("Debug mode: only processing the first patch")
                break

            if sc_mode == "prior" and args.tomo_bg_metacal is None:
                tomo_bg_metacal = bsc_samples[i_patch]

            wl_gamma_patch, gc_count_patch = observation.forward_model_cosmogrid(
                perm_dir,
                conf=msfm_conf,
                noisy=not args.noiseless,
                noise_only=args.noise_only,
                i_patch=i_patch,
                # lensing
                with_lensing=args.with_lensing,
                tomo_Aia=args.tomo_Aia,
                bta=args.bta,
                tomo_bg_metacal=tomo_bg_metacal,
                reduced_shear=args.reduced_shear,
                # clustering
                with_clustering=args.with_clustering,
                tomo_bg=args.tomo_bg,
                tomo_cg=args.tomo_cg,
                survey_sys=args.contaminate_survey_systematics,
                noise_seed=args.np_seed,
            )

            obs_map, obs_cl_raw, _ = observation.forward_model_observation_map(
                wl_gamma_map=wl_gamma_patch,
                gc_count_map=gc_count_patch,
                conf=msfm_conf,
                apply_norm=False,
                with_padding=True,
                nest_in=False,
            )

            obs_maps.append(obs_map)
            obs_cls_raw.append(obs_cl_raw)

        obs_maps = np.stack(obs_maps, axis=0)
        obs_cls_raw = np.stack(obs_cls_raw, axis=0)

        # save the results
        cosmo_name = os.path.basename(args.dir_in)
        out_file = os.path.join(args.dir_out, f"{cosmo_name}_obs_maps{args.suffix_out}_{index:04}.h5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset(name="obs/maps", data=obs_maps)
            f.create_dataset(name="obs/cls_raw", data=obs_cls_raw)
        LOGGER.info(f"Saved results to {out_file}")

        yield index


def merge(indices, args):
    args = setup(args)
    msfm_conf = files.load_config(args.msfm_config)
    n_patches = msfm_conf["analysis"]["n_patches"]

    cosmo_name = os.path.basename(args.dir_in)
    out_file = os.path.join(args.dir_out, f"{cosmo_name}_obs_maps{args.suffix_out}.h5")

    with h5py.File(out_file, "w") as f_merged:
        for index in LOGGER.progressbar(indices, desc="merging files", at_level="info"):
            in_file = os.path.join(args.dir_out, f"{cosmo_name}_obs_maps{args.suffix_out}_{index:04}.h5")
            with h5py.File(in_file, "r") as f_in:
                obs_maps = f_in["obs/maps"][:]
                obs_cls_raw = f_in["obs/cls_raw"][:]

            if index == indices[0]:
                f_merged.create_dataset(
                    "obs/maps", shape=(len(indices) * obs_maps.shape[0],) + obs_maps.shape[1:], dtype=np.float32
                )
                f_merged.create_dataset(
                    "obs/cls_raw",
                    shape=(len(indices) * obs_cls_raw.shape[0],) + obs_cls_raw.shape[1:],
                    dtype=np.float32,
                )

            f_merged["obs/maps"][index * n_patches : (index + 1) * n_patches, :] = obs_maps
            f_merged["obs/cls_raw"][index * n_patches : (index + 1) * n_patches, :] = obs_cls_raw

    LOGGER.info(f"Merged all files into {out_file}")

    # only remove the files after the above loop has terminated successfully
    for index in indices:
        in_file = os.path.join(args.dir_out, f"{cosmo_name}_obs_maps{args.suffix_out}_{index:04}.h5")
        os.remove(in_file)
    LOGGER.info(f"Removed temporary files")
