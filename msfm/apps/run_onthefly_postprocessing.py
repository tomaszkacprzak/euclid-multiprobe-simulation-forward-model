# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen

Transform the full sky weak lensing signal and intrinsic alignment maps into multiple survey footprint cut-outs and
store them in .tfrecord files. The parallelization is done over the .tfrecord files, every jobarray element corresponds
to one.

For the grid, the main loop runs over the cosmologies.

Meant for
 - Euler (CPU nodes, local scratch)
 - esub jobarrays
 - Read the CosmoGrid directly from the SAN
 - CosmoGridV1.1
"""

import numpy as np
import os, argparse, warnings, time, yaml, sys, io
import webdataset
import torch


from msfm.utils import (
    logger,
    imports,
    filenames,
    input_output,
    files,
    lensing,
    clustering,
    cosmogrid,
    postprocessing,
    power_spectra,
    scales,
    redshift,
    parameters,
    configuration,
    prior,
)

hp = imports.import_healpy()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)



def setup(args):
    description = "Postprocess the CosmoGrid projections into forward-modeled survey footprints in webdataset tar files"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "command",
        type=str,
        default='wds',
        choices=('wds', 'test'),
        help="command to run",
    )
    parser.add_argument(
        "--n_files",
        type=int,
        default=2500,
        help="number of webdataset tar files to produce, this should be equal to the number of tasks in esub",
    )
    parser.add_argument(
        "--dir_in",
        type=str,
        required=True,
        help="input root dir of the full sky CosmoGrid projections",
    )
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
        "--cosmogrid_version",
        type=str,
        default="1.1",
        choices=["1.1", "1"],
        help="version of the input CosmoGrid",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default="",
        help="Optional suffix to be appended to the end of the filename, for example to distinguish different runs",
    )
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=120,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )

    parser.add_argument(
        "--indices", 
        type=str, 
        default="0", 
        help="Indices to process, format: 0,1,2,4 or start>stop. Default is 0.")

    
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    # paths
    args.config = os.path.abspath(args.config)

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    # compute
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    return args




def main(indices, args):
    
    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")

    # I/O delay
    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("debug mode")
    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    # configuration
    conf = files.load_config(args.config)
    with open(os.path.join(args.dir_out, "config.yaml"), "w") as f:
        yaml.dump(conf, f)

    # directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])

    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]
    cosmo_dirs_in = [os.path.join(args.dir_in, "grid", cosmo_dir) for cosmo_dir in cosmo_dirs]

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_cosmos = 2501 # 2500 grid and 1 fiducial
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_signal = conf["analysis"]["grid"]["n_noise_per_signal"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo
    LOGGER.info(
        f"For every cosmology, theres {n_examples_per_cosmo} examples: {n_patches} patches times {n_perms_per_cosmo} permutations"
    )

    # modeling
    baryonified = conf["analysis"]["modelling"]["baryonified"]

    # webdataset tar files
    n_cosmos_per_file = 1
    n_examples_per_file = n_examples_per_cosmo
    LOGGER.info(f"The number of files implies {n_cosmos_per_file} cosmological parameters per webdataset tar file")

    LOGGER.info(
        f"In total, there are n_examples_per_cosmo * n_cosmos_per_file = {n_examples_per_cosmo} * {n_cosmos_per_file}"
        f" = {n_examples_per_file} examples per file"
    )

    # analysis files
    pixel_file = files.load_pixel_file(conf)

    # constants
    full_sky_samples = {"γg":'WL', "γa":'WL', "ds":'WL', "γd":'WL', "dg":'GC', "qg":'GC'}

    LOGGER.info(f"Starting the main loop trough indices {indices}")

    # index corresponds to a webdataset tar file ###########################################################################
    for index in indices:
        LOGGER.info(f"Starting index {index}")
        LOGGER.timer.start("index")

        if args.debug:
            args.dir_out = os.path.join(args.dir_out, "debug")
            os.makedirs(args.dir_out, exist_ok=True)

        wds_file = filenames.get_filename_webdataset(
            args.dir_out,
            tag=conf["survey"]["name"] + args.file_suffix,
            index=index,
            simset="grid",
            with_bary=baryonified,
        )
        LOGGER.info(f"Index {index} is writing to {wds_file}")

        # index for the cosmological parameters
        i_cosmo_start = index * n_cosmos_per_file
        i_cosmo_end = (index + 1) * n_cosmos_per_file
        LOGGER.info(f"And includes {cosmo_dirs[i_cosmo_start : i_cosmo_end]}")

        # initialize the webdataset tar file writer
        num_total_examples = 0
        with webdataset.TarWriter(wds_file, encoder=True) as file_writer:

            # loop over the cosmological parameters
            for i_cosmo, cosmo_dir_in in LOGGER.progressbar(
                zip(range(i_cosmo_start, i_cosmo_end), cosmo_dirs_in[i_cosmo_start:i_cosmo_end]),
                at_level="debug",
                desc="Looping through cosmologies\n",
                total=i_cosmo_end - i_cosmo_start,
            ):
                LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")


                # constants
                cosmo = prior.get_hard_parameters(conf, cosmo_params_info, i_cosmo)
                i_sobol = int(cosmo_dir_in[-7:-1])
                n_patches = conf["analysis"]["n_patches"]
                n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
                # iterate over total maps set, which has n_cosmos * n_perms_per_cosmo * n_patches examples
                i_signal = i_cosmo * n_perms_per_cosmo * n_patches 
                

                # loop over permutations for this cosmology
                for i_perm in LOGGER.progressbar(range(n_perms_per_cosmo),
                    at_level="debug",
                    desc="Looping through the per cosmology signal maps",
                    total=n_perms_per_cosmo,
                ):  

                                        
                    LOGGER.info(f"Permutation {i_perm+1: 2d}/{n_perms_per_cosmo: 2d} for cosmology {i_cosmo+1: 2d}/{n_cosmos_per_file: 2d} for file {wds_file}")
                    LOGGER.timer.start("permutation")

                    ##
                    ## Main magic - get postprocessed full sky maps
                    ##
                    full_maps_file = postprocessing._get_full_sky_perm(args, conf, cosmo_dir_in, i_perm)
                    full_sky_maps = get_postprocessed_maps(conf, full_maps_file)

                    # write patches
                    for i_patch in range(n_patches):

                        patch_maps = {}
                        for m_name in full_sky_maps.keys():

                            patch_maps[m_name] = []
                            for i_z, m in enumerate(full_sky_maps[m_name]):
                                patch_map_ = postprocessing.full_sky_to_patch(m, conf, pixel_file, i_z, i_patch, sample=full_sky_samples[m_name])
                                patch_maps[m_name].append(patch_map_[..., np.newaxis]) # shape n_pix, n_z_bins
                            patch_maps[m_name] = np.concatenate(patch_maps[m_name], axis=-1)

                        # build output dict to be stored
                        dict_out = {
                                "__key__": f"{i_signal:09d}",
                                "gamma_g.pth": torch_bytes(torch.from_numpy(patch_maps["γg"])),
                                "gamma_a.pth": torch_bytes(torch.from_numpy(patch_maps["γa"])),
                                "gamma_d.pth": torch_bytes(torch.from_numpy(patch_maps["γd"])),
                                "ds.pth": torch_bytes(torch.from_numpy(patch_maps["ds"])),
                                "dg.pth": torch_bytes(torch.from_numpy(patch_maps["dg"])),
                                "qg.pth": torch_bytes(torch.from_numpy(patch_maps["qg"])),
                                "cosmo.pth": torch_bytes(torch.from_numpy(cosmo)),
                                "i_sobol.index": int(i_sobol),
                                "i_signal.index": int(i_signal),
                                "n_params.count": int(cosmo.shape[0]),
                                "n_pix.count": int(patch_maps["γg"].shape[0]),
                                "n_z_WL.count": int(patch_maps["γg"].shape[1]),
                                "n_z_GC.count": int(patch_maps["dg"].shape[1]),
                            }
                        # writeout to webdataset
                        file_writer.write(dict_out)
                        del_dict(dict_out)

                        i_signal += 1
                        LOGGER.info(f"Writing example to {wds_file} i_cosmo={i_cosmo:>5d} i_perm={i_perm:>2d}, i_patch={i_patch:>2d}, i_signal={i_signal:>8d}")
                        for key in patch_maps.keys():
                            LOGGER.debug(f"{key}.shape={patch_maps[key].shape}, dtype={patch_maps[key].dtype}")

                    LOGGER.info(f"Done with permutation {i_perm:04d} time taken {LOGGER.timer.elapsed('permutation')}")

                                    

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        return num_total_examples


        
def get_postprocessed_maps(conf, full_maps_file):

    # filepaths
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # constants
    n_side = conf["analysis"]["n_side"]
    kappa2gamma_fac, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(3 * n_side - 1)
    z_bins_WL = conf["survey"]["WL"]["z_bins"]
    z_bins_GC = conf["survey"]["GC"]["z_bins"]


    # container
    full_sky_maps = {"γg": [], "γa": [], "γd": [], "ds": [], "dg": [], "qg": []}

    # loop over lensing bins
    for i_z, z_bin in enumerate(z_bins_WL):

        ##
        ## Lensing shear
        ##

        kg = postprocessing._read_full_sky_bin(conf, full_maps_file, "kg", z_bin)
        # kappa to shear conversion for lensing signal
        g1_, g2_ = lensing.kappa_to_gamma(kg, hp_datapath, kappa2gamma_fac, n_side)
        γg_ = g1_ + 1j*g2_
        full_sky_maps["γg"].append(γg_.astype(np.complex64))

        ##
        ## Linear intrinsic alignment
        ##

        ia = postprocessing._read_full_sky_bin(conf, full_maps_file, "ia", z_bin)
        # kappa to shear conversion for intrinsic alignment
        g1_, g2_ = lensing.kappa_to_gamma(ia, hp_datapath, kappa2gamma_fac, n_side)
        γa_ = g1_ + 1j*g2_
        full_sky_maps["γa"].append(γa_.astype(np.complex64)) 

        ##
        ## Source sample galaxy counts
        ##

        # source sample galaxy counts for shape noise
        ds_ = postprocessing._read_full_sky_bin(conf, full_maps_file, "dg", z_bin)
        full_sky_maps["ds"].append(ds_.astype(np.float32))

        ##
        ## Delta-NLA intrinsic alignment
        ##

        # delta-NLA component approximation
        γd_ = γa_ * ds_ # approximation
        full_sky_maps["γd"].append(γd_.astype(np.complex64))

    # loop over clustering bins
    for i_z, z_bin in enumerate(z_bins_GC):

        ##
        ## Galaxy counts
        ##

        dg_ = postprocessing._read_full_sky_bin(conf, full_maps_file, "dg", z_bin)
        full_sky_maps["dg"].append(dg_.astype(np.float32))


        ##
        ## Quadratic galaxy counts
        ##

        # quadratic galaxy counts for shape noise
        qg_ = (dg_**2) # approximation
        full_sky_maps["qg"].append(qg_.astype(np.float32))

    return full_sky_maps

def del_dict(dict_):

    for key in list(dict_.keys()):
        del(dict_[key])
    del(dict_)


def torch_bytes(x: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(x, buffer)
    return buffer.getvalue()


if __name__ == "__main__":

    args = setup(sys.argv[1:])

    if args.command == 'wds':

        indices = configuration.get_indices(args.indices)
        main(indices=indices, args=args)

    elif args.command == 'test_pipeline':

        pass

        # # test the onthefly_pipeline

        # from msfm.onthefly_pipeline import OntheflyPipeline
        # onthefly_pipeline = OntheflyPipeline(conf=conf)
        # tfr_pattern = os.path.join(args.dir_out, "*.tfrecord")
        # dset = onthefly_pipeline.get_dset(tfr_pattern=tfr_pattern, local_batch_size=2)
        # for data_vectors, cosmo, index in dset:
        #     print(data_vectors.shape, cosmo, index)
        #     break

