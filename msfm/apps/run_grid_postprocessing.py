# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen

Transform the full sky weak lensing signal and intrinsic alignment maps into multiple survey footprint cut-outs and
store them in .tar WebDataset shards. The parallelization is done over the .tar files, every jobarray element corresponds
to one.

For the grid, the main loop runs over the cosmologies.

Meant for
 - Euler (CPU nodes, local scratch)
 - esub jobarrays
 - Read the CosmoGrid directly from the SAN
 - CosmoGridV1.1
"""

import numpy as np
import webdataset as wds
import os, argparse, warnings, time, yaml, h5py, pickle, glob, sys, itertools


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
    webdatasets,
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


def _batched(samples, batch_size):
    """Yield dictionaries of WebDataset examples stacked into cosmology batches."""
    iterator = iter(samples)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield {key: np.stack([sample[key].numpy() for sample in batch], axis=0) for key in batch[0]}


def setup(args):
    description = "Postprocess the CosmoGrid projections into forward-modeled survey footprints in .tar WebDataset shards"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "--n_files",
        type=int,
        default=2500,
        help="number of .tar WebDataset shards to produce, this should be equal to the number of tasks in esub",
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
    n_cosmos = conf["analysis"]["grid"]["n_cosmos"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_signal = conf["analysis"]["grid"]["n_noise_per_signal"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_signal
    LOGGER.info(
        f"For every cosmology, theres {n_examples_per_cosmo} examples: "
        f"{n_patches} patches times {n_perms_per_cosmo} permutations times {n_noise_per_signal} noise realizations"
    )

    # modeling
    configuration.print_and_check_modeling_in_config(conf)

    baryonified = conf["analysis"]["modelling"]["baryonified"]

    store_cross_maps = conf["analysis"]["modelling"]["store_cross_maps"]
    store_lensing = conf["analysis"]["modelling"]["WL"]["store"]
    store_clustering = conf["analysis"]["modelling"]["GC"]["store"]

    extended_nla = conf["analysis"]["modelling"]["WL"]["extended_nla"]

    power_law_biasing = conf["analysis"]["modelling"]["GC"]["power_law_biasing"]
    per_bin_biasing = conf["analysis"]["modelling"]["GC"]["per_bin_biasing"]
    quadratic_biasing = conf["analysis"]["modelling"]["GC"]["quadratic_biasing"]

    astro_params = conf["analysis"]["params"]["ia"]["nla"]
    if extended_nla:
        astro_params += conf["analysis"]["params"]["ia"]["tatt"]
    astro_params += conf["analysis"]["params"]["bg"]["linear"]
    if quadratic_biasing:
        astro_params += conf["analysis"]["params"]["bg"]["quadratic"]
    if conf["analysis"]["modelling"]["WL"]["source_clustering"] == "prior":
        astro_params += conf["analysis"]["params"]["sc"]
    LOGGER.info(f"Sampling the astrophysical parameters {astro_params} from a Latin hypercube")

    astro_priors = parameters.get_prior_intervals(astro_params, conf=conf)

    # .tar WebDataset shards
    if n_cosmos % args.n_files == 0:
        n_cosmos_per_file = n_cosmos // args.n_files
        n_examples_per_file = n_examples_per_cosmo * n_cosmos_per_file
        LOGGER.info(f"The number of files implies {n_cosmos_per_file} cosmological parameters per .tar WebDataset shard")
    else:
        raise ValueError(
            f"The total number of cosmologies {n_cosmos} has to be evenly divisible by the number of files {args.n_files}"
        )
    LOGGER.info(
        f"In total, there are n_examples_per_cosmo * n_cosmos_per_file = {n_examples_per_cosmo} * {n_cosmos_per_file}"
        f" = {n_examples_per_file} examples per file"
    )

    # analysis files
    pixel_file = files.load_pixel_file(conf)
    noise_file = files.load_noise_file(conf)

    # transforms
    lensing_transform = _get_lensing_transform(conf, pixel_file)
    clustering_transform = _get_clustering_transform(conf, pixel_file)

    LOGGER.warning(f"Starting the main loop trough indices {indices}")

    # index corresponds to a .tar WebDataset shard ###########################################################################
    for index in indices:
        LOGGER.warning(f"Starting index {index}")
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

        num_total_examples = 0
        with wds.TarWriter(wds_file) as sink:
            # loop over the cosmological parameters
            for i_cosmo, cosmo_dir_in in LOGGER.progressbar(
                zip(range(i_cosmo_start, i_cosmo_end), cosmo_dirs_in[i_cosmo_start:i_cosmo_end]),
                at_level="debug",
                desc="Looping through cosmologies\n",
                total=i_cosmo_end - i_cosmo_start,
            ):
                LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")

                state_file = os.path.join(args.dir_out, f"program_state{i_cosmo:06}" + args.file_suffix + ".pkl")

                i_sobol, cosmo = prior.extend_sobol_sequence(conf, cosmo_params_info, i_cosmo)
                astro_samples = prior.sample_astro_parameters(astro_params, i_cosmo, n_examples_per_cosmo, n_noise_per_signal, astro_priors)
                i_sobol = int(cosmo_dir_in[-7:-1])
                n_patches = conf["analysis"]["n_patches"]
                n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
                # rng = np.random.default_rng()

                store_lensing = conf["analysis"]["modelling"]["WL"]["store"]
                store_clustering = conf["analysis"]["modelling"]["GC"]["store"]
                samples = []
                if store_lensing:
                    samples.append("WL")
                if store_clustering:
                    samples.append("GC")

                for i_perm in LOGGER.progressbar(range(n_perms_per_cosmo),
                    at_level="info",
                    desc="Looping through the per cosmology signal maps",
                    total=n_perms_per_cosmo,
                ):  
                                        
                    LOGGER.info(f"Starting permutation {i_perm:04d}/{n_perms_per_cosmo} for cosmology {i_cosmo}/{n_cosmos_per_file} for file {wds_file}")
                    LOGGER.timer.start("permutation")

                    rng_perm = np.random.default_rng(int(conf['master_seed']) + i_cosmo * n_perms_per_cosmo + i_perm)

                    full_maps_file = postprocessing._get_full_sky_perm(args, conf, cosmo_dir_in, i_perm)
                    bsc_samples = (
                        astro_samples[:, -1]
                        if conf["analysis"]["modelling"]["WL"]["source_clustering"] == "prior"
                        else None
                    )

                    container_data_vecs = {}
                    for sample in samples:
                        LOGGER.timer.start("sample")
                        LOGGER.info(f"Starting with sample {sample}")

                        # sample specific
                        in_map_types = conf["survey"][sample]["map_types"]["input"]
                        out_map_types = conf["survey"][sample]["map_types"]["output"]
                        z_bins = conf["survey"][sample]["z_bins"]
                        
                        for in_map_type, out_map_type in zip(in_map_types, out_map_types):
                            LOGGER.info(f"Starting with map type {in_map_type} -> {out_map_type}")
                            LOGGER.timer.start("map_type")

                            for i_z, z_bin in enumerate(z_bins):

                                full_sky_bin = postprocessing._read_full_sky_bin(conf, full_maps_file, in_map_type, z_bin)

                                if sample == "WL":
                                    data_vecs = postprocessing.postprocess_wl_bin(
                                        conf,
                                        full_sky_bin,
                                        in_map_type,
                                        out_map_type,
                                        i_z,
                                        "grid",
                                        pixel_file,
                                        noise_file,
                                        full_maps_file,
                                        bgs_key=f"cosmo_{i_sobol:06d}",
                                        i_perm=i_perm,
                                        bsc_samples=bsc_samples,
                                        rng=rng_perm
                                    )
                                elif sample == "GC":
                                    data_vecs = postprocessing.postprocess_gc_bin(
                                        conf,
                                        full_sky_bin,
                                        in_map_type,
                                        out_map_type,
                                        i_z,
                                        "grid",
                                        pixel_file,
                                        i_sobol=i_sobol,
                                        rng=rng_perm,
                                    )

                                # store to temporary container
                                container_data_vecs.setdefault(z_bin, {}) # initialize if not exists
                                container_data_vecs[z_bin][out_map_type] = data_vecs

                        LOGGER.debug(f'i_perm={i_perm}, strting with {n_patches} patches')

                        def concat_probe_zbins(probe):
                            xs = []
                            for z_bin in container_data_vecs:
                                if probe in container_data_vecs[z_bin]:
                                    xs.append(container_data_vecs[z_bin][probe][..., np.newaxis])
                            if len(xs) == 0:
                                LOGGER.debug(f"concat_probe_zbins: no data for probe {probe}")
                                return [None] * n_patches
                            else:
                                LOGGER.debug(f"concat_probe_zbins: concatenating {len(xs)} z bins for probe {probe}")
                                return np.concatenate(xs, axis=-1)

                        kg_examples  = concat_probe_zbins("kg")
                        ia_examples  = concat_probe_zbins("ia")
                        ds_examples  = concat_probe_zbins("ds")
                        sn_examples  = concat_probe_zbins("sn")
                        dg_examples  = concat_probe_zbins("dg")
                        qdg_examples = concat_probe_zbins("dg2")

                        num_processed_examples = 0
                        for i_patch in range(n_patches):

                            i_signal = i_patch + i_perm * n_patches
                            LOGGER.debug(f"i_perm={i_perm}, i_patch={i_patch}, i_signal={i_signal}")

                            kg = kg_examples[i_patch]
                            ia = ia_examples[i_patch]
                            ds = ds_examples[i_patch]
                            sn_samples = sn_examples[i_patch]
                            dg = dg_examples[i_patch]
                            qdg = qdg_examples[i_patch]

                            astro_sample = astro_samples[i_signal]
                            cosmo_sample = np.concatenate([cosmo, astro_sample])

                            # to keep the indexing identical
                            if conf["analysis"]["modelling"]["WL"]["source_clustering"] == "prior":
                                astro_sample = astro_sample[:-1]

                            # lensing
                            if extended_nla:
                                Aia, n_Aia, bta = astro_sample[:3]
                            else:
                                Aia, n_Aia = astro_sample[:2]
                                bta = None

                            # clustering
                            if power_law_biasing:
                                if quadratic_biasing:
                                    bg, n_bg, qbg, n_qbg = astro_sample[-4:]
                                    tomo_qbg = redshift.get_tomo_amplitudes_according_to_config(conf, qbg, n_qbg, "gc")
                                else:
                                    bg, n_bg = astro_sample[-2:]
                                    tomo_qbg = None
                                tomo_bg = redshift.get_tomo_amplitudes_according_to_config(conf, bg, n_bg, "gc")
                            elif per_bin_biasing:
                                n_gc_bins = len(conf["survey"]["GC"]["z_bins"])
                                if quadratic_biasing:
                                    # bg1, bg2, bg3, bg4, qbg1, qbg2, qbg3, qbg4 = astro_sample[-8:]
                                    # tomo_qbg = np.array([qbg1, qbg2, qbg3, qbg4])
                                    tomo_qbg = np.array(astro_sample[-n_gc_bins:])
                                    tomo_bg = np.array(astro_sample[-2*n_gc_bins:-n_gc_bins])
                                else:
                                    # bg1, bg2, bg3, bg4 = astro_sample[-n_gc_bins:]
                                    # tomo_bg = np.array([bg1, bg2, bg3, bg4])    
                                    tomo_bg = np.array(astro_sample[-n_gc_bins:])
                                    tomo_qbg = None

                            else:
                                raise ValueError(f"Unsupported configuration of clustering bias")

                            kg, sn_samples, alm_kg, alm_sn_samples = (
                                lensing_transform(kg, ia, ds, sn_samples, Aia, n_Aia, bta, np_seed=None)
                                if store_lensing
                                else (None, None, None, None)
                            )
                            dg, pn_samples, alm_dg, alm_pn_samples = (
                                clustering_transform(dg, tomo_bg, qdg, tomo_qbg, np_seed=None)
                                if store_clustering
                                else (None, None, None, None)
                            )

                            # cross-probe maps
                            xg = None
                            xn_samples = None
                            if store_cross_maps and store_lensing and store_clustering:
                                data_vec_pix = pixel_file[0]
                                n_side = conf["analysis"]["n_side"]

                                n_z_wl = alm_kg.shape[1]
                                n_z_gc = alm_dg.shape[1]
                                n_z_cross = n_z_wl * n_z_gc

                                xg = np.zeros((kg.shape[0], n_z_cross), dtype=np.float32)
                                xn_samples = np.zeros((n_noise_per_signal, kg.shape[0], n_z_cross), dtype=np.float32)
                                ix = 0
                                for i in LOGGER.progressbar(
                                    range(n_z_wl), desc="cross bins", total=n_z_wl, at_level="debug"
                                ):
                                    for j in range(n_z_gc):
                                        alm_cross = np.sqrt(alm_kg[:, i] * alm_dg[:, j])
                                        map_cross = hp.alm2map(alm_cross, nside=n_side, pol=False)
                                        xg[:, ix] = hp.reorder(map_cross, r2n=True)[data_vec_pix]

                                        for k in range(n_noise_per_signal):
                                            alm_cross_noise = np.sqrt(alm_sn_samples[k][:, i] * alm_pn_samples[k][:, j])
                                            map_cross_noise = hp.alm2map(alm_cross_noise, nside=n_side, pol=False)
                                            xn_samples[k, :, ix] = hp.reorder(map_cross_noise, r2n=True)[data_vec_pix]

                                        ix += 1

                            # power spectra
                            cls = power_spectra.run_alm_to_cl(alm_kg, alm_sn_samples, alm_dg, alm_pn_samples)

                            sample = webdatasets.encode_grid_sample(
                                kg, sn_samples, dg, pn_samples, cls, cosmo_sample, i_sobol, i_signal, xg, xn_samples
                            )
                            sample["__key__"] = f"grid_{i_sobol:06d}_{i_signal:06d}"

                            webdatasets.verify_grid_sample(
                                sample,
                                n_noise_per_signal,
                                kg,
                                sn_samples,
                                dg,
                                pn_samples,
                                cosmo_sample,
                                i_sobol,
                                i_signal,
                                cls,
                                xg,
                                xn_samples,
                            )

                            num_processed_examples += 1
                            LOGGER.debug(f"Writing example to {wds_file} i_perm={i_perm}, i_patch={i_patch} i_signal={i_signal} kg.shape={kg.shape}, sn_samples.shape={sn_samples.shape}, dg.shape={dg.shape}, pn_samples.shape={pn_samples.shape}")
                            sink.write(sample)

                        LOGGER.info(f"Done with permutation {i_perm:04d} time taken {LOGGER.timer.elapsed('permutation')}")

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        return num_total_examples
        


def _data_vector_smoothing(dv, l_min, l_max, theta_fwhm, np_seed, conf, pixel_file, mask):
    # Gaussian Random Field
    if conf["analysis"]["modelling"]["degrade_to_grf"]:
        dv, alm = scales.data_vector_to_grf_data_vector(
            np_seed,
            dv,
            data_vec_pix=pixel_file[0],
            n_side=conf["analysis"]["n_side"],
            l_min=l_min,
            l_max=l_max,
            theta_fwhm=theta_fwhm,
            arcmin=True,
            mask=mask,
            conf=conf,
            hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
        )
    # standard smoothing with a Gaussian kernel
    else:
        dv, alm = scales.data_vector_to_smoothed_data_vector(
            dv,
            data_vec_pix=pixel_file[0],
            n_side=conf["analysis"]["n_side"],
            l_min=l_min,
            l_max=l_max,
            theta_fwhm=theta_fwhm,
            arcmin=True,
            mask=mask,
            conf=conf,
            hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
        )

    return dv, alm


def _get_lensing_transform(conf, pixel_file):
    extended_nla = conf["analysis"]["modelling"]["WL"]["extended_nla"]

    tomo_z_wl, tomo_nz_wl = files.load_redshift_distributions("WL", conf)
    m_bias_dist = lensing.get_m_bias_distribution(conf)
    wl_mask = files.get_tomo_dv_masks(conf)["WL"]

    def lensing_smoothing(kg, np_seed):
        kg, alm = _data_vector_smoothing(
            kg,
            conf["analysis"]["scale_cuts"]["WL"]["l_min"],
            conf["analysis"]["scale_cuts"]["WL"]["l_max"],
            conf["analysis"]["scale_cuts"]["WL"]["theta_fwhm"],
            np_seed,
            conf,
            pixel_file,
            wl_mask,
        )

        return kg, alm

    def lensing_transform(kg, ia, ds, sn_samples, Aia, n_Aia, bta, np_seed=None):
        # intrinsic alignment
        tomo_Aia = redshift.get_tomo_amplitudes(
            Aia,
            n_Aia,
            tomo_z_wl,
            tomo_nz_wl,
            z0=conf["survey"]["WL"]["z0"],
            truncate_nz=conf["analysis"]["modelling"]["WL"]["nla"]["truncate_nz"],
            z_min_quantile=conf["analysis"]["modelling"]["WL"]["nla"]["z_min_quantile"],
            z_max_quantile=conf["analysis"]["modelling"]["WL"]["nla"]["z_max_quantile"],
        )
        LOGGER.debug(f"Per z bin Aia = {tomo_Aia}")

        if extended_nla:
            # first two TATT terms like in eq. (19) in https://arxiv.org/pdf/2105.13544
            # NOTE ds already contains the ia map (in postprocessing.py)
            kg = kg + tomo_Aia * (ia + bta * ds)
        else:
            # standard NLA
            kg = kg + tomo_Aia * ia

        # fixing this in the WebDataset shards simplifies reproducibility
        m_bias = m_bias_dist.sample()
        kg *= 1.0 + m_bias

        kg *= wl_mask
        kg, alm_kg = lensing_smoothing(kg, np_seed)

        smooth_sn_samples, alm_sn_samples = [], []
        for i, shape_noise in enumerate(sn_samples):
            shape_noise *= wl_mask

            smooth_sn, alm_sn = lensing_smoothing(shape_noise, np_seed)

            smooth_sn_samples.append(smooth_sn)
            alm_sn_samples.append(alm_sn)

        sn_samples = np.stack(smooth_sn_samples, axis=0)
        alm_sn_samples = np.stack(alm_sn_samples, axis=0)

        return kg, sn_samples, alm_kg, alm_sn_samples

    return lensing_transform


def _get_clustering_transform(conf, pixel_file):
    n_side = conf["analysis"]["n_side"]
    n_noise_per_signal = conf["analysis"]["grid"]["n_noise_per_signal"]

    # modeling
    quadratic_biasing = conf["analysis"]["modelling"]["GC"]["quadratic_biasing"]

    gc_mask = files.get_tomo_dv_masks(conf)["GC"]
    tomo_n_gal_gc = np.array(conf["survey"]["GC"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

    # survey systematics
    if conf["analysis"]["modelling"]["GC"]["survey_systematics_map"]:
        tomo_gc_sys_dv = files.get_clustering_systematics(conf, pixel_type="data_vector")
    else:
        tomo_gc_sys_dv = None

    def clustering_smoothing(dg, np_seed):
        dg, alm = _data_vector_smoothing(
            dg,
            conf["analysis"]["scale_cuts"]["GC"]["l_min"],
            conf["analysis"]["scale_cuts"]["GC"]["l_max"],
            conf["analysis"]["scale_cuts"]["GC"]["theta_fwhm"],
            np_seed,
            conf,
            pixel_file,
            gc_mask,
        )

        return dg, alm

    def clustering_transform(
        # linear
        dg,
        tomo_bg,
        # quadratic
        qdg=None,
        tomo_qdg=None,
        # noise
        np_seed=None,
    ):
        assert (not quadratic_biasing and ((qdg is None) or (tomo_qdg is None))) or (
            quadratic_biasing and (qdg is not None) and (tomo_qdg is not None)
        ), f"The galaxy biasing setup must be consistent"
        LOGGER.debug(f"Per z bin linear bias = {tomo_bg}")

        if quadratic_biasing:
            LOGGER.debug(f"Per z bin quadratic bias = {tomo_qdg}")

        # the distinction between linear and quadratic biasing is done in main with conditional None values
        dg = clustering.galaxy_density_to_count(
            tomo_n_gal_gc,
            # linear
            dg,
            tomo_bg,
            # quadratic
            qdg,
            tomo_qdg,
            # misc
            systematics_map=tomo_gc_sys_dv,
            mask=gc_mask,
        )

        # draw noise, mask, smooth
        pn_samples = clustering.galaxy_count_to_noise(dg, n_noise_per_signal, np_seed=np_seed)

        smooth_pn_samples, alm_pn_samples = [], []
        for i, pn in enumerate(pn_samples):
            pn *= gc_mask

            smooth_pn, alm_smooth_pn = clustering_smoothing(pn, np_seed + i if np_seed is not None else None)

            smooth_pn_samples.append(smooth_pn)
            alm_pn_samples.append(alm_smooth_pn)

        pn_samples = np.stack(smooth_pn_samples, axis=0)
        alm_pn_samples = np.stack(alm_pn_samples, axis=0)

        # noiseless
        dg, alm_dg = clustering_smoothing(dg, np_seed)

        # shapes (n_pix, n_z_gc), (n_noise_per_signal, n_pix, n_z_gc)
        return dg, pn_samples, alm_dg, alm_pn_samples

    return clustering_transform





def merge(indices, args):
    args = setup(args)
    conf = files.load_config(args.config)

    n_cosmos = conf["analysis"]["grid"]["n_cosmos"]
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_signal = conf["analysis"]["grid"]["n_noise_per_signal"]
    n_signal_per_cosmo = n_patches * n_perms_per_cosmo

    webdataset_pattern = filenames.get_filename_webdataset(
        args.dir_out,
        tag=conf["survey"]["name"] + args.file_suffix,
        with_bary=conf["analysis"]["modelling"]["baryonified"],
        index=None,
        simset="grid",
        return_pattern=True,
    )
    webdataset_files = sorted(glob.glob(webdataset_pattern))

    cls_samples = (webdatasets.decode_grid_cls_sample(sample) for sample in wds.WebDataset(webdataset_files, shardshuffle=False))
    cls_dset = _batched(cls_samples, n_signal_per_cosmo)

    # separate folder on the same level as WebDataset shards
    if args.debug:
        n_cosmos = 10
        cls_dset = itertools.islice(cls_dset, n_cosmos)
        out_dir = os.path.join(args.dir_out, "../../cls/debug")
    else:
        out_dir = os.path.join(args.dir_out, "../../cls")
    os.makedirs(out_dir, exist_ok=True)
    LOGGER.info(f"Saving the results in {out_dir}")

    with h5py.File(os.path.join(out_dir, "grid_cls.h5"), "w") as f:
        for i, example in LOGGER.progressbar(
            enumerate(cls_dset),
            total=n_cosmos,
            desc="Looping through the different cosmologies in the WebDataset shards",
            at_level="info",
        ):
            cls = example["cls"]
            cosmo = example["cosmo"]
            i_sobol = example["i_sobol"]
            i_signal = example["i_signal"]

            # concatenate the noise realizations along the same axis as the examples
            cls = np.concatenate([cls[:, i, ...] for i in range(cls.shape[1])], axis=0)

            # perform the binning (all examples of a single cosmology at once)
            binned_cls, bin_edges = power_spectra.bin_according_to_config(cls, conf)

            # tiling has the same form as the above concatenation
            cosmo = np.tile(cosmo, (n_noise_per_signal, 1))
            i_sobol = np.tile(i_sobol, n_noise_per_signal)
            i_signal = np.tile(i_signal, n_noise_per_signal)

            # noise is treated separately because it's along a separate dimension in the WebDataset shards. This here preserves
            # the order imposed above in power_spectrum = ...
            i_noise = np.arange(n_noise_per_signal)
            i_noise = np.repeat(i_noise, n_signal_per_cosmo)

            if i == 0:
                f.create_dataset("cls/raw", shape=(n_cosmos,) + cls.shape, dtype="f4")
                f.create_dataset("cls/binned", shape=(n_cosmos,) + binned_cls.shape, dtype="f4")
                f.create_dataset("cls/bin_edges", shape=(n_cosmos,) + bin_edges.shape, dtype="f4")
                f.create_dataset("cosmo", shape=(n_cosmos,) + cosmo.shape, dtype="f4")
                f.create_dataset("i_sobol", shape=(n_cosmos,) + i_sobol.shape, dtype="i4")
                f.create_dataset("i_signal", shape=(n_cosmos,) + i_signal.shape, dtype="i4")
                f.create_dataset("i_noise", shape=(n_cosmos,) + i_noise.shape, dtype="i4")

            f["cls/raw"][i] = cls
            f["cls/binned"][i] = binned_cls
            f["cls/bin_edges"][i] = bin_edges
            f["cosmo"][i] = cosmo
            f["i_sobol"][i] = i_sobol
            f["i_signal"][i] = i_signal
            f["i_noise"][i] = i_noise

    LOGGER.info(f"Done with merging of the grid power spectra")

if __name__ == "__main__":

    args = setup(sys.argv[1:])
    indices = configuration.get_indices(args.indices)
    main(indices=indices, args=args)


# Code graveyard


# def resources(args):
#     args = setup(args)

#     if args.cluster == "perlmutter":
#         # because of hyperthreading, there's a total of 256 threads per node
#         resources = {
#             "main_time": 8,
#             "main_n_cores": 8,
#             "main_memory": 1952,
#             "main_scratch": 0,
#             "merge_time": 16,
#             "merge_n_cores": 32,
#             "merge_memory": 1952,
#             "merge_scratch": 0,
#         }
#     elif args.cluster == "euler":
#         resources = {"main_time": 4, "main_memory": 4096, "main_n_cores": 4, "merge_memory": 4096, "merge_n_cores": 16}

#         if args.from_san:
#             # in MB. One projected_probes_maps_v11dmb.h5 should be around 1 GB
#             resources["main_scratch"] = 4096
#         else:
#             resources["main_scratch"] = 0

#     return resources
