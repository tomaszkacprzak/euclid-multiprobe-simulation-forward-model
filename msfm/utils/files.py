# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Functions to handle the configuration and read in the survey files on the data vector pixels, masks and noise
"""

import os, h5py, warnings
import numpy as np

from msfm.utils import logger, input_output, filenames, scales, maps, imports

hp = imports.import_healpy()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def load_config(conf=None):
    """Loads or passes through a config

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Raises:
        ValueError: When an invalid conf is passed

    Returns:
        dict: A configuration dictionary
    """
    # load the default config within this repo
    if conf is None:
        file_dir = os.path.dirname(__file__)
        repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
        conf = os.path.join(repo_dir, "configs/config.yaml")
        LOGGER.warning(f"Loading the default config from {conf}")
        conf = input_output.read_yaml(conf)

    # load a config specified by a path
    elif isinstance(conf, str):
        conf = input_output.read_yaml(conf)

    # pass through an existing config
    elif isinstance(conf, dict):
        pass

    else:
        raise ValueError(f"conf {conf} must be None, a str specifying the path to the .yaml file, or the read dict")

    return conf


def load_pixel_file(conf=None):
    """Loads the .h5 file that contains the pixel indices associated with the survey like the different patches. That
    file is generated in notebooks/survey_file_gen/pixel_file.ipynb. If the conf argument is not passed, the default
    within the directory where this file resides is used.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        data_vec_pix: data vector pixels including padding in NEST ordering (non-tomographic).
        patches_pix_dict: For "metacal" (tomographic) and "maglim" (non-tomographic), four patch indices in RING
            ordering to cut out from the full sky maps.
        corresponding_pix_dict: For "metacal" (tomographic) and "maglim" (non-tomographic), needed to convert the
            pixels in RING ordering to NEST inside the datavector.
        gamma2_signs: Signs for gamma2 that come from mirroring the survey patch, needed for Metacal only.
    """
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    pixel_file = os.path.join(repo_dir, conf["files"]["pixels"])

    with h5py.File(pixel_file, "r") as f:
        # pixel indices of padded data vector
        data_vec_pix = f["data_vec"][:]

        # Metacal sample: weak lensing
        metacal_tomo_patches_pix = []
        metacal_tomo_corresponding_pix = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (4, pix_in_bin)
            patches_pix = f[f"metacal/patches/{z_bin}"][:]
            # shape (pix_in_bin,)
            corresponding_pix = f[f"metacal/patch_to_data_vec/{z_bin}"][:]

            metacal_tomo_patches_pix.append(patches_pix)
            metacal_tomo_corresponding_pix.append(corresponding_pix)

        # to correct the shear for patch cut outs that have been mirrored
        gamma2_signs = f["metacal/gamma_2_sign"][:]

        # Maglim sample: galaxy clustering
        maglim_tomo_patches_pix = []
        maglim_tomo_corresponding_pix = []
        for z_bin in conf["survey"]["maglim"]["z_bins"]:
            patches_pix = f[f"maglim/patches/{z_bin}"][:]
            corresponding_pix = f[f"maglim/patch_to_data_vec/{z_bin}"][:]

            maglim_tomo_patches_pix.append(patches_pix)
            maglim_tomo_corresponding_pix.append(corresponding_pix)

    LOGGER.debug(f"Loaded the pixel file {pixel_file}")

    # package into dictionaries
    patches_pix_dict = {}
    patches_pix_dict["metacal"] = metacal_tomo_patches_pix
    patches_pix_dict["maglim"] = maglim_tomo_patches_pix

    corresponding_pix_dict = {}
    corresponding_pix_dict["metacal"] = metacal_tomo_corresponding_pix
    corresponding_pix_dict["maglim"] = maglim_tomo_corresponding_pix

    return data_vec_pix, patches_pix_dict, corresponding_pix_dict, gamma2_signs


def get_clustering_systematics(conf=None, pixel_type="data_vector", apply_smoothing=False):
    """Per (maglim) tomographic bin survey systematics maps packaged as data vectors, such that the maps can be
    multiplied on that level.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        pixel_type (str, optional): Either "map" or "data_vector", determines whether the systematics map is returned
            as a full sky healpy map or in data vector format.

    Returns:
        list: len = n_z_maglim
    """
    assert pixel_type in ["map", "data_vector"]

    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    pixel_file = os.path.join(repo_dir, conf["files"]["pixels"])

    with h5py.File(pixel_file, "r") as f:
        tomo_sys = []
        for z_bin in conf["survey"]["maglim"]["z_bins"]:
            tomo_sys.append(f[f"maglim/systematics/{pixel_type}/{z_bin}"][:])

    if apply_smoothing:
        # constants
        data_vec_pix, patches_pix_dict, _, _ = load_pixel_file(conf)
        n_side = conf["analysis"]["n_side"]
        n_pix = hp.nside2npix(n_side)
        tomo_l_min = conf["analysis"]["scale_cuts"]["maglim"]["l_min"]
        tomo_theta_fwhm = conf["analysis"]["scale_cuts"]["maglim"]["theta_fwhm"]

        for sys, l_min, theta_fwhm in zip(tomo_sys, tomo_l_min, tomo_theta_fwhm):
            if pixel_type == "map":
                # populate the survey footprint
                base_patch_pix = patches_pix_dict["maglim"][0]
                sys_map = np.zeros(n_pix)
                sys_map[base_patch_pix] = sys
                sys = scales.map_to_smoothed_map(sys_map, n_side, l_min, theta_fwhm=theta_fwhm)

            elif pixel_type == "data_vector":
                sys = scales.data_vector_to_smoothed_data_vector(
                    sys, data_vec_pix, n_side, l_min, theta_fwhm=theta_fwhm
                )

            else:
                raise ValueError(f"Unsupported pixel_type = {pixel_type}")

    # shape (n_pix, n_z_maglim)
    return np.stack(tomo_sys, axis=-1)


def get_tomo_dv_masks(conf=None):
    """Masks the data vectors for the different tomographic bins. (NEST ordering)

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        dict: For "metacal" (tomographic) and "maglim" (non-tomographic), mask array of shape (n_pix, n_z_bins) that
            is zero for the padding and one for the data.
    """
    data_vec_pix, _, corresponding_pix_dict, _ = load_pixel_file(conf)

    masks_metacal = []
    # loop over the tomographic bins
    for pix in corresponding_pix_dict["metacal"]:
        mask = np.zeros(len(data_vec_pix), dtype=np.int32)
        # loop over individual pixels
        for p in pix:
            mask[p] = 1
        masks_metacal.append(mask)

    masks_maglim = []
    # loop over the tomographic bins
    for pix in corresponding_pix_dict["maglim"]:
        mask = np.zeros(len(data_vec_pix), dtype=np.int32)
        # loop over individual pixels
        for p in pix:
            mask[p] = 1
        masks_maglim.append(mask)

    masks_dict = {
        "metacal": np.array(masks_metacal).T,
        "maglim": np.array(masks_maglim).T,
    }

    return masks_dict


def get_dv_mask(conf=None):
    masks_dict = get_tomo_dv_masks(conf)

    assert np.all(masks_dict["metacal"] == masks_dict["maglim"]), "The masks for metacal and maglim should be the same"
    assert np.all(
        masks_dict["metacal"] == masks_dict["metacal"][:, 0][:, None]
    ), "The mask should be the same for all tomographic bins"

    return masks_dict["metacal"][:, 0].astype(bool)


def get_tomo_masks(conf=None, nest_out=True):
    conf = load_config(conf)

    n_pix = hp.nside2npix(conf["analysis"]["n_side"])
    data_vec_pix, _, _, _ = load_pixel_file(conf)
    dv_masks_dict = get_tomo_dv_masks(conf)

    masks_dict = {}
    for sample in dv_masks_dict.keys():
        dv_masks = dv_masks_dict[sample]
        masks = np.zeros((n_pix, dv_masks.shape[-1]))
        masks[data_vec_pix] = dv_masks

        if nest_out == False:
            masks = maps.tomographic_reorder(masks, n2r=True)

        masks_dict[sample] = masks

    return masks_dict


def get_mask(conf=None, nest_out=True):
    masks_dict = get_tomo_masks(conf, nest_out)

    assert np.all(masks_dict["metacal"] == masks_dict["maglim"]), "The masks for metacal and maglim should be the same"
    assert np.all(
        masks_dict["metacal"] == masks_dict["metacal"][:, 0][:, None]
    ), "The mask should be the same for all tomographic bins"

    return masks_dict["metacal"][:, 0].astype(bool)


def load_noise_file(conf=None):
    """Loads the .h5 file that contains the noise information of the survey. That
    file is generated in notebooks/survey_file_gen/noise_file.ipynb

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        tomo_gamma_cat: list for the tomographic bins containing all of the gamma values for the galaxies in the survey
    """
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    noise_file = os.path.join(repo_dir, conf["files"]["noise"])

    with h5py.File(noise_file, "r") as f:
        tomo_gamma_cat = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (n_gal, 3) with e1, e2, w
            gamma_cat = f[f"{z_bin}/cat"][:]

            tomo_gamma_cat.append(gamma_cat)
    LOGGER.info(f"Loaded the noise file")

    return tomo_gamma_cat


def load_redshift_distributions(galaxy_sample, conf=None):
    """Load the redshift distributions from disk to memory.

    Args:
        galaxy_sample (str): Either "metacal" or "maglim".
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        list: Per redshift bin z an nz values of the distribution.
    """
    assert galaxy_sample in ["maglim", "metacal"]

    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    redshift_dir = os.path.join(repo_dir, conf["dirs"]["redshift_distributions"])

    n_z_bins = len(conf["survey"][galaxy_sample]["z_bins"])

    tomo_z = []
    tomo_nz = []
    for i_tomo in range(1, n_z_bins + 1):
        z_dist_file = filenames.get_filename_z_distribution(redshift_dir, galaxy_sample, i_tomo)
        z_dist = np.loadtxt(z_dist_file)

        tomo_z.append(z_dist[:, 0])
        tomo_nz.append(z_dist[:, 1])

    return tomo_z, tomo_nz


def read_metacal_bias(key, conf=None):
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    metacal_bias_file = os.path.join(repo_dir, conf["files"]["metacal_bias"])
    with h5py.File(metacal_bias_file, "r") as f:
        metacal_bias = f[key][:]

    return np.array(metacal_bias)
