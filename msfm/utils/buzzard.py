import os, h5py
import numpy as np
import healpy as hp

from msfm.utils import files, parameters

def get_cosmo(conf=None, params=None):
    conf = files.load_config(conf)
    params = parameters.get_parameters(params, conf)

    buzzard_cosmo = {
        "Om": 0.286, 
        "s8": 0.82, 
        "w0": -1, 
        "Aia": 0.0, 
        "n_Aia": np.nan, 
        "bta": 0.0, 
        "bg1": np.nan, 
        "bg2": np.nan, 
        "bg3": np.nan, 
        "bg4": np.nan
    }

    cosmo = {}
    for param in params:
        cosmo[param] = buzzard_cosmo[param]

    return cosmo

def get_filenames(base_dir="/pscratch/sd/j/jbucko/DESY3/mock_observations/lensing/buzzard_flock"):
    # TODO move hardcoded definitions to the config?
    I = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    J = [0, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 11, 11]
    K = ["a"] + 7 * ["a", "b"]

    lensing_files = []
    clustering_files = []
    for i, j, k in zip(I, J, K):
        lensing_file = f"{i}/DESY3_mock_observation_buzzard_flock_v14_shear_noise+WL_iseed_42_varied.h5"
        clustering_file = f"{i}/DESY3_mock_observation_Buzzard_{j}_Y3{k}.h5"

        lensing_files.append(os.path.join(base_dir, lensing_file))
        clustering_files.append(os.path.join(base_dir, clustering_file))

    return I, lensing_files, clustering_files


def get_lensing_map(lensing_file, nest_in=False, plot_diagnostics=False):
    with h5py.File(lensing_file, "r") as f_in:
        gamma1 = []
        gamma2 = []
        for j in range(1, 5):
            gamma1.append(f_in[f"metacal/raw_gamma1_bin{j}"])
            gamma2.append(f_in[f"metacal/raw_gamma2_bin{j}"])
        gamma1 = np.stack(gamma1, axis=-1)
        gamma2 = np.stack(gamma2, axis=-1)

        wl_gamma_map = np.stack([gamma1, gamma2], axis=-1)

    if plot_diagnostics:
        hp.mollview(gamma1[:, 0], nest=nest_in, title="Buzzard gamma1")
        hp.mollview(gamma2[:, 0], nest=nest_in, title="Buzzard gamma2")

    return wl_gamma_map


def get_clustering_map(clustering_file, nest_in=False, plot_diagnostics=False):
    with h5py.File(clustering_file, "r") as f_in:
        gc_count_map = []
        for i in range(1, 5):
            gc_count_map.append(f_in[f"maglim/galaxy_counts_bin{i}"][:])
        gc_count_map = np.stack(gc_count_map, axis=-1)

    if plot_diagnostics:
        hp.mollview(gc_count_map[:, 0], nest=nest_in, title="Buzzard galaxy counts")

    return gc_count_map



