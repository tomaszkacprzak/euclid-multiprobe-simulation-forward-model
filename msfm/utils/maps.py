# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

""" 
Created April 2023
Author: Arne Thomsen

Tools to handle maps like conversion to and from data vectors, partially made more efficient using numba.
"""

import numpy as np
from numba import njit

from msfm.utils import logger, imports

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


@njit
def make_normallized_maps(gal_pix, e1, e2, w, n_pix):
    """
    Args:
        gal_pix (np.ndarray): shape (n_gal,), pixel ids of all of the galaxies
        e1 (np.ndarray): shape (n_gal,), ellipticities 1
        e2 (np.ndarray): shape (n_gal,), ellipticities 2
        w (np.ndarray): shape (n_gal,) Metacalibration weight
        n_pix (int): number of pixels in the maps

    Returns:
        e1_map, e2_map, abs_map, w_map, n_map: maps of size n_pix
    """
    assert gal_pix.shape == e1.shape == e2.shape == w.shape

    # maps for accumulation
    e1_map = np.zeros(n_pix)
    e2_map = np.zeros(n_pix)
    abs_map = np.zeros(n_pix)
    w_map = np.zeros(n_pix)
    n_map = np.zeros(n_pix)

    # loop over the whole catalog
    for pix_id, gamma1, gamma2, weight in zip(gal_pix, e1, e2, w):
        e1_map[pix_id] += gamma1 * weight
        e2_map[pix_id] += gamma2 * weight
        abs_map[pix_id] += weight * np.abs(gamma1 + gamma2 * 1j)
        w_map[pix_id] += weight
        n_map[pix_id] += 1

    # avoid division by zero
    w_mask = w_map != 0

    # normalize
    e1_map[w_mask] /= w_map[w_mask]
    e2_map[w_mask] /= w_map[w_mask]
    abs_map[w_mask] /= w_map[w_mask]

    return e1_map, e2_map, abs_map, w_map, n_map


@njit
def numba_assign(full_sky, data, indices):
    """Assigns data to indices in m

    Args:
        m (ndarray): A 1D array with a length of at least max(indices) + 1
        data (ndarray): The data to assign to
        indices (ndarray): The indices of data in m

    Returns:
        ndarray: m with the data assigned
    """
    n_indices = len(indices)
    for i in range(n_indices):
        full_sky[indices[i]] = data[i]
    return full_sky


@njit
def numba_transfer_map(full_sky, old_pix, new_pix):
    """This functions cuts out old_pix from full_sky map and asign it to new_pix on a new map

    Args:
        full_sky (ndarray): full sky map to cut out
        old_pix (ndarray): pixel position on the full sky map
        new_pix (ndarray): pixels where the data should be transfered to

    Returns:
        ndarray: the map with the filled in data
    """

    # prepare
    m = np.zeros_like(full_sky)
    n_data_pix = old_pix.shape[0]

    # assign
    for i in range(n_data_pix):
        m[new_pix[i]] = full_sky[old_pix[i]]

    return m


@njit
def map_to_data_vec(hp_map, data_vec_len, corresponding_pix, cutout_pix, remove_mean=False, divide_by_mean=False):
    """
    This function makes cutouts from full sky maps to a nice data vector that can be fed into a DeepSphere network

    Args:
        hp_map (np.ndarray): The full sky healpy map one should make a cutout from
        data_vec_len (int): length of the full data vec (including padding)
        corresponding_pix (np.ndarray): pixels inside the data vec that should be populated (excludes padding)
        cutout_pix (np.ndarray): pixels that should be cut out from the map (excludes padding)
        remove_mean (bool): Remove the mean within the footprint, that is without including the padding. This is
            applied to weak lensing maps.
        divide_by_mean (bool): Remove the mean and divide by the mean within the footprint, that is without including
            the padding. This is applied to galaxy clustering maps.

    Returns:
        np.ndarray: the data vec
    """
    assert not (remove_mean and divide_by_mean), "Only one of remove_mean and dividie_by_mean can be true"

    # within the patch, not over the full sky
    if hp_map.dtype == np.float32:

        if remove_mean or divide_by_mean:
            hp_mean = np.float32(np.mean(hp_map[cutout_pix]))
            if remove_mean:
                hp_map = hp_map - hp_mean
            if divide_by_mean:
                hp_map = (hp_map - hp_mean) / hp_mean

        data_vec = np.zeros(data_vec_len, dtype=np.float32)


    elif hp_map.dtype == np.complex64:

        hp_map_cutout = hp_map[cutout_pix]
        hp_map1 = np.real(hp_map_cutout)
        hp_map2 = np.imag(hp_map_cutout)

        if remove_mean or divide_by_mean:

            hp_mean1, hp_mean2 = np.float32(np.mean(hp_map1[cutout_pix])), np.float32(np.mean(hp_map2[cutout_pix]))
            if remove_mean:
                hp_map1, hp_map2 = hp_map1 - hp_mean1, hp_map2 - hp_mean2
            if divide_by_mean:
                hp_map1, hp_map2 = (hp_map1 - hp_mean1) / hp_mean1, (hp_map2 - hp_mean2) / hp_mean2 
            
            hp_map = hp_map1 + 1j*hp_map2

        data_vec = np.zeros(data_vec_len, dtype=np.complex64)

    n_indices = corresponding_pix.shape[0]
    assert corresponding_pix.shape == cutout_pix.shape

    # assign
    for i in range(n_indices):
        data_vec[corresponding_pix[i]] = hp_map[cutout_pix[i]]

    return data_vec


@njit
def data_vec_to_map(data_vec, n_pix, corresponding_pix, cutout_pix):
    """
    This function makes cutouts from full sky maps to a nice data vector that can be fed into a DeepSphere network

    Args:
        hp_map (np.ndarray): The full sky healpy map one should make a cutout from
        data_vec_len (int): length of the full data vec (including padding)
        corresponding_pix (np.ndarray): pixels inside the data vec that should be populated (excludes padding)
        cutout_pix (np.ndarray): pixels that should be cut out from the map (excludes padding)

    Returns:
        np.ndarray: the data vec
    """
    hp_map = np.zeros(n_pix, dtype=np.float32)
    n_indices = corresponding_pix.shape[0]

    assert corresponding_pix.shape == cutout_pix.shape

    # assign
    for i in range(n_indices):
        hp_map[cutout_pix[i]] = data_vec[corresponding_pix[i]]

    return hp_map


@njit
def patch_to_data_vec(patch, data_vec_len, corresponding_pix):
    """
    This function makes cutouts from full sky maps to a nice data vector that can be fed into a DeepSphere network

    Args:
        hp_map (np.ndarray): The full sky healpy map one should make a cutout from
        data_vec_len (int): length of the full data vec (including padding)
        corresponding_pix (np.ndarray): pixels inside the data vec that should be populated (excludes padding)
        cutout_pix (np.ndarray): pixels that should be cut out from the map (excludes padding)
        remove_mean (bool): Remove the mean within the footprint, that is without including the padding

    Returns:
        np.ndarray: the data vec
    """
    data_vec = np.zeros(data_vec_len, dtype=np.float32)
    n_indices = corresponding_pix.shape[0]

    assert corresponding_pix.shape == patch.shape

    # assign
    for i in range(n_indices):
        data_vec[corresponding_pix[i]] = patch[i]

    return data_vec


def tomographic_reorder(map_in, r2n=False, n2r=False):
    """Like hp.reorder, but for tomographic maps.

    Args:
        map_in (np.ndarray): Tomographic input maps of shape (n_pix, n_z_bins).
        r2n (bool, optional): RING to NEST. Defaults to False.
        n2r (bool, optional): NEST to RING. Defaults to False.

    Returns:
        np.ndarray: Reordered tomographic maps.
    """
    assert not (r2n and n2r), "Only one of r2n and n2r can be true"

    for i in range(map_in.shape[1]):
        map_in[:, i] = hp.reorder(map_in[:, i], r2n=r2n, n2r=n2r)

    return map_in
