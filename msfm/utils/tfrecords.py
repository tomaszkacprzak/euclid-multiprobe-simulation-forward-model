# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

This file is based off
https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/utils_tfrecords.py
by Tomasz Kacprzak and
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/data.py
by Janis Fluri and see
https://www.tensorflow.org/tutorials/load_data/tfrecord
https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
"""

import warnings
import tensorflow as tf
import numpy as np
import torch

from msfm.utils import logger, cross_statistics

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)



def parse_forward_grid(kg, sn_realz, dg, pn_realz, cls, cosmo, i_sobol, i_signal, xg=None, xn_realz=None):
    """The grid cosmologies contain all of the maps and labels.

    Args:
        kg (np.ndarray): shape(n_pix, n_z_metacal), includes the sum of an original kg and ia map.
        sn_realz (np.ndarray): shape(n_noise, n_pix, n_z_metacal), shape noise consistent with the kg map.
        dg (np.ndarray): shape (n_pix, n_z_maglim), a map of galaxy counts (not just density contrast).
        pn_realz (np.ndarray): shape(n_noise, n_pix, n_z_maglim), poisson noise consistent with the dg map.
        cosmo (np.ndarray): shape(n_params) to be used as a label.
        cls (np.ndarray): Auto and cross bin (both in terms of the tomographic bins, and the two probes) power spectra.
            The shape is (n_noise, n_cls, n_z_cross).
        i_sobol (int): Seed within the Sobol sequence.
        i_signal (int): Example index, which is determined by the simulation run and the patch.
        xg (np.ndarray, optional): shape(n_pix, n_z_cross), cross-maps between kg and dg. Defaults to None.
        xn_realz (np.ndarray, optional): shape(n_noise, n_pix, n_z_cross), noise realizations for the cross-maps.
            Defaults to None.

    Returns:
        tf.train.Example: Example containing all of these tensors.
    """
    # LOGGER.warning(f"Tracing parse_forward_grid")

    features = {
        # labels
        "cosmo": _bytes_feature(tf.io.serialize_tensor(cosmo)),
        "n_params": _int64_feature(cosmo.shape[0]),
        "i_sobol": _int64_feature(i_sobol),
        "i_signal": _int64_feature(i_signal),
    }

    if cls is not None:
        features["cls"] = _bytes_feature(tf.io.serialize_tensor(cls))
        features["n_noise"] = _int64_feature(cls.shape[0])
        features["n_cls"] = _int64_feature(cls.shape[1])
        features["n_z_cross"] = _int64_feature(cls.shape[2])

    if kg is not None and sn_realz is not None:
        assert kg.shape == sn_realz.shape[1:]
        features["n_pix"] = _int64_feature(kg.shape[0])
        features["n_z_metacal"] = _int64_feature(kg.shape[1])
        for i, sn in enumerate(sn_realz):
            features[f"kg_{i}"] = _bytes_feature(tf.io.serialize_tensor(kg + sn)) # here we add the noise map to the signal map

    if dg is not None and pn_realz is not None:
        assert dg.shape == pn_realz.shape[1:]
        if kg is None:
            features["n_pix"] = _int64_feature(dg.shape[0])
        else:
            assert kg.shape[0] == dg.shape[0]
        features["n_z_maglim"] = _int64_feature(dg.shape[1])
        for i, pn in enumerate(pn_realz):
            features[f"dg_{i}"] = _bytes_feature(tf.io.serialize_tensor(dg + pn))

    # cross-maps
    if (xg is not None) and (xn_realz is not None):
        features["n_z_cross_map"] = _int64_feature(xg.shape[1])

        for i, xn in enumerate(xn_realz):
            features[f"xg_{i}"] = _bytes_feature(tf.io.serialize_tensor(xg + xn))

    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parse_inverse_grid(
    serialized_example,
    noise_indices=[0],
    # shapes
    n_pix=None,
    n_z_metacal=None,
    n_z_maglim=None,
    n_z_cross_map=None,
    n_z_cross=None,
    n_params=None,
    n_noise=None,
    n_cls=None,
    # probes
    with_lensing=True,
    with_clustering=True,
    with_cross=False,
    return_maps=True,
    return_cls=True,
):
    """Use the same structure as in in the forward pass above. Note that n_pix, n_z_bins and n_params have to be passed
    as function arguments to ensure that the function can be converted to a graph.

    Args:
        serialized_example (tf.train.Example.SerializeToString()): The stored data.
        noise_indices (Union[list, range], optional): Realizations corresponding to these noise indices are returned.
            Defaults to [0].
        n_pix (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_metacal (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_maglim (int, optional): Fixes the size of the tensors. Defaults to None.
        n_params (int, optional): Fixes the size of the tensors. Defaults to None.
        with_lensing (bool, optional): Whether to return the weak lensing maps. Defaults to True.
        with_clustering (bool, optional): Whether to return the galaxy clustering maps. Defaults to True.
        with_cross_only (bool, optional): Whether to return only the cross maps. Defaults to False.
        return_cls (bool, optional): Whether to return the cls. Defaults to True.
    Returns:
        dict: Dictionary containing the tensors for the different fields, the cosmological parameters and indices
        i_sobol and i_signal.
    """
    # LOGGER.warning(f"Tracing parse_inverse_grid")

    features = {
        # tensor shapes
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_params": tf.io.FixedLenFeature([], tf.int64),
        # labels
        "cosmo": tf.io.FixedLenFeature([], tf.string),
        "i_sobol": tf.io.FixedLenFeature([], tf.int64),
        "i_signal": tf.io.FixedLenFeature([], tf.int64),
    }

    if return_cls:
        features["cls"] = tf.io.FixedLenFeature([], tf.string)
        features["n_noise"] = tf.io.FixedLenFeature([], tf.int64)
        features["n_cls"] = tf.io.FixedLenFeature([], tf.int64)
        features["n_z_cross"] = tf.io.FixedLenFeature([], tf.int64)

    if return_maps:
        if with_lensing:
            for i in noise_indices:
                features[f"kg_{i}"] = tf.io.FixedLenFeature([], tf.string)

        if with_clustering:
            for i in noise_indices:
                features[f"dg_{i}"] = tf.io.FixedLenFeature([], tf.string)

        if with_cross:
            for i in noise_indices:
                features[f"xg_{i}"] = tf.io.FixedLenFeature([], tf.string)

    if with_lensing and (return_maps or return_cls):
        features["n_z_metacal"] = tf.io.FixedLenFeature([], tf.int64)

    if with_clustering and (return_maps or return_cls):
        features["n_z_maglim"] = tf.io.FixedLenFeature([], tf.int64)

    if with_cross and return_maps:
        features["n_z_cross_map"] = tf.io.FixedLenFeature([], tf.int64)

    serialized_data = tf.io.parse_single_example(serialized_example, features)

    # output container
    output_data = {}

    cosmo = tf.io.parse_tensor(serialized_data["cosmo"], out_type=tf.float32)
    if n_params is None:
        cosmo = tf.reshape(cosmo, shape=(serialized_data["n_params"],))
    else:
        cosmo = tf.ensure_shape(cosmo, shape=(n_params,))
    output_data["cosmo"] = cosmo

    for i in noise_indices:
        if return_maps:
            if with_lensing:
                output_data = _parse_and_reshape_data_vector(
                    output_data, serialized_data, f"kg_{i}", f"kg_{i}", n_pix, n_z_metacal, "n_z_metacal"
                )

            if with_clustering:
                output_data = _parse_and_reshape_data_vector(
                    output_data, serialized_data, f"dg_{i}", f"dg_{i}", n_pix, n_z_maglim, "n_z_maglim"
                )

            if with_cross:
                output_data = _parse_and_reshape_data_vector(
                    output_data, serialized_data, f"xg_{i}", f"xg_{i}", n_pix, n_z_cross_map, "n_z_cross_map"
                )

        if return_cls:
            n_z_mc = _parse_none_value(serialized_data, "n_z_metacal", n_z_metacal) if with_lensing else 0
            n_z_ml = _parse_none_value(serialized_data, "n_z_maglim", n_z_maglim) if with_clustering else 0
            bin_indices, _ = cross_statistics.get_cross_bin_indices(
                n_z_mc,
                n_z_ml,
                with_lensing,
                with_clustering,
                with_cross_z=True,
                with_cross_probe=(with_lensing and with_clustering),
            )

            _parse_and_reshape_cls(
                output_data,
                serialized_data,
                f"cls",
                f"cl_{i}",
                n_noise,
                n_cls,
                n_z_cross,
                i,
                bin_indices,
            )

    # indices
    output_data["i_sobol"] = serialized_data["i_sobol"]
    output_data["i_signal"] = serialized_data["i_signal"]

    return output_data


def parse_inverse_grid_cls(
    serialized_example,
    # shapes
    n_noise=None,
    n_cls=None,
    n_z_cross=None,
    n_params=None,
):
    """
    Use the same structure as in in the forward pass above, but only return the data associated with the power spectra.
    Note that n_noise, n_cls, n_z_cross and n_params have to be passed as function arguments to ensure that the
    function can be converted to a graph.

    Args:
        serialized_example (tf.train.Example.SerializeToString()): The stored data.
        n_noise (int, optional): Number of noise realizations to return, where the noise index always runs from 0 to
            n_noise - 1. Defaults to 1.
        n_cls (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_cross (int, optional): Fixes the size of the tensors. Defaults to None.
        n_params (int, optional): Fixes the size of the tensors. Defaults to None.

    Returns:
        tf.tensors, int: Tensors containing the different fields, the cosmological parameters and indices i_sobol and
            i_signal.
    """

    features = {
        "cls": tf.io.FixedLenFeature([], tf.string),
        # tensor shapes
        "n_noise": tf.io.FixedLenFeature([], tf.int64),
        "n_cls": tf.io.FixedLenFeature([], tf.int64),
        "n_z_cross": tf.io.FixedLenFeature([], tf.int64),
        "n_params": tf.io.FixedLenFeature([], tf.int64),
        # labels
        "cosmo": tf.io.FixedLenFeature([], tf.string),
        "i_sobol": tf.io.FixedLenFeature([], tf.int64),
        "i_signal": tf.io.FixedLenFeature([], tf.int64),
    }

    serialized_data = tf.io.parse_single_example(serialized_example, features)

    # output container
    output_data = {}

    # power spectra
    cls = tf.io.parse_tensor(serialized_data["cls"], out_type=tf.float32)
    if n_noise is None and n_cls is None and n_z_cross is None:
        cls = tf.reshape(
            cls, shape=(serialized_data["n_noise"], serialized_data["n_cls"], serialized_data["n_z_cross"])
        )
    else:
        cls = tf.ensure_shape(cls, shape=(n_noise, n_cls, n_z_cross))
    output_data["cls"] = cls

    # cosmology label
    cosmo = tf.io.parse_tensor(serialized_data["cosmo"], out_type=tf.float32)
    if n_params is None:
        cosmo = tf.reshape(cosmo, shape=(serialized_data["n_params"],))
    else:
        cosmo = tf.ensure_shape(cosmo, shape=(n_params,))
    output_data["cosmo"] = cosmo

    # indices
    output_data["i_sobol"] = serialized_data["i_sobol"]
    output_data["i_signal"] = serialized_data["i_signal"]

    return output_data


def parse_forward_fiducial(
    cosmo_pert_labels,
    kg_perts,
    dg_perts,
    # lensing
    ia_pert_labels,
    ia_perts,
    sn_realz,
    # clustering
    bg_pert_labels,
    bg_perts,
    pn_realz,
    # power spectra
    cl_perts,
    cl_ia_perts,
    cl_bg_perts,
    # label
    i_signal,
):
    """The fiducials don't need a label and contain the perturbation for the delta loss with
    n_perts = 2 * n_params + 1

    Args:
        cosmo_pert_labels (list): Dictionary keys of length n_cosmo_perts and string elements. These are the
            cosmological parameters and common to both kg and dg.
        kg_perts (list): Kappa perturbations of length n_perts and elements of shape(n_pix, n_z_metacal).
        dg_perts (list): Delta perturbations of length n_perts and elements of shape(n_pix, n_z_maglim).
        ia_pert_labels (list): Dictionary keys for the intrinsic alignment perturbations, which only affect kg.
        ia_perts (list): Same length as ia_pert_labels, these are the perturbed kg tensors.
        sn_realz (np.ndarray): Shape noise realizations of shape(n_noise, n_pix, n_z_metacal).
        bg_pert_labels (list): Dictionary keys for the galaxy clustering perturbations, which only affect dg.
        bg_perts (list): Same length as bg_pert_labels, these are the perturbed dg tensors.
        pn_realz (np.ndarray): Poisson noise realizations of shape(n_noise, n_pix, n_z_maglim).
        cls (np.ndarray): Auto and cross bin (both in terms of the tomographic bins, and the two probes) power spectra.
            The shape is (n_noise, n_cls, n_z_cross).
        i_signal (int): example index (comes from simulation run and the patch), there are
            n_perms_per_cosmo * n_patches.

    Returns:
        tf.train.Example: Example containing all of these tensors.
    """
    # LOGGER.warning(f"Tracing parse_forward_fiducial")

    # the number of perturbations is the same
    assert len(kg_perts) == len(dg_perts) == len(cosmo_pert_labels)
    assert len(ia_pert_labels) == len(ia_perts)
    assert len(bg_pert_labels) == len(bg_perts)

    # the data vector dimension matches (while n_z does not)
    for kg_pert, dg_pert in zip(kg_perts, dg_perts):
        assert kg_pert.shape[0] == dg_pert.shape[0] == sn_realz.shape[1] == pn_realz.shape[1]

    assert (
        len(sn_realz) == len(pn_realz) == cl_perts.shape[1]
    ), "the number of noise realizations has to be identical for sn, pn and the cls"

    # define the structure of a single example
    features = {
        # tensor shapes
        "n_pix": _int64_feature(kg_perts[0].shape[0]),
        "n_z_metacal": _int64_feature(kg_perts[0].shape[1]),
        "n_z_maglim": _int64_feature(dg_perts[0].shape[1]),
        # label
        "i_signal": _int64_feature(i_signal),
        # power spectra
        "cls": _bytes_feature(tf.io.serialize_tensor(cl_perts[0])),
        "n_noise": _int64_feature(cl_perts.shape[1]),
        "n_cls": _int64_feature(cl_perts.shape[2]),
        "n_z_cross": _int64_feature(cl_perts.shape[3]),
    }

    # cosmological perturbations (kappa and delta)
    for label, kg_pert, dg_pert, cl_pert in zip(cosmo_pert_labels, kg_perts, dg_perts, cl_perts):
        features[f"kg_{label}"] = _bytes_feature(tf.io.serialize_tensor(kg_pert))
        features[f"dg_{label}"] = _bytes_feature(tf.io.serialize_tensor(dg_pert))
        features[f"cl_{label}"] = _bytes_feature(tf.io.serialize_tensor(cl_pert))

    # intrinsic alignment perturbations (kappa)
    for label, ia_pert, cl_ia_pert in zip(ia_pert_labels, ia_perts, cl_ia_perts):
        features[f"kg_{label}"] = _bytes_feature(tf.io.serialize_tensor(ia_pert))
        features[f"cl_{label}"] = _bytes_feature(tf.io.serialize_tensor(cl_ia_pert))

    # shape noise realizations
    for i, sn in enumerate(sn_realz):
        features[f"sn_{i}"] = _bytes_feature(tf.io.serialize_tensor(sn))

    # galaxy biasing (delta)
    for label, bg_pert, cl_bg_pert in zip(bg_pert_labels, bg_perts, cl_bg_perts):
        features[f"dg_{label}"] = _bytes_feature(tf.io.serialize_tensor(bg_pert))
        features[f"cl_{label}"] = _bytes_feature(tf.io.serialize_tensor(cl_bg_pert))

    # poisson noise realizations
    for i, pn in enumerate(pn_realz):
        features[f"pn_{i}"] = _bytes_feature(tf.io.serialize_tensor(pn))

    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parse_inverse_fiducial(
    serialized_example,
    pert_labels,
    noise_indices=[0],
    # shapes
    n_pix=None,
    n_z_metacal=None,
    n_z_maglim=None,
    n_noise=None,
    n_cls=None,
    n_z_cross=None,
    # probes
    with_lensing=True,
    with_clustering=True,
    return_maps=True,
    return_cls=True,
):
    """Use the same structure as in in the forward pass above. Note that n_pix and n_z_bins have to be passed as
    arguments to ensure that the function can be converted to a graph.

    Args:
        serialized_example (tf.train.Example.SerializeToString()): The data loaded from the .tfrecord file.
        pert_labels (list): List of strings that contain the labels defining the keys. These include all parameters,
            so cosmological and astrophysics (intrinsic alignment and galaxy clustering).
        noise_indices (Union[list, range], optional): Realizations corresponding to these noise indices are returned.
            Defaults to [0].
        n_pix (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_metacal (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_maglim (int, optional): Fixes the size of the tensors. Defaults to None.
        with_lensing (bool, optional): Whether the weak lensing maps should be returned or not. Defaults to True.
        with_clustering (bool, optional): Whether the galaxy clustering maps should be returned or not. Defaults to
            True.

    Returns:
        dict, int: Dictionary of datavectors (fiducial, perturbations and shape noise) and the patch index (i_signal).
    """
    # LOGGER.warning(f"Tracing parse_inverse_fiducial")

    features = {
        # tensor shapes, not recommended as reshaping with respect to them leads to a None shape in tf.function
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_metacal": tf.io.FixedLenFeature([], tf.int64),
        "n_z_maglim": tf.io.FixedLenFeature([], tf.int64),
        "n_noise": tf.io.FixedLenFeature([], tf.int64),
        "n_cls": tf.io.FixedLenFeature([], tf.int64),
        "n_z_cross": tf.io.FixedLenFeature([], tf.int64),
        # label
        "i_signal": tf.io.FixedLenFeature([], tf.int64),
    }

    # all perturbation parameters
    for label in pert_labels:
        if return_maps:
            # kappa: cosmological + intrinsic alignment parameters
            if with_lensing and (not "bg" in label):
                features[f"kg_{label}"] = tf.io.FixedLenFeature([], tf.string)

            # delta: cosmological + galaxy clustering parameters
            if with_clustering and (not "Aia" in label):
                features[f"dg_{label}"] = tf.io.FixedLenFeature([], tf.string)

        features[f"cl_{label}"] = tf.io.FixedLenFeature([], tf.string)

    if return_maps:
        # all desired noise realizations
        for i in noise_indices:
            if with_lensing:
                # shape noise
                features[f"sn_{i}"] = tf.io.FixedLenFeature([], tf.string)

            if with_clustering:
                # poisson noise
                features[f"pn_{i}"] = tf.io.FixedLenFeature([], tf.string)

    serialized_data = tf.io.parse_single_example(serialized_example, features)

    # output container
    output_data = {}

    bin_indices, _ = cross_statistics.get_cross_bin_indices(
        _parse_none_value(serialized_data, "n_z_metacal", n_z_metacal) if with_lensing else 0,
        _parse_none_value(serialized_data, "n_z_maglim", n_z_maglim) if with_clustering else 0,
        with_lensing,
        with_clustering,
        with_cross_z=True,
        with_cross_probe=(with_lensing and with_clustering),
    )

    # all perturbation parameters
    for label in pert_labels:
        if return_maps:
            # kappa: cosmological + intrinsic alignment parameters
            if with_lensing and (not "bg" in label):
                output_data = _parse_and_reshape_data_vector(
                    output_data, serialized_data, f"kg_{label}", f"kg_{label}", n_pix, n_z_metacal, "n_z_metacal"
                )

            # delta: cosmological + galaxy clustering parameters
            if with_clustering and (not "Aia" in label):
                output_data = _parse_and_reshape_data_vector(
                    output_data, serialized_data, f"dg_{label}", f"dg_{label}", n_pix, n_z_maglim, "n_z_maglim"
                )

        if return_cls:
            _parse_and_reshape_cls(
                output_data,
                serialized_data,
                f"cl_{label}",
                f"cl_{label}",
                n_noise,
                n_cls,
                n_z_cross,
                noise_indices,
                bin_indices,
            )

    if return_maps:
        # all desired noise realizations
        for i in noise_indices:
            # shape noise
            if with_lensing:
                output_data = _parse_and_reshape_data_vector(
                    output_data, serialized_data, f"sn_{i}", f"sn_{i}", n_pix, n_z_metacal, "n_z_metacal"
                )

            # poisson noise
            if with_clustering:
                output_data = _parse_and_reshape_data_vector(
                    output_data, serialized_data, f"pn_{i}", f"pn_{i}", n_pix, n_z_maglim, "n_z_maglim"
                )

    # indices
    output_data["i_signal"] = serialized_data["i_signal"]

    return output_data


def parse_inverse_fiducial_cls(
    serialized_example,
    # shapes
    n_noise=None,
    n_cls=None,
    n_z_cross=None,
):
    """
    Use the same structure as in in the forward pass above, but only return the data associated with the power spectra.
    Note that n_noise, n_cls, n_z_cross and n_params have to be passed as function arguments to ensure that the
    function can be converted to a graph.

    Args:
        serialized_example (tf.train.Example.SerializeToString()): The data loaded from the .tfrecord file.
        n_noise (int, optional): Number of noise realizations to return, where the noise index always runs from 0 to
            n_noise - 1. Defaults to 1.
        n_cls (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_cross (int, optional): Fixes the size of the tensors. Defaults to None.
        n_params (int, optional): Fixes the size of the tensors. Defaults to None.

    Returns:
        dict, int: Dictionary of datavectors (fiducial, perturbations and shape noise) and the patch index (i_signal).
    """

    features = {
        "cls": tf.io.FixedLenFeature([], tf.string),
        # tensor shapes
        "n_noise": tf.io.FixedLenFeature([], tf.int64),
        "n_cls": tf.io.FixedLenFeature([], tf.int64),
        "n_z_cross": tf.io.FixedLenFeature([], tf.int64),
        # labels
        "i_signal": tf.io.FixedLenFeature([], tf.int64),
    }

    serialized_data = tf.io.parse_single_example(serialized_example, features)

    # output container
    output_data = {}

    # power spectra
    cls = tf.io.parse_tensor(serialized_data["cls"], out_type=tf.float32)
    if n_noise is None and n_cls is None and n_z_cross is None:
        cls = tf.reshape(
            cls, shape=(serialized_data["n_noise"], serialized_data["n_cls"], serialized_data["n_z_cross"])
        )
    else:
        cls = tf.ensure_shape(cls, shape=(n_noise, n_cls, n_z_cross))
    output_data["cls"] = cls

    # indices
    output_data["i_signal"] = serialized_data["i_signal"]

    return output_data


# helper functions ####################################################################################################


def _parse_and_reshape_data_vector(out_dict, serialized_data, key_in, key_out, n_pix, n_z_bins, n_z_bins_label):
    tensor = tf.io.parse_tensor(serialized_data[key_in], out_type=tf.float32)

    if (n_pix is None) or (n_z_bins is None):
        # reshape allows for None shapes within the graph, but is slower
        tensor = tf.reshape(tensor, shape=(serialized_data["n_pix"], serialized_data[n_z_bins_label]))
    else:
        # tf.ensure_shape fixes the shape inside the graph
        tensor = tf.ensure_shape(tensor, shape=(n_pix, n_z_bins))

    out_dict[key_out] = tensor

    return out_dict


def _parse_and_reshape_cls(
    out_dict, serialized_data, key_in, key_out, n_noise, n_cls, n_z_cross, noise_indices, bin_indices
):
    cls = tf.io.parse_tensor(serialized_data[key_in], out_type=tf.float32)

    if n_noise is None and n_cls is None and n_z_cross is None:
        cls = tf.reshape(
            cls, shape=(serialized_data["n_noise"], serialized_data["n_cls"], serialized_data["n_z_cross"])
        )
    else:
        cls = tf.ensure_shape(cls, shape=(n_noise, n_cls, n_z_cross))

    cls = tf.gather(cls, noise_indices, axis=0)
    cls = tf.gather(cls, bin_indices, axis=-1)

    out_dict[key_out] = cls

    return out_dict


def _parse_none_value(serialized_example, key, value):
    if value is None:
        value = serialized_example[key]
    return value


# features ############################################################################################################


# https://www.tensorflow.org/tutorials/load_data/tfrecord#data_types_for_tftrainexample
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


## verification ########################################################################################################

def verify_tfrecord(
    serialized,
    n_noise_per_signal,
    kg,
    sn_samples,
    dg,
    pn_samples,
    cosmo,
    i_sobol,
    i_signal,
    cls,
    xg=None,
    xn_samples=None,
):
    with_cross_probe = xg is not None and xn_samples is not None
    with_lensing = kg is not None and sn_samples is not None
    with_clustering = dg is not None and pn_samples is not None

    inv_tfr = parse_inverse_grid(
        serialized,
        range(n_noise_per_signal),
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross=with_cross_probe,
        return_cls=cls is not None,
    )

    for i_noise in range(n_noise_per_signal):
        if with_lensing:
            assert np.allclose(inv_tfr[f"kg_{i_noise}"], kg + sn_samples[i_noise])
        if with_clustering:
            assert np.allclose(inv_tfr[f"dg_{i_noise}"], dg + pn_samples[i_noise])
        if cls is not None:
            assert np.allclose(inv_tfr[f"cl_{i_noise}"], cls[i_noise])
        if with_cross_probe:
            assert np.allclose(inv_tfr[f"xg_{i_noise}"], xg + xn_samples[i_noise])
    assert np.allclose(inv_tfr["cosmo"], cosmo)
    assert np.allclose(inv_tfr["i_sobol"], i_sobol)
    assert np.allclose(inv_tfr["i_signal"], i_signal)
    LOGGER.debug("Decoded the map part of the .tfrecord successfully")

    if cls is not None:
        inv_cls = parse_inverse_grid_cls(serialized)

        assert np.allclose(inv_cls["cls"], cls)
        assert np.allclose(inv_cls["cosmo"], cosmo)
        assert np.allclose(inv_cls["i_sobol"], i_sobol)
        assert np.allclose(inv_cls["i_signal"], i_signal)
        LOGGER.debug("Decoded the cls part of the .tfrecord successfully")


## onthefly ############################################################################################################


def parse_forward_onthefly(γg, γa, γd, ds, dg, qg, cosmo, i_sobol, i_signal):
    """The grid cosmologies contain all of the maps and labels.

    Args:
        γg (np.ndarray): shape(n_pix, n_z_WL), lensing shear map.
        γa (np.ndarray): shape(n_pix, n_z_WL), linear IA map.
        γd (np.ndarray): shape(n_pix, n_z_WL), delta-NLA IA map.
        ds (np.ndarray): shape(n_pix, n_z_WL), delta of source galaxies.
        dg (np.ndarray): shape(n_pix, n_z_GC), linear galaxy counts.
        qg (np.ndarray): shape(n_pix, n_z_GC), quadratic galaxy counts.
        cosmo (np.ndarray): shape(n_params) to be used as a label.
        i_sobol (int): Seed within the Sobol sequence.
        i_signal (int): Example index, which is determined by the simulation run and the patch.

    Returns:
        tf.train.Example: Example containing all of these tensors.
    """
    # LOGGER.warning(f"Tracing parse_forward_grid")

    features = {
        # labels
        "cosmo": _bytes_feature(tf.io.serialize_tensor(cosmo)),
        "n_params": _int64_feature(cosmo.shape[0]),
        "i_sobol": _int64_feature(i_sobol),
        "i_signal": _int64_feature(i_signal),

        "n_pix": _int64_feature(γg.shape[0]),
        "n_z_WL": _int64_feature(γg.shape[1]),
        "n_z_GC": _int64_feature(dg.shape[1]),

        "γg": _bytes_feature(tf.io.serialize_tensor(γg)),
        "γa": _bytes_feature(tf.io.serialize_tensor(γa)),
        "γd": _bytes_feature(tf.io.serialize_tensor(γd)),
        "ds": _bytes_feature(tf.io.serialize_tensor(ds)),
        "dg": _bytes_feature(tf.io.serialize_tensor(dg)),
        "qg": _bytes_feature(tf.io.serialize_tensor(qg)),
    }
    
    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


import numpy as np
import tensorflow as tf


# Defaults used when parsing from a tf.data pipeline.
# These MUST match the dtype used when writing with tf.io.serialize_tensor.
# If your arrays are float64, change these defaults to tf.float64,
# or pass tensor_dtypes explicitly as shown in verify_tfrecord_onthefly.
_DEFAULT_TENSOR_DTYPES = {
    "cosmo": tf.float32,
    "γg": tf.complex64,
    "γa": tf.complex64,
    "γd": tf.complex64,
    "ds": tf.float32,
    "dg": tf.float32,
    "qg": tf.float32,
}


def parse_inverse_onthefly(serialized_example, tensor_dtypes=None):
    """Parse one serialized TFRecord example written by parse_forward_onthefly.

    Args:
        serialized_example: A scalar string Tensor or raw serialized Example bytes.
        tensor_dtypes: Optional dict mapping tensor field names to TensorFlow dtypes.
            This is useful because tf.io.parse_tensor needs the dtype explicitly.
            Example:
                {
                    "cosmo": tf.float64,
                    "γg": tf.float32,
                    ...
                }

    Returns:
        dict: Parsed tensors and integer metadata.
    """
    dtypes = dict(_DEFAULT_TENSOR_DTYPES)
    if tensor_dtypes is not None:
        dtypes.update({k: tf.as_dtype(v) for k, v in tensor_dtypes.items()})

    feature_description = {
        # labels / metadata
        "cosmo": tf.io.FixedLenFeature([], tf.string),
        "n_params": tf.io.FixedLenFeature([], tf.int64),
        "i_sobol": tf.io.FixedLenFeature([], tf.int64),
        "i_signal": tf.io.FixedLenFeature([], tf.int64),

        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_WL": tf.io.FixedLenFeature([], tf.int64),
        "n_z_GC": tf.io.FixedLenFeature([], tf.int64),

        # maps
        "γg": tf.io.FixedLenFeature([], tf.string),
        "γa": tf.io.FixedLenFeature([], tf.string),
        "γd": tf.io.FixedLenFeature([], tf.string),
        "ds": tf.io.FixedLenFeature([], tf.string),
        "dg": tf.io.FixedLenFeature([], tf.string),
        "qg": tf.io.FixedLenFeature([], tf.string),
    }

    features = tf.io.parse_single_example(serialized_example, feature_description)

    n_params = features["n_params"]
    n_pix = features["n_pix"]
    n_z_WL = features["n_z_WL"]
    n_z_GC = features["n_z_GC"]

    wl_shape = tf.stack([n_pix, n_z_WL])
    gc_shape = tf.stack([n_pix, n_z_GC])

    cosmo = tf.io.parse_tensor(features["cosmo"], out_type=dtypes["cosmo"])
    γg = tf.io.parse_tensor(features["γg"], out_type=dtypes["γg"])
    γa = tf.io.parse_tensor(features["γa"], out_type=dtypes["γa"])
    γd = tf.io.parse_tensor(features["γd"], out_type=dtypes["γd"])
    ds = tf.io.parse_tensor(features["ds"], out_type=dtypes["ds"])
    dg = tf.io.parse_tensor(features["dg"], out_type=dtypes["dg"])
    qg = tf.io.parse_tensor(features["qg"], out_type=dtypes["qg"])

    # Restore expected shapes using the stored metadata.
    cosmo = tf.reshape(cosmo, tf.stack([n_params]))

    γg = tf.reshape(γg, wl_shape)
    γa = tf.reshape(γa, wl_shape)
    γd = tf.reshape(γd, wl_shape)
    ds = tf.reshape(ds, wl_shape)

    dg = tf.reshape(dg, gc_shape)
    qg = tf.reshape(qg, gc_shape)

    # Give TensorFlow static rank information, useful in tf.data pipelines.
    cosmo.set_shape([None])
    for x in (γg, γa, γd, ds, dg, qg):
        x.set_shape([None, None])

    return {
        "cosmo": cosmo,
        "n_params": n_params,
        "i_sobol": features["i_sobol"],
        "i_signal": features["i_signal"],
        "n_pix": n_pix,
        "n_z_WL": n_z_WL,
        "n_z_GC": n_z_GC,
        "γg": γg,
        "γa": γa,
        "γd": γd,
        "ds": ds,
        "dg": dg,
        "qg": qg,
    }

_DEFAULT_TENSOR_DTYPES_TORCH = {
    "cosmo": torch.float32,
    "γg": torch.complex64,
    "γa": torch.complex64,
    "γd": torch.complex64,
    "ds": torch.float32,
    "dg": torch.float32,
    "qg": torch.float32,
}

def parse_inverse_onthefly_torch(serialized_example, tensor_dtypes=None):
    """Parse one serialized TFRecord example written by parse_forward_onthefly.

    Args:
        serialized_example: A scalar string Tensor or raw serialized Example bytes.
        tensor_dtypes: Optional dict mapping tensor field names to TensorFlow dtypes.
            This is useful because tf.io.parse_tensor needs the dtype explicitly.
            Example:
                {
                    "cosmo": tf.float64,
                    "γg": tf.float32,
                    ...
                }

    Returns:
        dict: Parsed tensors and integer metadata.
    """

    


    return {
        "cosmo": cosmo,
        "n_params": n_params,
        "i_sobol": features["i_sobol"],
        "i_signal": features["i_signal"],
        "n_pix": n_pix,
        "n_z_WL": n_z_WL,
        "n_z_GC": n_z_GC,
        "γg": γg,
        "γa": γa,
        "γd": γd,
        "ds": ds,
        "dg": dg,
        "qg": qg,
    }


def verify_tfrecord_onthefly(
    serialized_example,
    γg,
    γa,
    γd,
    ds,
    dg,
    qg,
    cosmo,
    i_sobol,
    i_signal,
):
    """Verify that one serialized example round-trips correctly.

    Args:
        serialized_example: Either a serialized Example byte string,
            a scalar tf.string Tensor, or a tf.train.Example object.
        γg, γa, γd, ds, dg, qg, cosmo: Original arrays.
        i_sobol, i_signal: Original integer metadata.

    Returns:
        bool: True if verification passes.

    Raises:
        AssertionError: If any metadata, shape, dtype, or value differs.
    """

    # Allow passing the tf.train.Example returned by parse_forward_onthefly directly.
    if isinstance(serialized_example, tf.train.Example):
        serialized_example = serialized_example.SerializeToString()

    expected = {
        "γg": np.asarray(γg),
        "γa": np.asarray(γa),
        "γd": np.asarray(γd),
        "ds": np.asarray(ds),
        "dg": np.asarray(dg),
        "qg": np.asarray(qg),
        "cosmo": np.asarray(cosmo),
    }

    # Basic consistency checks on the original inputs.
    if expected["γg"].ndim != 2:
        raise AssertionError(f"γg must be 2D, got shape {expected['γg'].shape}")

    if expected["cosmo"].ndim != 1:
        raise AssertionError(f"cosmo must be 1D, got shape {expected['cosmo'].shape}")

    for key in ("γa", "γd", "ds"):
        if expected[key].shape != expected["γg"].shape:
            raise AssertionError(
                f"{key} shape mismatch: got {expected[key].shape}, "
                f"expected {expected['γg'].shape}"
            )

    if expected["qg"].shape != expected["dg"].shape:
        raise AssertionError(
            f"qg shape mismatch: got {expected['qg'].shape}, "
            f"expected {expected['dg'].shape}"
        )

    if expected["dg"].shape[0] != expected["γg"].shape[0]:
        raise AssertionError(
            f"dg and γg must have the same n_pix: "
            f"dg has {expected['dg'].shape[0]}, γg has {expected['γg'].shape[0]}"
        )

    # Infer the exact dtypes from the originals for verification.
    # This avoids failing when your stored arrays are float64 instead of float32.
    tensor_dtypes = {
        key: tf.as_dtype(arr.dtype)
        for key, arr in expected.items()
    }

    parsed = parse_inverse_onthefly(
        serialized_example,
        tensor_dtypes=tensor_dtypes,
    )

    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if tf.is_tensor(x):
            return x.numpy()
        return np.asarray(x)

    def _to_int(x):
        if tf.is_tensor(x):
            return int(x.numpy())
        return int(x)

    expected_n_pix = expected["γg"].shape[0]
    expected_n_z_WL = expected["γg"].shape[1]
    expected_n_z_GC = expected["dg"].shape[1]
    expected_n_params = expected["cosmo"].shape[0]

    metadata_checks = {
        "n_pix": expected_n_pix,
        "n_z_WL": expected_n_z_WL,
        "n_z_GC": expected_n_z_GC,
        "n_params": expected_n_params,
        "i_sobol": int(i_sobol),
        "i_signal": int(i_signal),
    }

    for key, expected_value in metadata_checks.items():
        got_value = _to_int(parsed[key])
        if got_value != expected_value:
            raise AssertionError(
                f"{key} mismatch: got {got_value}, expected {expected_value}"
            )

    for key, expected_array in expected.items():
        got_array = _to_numpy(parsed[key])

        if got_array.shape != expected_array.shape:
            raise AssertionError(
                f"{key} shape mismatch: got {got_array.shape}, "
                f"expected {expected_array.shape}"
            )

        if got_array.dtype != expected_array.dtype:
            raise AssertionError(
                f"{key} dtype mismatch: got {got_array.dtype}, "
                f"expected {expected_array.dtype}"
            )

        if np.issubdtype(expected_array.dtype, np.inexact):
            np.testing.assert_allclose(
                got_array,
                expected_array,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
                err_msg=f"{key} values differ",
            )
        else:
            np.testing.assert_array_equal(
                got_array,
                expected_array,
                err_msg=f"{key} values differ",
            )

    return True