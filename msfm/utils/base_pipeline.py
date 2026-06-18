# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Parent class of the fiducial and grid pipelines
"""

import tensorflow as tf
import numpy as np
import healpy as hp
import warnings

from msfm.utils import files, lensing, parameters, logger, cross_statistics

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class MSFMpipeline:
    """Parent class of the fiducial and grid pipeline"""

    def __init__(
        self,
        conf: dict,
        # cosmology
        params: list = None,
        with_lensing: bool = True,
        with_clustering: bool = True,
        with_cross: bool = False,
        # format
        apply_norm: bool = True,
        with_padding: bool = True,
        z_bin_inds: list = None,
        return_maps: bool = True,
        return_cls: bool = True,
        # noise
        apply_m_bias: bool = True,
        shape_noise_scale: float = 1.0,
        poisson_noise_scale: float = 1.0,
    ):
        """Shared parameters are set up here.

        Args:
            conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
                passed through) or None (the default config is loaded). Defaults to None.
            params (list): List of the cosmological parameters of interest. Fiducial: perturbations, grid: labels.
            with_lensing (bool, optional): Whether to include the kappa maps. Defaults to True.
            with_clustering (bool, optional): Whether to include the delta maps. Defaults to True.
            with_cross (bool, optional): Whether to include the cross-correlation between lensing and clustering. 
                Defaults to False.
            apply_norm (bool, optional): Whether to rescale the maps to approximate unit range. Defaults to True.
            with_padding (bool, optional): Whether to include the padding of the data vectors (the healpy DeepSphere \
                networks) need this. Defaults to True.
            z_bin_inds (list, optional): Specify the indices of the redshift bins to be included. Note that this is
                mainly meant for testing purposes and is inefficient, since all redshift bins are loaded from the
                WebDataset shards nonetheless. Defaults to None, then all redshift bins are kept.
            return_maps (bool, optional): Whether to return the maps. Defaults to True.
            return_cls (bool, optional): Whether to return the cls. Defaults to True.
            apply_m_bias (bool, optional): Whether to include the multiplicative shear bias. Defaults to True.
            shape_noise_scale (float, optional): Factor by which to multiply the shape noise. This could also be a
                tf.Variable to change it according to a schedule during training. Set to None to not include any shape
                noise. Defaults to 1.0.
            poisson_noise_scale (float, optional): Factor by which to multiply the Poisson noise. This could also be a
                tf.Variable to change it according to a schedule during training. Set to None to not include any 
                Poisson noise. Defaults to 1.0.
        """
        # general constants
        self.conf = files.load_config(conf)
        self.params = parameters.get_parameters(params, self.conf)

        # function arguments
        self.apply_norm = apply_norm
        self.shape_noise_scale = shape_noise_scale
        self.poisson_noise_scale = poisson_noise_scale
        if self.shape_noise_scale != 1.0 or self.poisson_noise_scale != 1.0:
            LOGGER.warning(f"The noise scaling is only implemented for the maps, not the power spectra")
        self.with_padding = with_padding
        if isinstance(z_bin_inds, (list, np.ndarray, tf.Tensor)):
            self.z_bin_inds = tf.constant(z_bin_inds, dtype=tf.int32)
        elif z_bin_inds is None:
            self.z_bin_inds = z_bin_inds
        else:
            raise TypeError(f"z_bin_inds = {z_bin_inds} must be None, a list, array or tensor")
        self.return_maps = return_maps
        self.return_cls = return_cls
        assert self.return_maps or self.return_cls, "At least one of return_maps and return_cls must be True"

        self.n_z_WL = len(self.conf["survey"]["WL"]["z_bins"])
        self.n_z_GC = len(self.conf["survey"]["GC"]["z_bins"])

        # pixel file
        self.data_vec_pix, _, _, _ = files.load_pixel_file(self.conf)
        self.n_dv_pix = len(self.data_vec_pix)

        masks_dict = files.get_tomo_dv_masks(self.conf)
        self.masks_WL = tf.constant(masks_dict["WL"], dtype=tf.float32)
        self.masks_GC = tf.constant(masks_dict["GC"], dtype=tf.float32)

        if not self.with_padding:
            # only keep indices that are in all (per tomographic bin and galaxy sample) masks
            self.mask_total = tf.reduce_prod(tf.concat([self.masks_WL, self.masks_GC], axis=-1), axis=-1)
            self.mask_total = tf.cast(self.mask_total, dtype=tf.bool)
            self.patch_pix = tf.boolean_mask(self.data_vec_pix, self.mask_total, axis=0)
            self.n_patch_pix = len(self.patch_pix)

        # lensing
        self.with_lensing = with_lensing
        self.apply_m_bias = apply_m_bias
        if apply_m_bias:
            self.m_bias_dist = lensing.get_m_bias_distribution(self.conf)
        else:
            self.m_bias_dist = None
        self.normalize_lensing = lambda lensing_dv: lensing_dv / tf.constant(
            self.conf["analysis"]["normalization"]["WL"], dtype=tf.float32
        )

        # clustering
        self.with_clustering = with_clustering
        self.tomo_n_gal_maglim = tf.constant(self.conf["survey"]["GC"]["n_gal"]) * hp.nside2pixarea(
            self.conf["analysis"]["n_side"], degrees=True
        )
        self.normalize_clustering = lambda clustering_dv: clustering_dv / tf.constant(
            self.conf["analysis"]["normalization"]["GC"], dtype=tf.float32
        )

        self.with_cross = with_cross
        if self.with_cross:
            assert not (
                self.with_lensing or self.with_clustering
            ), "with_cross can only be True if both with_lensing and with_clustering are False"

        # power spectra
        self.n_cls = 3 * self.conf["analysis"]["n_side"]
        self.n_z_cross = len(
            cross_statistics.get_cross_bin_indices(
                self.n_z_WL,
                self.n_z_GC,
                True,
                True,
                True,
                True,
            )[0]
        )

    def padded_dv_to_non_padded_patch(self, data_vector):
        nest_patch = tf.gather(
            data_vector, hp.ring2nest(nside=self.conf["analysis"]["n_side"], ipix=self.base_patch_pix), axis=1
        )

        return nest_patch
