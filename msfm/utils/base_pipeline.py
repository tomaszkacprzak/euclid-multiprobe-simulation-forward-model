# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Parent class of the fiducial and grid pipelines
"""

from typing import Union

import torch
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
        with_WL: bool = True,
        with_GC: bool = True,
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
        device: Union[str, torch.device] = "cpu",
    ):
        """Shared parameters are set up here.

        Args:
            conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
                passed through) or None (the default config is loaded). Defaults to None.
            params (list): List of the cosmological parameters of interest. Fiducial: perturbations, grid: labels.
            with_WL (bool, optional): Whether to include the kappa maps. Defaults to True.
            with_GC (bool, optional): Whether to include the delta maps. Defaults to True.
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
                torch.Tensor to change it according to a schedule during training. Set to None to not include any shape
                noise. Defaults to 1.0.
            poisson_noise_scale (float, optional): Factor by which to multiply the Poisson noise. This could also be a
                torch.Tensor to change it according to a schedule during training. Set to None to not include any
                Poisson noise. Defaults to 1.0.
            device (Union[str, torch.device], optional): Device for pipeline tensors derived from the configuration.
                Defaults to CPU to keep preprocessing deterministic and avoid implicit GPU memory use.
        """
        # general constants
        self.conf = files.load_config(conf)
        self.params = parameters.get_parameters(params, self.conf)
        self.device = torch.device(device)

        # function arguments
        self.apply_norm = apply_norm
        self.shape_noise_scale = shape_noise_scale
        self.poisson_noise_scale = poisson_noise_scale
        if self.shape_noise_scale != 1.0 or self.poisson_noise_scale != 1.0:
            LOGGER.warning(f"The noise scaling is only implemented for the maps, not the power spectra")
        self.with_padding = with_padding
        if isinstance(z_bin_inds, (list, np.ndarray, torch.Tensor)):
            self.z_bin_inds = torch.as_tensor(z_bin_inds, dtype=torch.int32, device=self.device)
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
        data_vec_pix, _, _, _ = files.load_pixel_file(self.conf)
        self.data_vec_pix = torch.as_tensor(data_vec_pix, dtype=torch.int64, device=self.device)
        self.n_dv_pix = len(self.data_vec_pix)

        masks_dict = files.get_tomo_dv_masks(self.conf)
        self.masks_WL = torch.as_tensor(masks_dict["WL"], dtype=torch.float32, device=self.device)
        self.masks_GC = torch.as_tensor(masks_dict["GC"], dtype=torch.float32, device=self.device)

        if not self.with_padding:
            # only keep indices that are in all (per tomographic bin and galaxy sample) masks
            self.mask_total = torch.prod(torch.cat([self.masks_WL, self.masks_GC], dim=-1), dim=-1)
            self.mask_total = self.mask_total.bool()
            self.patch_pix = self.data_vec_pix[self.mask_total]
            self.n_patch_pix = len(self.patch_pix)

        # lensing
        self.with_WL = with_WL
        self.apply_m_bias = apply_m_bias
        if apply_m_bias:
            self.m_bias_dist = lensing.get_m_bias_distribution(self.conf)
        else:
            self.m_bias_dist = None
        self.norm_WL = torch.as_tensor(
            self.conf["analysis"]["normalization"]["WL"], dtype=torch.float32, device=self.device
        )
        self.normalize_lensing = lambda lensing_dv: torch.as_tensor(lensing_dv, device=self.device) / self.norm_WL.to(
            dtype=torch.as_tensor(lensing_dv).dtype
        )

        # clustering
        self.with_GC = with_GC
        self.tomo_n_gal_maglim = torch.as_tensor(
            self.conf["survey"]["GC"]["n_gal"], dtype=torch.float32, device=self.device
        ) * hp.nside2pixarea(self.conf["analysis"]["n_side"], degrees=True)
        self.norm_GC = torch.as_tensor(
            self.conf["analysis"]["normalization"]["GC"], dtype=torch.float32, device=self.device
        )
        self.normalize_clustering = lambda clustering_dv: torch.as_tensor(
            clustering_dv, device=self.device
        ) / self.norm_GC.to(dtype=torch.as_tensor(clustering_dv).dtype)

        self.with_cross = with_cross
        if self.with_cross:
            assert not (
                self.with_WL or self.with_GC
            ), "with_cross can only be True if both with_WL and with_GC are False"

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
        data_vector = torch.as_tensor(data_vector, device=self.device)
        patch_pix = torch.as_tensor(self.base_patch_pix, dtype=torch.int64, device=self.device)
        nest_indices = torch.as_tensor(
            hp.ring2nest(nside=self.conf["analysis"]["n_side"], ipix=patch_pix.cpu().numpy()),
            dtype=torch.int64,
            device=self.device,
        )
        nest_patch = torch.index_select(data_vector, dim=1, index=nest_indices)

        return nest_patch
