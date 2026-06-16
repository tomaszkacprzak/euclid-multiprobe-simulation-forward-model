# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak

This file is loosely based off
grid_pipeline.py by Arne Thomsen
"""

import tensorflow as tf
import warnings
from typing import Union

from msfm.utils import logger, tfrecords, parameters
from msfm.utils.base_pipeline import MSFMpipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class OntheflyPipeline(MSFMpipeline):
    """
    Sets up a tf.data.Dataset for the grid cosmologies.
    """

    def __init__(
        self,
        conf: dict = None,
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
        return_cls: bool = False,
    ):
        """Set up the physics parameters of the pipeline.

        Args:
            conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
                passed through) or None (the default config is loaded). Defaults to None.
            params (list): List of the cosmological parameters of interest. Fiducial: perturbations, grid: labels, onthefly: labels.
            with_lensing (bool, optional): Whether to include the lensing maps. Defaults to True.
            with_clustering (bool, optional): Whether to include the clustering maps. Defaults to True.
            with_cross (bool, optional): Whether to include the cross-correlation between lensing and clustering. 
                Defaults to False.
            apply_norm (bool, optional): Whether to rescale the maps to approximate unit range. Defaults to True.
            with_padding (bool, optional): Whether to include the padding of the data vectors (the healpy DeepSphere \
                networks) need this. Defaults to True.
            z_bin_inds (list, optional): Specify the indices of the redshift bins to be included. Note that this is
                mainly meant for testing purposes and is inefficient, since all redshift bins are loaded from the
                .tfrecords nonetheless. Defaults to None, then all redshift bins are kept.
            return_maps (bool, optional): Whether to return the maps. Defaults to True.
            return_cls (bool, optional): Whether to return the cls. Defaults to True.
            return_only_cross_maps (bool, optional): Whether to return only the cross maps. Defaults to False.
        """
        super().__init__(
            conf=conf,
            params=params,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            z_bin_inds=z_bin_inds
        )

        # TODO: implement this

    def get_dset(
        self,
        tfr_pattern: str,
        local_batch_size: int,
        example_indices: Union[int, float, list, range] = None,
        # performance
        n_readers: int = 8,
        n_workers: int = None,
        n_prefetch: int = None,
        file_name_shuffle_buffer: int = 128,
        examples_shuffle_buffer: int = 128,
        # training/evaluation
        is_eval: bool = True,
        drop_remainder: bool = None,
        eval_seed: int = 33,
        file_name_shuffle_seed: int = 11,
        examples_shuffle_seed: int = 12,
        # distribution
        input_context: tf.distribute.InputContext = None,
        # nside downsampling
        downsample_nside: int = None,
        parent_output_idx=None,
    ) -> tf.data.Dataset:
        """Builds the tf.data.Dataset from the given file name pattern and performance related parameters.

        Args:
            tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
            local_batch_size (int): Local batch size. Can also be the string "cosmo". Then, every batch contains all of
                the realisations of exactly one cosmology.
            example_indices (int, float, list, range, optional): The noise indices to return. When this is an integer, the value is
                interpreted as range(noise_indices). When this is a float between 0 and 1, it is interpreted as the
                train/vali split ratio along the available noise indices where `is_eval` determines which half is chosen.
                Python lists and ranges are also accepted and not modified.
                Defaults to None, then all noise indices are returned.
            n_readers (int, optional): Number of parallel readers, i_e. different input files read concurrently. This
                should be roughly less than a tenth of the number of files. Large values cost a lot of RAM, especially
                in the distributed setting. Defaults to 4.
            n_workers (int, optional): Number of parallel workers for the file reading, file parsing and preprocessing
                augmentations. Defaults to None, then tf.data.AUTOTUNE is used. Note that this may lead to unexpected
                RAM usage, especially if there's more than one dataset within the same script.
            n_prefetch (int, optional): Number of dataset elements to prefetch.
            is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeatedly, such that one can
                go through it deterministically exactly once. Defaults to True.
            eval_seed (int, optional): Fixed seed for evaluation. Defaults to 32.
            file_name_shuffle_seed (int, optional): Defaults to 17.
            examples_shuffle_seed (int, optional): Defaults to 67.
            input_context (Union[tf.distribute.InputContext, deep_lss.utils.distribute.HorovodStrategy], optional):
                Custom input_context attribute of my HorovodStrategy class or when using the TensorFlow builtin
                distribution strategies, this is passed to the dataset_fn like in
                https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
                Then, the dataset is sharded. Defaults to None for a non distributed dataset.

                Example usage:
                    def dataset_fn(input_context):
                        dset = fiducial_pipeline.get_grid_dset(
                            tfr_pattern,
                            local_batch_size,
                            input_context=input_context,
                        )

        Returns:
            tf.data.Dataset: A deterministic dataset that goes through the grid cosmologies in the order of the sobol
                seeds. The output is a tuple like (data_vectors, cosmo, index), where data_vectors is a tensor of shape
            (batch_size, n_pix, n_z_WL + n_z_GC), cosmo is a label distributed on the Sobol sequence and index
            is a tuple containing (i_sobol, i_signal, i_noise).
        """

        # TODO: implement this

        

        LOGGER.info(f"Successfully generated the grid set with element_spec {dset.element_spec}")
        return dset

