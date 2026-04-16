# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

This file is loosely based off
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
by Janis Fluri
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


class GridPipeline(MSFMpipeline):
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
            with_cross=with_cross,
            apply_norm=apply_norm,
            with_padding=with_padding,
            z_bin_inds=z_bin_inds,
            return_maps=return_maps,
            return_cls=return_cls,
            # these are fixed in the .tfrecord files
            apply_m_bias=False,
            shape_noise_scale=1.0,
            poisson_noise_scale=1.0,
        )

        # used to return the correct labels
        self.all_params = parameters.get_parameters(conf=conf)

        # used to reshape the stored tensors, and for nothing else
        self.n_all_params = len(self.all_params)

        self.n_noise_total = self.conf["analysis"]["grid"]["n_noise_per_signal"]
        self.n_signal_total = self.conf["analysis"]["n_patches"] * self.conf["analysis"]["grid"]["n_perms_per_cosmo"]

    def _parse_indices(
        self, indices: Union[int, float, list, range], name: str, fallback_length: int, is_eval: bool = False
    ) -> list:
        if indices is None:
            parsed_indices = list(range(fallback_length))
            LOGGER.info(f"Including all {len(parsed_indices)} {name} = {parsed_indices}")
            return parsed_indices
        if isinstance(indices, float):
            assert 0.0 < indices < 1.0, f"for a float, {name} = {indices} must be between 0 and 1"
            split_idx = int(indices * fallback_length)
            if is_eval:
                parsed_indices = list(range(split_idx, fallback_length))
                LOGGER.warning(f"Using validation split ({1.0 - indices:<.2%})")
            else:
                parsed_indices = list(range(0, split_idx))
                LOGGER.warning(f"Using training split ({indices:<.2%})")
        elif isinstance(indices, int):
            assert indices >= 1, f"for an integer, {name} = {indices} must be >= 1"
            parsed_indices = list(range(indices))
        elif isinstance(indices, list):
            assert len(indices) >= 1, f"{name} = {indices} must be a list of length >= 1"
            assert all(isinstance(i, int) for i in indices), f"All elements in {name} must be integers"
            parsed_indices = indices
        elif isinstance(indices, range):
            parsed_indices = list(indices)
        else:
            raise TypeError(f"{name} = {indices} must be an integer, float, a list of integers or a range")

        LOGGER.info(f"Including {len(parsed_indices)} {name} = {parsed_indices}")

        return parsed_indices

    def get_dset(
        self,
        tfr_pattern: str,
        local_batch_size: int,
        noise_indices: Union[int, float, list, range] = None,
        signal_indices: Union[int, float, list, range] = None,
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
    ) -> tf.data.Dataset:
        """Builds the tf.data.Dataset from the given file name pattern and performance related parameters.

        Args:
            tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
            local_batch_size (int): Local batch size. Can also be the string "cosmo". Then, every batch contains all of
                the realisations of exactly one cosmology.
            noise_indices (int, float, list, range, optional): The noise indices to return. When this is an integer, the value is
                interpreted as range(noise_indices). When this is a float between 0 and 1, it is interpreted as the
                train/vali split ratio along the available noise indices where `is_eval` determines which half is chosen.
                Python lists and ranges are also accepted and not modified.
                Defaults to None, then all noise indices are returned.
            signal_indices (int, float, list, range, optional): The signal indices to return. When this is an integer, the
                value is interpreted as range(signal_indices). When this is a float between 0 and 1, it is interpreted as the
                train/vali split ratio along the available signal indices where `is_eval` determines which half is chosen.
                Python lists and ranges are also accepted and not modified.
                Defaults to None, then all signal indices are returned.
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
            (batch_size, n_pix, n_z_metacal + n_z_maglim), cosmo is a label distributed on the Sobol sequence and index
            is a tuple containing (i_sobol, i_signal, i_noise).
        """

        if is_eval:
            tf.random.set_seed(eval_seed)

        # parallelization
        if n_workers is None:
            LOGGER.info(f"n_workers is not set, using tf.data.AUTOTUNE. This might produce unexpected RAM usage.")
            n_file_workers = tf.data.AUTOTUNE
            n_parse_workers = tf.data.AUTOTUNE
            n_augment_workers = tf.data.AUTOTUNE
        else:
            n_file_workers = n_readers
            n_parse_workers = max((n_workers - n_readers) // 2, 1)
            n_augment_workers = max((n_workers - n_readers) // 2, 1)
            LOGGER.info(
                f"Using n_file_workers = {n_file_workers}, n_parse_workers = {n_parse_workers}, "
                f"n_augment_workers = {n_augment_workers}"
            )

        # batching
        if drop_remainder is None:
            if is_eval:
                drop_remainder = False
            else:
                drop_remainder = True
            LOGGER.info(f"drop_remainder is not set, using drop_remainder = {drop_remainder}")

        # indexing
        noise_indices = self._parse_indices(noise_indices, "noise_indices", self.n_noise_total, is_eval=is_eval)
        signal_indices = self._parse_indices(signal_indices, "signal_indices", self.n_signal_total, is_eval=is_eval)
        self.n_noise = len(noise_indices)
        self.n_signal = len(signal_indices)

        # get the file names and dataset them
        dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=(not is_eval), seed=file_name_shuffle_seed)

        # shard for distributed evaluation
        if input_context is not None:
            # NOTE that for the builtin MirroredStrategy, input_context.num_input_pipelines = 1 and
            # input_context.input_pipeline_id = 0, indicating that no sharding happens
            # NOTE My HorovodStrategy is written to be compatible with this

            # Taken from https://www.tensorflow.org/tutorials/distribute/input#usage_2
            dset = dset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            LOGGER.info(f"Sharding the dataset over the .tfrecord files according to the input context")

        # repeat and shuffle the files
        if not is_eval:
            dset = dset.repeat()
            dset = dset.shuffle(file_name_shuffle_buffer, seed=file_name_shuffle_seed)
            LOGGER.info(f"Shuffling file names with shuffle_buffer = {file_name_shuffle_buffer}")

        # interleave, block_length is the number of files every reader reads
        if local_batch_size == "cosmo":
            assert n_readers == 1, f"Can only read from a single file concurrently when local_batch_size = 'cosmo'"
            assert is_eval, f"The 'cosmo' batching is only for validation"

        if signal_indices is not None:

            def interleave_func(file):
                return (
                    tf.data.TFRecordDataset(file)
                    .enumerate()
                    .filter(lambda i, ex: tf.reduce_any(tf.equal(i, tf.constant(signal_indices, dtype=tf.int64))))
                    .map(lambda i, ex: ex)
                )

        else:
            interleave_func = tf.data.TFRecordDataset

        dset = dset.interleave(
            interleave_func,
            cycle_length=n_readers,
            block_length=1,
            num_parallel_calls=n_file_workers,
            deterministic=is_eval,
        )
        LOGGER.info(f"Interleaving with n_readers = {n_readers}")

        # parse, output signature (data_vectors, index), where data_vectors is a dict
        dset = dset.map(
            lambda serialized_example: tfrecords.parse_inverse_grid(
                serialized_example,
                noise_indices,
                # dimensions
                n_pix=self.n_dv_pix,
                n_z_metacal=self.n_z_metacal,
                n_z_maglim=self.n_z_maglim,
                n_z_cross=self.n_z_cross,
                n_params=self.n_all_params,
                n_noise=self.n_noise_total,
                n_cls=self.n_cls,
                # map types
                with_lensing=self.with_lensing,
                with_clustering=self.with_clustering,
                with_cross=self.with_cross,
                # outputs
                return_maps=self.return_maps,
                return_cls=self.return_cls,
            ),
            num_parallel_calls=n_parse_workers,
        )

        # map a single example to len(noise_indices) examples corresponding to different noise realizations
        # NOTE that interleaving with cycle_lengths > 1 doesn't improve performance, so we use flat_map
        dset = dset.flat_map(lambda data_vectors: self._split_noise_realizations(data_vectors, noise_indices))

        # shuffle the examples
        if not is_eval:
            dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)
            LOGGER.info(f"Shuffling examples with shuffle_buffer = {examples_shuffle_buffer}")

        # batch (first, for vectorization)
        if local_batch_size == "cosmo":
            local_batch_size = len(signal_indices) * len(noise_indices)
            LOGGER.info(f"The dset is batched by cosmology")
        dset = dset.batch(local_batch_size, drop_remainder=drop_remainder)
        LOGGER.info(f"Batching into {local_batch_size} elements locally")

        # augmentations (all in one function, to make parallelization faster)
        dset = dset.map(
            self._augmentations,
            num_parallel_calls=n_augment_workers,
        )

        # prefetch
        if n_prefetch != 0:
            if n_prefetch is None:
                n_prefetch = tf.data.AUTOTUNE
            dset = dset.prefetch(n_prefetch)
            LOGGER.info(f"Prefetching {n_prefetch} elements")

        LOGGER.info(f"Successfully generated the grid validation set with element_spec {dset.element_spec}")
        return dset

    def _split_noise_realizations(self, data_vectors: dict, noise_indices: Union[list, range]) -> tf.data.Dataset:
        """Split the dictionary stored within the .tfrecord files into the separate noise realizations stored within.
        In this way, a single element of the dataset is mapped to a new dataset. Therefore, this function should be
        applied as flat_map or interleave.

        Args:
            data_vectors (dict): Full dictionary containing all noisy kg and dg maps, i_sobol and i_signal indices.
            noise_indices (list, range): The noise indices to return.

        Returns:
            tf.data.Dataset: Dataset containing the separate noise realizations.
        """

        if self.return_maps:
            # separate the noise realizations
            if self.with_lensing:
                kg = []
                for i in noise_indices:
                    kg.append(data_vectors.pop(f"kg_{i}"))

            if self.with_clustering:
                dg = []
                for i in noise_indices:
                    dg.append(data_vectors.pop(f"dg_{i}"))

            if self.with_cross:
                xg = []
                for i in noise_indices:
                    xg.append(data_vectors.pop(f"xg_{i}"))

        if self.return_cls:
            cl = []
            for i in noise_indices:
                cl.append(data_vectors.pop(f"cl_{i}"))

        # repeat as often as there are different noise realizations
        for key in data_vectors.keys():
            # no action is necessary for the cls. They're already in this format right out of the .tfrecords
            if not "cl" in key:
                data_vectors[key] = tf.repeat(tf.expand_dims(data_vectors[key], axis=0), len(noise_indices), axis=0)

        if self.return_maps:
            # update the dictionary
            if self.with_lensing:
                data_vectors["kg"] = kg
            if self.with_clustering:
                data_vectors["dg"] = dg
            if self.with_cross:
                data_vectors["xg"] = xg

        if self.return_cls:
            data_vectors["cl"] = cl

        data_vectors["i_noise"] = list(noise_indices)

        # return a dataset containing n_examples elements
        return tf.data.Dataset.from_tensor_slices(data_vectors)

    def _augmentations(self, data_vectors: dict) -> tf.Tensor:
        """Applies random augmentations and general pre-processing to the maps. This includes in order:

        lensing
        - Add the chosen shape noise realization to the kappa maps
        - Reversibly normalize to roughly unit values
        - Mask the resulting data vector

        clustering
        - Reversibly normalize to roughly unit values
        - Mask the resulting data vector

        Concatenate both along the z bin axis.

        Args:
            data_vectors (dict): Depending on with_clustering and with_lensing, contains the tensors kg (sum of signal
                and intrinsic alignment) and sn (single realization) of shape (n_pix, n_z_metacal) and dg of shape
                (n_pix, n_z_maglim).
            index (tuple): A tuple of two integers (i_sobol, i_noise).

        Returns:
            tuple: (out_tensor, cosmo, index) the elements of the dataset, where out_tensor has shape
            (batch_size, n_pix, n_z_metacal + n_z_maglim), cosmo is a label distributed on the Sobol sequence and index
            is a tuple containing (i_sobol, i_signal, i_noise).
        """
        LOGGER.warning(f"Tracing _augmentations")
        LOGGER.info(f"Running on the data_vectors.keys() = {data_vectors.keys()}")

        # to be explicit
        with tf.device("/CPU:0"):
            # label, cosmo params
            cosmo = data_vectors.pop("cosmo")
            cosmo = tf.gather(cosmo, [self.all_params.index(param) for param in self.params], axis=1)

            if self.return_maps:
                if self.with_lensing:
                    # normalization
                    if self.apply_norm:
                        data_vectors["kg"] = self.normalize_lensing(data_vectors["kg"])

                    # masking
                    data_vectors["kg"] *= self.masks_metacal

                    map_tensor = data_vectors["kg"]

                if self.with_clustering:
                    # normalization
                    if self.apply_norm:
                        data_vectors["dg"] = self.normalize_clustering(data_vectors["dg"])

                    # masking
                    data_vectors["dg"] *= self.masks_maglim

                    map_tensor = data_vectors["dg"]

                if self.with_cross:
                    # NOTE no normalization

                    # masking NOTE this assumes a single mask per tomographic bin
                    mask = tf.math.reduce_prod(self.masks_metacal, axis=-1) * tf.math.reduce_prod(
                        self.masks_maglim, axis=-1
                    )
                    mask = tf.expand_dims(mask, axis=-1)
                    data_vectors["xg"] *= mask

                    map_tensor = data_vectors["xg"]

                if self.with_lensing and self.with_clustering:
                    # concatenate along the tomography axis
                    map_tensor = tf.concat([data_vectors["kg"], data_vectors["dg"]], axis=-1)

                if not self.with_padding:
                    LOGGER.info(f"Removing the padding")
                    map_tensor = tf.boolean_mask(map_tensor, self.mask_total, axis=1)

                # potentially discard the unwanted redshift bins
                if self.z_bin_inds is not None:
                    LOGGER.warning(f"Discarding all redshift bins except {self.z_bin_inds}")
                    map_tensor = tf.gather(map_tensor, self.z_bin_inds, axis=-1)
            else:
                map_tensor = None

            if self.return_cls:
                cl_tensor = data_vectors.pop("cl")
            else:
                cl_tensor = None

        # gather the indices
        i_sobol = data_vectors.pop("i_sobol")
        i_signal = data_vectors.pop("i_signal")
        i_noise = data_vectors.pop("i_noise")

        return map_tensor, cl_tensor, cosmo, (i_sobol, i_signal, i_noise)
