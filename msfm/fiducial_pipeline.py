# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

This file is loosely based off
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
by Janis Fluri
"""

import glob
import random
import tensorflow as tf
import warnings
from typing import Union

import webdataset as wds

from msfm.utils import logger, webdatasets, parameters
from msfm.utils.base_pipeline import MSFMpipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class FiducialPipeline(MSFMpipeline):
    """
    Sets up a tf.data.Dataset for the fiducial cosmology and its per parameter perturbations.
    """

    def __init__(
        self,
        conf: dict = None,
        # cosmology
        params: list = None,
        with_lensing: bool = True,
        with_clustering: bool = True,
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
        """Set up the physics parameters of the pipeline.

        Args:
            conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
                passed through) or None (the default config is loaded). Defaults to None.
            params (list): List of the cosmological parameters of interest. Fiducial: perturbations, grid: labels.
            with_lensing (bool, optional): Whether to include the kappa maps. Defaults to True.
            with_clustering (bool, optional): Whether to include the delta maps. Defaults to True.
            apply_norm (bool, optional): Whether to rescale the maps to approximate unit range. Defaults to True.
            with_padding (bool, optional): Whether to include the padding of the data vectors (the healpy DeepSphere \
                networks) need this. Defaults to True.
            z_bin_inds (list, optional): Specify the indices of the redshift bins to be included. Note that this is
                mainly meant for testing purposes and is inefficient, since all redshift bins are loaded from the
                WebDataset shards nonetheless. Defaults to None, then all redshift bins are kept.
            return_maps (bool, optional): Whether to return the maps. Defaults to True.
            return_maps (bool, optional): Whether to return the cls. Defaults to True.
            apply_m_bias (bool, optional): Whether to include the multiplicative shear bias. Defaults to True.
            shape_noise_scale (float, optional): Factor by which to multiply the shape noise. This could also be a
                tf.Variable to change it according to a schedule during training. Set to None to not include any shape
                noise. Defaults to 1.0.
            poisson_noise_scale (float, optional): Factor by which to multiply the Poisson noise. This could also be a
                tf.Variable to change it according to a schedule during training. Set to None to not include any 
                Poisson noise. Defaults to 1.0.
        """
        super().__init__(
            conf=conf,
            # cosmology
            params=params,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            # format
            apply_norm=apply_norm,
            with_padding=with_padding,
            z_bin_inds=z_bin_inds,
            return_maps=return_maps,
            return_cls=return_cls,
            # noise
            apply_m_bias=apply_m_bias,
            shape_noise_scale=shape_noise_scale,
            poisson_noise_scale=poisson_noise_scale,
        )

        # perturbations of cosmo, ia, and bg parameters
        self.pert_labels = parameters.get_fiducial_perturbation_labels(self.params)

        # for the power spectra
        self.n_noise = self.conf["analysis"]["fiducial"]["n_noise_per_example"]

    def get_dset(
        self,
        pattern: str = None,
        local_batch_size: int = None,
        noise_indices: Union[int, list, range] = 1,
        # performance
        is_cached: bool = False,
        n_readers: int = 8,
        n_workers: int = None,
        n_prefetch: int = None,
        file_name_shuffle_buffer: int = 16,
        examples_shuffle_buffer: int = 64,
        # training/evaluation
        is_eval: bool = False,
        drop_remainder: bool = None,
        eval_seed: int = 32,
        file_name_shuffle_seed: int = 17,
        examples_shuffle_seed: int = 67,
        # distribution
        input_context: tf.distribute.InputContext = None,
        tfr_pattern: str = None,
    ) -> tf.data.Dataset:
        """Builds the tf.data.Dataset from the given file name pattern and performance related parameters.

        Args:
            pattern (str): Glob pattern of the fiducial WebDataset .tar shards.
            local_batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch
                size.
            noise_indices (int, optional): The noise indices to return. When this is an integer, the value is
                interpreted as range(noise_indices). Python lists and ranges are also accepted and not modified.
                Defaults to 1, then only the single noise realization with index 0 is returned.
            is_cached (bool): Whether to cache on the level on the deserialized tensors. This is only feasible if all of
                the fiducial WebDataset shards fit into RAM. Defaults to False.
            n_readers (int, optional): Number of parallel readers, i.e. different input files read concurrently. This
                should be roughly less than a tenth of the number of files. Large values cost a lot of RAM, especially
                in the distributed setting. Defaults to 4.
            n_workers (int, optional): Number of parallel workers for the file reading, file parsing and preprocessing
                augmentations. Defaults to None, then tf.data.AUTOTUNE is used. Note that this may lead to unexpected
                RAM usage, especially if there's more than one dataset within the same script.
            n_prefetch (int, optional): Number of dataset elements to prefetch. Defaults to None, then tf.data.AUTOTUNE
                is used.
            file_name_shuffle_buffer (int, optional): Size of the shuffle buffer for the .tfrecord files. Defaults to
                128.
            examples_shuffle_buffer (int, optional): Size of the shuffle buffer for the non-batched examples. Defaults
                to 128.
            is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeatedly, such that one can
                go through it deterministically exactly once. Defaults to False.
            drop_remainder (bool, optional): Whether to drop the remainder of the dataset when the batch size does not
                evenly divide the number of samples. If None, then it is set to False for evaluation and True for
                training. Defaults to None.
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
                        dset = fiducial_pipeline.get_fiducial_dset(
                            pattern,
                            params,
                            batch_size,
                            input_context=input_context,
                        )

        Returns:
            tf.data.Dataset: A dataset that returns samples with a given batchsize in the right ordering for the delta
            loss. The index label consists of (i_example, i_noise)
        """

        if is_eval:
            tf.random.set_seed(eval_seed)

            # parameters that are not used
            file_name_shuffle_seed = None
            examples_shuffle_buffer = None
            file_name_shuffle_seed = None
            examples_shuffle_seed = None

            LOGGER.warning(
                f"Evaluation mode is activated: the random seed is fixed, the shuffle arguments ignored, and the "
                f"dataset is not repeated"
            )

        # parallelization
        if n_workers is None:
            LOGGER.info(f"n_workers is not set, using tf.data.AUTOTUNE. This might produce unexpected RAM usage.")
            n_augment_workers = tf.data.AUTOTUNE
        else:
            n_augment_workers = max((n_workers - n_readers) // 2, 1)
            LOGGER.info(f"Using n_augment_workers = {n_augment_workers}")

        # batching
        if drop_remainder is None:
            if is_eval:
                drop_remainder = False
            else:
                drop_remainder = True
            LOGGER.info(f"drop_remainder is not set, using drop_remainder = {drop_remainder}")

        # noise indexing
        if isinstance(noise_indices, int):
            assert noise_indices >= 1, f"for an integer, noise_indices = {noise_indices} must be >= 1"
            noise_indices = range(noise_indices)
        elif isinstance(noise_indices, list):
            assert len(noise_indices) >= 1, f"noise_indices = {noise_indices} must be a list of length >= 1"
            assert all(isinstance(i, int) for i in noise_indices), "All elements in noise_indices must be integers"
        elif isinstance(noise_indices, range):
            pass
        else:
            raise TypeError(f"noise_indices = {noise_indices} must be an integer, a list of integers or a range")
        LOGGER.info(f"Including noise_indices = {list(noise_indices)}")

        if pattern is None:
            if tfr_pattern is None:
                raise ValueError("Either pattern or the deprecated tfr_pattern alias must be provided")
            warnings.warn("tfr_pattern is deprecated; use pattern for WebDataset shards", DeprecationWarning)
            pattern = tfr_pattern
        elif tfr_pattern is not None:
            raise ValueError("Provide only one of pattern or deprecated tfr_pattern")

        # get the file names
        file_names = sorted(file_name for file_name in glob.glob(pattern) if file_name.endswith(".tar"))
        if not file_names:
            raise FileNotFoundError(f"No WebDataset .tar shards match pattern {pattern!r}")
        LOGGER.info(f"Resolved {len(file_names)} WebDataset .tar shards before distributed sharding")

        if is_eval:
            LOGGER.info("Evaluation mode keeps WebDataset shards in sorted order and does not repeat")
        else:
            random.Random(file_name_shuffle_seed).shuffle(file_names)
            LOGGER.info(f"Shuffled WebDataset shard file list with file_name_shuffle_seed = {file_name_shuffle_seed}")

        # shard for distributed training
        if input_context is not None:
            # NOTE that for the builtin MirroredStrategy, input_context.num_input_pipelines = 1 and
            # input_context.input_pipeline_id = 0, indicating that no sharding happens
            # NOTE My HorovodStrategy is written to be compatible with this

            # Taken from https://www.tensorflow.org/tutorials/distribute/input#usage_2
            n_file_names_before_sharding = len(file_names)
            file_names = file_names[input_context.input_pipeline_id :: input_context.num_input_pipelines]
            LOGGER.info(
                f"Sharding the dataset over the WebDataset shards according to the input context "
                f"(pipeline {input_context.input_pipeline_id}/{input_context.num_input_pipelines}): "
                f"{n_file_names_before_sharding} -> {len(file_names)} shards"
            )
        else:
            LOGGER.info(f"Using all {len(file_names)} WebDataset .tar shards after distributed sharding")

        output_signature = {
            key: tf.TensorSpec(shape=None, dtype=tf.float32) for key in self._fiducial_float_keys(noise_indices)
        }
        output_signature["i_signal"] = tf.TensorSpec(shape=(), dtype=tf.int64)

        def generator():
            while True:
                for file_name in file_names:
                    for sample in wds.WebDataset([file_name], shardshuffle=False):
                        decoded = webdatasets.decode_fiducial_sample(
                            sample,
                            self.pert_labels,
                            noise_indices,
                            self.n_dv_pix,
                            self.n_z_WL,
                            self.n_z_GC,
                            self.n_noise,
                            self.n_cls,
                            self.n_z_cross,
                            self.with_lensing,
                            self.with_clustering,
                            self.return_maps,
                            self.return_cls,
                        )
                        yield {key: tf.convert_to_tensor(value) for key, value in decoded.items()}
                if is_eval or is_cached:
                    break

        dset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        LOGGER.info(f"Reading WebDataset shards with n_readers = {n_readers}")

        if is_cached:
            dset = dset.cache()
            dset = dset.repeat()
            LOGGER.warning(f"Caching the dataset")
            # TODO
            LOGGER.error(f"CACHING SEEMS TO PRODUCE BUGGY BEHAVIOR")

        # map a single example to len(noise_indices) examples corresponding to different noise realizations
        # NOTE that interleaving with cycle_lengths > 1 doesn't improve performance, so we use flat_map
        dset = dset.flat_map(lambda data_vectors: self._split_noise_realizations(data_vectors, noise_indices))

        # shuffle the examples
        if (not is_eval) and (examples_shuffle_buffer is not None) and (examples_shuffle_buffer > 0):
            dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)
            LOGGER.info(f"Shuffling examples with shuffle_buffer = {examples_shuffle_buffer}")
        elif not is_eval:
            LOGGER.warning(f"Examples are not shuffled, which is underisable for is_eval = {is_eval}")

        # batch (first, for vectorization)
        dset = dset.batch(local_batch_size, drop_remainder=drop_remainder)
        LOGGER.info(f"Batching into {local_batch_size} elements locally with drop_remainder = {drop_remainder}")

        # augmentations (all in one function, to make parallelization faster)
        dset = dset.map(
            self._augmentations,
            num_parallel_calls=n_augment_workers,
            deterministic=is_eval,
        )

        # prefetch
        if n_prefetch != 0:
            if n_prefetch is None:
                n_prefetch = tf.data.AUTOTUNE
            dset = dset.prefetch(n_prefetch)
            LOGGER.info(f"Prefetching {n_prefetch} elements")

        LOGGER.info(f"Successfully generated the fiducial training set with element_spec {dset.element_spec}")
        return dset

    def _fiducial_float_keys(self, noise_indices: Union[list, range]) -> list:
        keys = []
        if self.return_maps:
            for label in self.pert_labels:
                if self.with_lensing and "bg" not in label:
                    keys.append(f"kg_{label}")
                if self.with_clustering and "Aia" not in label:
                    keys.append(f"dg_{label}")
            for i in noise_indices:
                if self.with_lensing:
                    keys.append(f"sn_{i}")
                if self.with_clustering:
                    keys.append(f"pn_{i}")
        if self.return_cls:
            keys.extend(f"cl_{label}" for label in self.pert_labels)
        return keys

    def _split_noise_realizations(self, data_vectors: dict, noise_indices: Union[list, range]) -> tf.data.Dataset:
        """Split the dictionary stored within the .tfrecord files into the separate noise realizations stored within.
        For this, the signal maps are copied in memory and paired with the noise realizations. So a single element
        of the dataset is mapped to a new dataset containing len(noise_indices) examples. Therefore, this function
        should be applied as flat_map or interleave.

        Args:
            data_vectors (dict): Full dictionary containing all kg perturbations, dg perturbations, i_example and
                i_noise indices.
            noise_indices (list, range): The noise indices to return.

        Returns:
            tf.data.Dataset: Dataset containing the separate noise realizations.
        """

        if self.return_maps:
            # separate the noise realizations
            if self.with_lensing:
                sn = []
                for i in noise_indices:
                    sn.append(data_vectors.pop(f"sn_{i}"))

            if self.with_clustering:
                pn = []
                for i in noise_indices:
                    pn.append(data_vectors.pop(f"pn_{i}"))

        # repeat the signal as often as there are different noise realizations
        for key in data_vectors.keys():
            # no action is necessary for the cls. They're already in this format right out of the WebDataset shards
            if not "cl" in key:
                data_vectors[key] = tf.repeat(tf.expand_dims(data_vectors[key], axis=0), len(noise_indices), axis=0)

        if self.return_maps:
            # update the dictionary
            if self.with_lensing:
                data_vectors["sn"] = sn
            if self.with_clustering:
                data_vectors["pn"] = pn
        data_vectors["i_noise"] = list(noise_indices)

        # return a dataset containing len(noise_indices) elements
        return tf.data.Dataset.from_tensor_slices(data_vectors)

    def _augmentations(self, data_vectors: dict) -> tf.Tensor:
        """This function wraps _lensing_augmentations and _clustering_augmentations and implements the appropriate case
        distinction.

        Args:
            data_vectors (dict): Full dictionary containing all kg perturbations, dg perturbations, i_example and
                i_noise indices.
            index (tuple): Label (i_example, i_noise), which is only passed through.

        Raises:
            ValueError: If neither lensing nor clustering maps are selected.

        Returns:
            tuple: (out_tensor, index) the elements of the dataset, where index is a tuple (i_example, i_noise).
        """
        LOGGER.warning(f"Tracing _augmentations")
        LOGGER.info(f"Running on the data_vectors.keys() = {data_vectors.keys()}")

        # to be explicit
        with tf.device("/CPU:0"):
            if self.return_maps:
                if self.with_lensing and self.with_clustering:
                    kg_tensor = self._lensing_augmentations(data_vectors)
                    dg_tensor = self._clustering_augmentations(data_vectors)

                    # concatenate along the tomography axis
                    map_tensor = tf.concat([kg_tensor, dg_tensor], axis=-1)

                elif self.with_lensing:
                    assert not any(param in self.pert_labels for param in ["bg", "n_bg"])
                    map_tensor = self._lensing_augmentations(data_vectors)

                elif self.with_clustering:
                    assert not any(param in self.pert_labels for param in ["Aia", "n_Aia"])
                    map_tensor = self._clustering_augmentations(data_vectors)

                else:
                    raise ValueError(f"At least one of 'lensing' or 'clustering' maps need to be selected")

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
                cl = []
                for label in self.pert_labels:
                    cl.append(data_vectors[f"cl_{label}"])
                cl_tensor = tf.concat(cl, axis=0)
            else:
                cl_tensor = None

        # gather the indices
        i_example = data_vectors.pop("i_example")
        i_noise = data_vectors.pop("i_noise")

        return map_tensor, cl_tensor, (i_example, i_noise)

    def _lensing_augmentations(self, data_vectors: dict) -> tf.Tensor:
        """Applies random augmentations and general pre-processing to the weak lensing maps (kg). This includes in
        order:
            - Load the fiducial for galaxy clustering perturbations
            - Add random multiplicative shear bias (the additive one is negligible) to the kappa maps
            - Add the chosen shape noise realization to the kappa maps
            - Reversibly normalize to roughly unit values
            - Mask the resulting data vector (this is only required if an addition like an additive shear bias is
                applied)
            - Concatenate the batch and perturbations along axis 0 (for compatibility with the delta loss)

        Args:
            data_vectors (dict): Has keys "kg_{pert_label}" and "sn", which contain tensors of shape
                (batch_size, n_pix, n_z_WL)GC.

        Returns:
            tf.tensor: data_vectors of shape (n_perts * batch_size, n_pix, n_z_WL). The first batch_size elements
                correspond to the fiducial value, the second to the first pertuGCd
                perturbation, etc. (for compatibility with the delta loss).
        """
        LOGGER.warning(f"Tracing _lensing_augmentations")

        # shape noise
        sn = data_vectors.pop("sn")

        out_data_vectors = []
        for label in self.pert_labels:
            # galaxy bias perturbations
            if "bg" in label:
                # doesn't affect the convergence map
                data_vector = data_vectors[f"kg_fiducial"]

            # cosmology + intrinsic alignment perturbations
            else:
                data_vector = data_vectors[f"kg_{label}"]

            # shear bias
            if self.apply_m_bias and (self.m_bias_dist is not None):
                # only sample at the fiducial (which always comes first)
                if "fiducial" in label:
                    # shape (n_z_bins,)
                    m_bias = self.m_bias_dist.sample()

                # broadcast axis 0 of size n_pix
                data_vector *= 1.0 + m_bias
            else:
                LOGGER.warning(f"No multiplicative shear bias is applied")

            # shape noise
            if self.shape_noise_scale is not None:
                data_vector += self.shape_noise_scale * sn
            else:
                LOGGER.warning(f"No shape noise is added to the lensing maps")

            # normalization
            if self.apply_norm:
                data_vector = self.normalize_lensing(data_vector)

            # masking
            data_vector *= self.GC

            out_data_vectors.append(data_vector)

        # concatenate the perturbation axis
        return tf.concat(out_data_vectors, axis=0)

    def _clustering_augmentations(self, data_vectors: dict) -> tf.Tensor:
        """Applies random augmentations and general pre-processing to the clustering maps (dg). This includes in order:
            - Load the fiducial for intrinsic alignment perturbations
            - Mask the resulting data vector (just to be safe)
            - Concatenate the batch and perturbations along axis 0 (for compatibility with the delta loss)

        Args:
            data_vectors (dict): Has keys "dg_{pert_label}", which contain tensors of shape
                (batch_size, n_pix, n_z_maglim).

        Returns:
            tf.tensor: data_vectors of shape (n_perts * batch_size, n_pix, n_z_maglim). The first batch_size elements
                correspond to the fiducial value, the second to the first perturbation, the third to the second
                perturbation, etc. (for compatibility with the delta loss).
        """
        LOGGER.warning(f"Tracing _clustering_augmentations")

        # poisson noise
        pn = data_vectors.pop("pn")

        out_data_vectors = []
        for label in self.pert_labels:
            # intrinsic alignemnt perturbations
            if "Aia" in label:
                # doesn't affect the clustering map
                data_vector = data_vectors[f"dg_fiducial"]
            # cosmology perturbations
            else:
                data_vector = data_vectors[f"dg_{label}"]

            # poisson noise
            if self.poisson_noise_scale is not None:
                data_vector += self.poisson_noise_scale * pn
            else:
                LOGGER.warning(f"No poisson noise is added to the clustering maps")

            # normalization
            if self.apply_norm:
                data_vector = self.normalize_clustering(data_vector)

            # masking
            data_vector *= self.masks_maglim

            out_data_vectors.append(data_vector)

        # concatenate the perturbation axis
        return tf.concat(out_data_vectors, axis=0)
