# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak

This file is loosely based off
grid_pipeline.py by Arne Thomsen
"""

import glob
import warnings
from typing import Union

import numpy as np
import tensorflow as tf

from msfm.utils import logger, tfrecords, parameters
from msfm.utils.base_pipeline import MSFMpipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class _OntheflyTorchDataset:
    """Factory for a PyTorch iterable dataset for on-the-fly TFRecords.

    The TFRecords are parsed with the existing TensorFlow serializer/parser because
    the examples were written with ``tf.io.serialize_tensor``. The object returned
    to users is still a PyTorch dataset and yields PyTorch tensors.
    """

    def __new__(
        cls,
        pipeline,
        files,
        example_indices=None,
        is_eval=True,
        examples_shuffle_buffer=128,
        examples_shuffle_seed=12,
        n_examples_total=None,
    ):
        try:
            import torch
            from torch.utils.data import IterableDataset
        except ImportError as exc:
            raise ImportError("OntheflyPipeline.get_dset requires PyTorch to be installed.") from exc

        class DatasetImpl(IterableDataset):
            def __iter__(self_inner):
                rng = np.random.default_rng(examples_shuffle_seed)
                buffer = []
                for item in pipeline._iter_examples(
                    files, example_indices, is_eval, n_examples_total=n_examples_total
                ):
                    if is_eval:
                        yield item
                        continue
                    buffer.append(item)
                    if len(buffer) >= examples_shuffle_buffer:
                        rng.shuffle(buffer)
                        while buffer:
                            yield buffer.pop()
                if not is_eval:
                    rng.shuffle(buffer)
                    while buffer:
                        yield buffer.pop()

        return DatasetImpl()


class OntheflyPipeline(MSFMpipeline):
    """
    Sets up a PyTorch Dataset/DataLoader for the on-the-fly cosmologies.
    """

    def __init__(
        self,
        conf: dict = None,
        # cosmology
        params: list = None,
        with_lensing: bool = True,
        with_clustering: bool = True,
        with_padding: bool = True,
        z_bin_inds: list = None,
    ):
        """Set up the physics parameters of the pipeline."""
        super().__init__(
            conf=conf,
            params=params,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_padding=with_padding,
            z_bin_inds=z_bin_inds,
            return_maps=True,
            return_cls=False,
            apply_m_bias=False,
            shape_noise_scale=1.0,
            poisson_noise_scale=1.0,
        )

        assert self.with_lensing or self.with_clustering, "At least one of with_lensing and with_clustering must be True"
        self.all_params = parameters.get_parameters(conf=conf)
        self.n_all_params = len(self.all_params)
        self.param_indices = [self.all_params.index(param) for param in self.params]

    def _parse_example_indices(
        self, example_indices: Union[int, float, list, range], is_eval: bool, n_examples_total: int = None
    ):
        if example_indices is None:
            return None
        if isinstance(example_indices, float):
            assert 0.0 < example_indices < 1.0, "example_indices as a float must be between 0 and 1"
            assert n_examples_total is not None, "n_examples_total is required for float example_indices"
            split_idx = int(example_indices * n_examples_total)
            if is_eval:
                return set(range(split_idx, n_examples_total))
            return set(range(0, split_idx))
        if isinstance(example_indices, int):
            assert example_indices >= 1, "example_indices as an integer must be >= 1"
            return set(range(example_indices))
        if isinstance(example_indices, range):
            return set(example_indices)
        if isinstance(example_indices, list):
            assert all(isinstance(i, int) for i in example_indices), "All example_indices must be integers"
            return set(example_indices)
        raise TypeError("example_indices must be None, an integer, a list of integers or a range")

    def _iter_examples(self, files, example_indices, is_eval, n_examples_total=None):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("OntheflyPipeline.get_dset requires PyTorch to be installed.") from exc

        selected_indices = self._parse_example_indices(example_indices, is_eval, n_examples_total=n_examples_total)
        tensor_dtypes = {"cosmo": tf.float32}
        global_example_index = 0
        for file_name in files:
            for serialized in tf.data.TFRecordDataset(file_name).as_numpy_iterator():
                if selected_indices is not None and global_example_index not in selected_indices:
                    global_example_index += 1
                    continue
                parsed = tfrecords.parse_inverse_onthefly(serialized, tensor_dtypes=tensor_dtypes)
                parsed = {key: value.numpy() if hasattr(value, "numpy") else value for key, value in parsed.items()}
                yield self._example_to_torch(parsed, torch)
                global_example_index += 1

    def _example_to_torch(self, parsed, torch):
        channels = []

        if self.with_lensing:
            gamma = parsed["γg"]
            if self.z_bin_inds is not None:
                gamma = gamma[:, self.z_bin_inds.numpy()]
            channels.extend([gamma.real.T, gamma.imag.T])

        if self.with_clustering:
            dg = parsed["dg"]
            if self.z_bin_inds is not None:
                dg = dg[:, self.z_bin_inds.numpy()]
            channels.append(dg.T)

        # PyTorch convention: channels before spatial/pixel dimensions.
        map_tensor = np.concatenate(channels, axis=0).astype(np.float32, copy=False)

        if not self.with_padding:
            if self.with_lensing and self.with_clustering:
                mask = np.asarray(self.mask_total, dtype=bool)
            elif self.with_lensing:
                mask = np.asarray(tf.reduce_prod(self.masks_WL, axis=-1), dtype=bool)
            else:
                mask = np.asarray(tf.reduce_prod(self.masks_GC, axis=-1), dtype=bool)
            map_tensor = map_tensor[:, mask]

        cosmo = np.asarray(parsed["cosmo"])[self.param_indices].astype(np.float32, copy=False)
        index = (
            torch.as_tensor(int(parsed["i_sobol"]), dtype=torch.long),
            torch.as_tensor(int(parsed["i_signal"]), dtype=torch.long),
        )
        return torch.from_numpy(map_tensor), torch.from_numpy(cosmo), index

    def get_dset(
        self,
        tfr_pattern: str,
        local_batch_size: int,
        example_indices: Union[int, float, list, range] = None,
        n_readers: int = 8,
        n_workers: int = None,
        n_prefetch: int = None,
        file_name_shuffle_buffer: int = 128,
        examples_shuffle_buffer: int = 128,
        is_eval: bool = True,
        drop_remainder: bool = None,
        eval_seed: int = 33,
        file_name_shuffle_seed: int = 11,
        examples_shuffle_seed: int = 12,
        input_context: tf.distribute.InputContext = None,
        downsample_nside: int = None,
    ):
        """Build a PyTorch DataLoader from on-the-fly TFRecords.

        Batches contain ``(maps, cosmo, index)``. ``maps`` has PyTorch layout
        ``(batch_size, channels, n_pix)``. Lensing contributes two channels per
        selected WL bin (real and imaginary parts of ``γg``), and clustering
        contributes one channel per selected GC bin (``dg``).
        """
        if downsample_nside is not None:
            raise NotImplementedError("downsample_nside is not supported by OntheflyPipeline")
        if local_batch_size == "cosmo":
            raise ValueError("local_batch_size='cosmo' is not supported by OntheflyPipeline")
        if drop_remainder is None:
            drop_remainder = not is_eval

        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise ImportError("OntheflyPipeline.get_dset requires PyTorch to be installed.") from exc

        if is_eval:
            torch.manual_seed(eval_seed)

        files = sorted(glob.glob(tfr_pattern))
        if not files:
            raise FileNotFoundError(f"No TFRecord files match tfr_pattern={tfr_pattern}")
        if input_context is not None:
            files = files[input_context.input_pipeline_id :: input_context.num_input_pipelines]
            LOGGER.info("Sharding the dataset over the .tfrecord files according to the input context")
        if not is_eval:
            rng = np.random.default_rng(file_name_shuffle_seed)
            rng.shuffle(files)
            LOGGER.info(f"Shuffling file names with shuffle_buffer = {file_name_shuffle_buffer}")

        n_examples_total = None
        if isinstance(example_indices, float):
            n_examples_total = sum(1 for file_name in files for _ in tf.data.TFRecordDataset(file_name))

        dataset = _OntheflyTorchDataset(
            self,
            files,
            example_indices=example_indices,
            is_eval=is_eval,
            examples_shuffle_buffer=examples_shuffle_buffer,
            examples_shuffle_seed=examples_shuffle_seed,
            n_examples_total=n_examples_total,
        )
        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            num_workers=0 if n_workers is None else n_workers,
            drop_last=drop_remainder,
            prefetch_factor=n_prefetch if (n_workers and n_workers > 0 and n_prefetch is not None) else None,
        )
        LOGGER.info("Successfully generated the on-the-fly PyTorch DataLoader")
        return loader
