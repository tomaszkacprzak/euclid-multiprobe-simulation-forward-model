# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak

This file is loosely based off
grid_pipeline.py by Arne Thomsen
"""

import glob
import os
import random
import warnings
from typing import Union

import numpy as np
import tensorflow as tf

try:
    from torch.utils.data import IterableDataset
except ImportError:  # pragma: no cover - torch is an optional runtime dependency.
    IterableDataset = object

from msfm.utils import logger, parameters
from msfm.utils.base_pipeline import MSFMpipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class _OntheflyTFRecordTorchDataset(IterableDataset):
    """Iterable PyTorch dataset for on-the-fly TFRecords."""

    _WL_FIELDS = ("γg", "γa", "ds", "γd")
    _GC_FIELDS = ("dg", "qg")
    _COMPLEX_FIELDS = {"γg", "γa", "γd"}

    def __init__(
        self,
        files,
        with_lensing=True,
        with_clustering=True,
        z_bin_inds=None,
        example_indices=None,
        shuffle_examples=False,
        examples_shuffle_buffer=128,
        examples_shuffle_seed=12,
    ):
        super().__init__()
        self.files = list(files)
        self.with_lensing = with_lensing
        self.with_clustering = with_clustering
        self.z_bin_inds = None if z_bin_inds is None else np.asarray(z_bin_inds, dtype=np.int64)
        self.example_indices = None if example_indices is None else set(example_indices)
        self.shuffle_examples = shuffle_examples
        self.examples_shuffle_buffer = examples_shuffle_buffer
        self.examples_shuffle_seed = examples_shuffle_seed
        self._description = {
            "γg": "byte",
            "γa": "byte",
            "γd": "byte",
            "ds": "byte",
            "dg": "byte",
            "qg": "byte",
        }

    @staticmethod
    def _feature_bytes(value):
        if isinstance(value, bytes):
            return value
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(()).item()
        return bytes(value)

    @staticmethod
    def _parse_tensor(serialized_tensor):
        tensor_proto = tf.compat.v1.TensorProto()
        tensor_proto.ParseFromString(_OntheflyTFRecordTorchDataset._feature_bytes(serialized_tensor))
        return tf.make_ndarray(tensor_proto)

    @classmethod
    def _format_map(cls, value, z_bin_inds=None):
        arr = cls._parse_tensor(value)
        if z_bin_inds is not None:
            arr = arr[:, z_bin_inds]
        if np.iscomplexobj(arr):
            arr = np.stack((arr.real, arr.imag), axis=-1)
        else:
            arr = arr[..., np.newaxis]
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _iter_records(self, files):
        from tfrecord.reader import tfrecord_loader

        for file_name in files:
            index_path = f"{file_name}.index"
            index_path = index_path if os.path.exists(index_path) else None
            yield from tfrecord_loader(file_name, index_path, self._description)

    def _iter_selected_records(self, files):
        for i_example, record in enumerate(self._iter_records(files)):
            if self.example_indices is None or i_example in self.example_indices:
                yield record

    def _record_to_tuple(self, record):
        maps = []
        if self.with_lensing:
            maps.extend(
                self._format_map(record[field], self.z_bin_inds)
                for field in self._WL_FIELDS
            )
        if self.with_clustering:
            maps.extend(
                self._format_map(record[field], self.z_bin_inds)
                for field in self._GC_FIELDS
            )
        return tuple(maps)

    def __iter__(self):
        try:
            from torch.utils.data import get_worker_info

            worker_info = get_worker_info()
        except ImportError:  # pragma: no cover - torch is an optional runtime dependency.
            worker_info = None

        files = self.files
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        records = (self._record_to_tuple(record) for record in self._iter_selected_records(files))
        if not self.shuffle_examples:
            yield from records
            return

        rng = random.Random(self.examples_shuffle_seed)
        buffer = []
        for record in records:
            buffer.append(record)
            if len(buffer) >= self.examples_shuffle_buffer:
                i = rng.randrange(len(buffer))
                yield buffer.pop(i)
        while buffer:
            i = rng.randrange(len(buffer))
            yield buffer.pop(i)


class OntheflyPipeline(MSFMpipeline):
    """
    Sets up a PyTorch data loader for the on-the-fly cosmologies.
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

        if not (self.with_lensing or self.with_clustering):
            raise ValueError("At least one of with_lensing and with_clustering must be True")

        self.all_params = parameters.get_parameters(conf=conf)
        self.n_all_params = len(self.all_params)
        self.n_signal_total = self.conf["analysis"]["n_patches"] * self.conf["analysis"]["grid"]["n_perms_per_cosmo"]

    def _parse_indices(
        self, indices: Union[int, float, list, range], name: str, fallback_length: int, is_eval: bool = False
    ) -> list:
        if indices is None:
            return None
        if isinstance(indices, float):
            assert 0.0 < indices < 1.0, f"for a float, {name} = {indices} must be between 0 and 1"
            split_idx = int(indices * fallback_length)
            return list(range(split_idx, fallback_length)) if is_eval else list(range(0, split_idx))
        if isinstance(indices, int):
            assert indices >= 1, f"for an integer, {name} = {indices} must be >= 1"
            return list(range(indices))
        if isinstance(indices, list):
            assert len(indices) >= 1, f"{name} = {indices} must be a list of length >= 1"
            assert all(isinstance(i, int) for i in indices), f"All elements in {name} must be integers"
            return indices
        if isinstance(indices, range):
            return list(indices)
        raise TypeError(f"{name} = {indices} must be an integer, float, a list of integers or a range")

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
        """Build a PyTorch DataLoader from on-the-fly TFRecord files."""
        if downsample_nside is not None:
            raise NotImplementedError("downsample_nside is not supported by OntheflyPipeline")
        if local_batch_size == "cosmo":
            raise NotImplementedError("local_batch_size='cosmo' is not supported by OntheflyPipeline")

        from torch.utils.data import DataLoader
        from torch.utils.data._utils.collate import default_collate

        if drop_remainder is None:
            drop_remainder = not is_eval
            LOGGER.info(f"drop_remainder is not set, using drop_remainder = {drop_remainder}")

        files = sorted(glob.glob(tfr_pattern))
        if not files:
            raise FileNotFoundError(f"No TFRecord files match tfr_pattern = {tfr_pattern}")

        if input_context is not None:
            files = files[input_context.input_pipeline_id :: input_context.num_input_pipelines]
            LOGGER.info("Sharding the dataset over the .tfrecord files according to the input context")

        if not is_eval:
            rng = random.Random(file_name_shuffle_seed)
            rng.shuffle(files)
            LOGGER.info(f"Shuffling file names with shuffle_buffer = {file_name_shuffle_buffer}")
        elif eval_seed is not None:
            random.seed(eval_seed)

        example_indices = self._parse_indices(example_indices, "example_indices", self.n_signal_total, is_eval=is_eval)
        if example_indices is None:
            LOGGER.info("Including all example_indices")
        else:
            LOGGER.info(f"Including {len(example_indices)} example_indices = {example_indices}")

        dataset = _OntheflyTFRecordTorchDataset(
            files=files,
            with_lensing=self.with_lensing,
            with_clustering=self.with_clustering,
            z_bin_inds=None if self.z_bin_inds is None else self.z_bin_inds.numpy(),
            example_indices=example_indices,
            shuffle_examples=not is_eval,
            examples_shuffle_buffer=examples_shuffle_buffer,
            examples_shuffle_seed=examples_shuffle_seed,
        )

        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            drop_last=drop_remainder,
            num_workers=0 if n_workers is None else n_workers,
            prefetch_factor=n_prefetch if n_workers else None,
            collate_fn=lambda batch: tuple(default_collate(batch)),
        )

        LOGGER.info(f"Successfully generated the on-the-fly PyTorch DataLoader for {len(files)} TFRecord files")
        return loader
