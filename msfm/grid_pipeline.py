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
import numpy as np
import torch
from torch.utils.data import IterableDataset

import warnings
from typing import Union

import webdataset as wds

from msfm.utils import logger, webdatasets, parameters
from msfm.utils.base_pipeline import MSFMpipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class GridPipeline(MSFMpipeline):
    """
    Sets up a PyTorch/WebDataset loader for the grid cosmologies.
    """

    def __init__(
        self,
        conf: dict = None,
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
        return_cls: bool = False,
    ):
        """Set up the physics parameters of the pipeline.

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
            return_only_cross_maps (bool, optional): Whether to return only the cross maps. Defaults to False.
        """
        super().__init__(
            conf=conf,
            params=params,
            with_WL=with_WL,
            with_GC=with_GC,
            with_cross=with_cross,
            apply_norm=apply_norm,
            with_padding=with_padding,
            z_bin_inds=z_bin_inds,
            return_maps=return_maps,
            return_cls=return_cls,
            # these are fixed in the WebDataset samples
            apply_m_bias=False,
            shape_noise_scale=1.0,
            poisson_noise_scale=1.0,
        )

        # Base pipeline stores the probe flags under the historical names; keep readable aliases for WebDataset code.
        self.with_lensing = self.with_WL
        self.with_clustering = self.with_GC

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
        pattern: str = None,
        local_batch_size: int = None,
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
        input_context=None,
        shuffle: bool = None,
        repeat: bool = None,
        tfr_pattern: str = None,
        # nside downsampling
        downsample_nside: int = None,
        parent_output_idx=None,
    ) -> wds.WebLoader:
        """Builds a WebDataset-backed PyTorch WebLoader from the given file name pattern and performance parameters.

        Compatibility note: this method name is kept for existing callers, but it now returns a
        :class:`webdataset.WebLoader` whose batches are dictionaries of ``torch.Tensor`` objects instead of a
        framework-specific dataset yielding tuple outputs.

        Args:
            pattern (str): Glob pattern of the fiducial WebDataset .tar shards.
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
                augmentations. Defaults to None, then automatic worker tuning is used. Note that this may lead to unexpected
                RAM usage, especially if there's more than one dataset within the same script.
            n_prefetch (int, optional): Number of dataset elements to prefetch.
            is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeatedly, such that one can
                go through it deterministically exactly once. Defaults to True.
            eval_seed (int, optional): Fixed seed for evaluation. Defaults to 32.
            file_name_shuffle_seed (int, optional): Defaults to 17.
            examples_shuffle_seed (int, optional): Defaults to 67.
            input_context (distributed input context, optional):
                Custom input_context attribute for distributed loading; this is used to shard the file list across input pipelines.
                Then, the dataset is sharded. Defaults to None for a non distributed dataset.

                Example usage:
                    def dataset_fn(input_context):
                        dset = fiducial_pipeline.get_grid_dset(
                            pattern,
                            local_batch_size,
                            input_context=input_context,
                        )

        Returns:
            webdataset.WebLoader: An iterable PyTorch loader. Each batch is a dictionary containing torch tensors,
            including ``maps`` (when requested), ``cls`` (when requested), ``cosmo``, ``i_sobol``, ``i_signal``, and
            ``i_noise``.
        """

        if is_eval:
            torch.manual_seed(eval_seed)

        if shuffle is None:
            shuffle = not is_eval
        if repeat is None:
            repeat = not is_eval

        # parallelization
        if n_workers is None:
            n_workers = 0
            LOGGER.info("n_workers is not set, using single-process PyTorch loading")
        else:
            LOGGER.info(f"Using n_workers = {n_workers} for PyTorch loading")

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

        if pattern is None:
            if tfr_pattern is None:
                raise ValueError("Either pattern or the deprecated tfr_pattern alias must be provided")
            warnings.warn("tfr_pattern is deprecated; use pattern for WebDataset shards", DeprecationWarning)
            pattern = tfr_pattern
        elif tfr_pattern is not None:
            raise ValueError("Provide only one of pattern or deprecated tfr_pattern")

        # get the file names and dataset them
        file_names = sorted(file_name for file_name in glob.glob(pattern) if file_name.endswith(".tar"))
        if not file_names:
            raise FileNotFoundError(f"No WebDataset .tar shards match pattern {pattern!r}")
        LOGGER.info(f"Resolved {len(file_names)} WebDataset .tar shards before distributed sharding")

        if is_eval:
            LOGGER.info("Evaluation mode keeps WebDataset shards in sorted order and does not repeat")
        else:
            random.Random(file_name_shuffle_seed).shuffle(file_names)
            LOGGER.info(f"Shuffled WebDataset shard file list with file_name_shuffle_seed = {file_name_shuffle_seed}")

        # shard for distributed evaluation
        if input_context is not None:
            # NOTE that for the builtin MirroredStrategy, input_context.num_input_pipelines = 1 and
            # input_context.input_pipeline_id = 0, indicating that no sharding happens
            # NOTE My HorovodStrategy is written to be compatible with this

            # Taken from the distributed input sharding pattern
            n_file_names_before_sharding = len(file_names)
            file_names = file_names[input_context.input_pipeline_id :: input_context.num_input_pipelines]
            LOGGER.info(
                f"Sharding the dataset over the WebDataset shards according to the input context "
                f"(pipeline {input_context.input_pipeline_id}/{input_context.num_input_pipelines}): "
                f"{n_file_names_before_sharding} -> {len(file_names)} shards"
            )
        else:
            LOGGER.info(f"Using all {len(file_names)} WebDataset .tar shards after distributed sharding")

        # interleave, block_length is the number of files every reader reads
        if local_batch_size == "cosmo":
            assert n_readers == 1, f"Can only read from a single file concurrently when local_batch_size = 'cosmo'"
            assert is_eval, f"The 'cosmo' batching is only for validation"

        dataset = _GridWebDatasetIterable(
            file_names=file_names,
            pipeline=self,
            noise_indices=noise_indices,
            signal_indices=signal_indices,
            repeat=repeat,
            shuffle_examples=shuffle,
            examples_shuffle_seed=examples_shuffle_seed,
            examples_shuffle_buffer=examples_shuffle_buffer,
        )
        LOGGER.info(f"Reading {len(file_names)} WebDataset shards with PyTorch IterableDataset")

        if local_batch_size == "cosmo":
            local_batch_size = len(signal_indices) * len(noise_indices)
            LOGGER.info("The dset is batched by cosmology")

        def _collate_and_augment(samples):
            batch = self._collate_samples(samples)
            batch = self._augmentations(batch)

            if downsample_nside is not None and parent_output_idx is not None and batch.get("maps") is not None:
                parent_output_idx_t = torch.as_tensor(parent_output_idx, dtype=torch.long)
                n_pix_out = int(parent_output_idx_t.max().item()) + 1
                batch["maps"] = _unsorted_segment_mean(batch["maps"], parent_output_idx_t, n_pix_out, dim=1)

            return batch

        loader = wds.WebLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=n_workers,
            drop_last=drop_remainder,
            collate_fn=_collate_and_augment,
        )

        if n_prefetch not in (None, 0):
            LOGGER.info("n_prefetch is accepted for compatibility but WebLoader prefetching is controlled by workers")

        if downsample_nside is not None and parent_output_idx is not None:
            LOGGER.info(f"Downsampling maps to nside={downsample_nside} ({int(np.max(parent_output_idx)) + 1} pixels)")

        LOGGER.info("Successfully generated the grid WebLoader")
        return loader

    def _has_lensing(self) -> bool:
        return self.with_lensing if hasattr(self, "with_lensing") else self.with_WL

    def _has_clustering(self) -> bool:
        return self.with_clustering if hasattr(self, "with_clustering") else self.with_GC

    def _grid_float_keys(self, noise_indices: Union[list, range]) -> list:
        keys = ["cosmo"]
        if self.return_maps:
            for i in noise_indices:
                if self._has_lensing():
                    keys.append(f"kg_{i}")
                if self._has_clustering():
                    keys.append(f"dg_{i}")
                if self.with_cross:
                    keys.append(f"xg_{i}")
        if self.return_cls:
            keys.extend(f"cl_{i}" for i in noise_indices)
        return keys

    def _collate_samples(self, samples: list) -> dict:
        keys = samples[0].keys()
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}

    def _normalization_tensor(self, probe: str, reference: torch.Tensor) -> torch.Tensor:
        value = self.conf.get("analysis", {}).get("normalization", {}).get(probe, 1.0)
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

    def _augmentations(self, data_vectors: dict) -> dict:
        """Apply map/cls preprocessing with PyTorch and return a dictionary batch.

        Compatibility note: this method name is retained from the TensorFlow pipeline, but it now accepts and returns
        PyTorch tensor dictionaries. The output dictionary contains ``maps`` and/or ``cls`` plus label/index tensors.
        """
        cosmo = data_vectors.pop("cosmo")
        param_indices = torch.as_tensor([self.all_params.index(param) for param in self.params], dtype=torch.long)
        cosmo = torch.index_select(cosmo, dim=1, index=param_indices)

        if self.return_maps:
            map_tensor = None
            if self._has_lensing():
                if self.apply_norm:
                    data_vectors["kg"] = data_vectors["kg"] / self._normalization_tensor("WL", data_vectors["kg"])
                data_vectors["kg"] = data_vectors["kg"] * _as_torch(self.masks_WL, dtype=data_vectors["kg"].dtype)
                map_tensor = data_vectors["kg"]

            if self._has_clustering():
                if self.apply_norm:
                    data_vectors["dg"] = data_vectors["dg"] / self._normalization_tensor("GC", data_vectors["dg"])
                data_vectors["dg"] = data_vectors["dg"] * _as_torch(self.masks_GC, dtype=data_vectors["dg"].dtype)
                map_tensor = data_vectors["dg"]

            if self.with_cross:
                mask_wl = _as_torch(self.masks_WL, dtype=data_vectors["xg"].dtype)
                mask_gc = _as_torch(self.masks_GC, dtype=data_vectors["xg"].dtype)
                mask = torch.prod(mask_wl, dim=-1) * torch.prod(mask_gc, dim=-1)
                data_vectors["xg"] = data_vectors["xg"] * torch.unsqueeze(mask, dim=-1)
                map_tensor = data_vectors["xg"]

            if self._has_lensing() and self._has_clustering():
                map_tensor = torch.cat([data_vectors["kg"], data_vectors["dg"]], dim=-1)

            if not self.with_padding:
                mask_total = _as_torch(self.mask_total).bool()
                map_tensor = map_tensor[:, mask_total, :]

            if self.z_bin_inds is not None:
                z_bin_inds = _as_torch(self.z_bin_inds, dtype=torch.long)
                map_tensor = torch.index_select(map_tensor, dim=-1, index=z_bin_inds)
        else:
            map_tensor = None

        batch = {
            "cosmo": cosmo,
            "i_sobol": data_vectors.pop("i_sobol"),
            "i_signal": data_vectors.pop("i_signal"),
            "i_noise": data_vectors.pop("i_noise"),
        }
        if self.return_maps:
            batch["maps"] = map_tensor
        if self.return_cls:
            batch["cls"] = data_vectors.pop("cl")
        return batch


class _GridWebDatasetIterable(IterableDataset):
    """IterableDataset that expands each WebDataset grid sample over requested noise indices."""

    def __init__(
        self,
        file_names,
        pipeline,
        noise_indices,
        signal_indices,
        repeat,
        shuffle_examples,
        examples_shuffle_seed,
        examples_shuffle_buffer,
    ):
        self.file_names = file_names
        self.pipeline = pipeline
        self.noise_indices = list(noise_indices)
        self.signal_indices = set(signal_indices)
        self.repeat = repeat
        self.shuffle_examples = shuffle_examples
        self.examples_shuffle_seed = examples_shuffle_seed
        self.examples_shuffle_buffer = examples_shuffle_buffer

    def __iter__(self):
        epoch = 0
        while True:
            iterator = self._iter_once()
            if self.shuffle_examples:
                iterator = _shuffle_buffered(
                    iterator, self.examples_shuffle_buffer, self.examples_shuffle_seed + epoch
                )
            yield from iterator
            epoch += 1
            if not self.repeat:
                break

    def _iter_once(self):
        for file_name in self.file_names:
            for i_signal_in_file, sample in enumerate(wds.WebDataset([file_name], shardshuffle=False)):
                if i_signal_in_file not in self.signal_indices:
                    continue
                decoded = webdatasets.decode_grid_sample(
                    sample,
                    self.noise_indices,
                    n_pix=self.pipeline.n_dv_pix,
                    n_z_WL=self.pipeline.n_z_WL,
                    n_z_GC=self.pipeline.n_z_GC,
                    n_z_cross_map=self.pipeline.n_z_cross,
                    n_z_cross=self.pipeline.n_z_cross,
                    n_params=self.pipeline.n_all_params,
                    n_noise=self.pipeline.n_noise_total,
                    n_cls=self.pipeline.n_cls,
                    with_lensing=self.pipeline._has_lensing(),
                    with_clustering=self.pipeline._has_clustering(),
                    with_cross=self.pipeline.with_cross,
                    return_maps=self.pipeline.return_maps,
                    return_cls=self.pipeline.return_cls,
                )
                decoded = {key: _as_torch(value) for key, value in decoded.items()}
                for i_noise in self.noise_indices:
                    out = {
                        "cosmo": decoded["cosmo"],
                        "i_sobol": decoded["i_sobol"],
                        "i_signal": decoded["i_signal"],
                        "i_noise": torch.tensor(i_noise, dtype=torch.long),
                    }
                    if self.pipeline.return_maps:
                        if self.pipeline._has_lensing():
                            out["kg"] = decoded[f"kg_{i_noise}"]
                        if self.pipeline._has_clustering():
                            out["dg"] = decoded[f"dg_{i_noise}"]
                        if self.pipeline.with_cross:
                            out["xg"] = decoded[f"xg_{i_noise}"]
                    if self.pipeline.return_cls:
                        out["cl"] = decoded[f"cl_{i_noise}"]
                    yield out


def _as_torch(value, dtype=None):
    if isinstance(value, torch.Tensor):
        tensor = value
    elif hasattr(value, "numpy"):
        tensor = torch.as_tensor(value.numpy())
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _shuffle_buffered(iterator, buffer_size, seed):
    rng = random.Random(seed)
    buffer = []
    for item in iterator:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            idx = rng.randrange(len(buffer))
            yield buffer.pop(idx)
    while buffer:
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)


def _unsorted_segment_mean(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, dim: int = 0
) -> torch.Tensor:
    segment_ids = segment_ids.to(device=data.device, dtype=torch.long)
    moved = data.movedim(dim, 0)
    out = torch.zeros((num_segments, *moved.shape[1:]), dtype=data.dtype, device=data.device)
    index = segment_ids.view(-1, *([1] * (moved.ndim - 1))).expand_as(moved)
    out.scatter_add_(0, index, moved)
    counts = torch.zeros(num_segments, dtype=data.dtype, device=data.device)
    counts.scatter_add_(0, segment_ids, torch.ones_like(segment_ids, dtype=data.dtype))
    counts = counts.clamp_min(1).view(-1, *([1] * (moved.ndim - 1)))
    return (out / counts).movedim(0, dim)
