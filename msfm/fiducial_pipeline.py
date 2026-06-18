# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""PyTorch/WebDataset input pipeline for fiducial cosmology samples."""

from __future__ import annotations

import glob
import random
import warnings
from typing import Union

import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from msfm.utils import logger, parameters, webdatasets
from msfm.utils.base_pipeline import MSFMpipeline
from msfm.grid_pipeline import _as_torch

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class FiducialPipeline(MSFMpipeline):
    """Set up a PyTorch/WebDataset loader for fiducial and perturbation samples."""

    def __init__(
        self,
        conf: dict = None,
        params: list = None,
        with_lensing: bool = True,
        with_clustering: bool = True,
        apply_norm: bool = True,
        with_padding: bool = True,
        z_bin_inds: list = None,
        return_maps: bool = True,
        return_cls: bool = True,
        apply_m_bias: bool = True,
        shape_noise_scale: float = 1.0,
        poisson_noise_scale: float = 1.0,
    ):
        super().__init__(
            conf=conf,
            params=params,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            apply_norm=apply_norm,
            with_padding=with_padding,
            z_bin_inds=z_bin_inds,
            return_maps=return_maps,
            return_cls=return_cls,
            apply_m_bias=apply_m_bias,
            shape_noise_scale=shape_noise_scale,
            poisson_noise_scale=poisson_noise_scale,
        )
        self.pert_labels = parameters.get_fiducial_perturbation_labels(self.params)
        self.n_noise = self.conf["analysis"]["fiducial"]["n_noise_per_example"]

    def get_dset(
        self,
        pattern: str = None,
        local_batch_size: int = None,
        noise_indices: Union[int, list, range] = 1,
        is_cached: bool = False,
        n_readers: int = 8,
        n_workers: int = None,
        n_prefetch: int = None,
        file_name_shuffle_buffer: int = 16,
        examples_shuffle_buffer: int = 64,
        is_eval: bool = False,
        drop_remainder: bool = None,
        eval_seed: int = 32,
        file_name_shuffle_seed: int = 17,
        examples_shuffle_seed: int = 67,
        input_context=None,
    ) -> wds.WebLoader:
        """Build a WebDataset-backed PyTorch loader for fiducial samples."""
        if is_cached:
            LOGGER.warning("Caching is ignored by the PyTorch WebDataset fiducial loader")
        if is_eval:
            torch.manual_seed(eval_seed)
        if n_workers is None:
            n_workers = 0
        if drop_remainder is None:
            drop_remainder = not is_eval

        if isinstance(noise_indices, int):
            if noise_indices < 1:
                raise AssertionError("noise_indices must be >= 1")
            noise_indices = list(range(noise_indices))
        elif isinstance(noise_indices, range):
            noise_indices = list(noise_indices)
        elif isinstance(noise_indices, list):
            if not noise_indices or not all(isinstance(i, int) for i in noise_indices):
                raise AssertionError("noise_indices must be a non-empty list of integers")
        else:
            raise TypeError("noise_indices must be an integer, list of integers or range")

        if pattern is None:
            raise ValueError("pattern must be provided for WebDataset .tar shards")

        file_names = sorted(file_name for file_name in glob.glob(pattern) if file_name.endswith(".tar"))
        if not file_names:
            raise FileNotFoundError(f"No WebDataset .tar shards match pattern {pattern!r}")
        if not is_eval:
            random.Random(file_name_shuffle_seed).shuffle(file_names)
        if input_context is not None:
            file_names = file_names[input_context.input_pipeline_id :: input_context.num_input_pipelines]

        dataset = _FiducialWebDatasetIterable(
            file_names=file_names,
            pipeline=self,
            noise_indices=noise_indices,
            repeat=not is_eval,
            shuffle_examples=not is_eval,
            examples_shuffle_seed=examples_shuffle_seed,
            examples_shuffle_buffer=examples_shuffle_buffer,
        )
        return wds.WebLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=n_workers,
            drop_last=drop_remainder,
            collate_fn=lambda samples: self._augmentations(_collate_samples(samples)),
        )

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

    def _augmentations(self, data_vectors: dict) -> dict:
        out = {
            "i_signal": data_vectors.pop("i_signal"),
            "i_noise": data_vectors.pop("i_noise"),
        }
        if self.return_maps:
            maps = []
            for label in self.pert_labels:
                if self.with_lensing and "bg" not in label:
                    kg = data_vectors[f"kg_{label}"] + data_vectors["sn"]
                    kg = kg * _as_torch(self.masks_WL, dtype=kg.dtype)
                    maps.append(kg)
                if self.with_clustering and "Aia" not in label:
                    dg = data_vectors[f"dg_{label}"] + data_vectors["pn"]
                    dg = dg * _as_torch(self.masks_GC, dtype=dg.dtype)
                    maps.append(dg)
            map_tensor = torch.cat(maps, dim=0) if maps else None
            if map_tensor is not None and not self.with_padding:
                map_tensor = map_tensor[:, _as_torch(self.mask_total).bool(), :]
            if map_tensor is not None and self.z_bin_inds is not None:
                map_tensor = torch.index_select(map_tensor, -1, _as_torch(self.z_bin_inds, dtype=torch.long))
            out["maps"] = map_tensor
        if self.return_cls:
            out["cls"] = torch.cat([data_vectors[f"cl_{label}"] for label in self.pert_labels], dim=0)
        return out


class _FiducialWebDatasetIterable(IterableDataset):
    def __init__(self, file_names, pipeline, noise_indices, repeat, shuffle_examples, examples_shuffle_seed, examples_shuffle_buffer):
        self.file_names = file_names
        self.pipeline = pipeline
        self.noise_indices = list(noise_indices)
        self.repeat = repeat
        self.shuffle_examples = shuffle_examples
        self.examples_shuffle_seed = examples_shuffle_seed
        self.examples_shuffle_buffer = examples_shuffle_buffer

    def __iter__(self):
        epoch = 0
        while True:
            items = self._iter_once()
            if self.shuffle_examples:
                items = _shuffle_buffered(items, self.examples_shuffle_buffer, self.examples_shuffle_seed + epoch)
            yield from items
            epoch += 1
            if not self.repeat:
                break

    def _iter_once(self):
        for file_name in self.file_names:
            for sample in wds.WebDataset([file_name], shardshuffle=False):
                decoded = webdatasets.decode_fiducial_sample(
                    sample,
                    self.pipeline.pert_labels,
                    self.noise_indices,
                    self.pipeline.n_dv_pix,
                    self.pipeline.n_z_WL,
                    self.pipeline.n_z_GC,
                    self.pipeline.n_noise,
                    self.pipeline.n_cls,
                    self.pipeline.n_z_cross,
                    self.pipeline.with_lensing,
                    self.pipeline.with_clustering,
                    self.pipeline.return_maps,
                    self.pipeline.return_cls,
                )
                decoded = {key: _as_torch(value) for key, value in decoded.items()}
                for row, i_noise in enumerate(self.noise_indices):
                    out = {"i_signal": decoded["i_signal"], "i_noise": torch.tensor(i_noise, dtype=torch.long)}
                    if self.pipeline.return_maps:
                        for label in self.pipeline.pert_labels:
                            if self.pipeline.with_lensing and "bg" not in label:
                                out[f"kg_{label}"] = decoded[f"kg_{label}"]
                            if self.pipeline.with_clustering and "Aia" not in label:
                                out[f"dg_{label}"] = decoded[f"dg_{label}"]
                        if self.pipeline.with_lensing:
                            out["sn"] = decoded[f"sn_{i_noise}"]
                        if self.pipeline.with_clustering:
                            out["pn"] = decoded[f"pn_{i_noise}"]
                    if self.pipeline.return_cls:
                        for label in self.pipeline.pert_labels:
                            out[f"cl_{label}"] = decoded[f"cl_{label}"][row]
                    yield out


def _collate_samples(samples: list) -> dict:
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}


def _shuffle_buffered(iterator, buffer_size, seed):
    rng = random.Random(seed)
    buffer = []
    for item in iterator:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            yield buffer.pop(rng.randrange(len(buffer)))
    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))
