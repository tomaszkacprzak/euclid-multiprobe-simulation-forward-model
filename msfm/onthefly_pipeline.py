# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak

This file is loosely based off
grid_pipeline.py by Arne Thomsen
"""

import glob
import io
import warnings
from typing import Iterator, Optional, Sequence, Tuple

import torch

from msfm.utils import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


ONTHEFLY_FIELDS: Tuple[str, ...] = (
    "γg",
    "γa",
    "γd",
    "ds",
    "dg",
    "qg",
    "cosmo",
    "i_sobol",
    "i_signal",
    "n_params",
    "n_pix",
    "n_z_WL",
    "n_z_GC",
)

_DALI_EXTENSIONS: Tuple[str, ...] = (
    "γg.pth",
    "γa.pth",
    "γd.pth",
    "ds.pth",
    "dg.pth",
    "qg.pth",
    "cosmo.pth",
    "i_sobol.index",
    "i_signal.index",
    "n_params.count",
    "n_pix.count",
    "n_z_WL.count",
    "n_z_GC.count",
)

_TORCH_SERIALIZED_FIELDS = frozenset({"γg", "γa", "γd", "ds", "dg", "qg", "cosmo"})


class _DALIOntheflyLoader:
    """Iterable wrapper that converts DALI WebDataset batches to tuples of torch tensors."""

    def __init__(self, iterator, output_device: Optional[str] = None):
        self.iterator = iterator
        self.output_device = output_device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        for dali_batch in self.iterator:
            # DALIGenericIterator returns a list with one dict per pipeline.
            if isinstance(dali_batch, list):
                if len(dali_batch) != 1:
                    raise ValueError("OntheflyPipeline currently supports one DALI pipeline per loader")
                dali_batch = dali_batch[0]

            decoded = []
            for field in ONTHEFLY_FIELDS:
                samples = _split_batch(dali_batch[field])
                tensors = [_decode_sample(sample, field) for sample in samples]
                batch = torch.stack(tensors, dim=0)
                if self.output_device is not None:
                    batch = batch.to(self.output_device, non_blocking=True)
                decoded.append(batch)

            yield tuple(decoded)

    def __len__(self):
        return len(self.iterator)

    def reset(self) -> None:
        self.iterator.reset()


class OntheflyPipeline:
    """
    Sets up a PyTorch-compatible DALI WebDataset loader for on-the-fly cosmologies.
    """

    def __init__(self, num_threads: int = 4, device_id: int = 0, seed: int = 0):
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed

    def get_dset(
        self,
        webds_pattern: str,
        local_batch_size: int,
        output_device: Optional[str] = None,
        index_paths: Optional[Sequence[str]] = None,
        random_shuffle: bool = False,
        initial_fill: int = 1024,
        shard_id: int = 0,
        num_shards: int = 1,
        pad_last_batch: bool = False,
        read_ahead: bool = True,
    ) -> _DALIOntheflyLoader:
        """Build a PyTorch-compatible data loader over on-the-fly WebDataset tar files.

        The loader returns batches as a tuple of tensors in the same order as the
        fields stored by ``run_onthefly_postprocessing``:
        ``(γg, γa, γd, ds, dg, qg, cosmo, i_sobol, i_signal, n_params, n_pix, n_z_WL, n_z_GC)``.

        Args:
            webds_pattern: Glob pattern of the WebDataset tar files.
            local_batch_size: Batch size produced by this process.
            output_device: Optional torch device that receives the decoded tensors.
            index_paths: Optional DALI WebDataset index files. Pre-generating them
                with ``wds2idx`` avoids DALI scanning large tar files at startup.
            random_shuffle: Shuffle samples inside DALI's reader.
            initial_fill: DALI shuffle buffer size when ``random_shuffle`` is true.
            shard_id: Current distributed shard id.
            num_shards: Total number of distributed shards.
            pad_last_batch: Whether DALI should pad the final partial batch.
            read_ahead: Enable DALI reader read-ahead.

        Returns:
            Iterable PyTorch loader yielding tuples of tensors.
        """

        paths = sorted(glob.glob(webds_pattern))
        if len(paths) == 0:
            raise FileNotFoundError(f"No WebDataset tar files match pattern: {webds_pattern}")

        try:
            import nvidia.dali.fn as fn
            from nvidia.dali import pipeline_def
            from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
        except ImportError as exc:
            raise ImportError(
                "OntheflyPipeline requires NVIDIA DALI. Install the nvidia-dali package "
                "matching your CUDA version, e.g. nvidia-dali-cuda130."
            ) from exc

        @pipeline_def(
            batch_size=local_batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            seed=self.seed,
        )
        def _pipeline():
            return fn.readers.webdataset(
                paths=paths,
                index_paths=index_paths,
                ext=_DALI_EXTENSIONS,
                missing_component_behavior="error",
                random_shuffle=random_shuffle,
                initial_fill=initial_fill,
                shard_id=shard_id,
                num_shards=num_shards,
                pad_last_batch=pad_last_batch,
                read_ahead=read_ahead,
                name="OntheflyWebDatasetReader",
            )

        pipe = _pipeline()
        iterator = DALIGenericIterator(
            pipelines=[pipe],
            output_map=list(ONTHEFLY_FIELDS),
            reader_name="OntheflyWebDatasetReader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
        return _DALIOntheflyLoader(iterator, output_device=output_device)


def _split_batch(batch):
    """Return a Python list of samples from tensors returned by the DALI PyTorch plugin."""
    if isinstance(batch, (list, tuple)):
        return list(batch)
    if hasattr(batch, "as_tensor"):
        batch = batch.as_tensor()
    if isinstance(batch, torch.Tensor):
        return [batch[i] for i in range(batch.shape[0])]
    try:
        return [batch.at(i) for i in range(len(batch))]
    except AttributeError:
        return list(batch)


def _sample_to_bytes(sample) -> bytes:
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().contiguous().view(-1).numpy()
    elif hasattr(sample, "as_cpu"):
        sample = sample.as_cpu().as_array()
    elif hasattr(sample, "as_array"):
        sample = sample.as_array()

    if hasattr(sample, "tobytes"):
        return sample.tobytes()
    return bytes(sample)


def _decode_sample(sample, field: str) -> torch.Tensor:
    payload = _sample_to_bytes(sample)
    if field in _TORCH_SERIALIZED_FIELDS:
        tensor = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor

    text = payload.rstrip(b"\x00\n\r\t ").decode("utf-8")
    return torch.tensor(int(text), dtype=torch.int64)
