# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak

This file is loosely based off
 grid_pipeline.py by Arne Thomsen
"""

from __future__ import annotations

import glob
import io
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch

from msfm.utils import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


PTH_FIELDS = ("gg", "ga", "gd", "ds", "dg", "qg", "cosmo")
SCALAR_FIELDS = ("i_sobol", "i_signal", "n_params", "n_pix", "n_z_WL", "n_z_GC")
FIELD_NAMES = PTH_FIELDS + SCALAR_FIELDS
WDS_EXTENSIONS = tuple(f"{field}.pth" for field in PTH_FIELDS) + tuple(
    f"{field}.index" if field.startswith("i_") else f"{field}.count" for field in SCALAR_FIELDS
)


class _DaliTorchWebdatasetLoader:
    """Small iterable wrapper that decodes DALI WebDataset byte batches as torch tensors."""

    def __init__(self, dali_iterator, output_device: str | "torch.device"):
        import torch

        self.dali_iterator = dali_iterator
        self.output_device = torch.device(output_device)

    def __iter__(self) -> Iterator[tuple["torch.Tensor", ...]]:
        for dali_batch in self.dali_iterator:
            # DALIGenericIterator returns a list with one dict per pipeline.
            batch = dali_batch[0] if isinstance(dali_batch, list) else dali_batch
            yield tuple(self._decode_batch(batch[field], field) for field in FIELD_NAMES)

    def __len__(self) -> int:
        return len(self.dali_iterator)

    def reset(self) -> None:
        self.dali_iterator.reset()

    def _decode_batch(self, encoded_batch, field: str) -> "torch.Tensor":
        import torch

        samples = self._iter_samples(encoded_batch)
        if field in PTH_FIELDS:
            tensors = [self._load_pth_tensor(sample) for sample in samples]
        else:
            tensors = [self._load_scalar_tensor(sample) for sample in samples]
        return torch.stack(tensors, dim=0).to(self.output_device, non_blocking=True)

    @staticmethod
    def _iter_samples(encoded_batch) -> list["torch.Tensor"]:
        import torch

        if isinstance(encoded_batch, torch.Tensor):
            if encoded_batch.ndim == 1:
                return [encoded_batch]
            return [sample for sample in encoded_batch]
        return list(encoded_batch)

    @staticmethod
    def _bytes_from_uint8_tensor(encoded: "torch.Tensor") -> bytes:
        import torch

        encoded = encoded.detach().to("cpu", dtype=torch.uint8).contiguous().view(-1)
        return bytes(encoded.tolist())

    def _load_pth_tensor(self, encoded: "torch.Tensor") -> "torch.Tensor":
        import torch

        # torch.load consumes the PyTorch serialization bytes written by webdataset's .pth encoder.
        tensor = torch.load(io.BytesIO(self._bytes_from_uint8_tensor(encoded)), map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor

    def _load_scalar_tensor(self, encoded: "torch.Tensor") -> "torch.Tensor":
        import torch

        value = int(self._bytes_from_uint8_tensor(encoded).rstrip(b"\x00\n\r\t ").decode("ascii"))
        return torch.tensor(value, dtype=torch.int64)


class OntheflyPipeline:
    """Sets up a PyTorch iterable loader for on-the-fly WebDataset shards."""

    def __init__(
        self,
        num_threads: int = 4,
        device_id: int = 0,
        random_shuffle: bool = False,
        initial_fill: int = 1024,
        seed: int = -1,
    ):
        self.num_threads = num_threads
        self.device_id = device_id
        self.random_shuffle = random_shuffle
        self.initial_fill = initial_fill
        self.seed = seed

    def get_dset(
        self,
        webds_pattern: str,
        local_batch_size: int,
        output_device: str | "torch.device" = "cpu",
        num_shards: int = 1,
        shard_id: int = 0,
        index_paths: Sequence[str] | None = None,
    ) -> _DaliTorchWebdatasetLoader:
        """Build a PyTorch loader for on-the-fly WebDataset tar files.

        Args:
            webds_pattern: Glob pattern of the WebDataset tar files.
            local_batch_size: Batch size produced by this process.
            output_device: Final torch device, for example ``"cpu"`` or ``"cuda:0"``.
            num_shards: Number of DALI data shards for distributed loading.
            shard_id: Shard id handled by this process.
            index_paths: Optional DALI WebDataset index files. If omitted, DALI infers them.

        Returns:
            Iterable whose batches are tuples of tensors in this order:
            ``(gg, ga, gd, ds, dg, qg, cosmo, i_sobol, i_signal, n_params, n_pix, n_z_WL, n_z_GC)``.
        """
        try:
            import nvidia.dali.fn as fn
            from nvidia.dali import pipeline_def
            from nvidia.dali.plugin.pytorch import DALIGenericIterator
        except ImportError as exc:
            raise ImportError("OntheflyPipeline requires NVIDIA DALI for Python to be installed.") from exc

        paths = sorted(glob.glob(webds_pattern))
        if not paths:
            raise FileNotFoundError(f"No WebDataset tar files match pattern: {webds_pattern}")

        @pipeline_def(batch_size=local_batch_size, num_threads=self.num_threads, device_id=self.device_id, seed=self.seed)
        def _pipeline():
            return fn.readers.webdataset(
                paths=paths,
                index_paths=index_paths,
                ext=list(WDS_EXTENSIONS),
                missing_component_behavior="error",
                random_shuffle=self.random_shuffle,
                initial_fill=self.initial_fill,
                num_shards=num_shards,
                shard_id=shard_id,
                name="OntheflyWebdatasetReader",
            )

        pipe = _pipeline()
        iterator = DALIGenericIterator(
            [pipe],
            output_map=list(FIELD_NAMES),
            reader_name="OntheflyWebdatasetReader",
            auto_reset=True,
            dynamic_shape=True,
        )
        return _DaliTorchWebdatasetLoader(iterator, output_device=output_device)
