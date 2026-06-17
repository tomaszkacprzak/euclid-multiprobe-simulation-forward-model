# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak

This file is loosely based off
grid_pipeline.py by Arne Thomsen
"""

import glob
import io
import os
import tarfile
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from msfm.utils import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


WDS_FIELDS: Tuple[str, ...] = (
    "gamma_g.pth",
    "gamma_a.pth",
    "gamma_d.pth",
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


class _OntheflyDaliTorchIterator:
    """Convert DALI WebDataset batches into tuples of PyTorch tensors.

    DALI is responsible for high-throughput WebDataset tar reading. The on-the-fly
    postprocessing writer stores the map and cosmology tensors as serialized
    ``torch.save`` payloads (``*.pth``) and the metadata fields as ASCII encoded
    integers (``*.index``/``*.count``), so this thin wrapper performs only that
    final deserialization step and returns the fields in ``WDS_FIELDS`` order.
    """

    def __init__(
        self,
        dali_iterator,
        pth_fields: Sequence[str],
        int_fields: Sequence[str],
        output_device: str,
    ):
        self.dali_iterator = dali_iterator
        self.pth_fields = tuple(pth_fields)
        self.int_fields = tuple(int_fields)
        self.output_device = output_device

    def __iter__(self) -> "_OntheflyDaliTorchIterator":
        return self

    def __next__(self) -> Tuple[torch.Tensor, ...]:
        batch = next(self.dali_iterator)
        if len(batch) != 1:
            raise RuntimeError(f"Expected a single DALI pipeline, got {len(batch)} pipeline outputs")

        data = batch[0]
        outputs: List[torch.Tensor] = []
        for field in WDS_FIELDS:
            raw = data[field]
            if field in self.pth_fields:
                outputs.append(_deserialize_pth_batch(raw, output_device=self.output_device))
            elif field in self.int_fields:
                outputs.append(_deserialize_int_batch(raw, output_device=self.output_device))
            else:
                raise KeyError(f"Unexpected WebDataset field {field!r}")
        return tuple(outputs)

    def reset(self) -> None:
        self.dali_iterator.reset()


def _deserialize_pth_batch(raw: torch.Tensor, output_device: str) -> torch.Tensor:
    """Deserialize a batch of uint8 ``torch.save`` payloads and stack it."""

    tensors = []
    for sample in raw.cpu():
        payload = sample.numpy().tobytes()
        tensors.append(torch.load(io.BytesIO(payload), map_location=output_device, weights_only=False))
    return torch.stack(tensors, dim=0)


def _deserialize_int_batch(raw: torch.Tensor, output_device: str) -> torch.Tensor:
    """Deserialize a batch of ASCII integer payloads into an int64 tensor."""

    values = []
    for sample in raw.cpu():
        payload = sample.numpy().tobytes().rstrip(b"\x00")
        values.append(int(payload.decode("utf-8")))
    return torch.tensor(values, dtype=torch.int64, device=output_device)


def _tar_member_extension(member_name: str) -> str:
    """Return the WebDataset extension DALI uses for a tar member name."""

    filename = os.path.basename(member_name)
    return filename.split(".", 1)[1] if "." in filename else ""


def _first_sample_extensions(path: str) -> set:
    """Inspect the first sample in a WebDataset tar without scanning the whole shard."""

    sample_key = None
    extensions = set()
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            filename = os.path.basename(member.name)
            if filename.startswith(".") or "." not in filename:
                continue
            key = filename.split(".", 1)[0]
            if sample_key is None:
                sample_key = key
            elif key != sample_key:
                break
            extensions.add(_tar_member_extension(member.name))
    return extensions


def _validate_first_sample_components(path: str, extensions: Sequence[str]) -> None:
    """Raise a useful error before DALI reports an underful sample."""

    found_extensions = _first_sample_extensions(path)
    expected_extensions = set(extensions)
    missing_extensions = sorted(expected_extensions - found_extensions)
    if not missing_extensions:
        return

    legacy_gamma_extensions = {"γg.pth", "γa.pth", "γd.pth"}
    legacy_hint = ""
    if legacy_gamma_extensions & found_extensions:
        legacy_hint = (
            " The shard appears to use legacy non-ASCII gamma component names "
            "('γg.pth', 'γa.pth', 'γd.pth'). Regenerate it with the current "
            "run_onthefly_postprocessing writer, which stores DALI-compatible "
            "ASCII names ('gamma_g.pth', 'gamma_a.pth', 'gamma_d.pth')."
        )

    raise ValueError(
        f"The first WebDataset sample in {path!r} is missing components {missing_extensions}. "
        f"Found components are {sorted(found_extensions)}.{legacy_hint}"
    )


def _is_cuda_device(device: str) -> bool:
    """Return whether a torch device string refers to CUDA."""

    return torch.device(device).type == "cuda"


class OntheflyPipeline:
    """
    Sets up a PyTorch-compatible DALI loader for the on-the-fly WebDataset files.
    """

    def __init__(self):
        pass

    def get_dset(
        self,
        webds_pattern: str,
        local_batch_size: int,
        *,
        n_workers: int = 4,
        device_id: Optional[int] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        is_eval: bool = False,
        drop_last: bool = None,
        prefetch_queue_depth: int = 2,
        initial_fill: int = 1024,
        seed: int = 12,
        output_device: str = "cpu",
    ) -> Iterable[Tuple[torch.Tensor, ...]]:
        """Build a PyTorch data loader over on-the-fly WebDataset tar files.

        Args:
            webds_pattern: Glob pattern of the WebDataset tar files.
            local_batch_size: Batch size returned by the loader.
            n_workers: Number of DALI worker threads.
            device_id: DALI pipeline device id. If None, GPU 0 is used when CUDA is available and a CPU DALI
                pipeline is used otherwise.
            shard_id: Current shard id for distributed loading.
            num_shards: Number of distributed shards.
            is_eval: If True, read deterministically without random shuffling.
            drop_last: Whether to drop an incomplete final batch. Defaults to
                ``False`` for evaluation and ``True`` for training.
            prefetch_queue_depth: Number of DALI batches to prefetch.
            initial_fill: Shuffle buffer size used by DALI when training.
            seed: DALI random seed.
            output_device: Device for the returned PyTorch tensors. Use ``"cpu"``
                to keep batches in host memory, or a CUDA device such as
                ``"cuda:0"`` to deserialize tensors directly onto the GPU.

        Returns:
            An iterable yielding a tuple of tensors in this order:
            ``(γg, γa, γd, ds, dg, qg, cosmo, i_sobol, i_signal, n_params,
            n_pix, n_z_WL, n_z_GC)``.
        """

        paths = sorted(glob.glob(webds_pattern))
        if not paths:
            raise FileNotFoundError(f"No WebDataset tar files match pattern {webds_pattern!r}")
        if drop_last is None:
            drop_last = not is_eval

        cuda_available = torch.cuda.is_available()
        if device_id is None:
            device_id = 0 if cuda_available else None
        if _is_cuda_device(output_device) and not cuda_available:
            raise RuntimeError(
                f"output_device={output_device!r} requests CUDA, but torch.cuda.is_available() is False"
            )

        try:
            from nvidia.dali import fn, pipeline_def
            from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
        except ImportError as exc:
            raise ImportError("OntheflyPipeline requires NVIDIA DALI with the PyTorch plugin installed") from exc

        # The writer stores members as ``<sample_key>.<field>``, for example
        # ``000000001.gamma_g.pth``. DALI therefore needs the full field name
        # after the first dot, not only the terminal suffix (``pth``).
        extensions = WDS_FIELDS
        _validate_first_sample_components(paths[0], extensions)
        pth_fields = tuple(field for field in WDS_FIELDS if field.endswith(".pth"))
        int_fields = tuple(field for field in WDS_FIELDS if field.endswith((".index", ".count")))

        @pipeline_def
        def _webdataset_pipeline():
            # ``readers.webdataset`` returns one DataNode per requested
            # extension. Convert the returned list to a tuple so DALI sees
            # multiple pipeline outputs instead of a single nested DataNode.
            outputs = fn.readers.webdataset(
                paths=paths,
                ext=extensions,
                random_shuffle=not is_eval,
                initial_fill=initial_fill,
                missing_component_behavior="error",
                shard_id=shard_id,
                num_shards=num_shards,
                pad_last_batch=False,
                name="OntheflyWebDatasetReader",
            )
            return tuple(outputs)

        pipe = _webdataset_pipeline(
            batch_size=local_batch_size,
            num_threads=n_workers,
            device_id=device_id,
            prefetch_queue_depth=prefetch_queue_depth,
            seed=seed,
        )

        last_batch_policy = LastBatchPolicy.DROP if drop_last else LastBatchPolicy.PARTIAL
        dali_iterator = DALIGenericIterator(
            [pipe],
            output_map=list(WDS_FIELDS),
            reader_name="OntheflyWebDatasetReader",
            last_batch_policy=last_batch_policy,
            auto_reset=True,
        )
        LOGGER.info(f"Built DALI WebDataset loader from {len(paths)} files matching {webds_pattern}")
        return _OntheflyDaliTorchIterator(
            dali_iterator,
            pth_fields=pth_fields,
            int_fields=int_fields,
            output_device=output_device,
        )
