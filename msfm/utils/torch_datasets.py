"""PyTorch dataset helpers and batch schema documentation for MSFM samples.

The WebDataset-backed PyTorch loaders in :mod:`msfm.onthefly_pipeline` and
future Torch loaders near :mod:`msfm.grid_pipeline` should expose samples as
mappings whenever possible.  The logical unbatched sample schema is:

    kg: optional weak-lensing convergence map, shape ``(n_pix, n_z_wl)``.
    dg: optional galaxy-clustering density map, shape ``(n_pix, n_z_gc)``.
    xg: optional lensing/clustering cross map, shape ``(n_pix, n_z_cross)``.
    cls: optional angular power spectra, shape convention depends on the
        preprocessing configuration, commonly ``(n_ell, n_cls)`` or another
        fixed trailing shape shared by all samples in a batch.
    cosmo: cosmological parameter vector, shape ``(n_params,)``.
    i_sobol: Sobol-sequence cosmology index, scalar integer ``()``.
    i_signal: signal-realization index, scalar integer ``()``.

After batching with :func:`msfm_collate_fn`, every present tensor receives a
leading batch dimension ``batch_size``:

    kg -> ``(batch_size, n_pix, n_z_wl)``
    dg -> ``(batch_size, n_pix, n_z_gc)``
    xg -> ``(batch_size, n_pix, n_z_cross)``
    cls -> ``(batch_size, ...)`` where ``...`` is the unbatched ``cls`` shape
    cosmo -> ``(batch_size, n_params)``
    i_sobol -> ``(batch_size,)``
    i_signal -> ``(batch_size,)``

Floating arrays are converted to ``torch.float32`` by default, unless an
explicit ``float_dtype`` is provided. Metadata indices are always converted to
``torch.int64``. Optional keys may be absent or ``None``; absent keys stay
absent in the batch and all-``None`` keys are returned as ``None``. Mixed
``None``/tensor values for the same key are rejected because PyTorch cannot
represent ragged optional tensor batches without a project-specific sentinel.
"""

from collections.abc import Mapping, Sequence
from functools import partial

import torch
from torch.utils.data._utils.collate import default_collate

FLOAT_KEYS = frozenset({"kg", "dg", "xg", "cls", "cosmo"})
INDEX_KEYS = frozenset({"i_sobol", "i_signal", "i_noise", "i_example"})
ONTHEFLY_TUPLE_KEYS = (
    "gg",
    "ga",
    "gd",
    "ds",
    "dg",
    "qg",
    "cosmo",
    "i_sobol",
    "i_signal",
    "n_params",
    "n_pix",
    "n_z_wl",
    "n_z_gc",
)


def _as_tensor(value, *, key, float_dtype):
    """Convert one sample field to a tensor with the MSFM schema dtype."""
    if value is None:
        return None

    if key in INDEX_KEYS or key.startswith("i_") or key.startswith("n_"):
        return torch.as_tensor(value, dtype=torch.int64)

    tensor = torch.as_tensor(value)
    if key in FLOAT_KEYS or torch.is_floating_point(tensor):
        return tensor.to(dtype=float_dtype)
    return tensor


def _sample_to_mapping(sample, tuple_keys):
    """Normalize mapping and known tuple WebDataset samples to dictionaries."""
    if isinstance(sample, Mapping):
        return dict(sample)
    if isinstance(sample, tuple) and tuple_keys is not None:
        if len(sample) != len(tuple_keys):
            raise ValueError(f"Expected tuple sample of length {len(tuple_keys)}, got {len(sample)}")
        return dict(zip(tuple_keys, sample))
    return sample


def msfm_collate_fn(samples, *, float_dtype=torch.float32, tuple_keys=None):
    """Collate MSFM PyTorch samples while enforcing documented dtypes.

    Args:
        samples: Sequence of samples from a PyTorch ``Dataset`` or WebDataset.
            Mapping samples are preferred. Tuple samples can be supported by
            passing ``tuple_keys`` with the field names in tuple order.
        float_dtype: dtype for floating tensors. Defaults to ``torch.float32``.
        tuple_keys: Optional field names for tuple samples. Use
            :data:`ONTHEFLY_TUPLE_KEYS` for ``OntheflyPipeline`` tuples.

    Returns:
        A dictionary batch for mapping/known tuple samples, or PyTorch's default
        collation result for unsupported sample types.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    normalized = [_sample_to_mapping(sample, tuple_keys) for sample in samples]
    if not all(isinstance(sample, Mapping) for sample in normalized):
        return default_collate(samples)

    keys = set().union(*(sample.keys() for sample in normalized))
    batch = {}
    for key in keys:
        values = [sample.get(key) for sample in normalized]
        if all(value is None for value in values):
            batch[key] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"Cannot collate key {key!r}: only some samples contain a value")
        tensors = [_as_tensor(value, key=key, float_dtype=float_dtype) for value in values]
        batch[key] = torch.stack(tensors, dim=0)
    return batch


def make_msfm_collate_fn(*, float_dtype=torch.float32, tuple_keys=None):
    """Return a picklable ``DataLoader.collate_fn`` configured for MSFM batches."""
    return partial(msfm_collate_fn, float_dtype=float_dtype, tuple_keys=tuple_keys)
