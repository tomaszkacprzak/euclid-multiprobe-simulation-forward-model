# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""Utilities for serializing MSFM samples as WebDataset dictionaries.

WebDataset shards store each array as its own file-like payload. This module stores NumPy arrays as ``.npy`` bytes while keeping the logical field names and decoded output dictionaries stable for downstream readers.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch

from msfm.utils import cross_statistics, logger

LOGGER = logger.get_logger(__file__)

Sample = Dict[str, Any]

# WebDataset sample schema #############################################################################################

SCHEMA_VERSION = 1
SCHEMA_VERSION_KEY = "schema_version"
METADATA_SUFFIX = ".index"

COSMO_KEY = "cosmo"
CLS_KEY = "cls"
GRID_LENSING_MAP_KEY_PREFIX = "kg"
GRID_CLUSTERING_MAP_KEY_PREFIX = "dg"
GRID_CROSS_MAP_KEY_PREFIX = "xg"
FIDUCIAL_SHAPE_NOISE_KEY_PREFIX = "sn"
FIDUCIAL_POISSON_NOISE_KEY_PREFIX = "pn"
FIDUCIAL_CLS_KEY_PREFIX = "cl"

GRID_ARRAY_KEYS = (COSMO_KEY, CLS_KEY)
FIDUCIAL_ARRAY_KEYS = (CLS_KEY,)
OPTIONAL_MAP_KEY_PREFIXES = (GRID_CROSS_MAP_KEY_PREFIX,)

I_SOBOL_KEY = "i_sobol"
I_SIGNAL_KEY = "i_signal"
N_PARAMS_KEY = "n_params"
N_PIX_KEY = "n_pix"
N_Z_WL_KEY = "n_z_wl"
N_Z_GC_KEY = "n_z_gc"
N_Z_CROSS_MAP_KEY = "n_z_cross_map"
N_NOISE_KEY = "n_noise"
N_CLS_KEY = "n_cls"
N_Z_CROSS_KEY = "n_z_cross"
WITH_LENSING_KEY = "with_lensing"
WITH_CLUSTERING_KEY = "with_clustering"
WITH_CROSS_KEY = "with_cross"

GRID_METADATA_KEYS = (N_PARAMS_KEY, I_SOBOL_KEY, I_SIGNAL_KEY)
FIDUCIAL_METADATA_KEYS = (N_PIX_KEY, N_Z_WL_KEY, N_Z_GC_KEY, I_SIGNAL_KEY, N_NOISE_KEY, N_CLS_KEY, N_Z_CROSS_KEY)
CLS_METADATA_KEYS = (N_NOISE_KEY, N_CLS_KEY, N_Z_CROSS_KEY)
PROBE_FLAG_KEYS = (WITH_LENSING_KEY, WITH_CLUSTERING_KEY, WITH_CROSS_KEY)


def _resolve_sample_key(sample: Mapping[str, Any], key: str) -> str:
    if key in sample:
        return key
    normalised = _normalise_key(key)
    if normalised in sample:
        return normalised
    lower = normalised.lower()
    for sample_key in sample:
        if sample_key.lower() == lower:
            return sample_key
    raise KeyError(f"sample is missing key {key!r}")


def _has_array_key(sample: Mapping[str, Any], key: str) -> bool:
    """Return whether ``sample`` contains an encoded or already-decoded array key."""
    try:
        _resolve_sample_key(sample, key)
    except KeyError:
        return False
    return True


def _normalise_metadata_key(key: str) -> str:
    return key.lower()


def _metadata_key(key: str) -> str:
    return f"{_normalise_metadata_key(key)}{METADATA_SUFFIX}"


def _has_metadata_key(sample: Mapping[str, Any], key: str) -> bool:
    return any(candidate in sample for candidate in (key, _metadata_key(key), f"{key}{METADATA_SUFFIX}"))


def _missing_metadata_keys(sample: Mapping[str, Any], keys: Sequence[str]) -> Sequence[str]:
    return tuple(key for key in keys if not _has_metadata_key(sample, key))


def _missing_array_keys(sample: Mapping[str, Any], keys: Sequence[str]) -> Sequence[str]:
    return tuple(key for key in keys if not _has_array_key(sample, key))


def _raise_invalid_sample(kind: str, missing_metadata: Sequence[str], missing_arrays: Sequence[str]) -> None:
    details = []
    if missing_metadata:
        details.append(f"missing metadata keys: {', '.join(missing_metadata)}")
    if missing_arrays:
        details.append(f"missing array keys: {', '.join(_normalise_key(key) for key in missing_arrays)}")
    if details:
        raise KeyError(f"Invalid {kind} WebDataset sample: {'; '.join(details)}")


def validate_grid_sample(
    sample: Mapping[str, Any],
    noise_indices: Sequence[int] = (),
    *,
    with_lensing: bool = True,
    with_clustering: bool = True,
    with_cross: bool = False,
    return_maps: bool = True,
    return_cls: bool = True,
    require_cosmo: bool = True,
) -> None:
    """Validate that a grid WebDataset sample has the keys needed by the decoder."""
    metadata_keys = list(GRID_METADATA_KEYS if require_cosmo else (I_SOBOL_KEY, I_SIGNAL_KEY))
    array_keys = [COSMO_KEY] if require_cosmo else []

    if return_cls:
        metadata_keys.extend(CLS_METADATA_KEYS)
        array_keys.append(CLS_KEY)
    if return_maps:
        metadata_keys.append(N_PIX_KEY)
        if with_lensing:
            metadata_keys.append(N_Z_WL_KEY)
            array_keys.extend(f"{GRID_LENSING_MAP_KEY_PREFIX}_{i}" for i in noise_indices)
        if with_clustering:
            metadata_keys.append(N_Z_GC_KEY)
            array_keys.extend(f"{GRID_CLUSTERING_MAP_KEY_PREFIX}_{i}" for i in noise_indices)
        if with_cross:
            metadata_keys.append(N_Z_CROSS_MAP_KEY)
            array_keys.extend(f"{GRID_CROSS_MAP_KEY_PREFIX}_{i}" for i in noise_indices)

    _raise_invalid_sample(
        "grid",
        _missing_metadata_keys(sample, tuple(dict.fromkeys(metadata_keys))),
        _missing_array_keys(sample, tuple(dict.fromkeys(array_keys))),
    )


def validate_fiducial_sample(
    sample: Mapping[str, Any],
    pert_labels: Sequence[str] = (),
    noise_indices: Sequence[int] = (),
    *,
    with_lensing: bool = True,
    with_clustering: bool = True,
    return_maps: bool = True,
    return_cls: bool = True,
) -> None:
    """Validate that a fiducial WebDataset sample has the keys needed by the decoder."""
    metadata_keys = list(FIDUCIAL_METADATA_KEYS)
    array_keys = []

    if return_cls:
        array_keys.extend(f"{FIDUCIAL_CLS_KEY_PREFIX}_{label}" for label in pert_labels)
    if return_maps:
        if with_lensing:
            array_keys.extend(f"{GRID_LENSING_MAP_KEY_PREFIX}_{label}" for label in pert_labels if "bg" not in label)
            array_keys.extend(f"{FIDUCIAL_SHAPE_NOISE_KEY_PREFIX}_{i}" for i in noise_indices)
        if with_clustering:
            array_keys.extend(
                f"{GRID_CLUSTERING_MAP_KEY_PREFIX}_{label}" for label in pert_labels if "Aia" not in label
            )
            array_keys.extend(f"{FIDUCIAL_POISSON_NOISE_KEY_PREFIX}_{i}" for i in noise_indices)

    _raise_invalid_sample(
        "fiducial",
        _missing_metadata_keys(sample, tuple(dict.fromkeys(metadata_keys))),
        _missing_array_keys(sample, tuple(dict.fromkeys(array_keys))),
    )


# Encoding / decoding primitives ######################################################################################


def _normalise_key(key: str) -> str:
    """Return the WebDataset key used for an array payload."""
    return key if key.endswith(".npy") else f"{key}.npy"


def _encode_npy(array: np.ndarray) -> bytes:
    """Serialize a NumPy array to a byte string in ``.npy`` format."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _decode_npy(payload: Any) -> np.ndarray:
    """Deserialize a ``.npy`` payload produced by :func:`_encode_npy`."""
    if isinstance(payload, np.ndarray):
        return payload
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    with io.BytesIO(payload) as buffer:
        return np.load(buffer, allow_pickle=False)


def _set_array(sample: MutableMapping[str, Any], key: str, array: np.ndarray) -> None:
    sample[_normalise_key(key)] = _encode_npy(array)


def _get_array(sample: Mapping[str, Any], key: str, *, dtype: Optional[Any] = None) -> np.ndarray:
    array = _decode_npy(sample[_resolve_sample_key(sample, key)])

    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _to_tensor(value: Any, *, dtype: Optional[Any] = None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return int(value)


def _set_metadata(sample: MutableMapping[str, Any], key: str, value: Any) -> None:
    if isinstance(value, bool):
        value = int(value)
    sample[_metadata_key(key)] = int(value)


def _metadata(sample: Mapping[str, Any], key: str) -> int:
    for candidate in (key, _metadata_key(key), f"{key}{METADATA_SUFFIX}"):
        if candidate in sample:
            return _to_int(sample[candidate])
    raise KeyError(f"sample is missing metadata field {key!r}")


def _parse_none_value(sample: Mapping[str, Any], key: str, value: Optional[int]) -> int:
    return _metadata(sample, key) if value is None else value


def _with_shape(tensor: torch.Tensor, shape: Sequence[Optional[int]]) -> torch.Tensor:
    if all(dim is not None for dim in shape):
        expected = tuple(int(dim) for dim in shape if dim is not None)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"expected tensor shape {expected}, got {tuple(tensor.shape)}")
        return tensor

    reshape_shape = tuple(-1 if dim is None else int(dim) for dim in shape)
    return tensor.reshape(reshape_shape)


def _gather_axis(tensor: torch.Tensor, indices: Any, axis: int) -> torch.Tensor:
    if isinstance(indices, (int, np.integer)):
        return tensor.select(axis, int(indices))
    index_tensor = torch.as_tensor(indices, dtype=torch.int64, device=tensor.device)
    return torch.index_select(tensor, axis, index_tensor.reshape(-1))


def _decode_data_vector(
    output: MutableMapping[str, torch.Tensor],
    sample: Mapping[str, Any],
    key_in: str,
    key_out: str,
    n_pix: Optional[int],
    n_z_bins: Optional[int],
    n_z_bins_label: str,
    tensor_backend: str = "tensorflow",
) -> None:
    tensor = _to_tensor(_get_array(sample, key_in), dtype=torch.float32)
    shape = (
        (n_pix, n_z_bins)
        if n_pix is not None and n_z_bins is not None
        else (_metadata(sample, "n_pix"), _metadata(sample, n_z_bins_label))
    )
    output[key_out] = _with_shape(tensor, shape, tensor_backend=tensor_backend)


def _decode_cls(
    output: MutableMapping[str, torch.Tensor],
    sample: Mapping[str, Any],
    key_in: str,
    key_out: str,
    n_noise: Optional[int],
    n_cls: Optional[int],
    n_z_cross: Optional[int],
    noise_indices: Any,
    bin_indices: Any,
    tensor_backend: str = "tensorflow",
) -> None:
    cls = _to_tensor(_get_array(sample, key_in), dtype=torch.float32)
    shape = (
        (_metadata(sample, "n_noise"), _metadata(sample, "n_cls"), _metadata(sample, "n_z_cross"))
        if n_noise is None and n_cls is None and n_z_cross is None
        else (n_noise, n_cls, n_z_cross)
    )
    cls = _with_shape(cls, shape)
    cls = _gather_axis(cls, noise_indices, axis=0)
    cls = _gather_axis(cls, bin_indices, axis=-1)
    output[key_out] = cls


# Grid samples #########################################################################################################


def encode_grid_sample(kg, sn_realz, dg, pn_realz, cls, cosmo, i_sobol, i_signal, xg=None, xn_realz=None) -> Sample:
    """Encode a grid-cosmology sample as a WebDataset sample dictionary.

    The array field names mirror ``parse_forward_grid``: ``cosmo``, ``cls``,
    ``kg_{i}``, ``dg_{i}``, and optionally ``xg_{i}``.  Arrays are stored under
    ``<field>.npy`` keys and scalar metadata keeps the stable field names.
    """
    sample: Sample = {}
    _set_metadata(sample, SCHEMA_VERSION_KEY, SCHEMA_VERSION)
    _set_metadata(sample, N_PARAMS_KEY, int(np.asarray(cosmo).shape[0]))
    _set_metadata(sample, I_SOBOL_KEY, i_sobol)
    _set_metadata(sample, I_SIGNAL_KEY, i_signal)
    _set_metadata(sample, WITH_LENSING_KEY, kg is not None and sn_realz is not None)
    _set_metadata(sample, WITH_CLUSTERING_KEY, dg is not None and pn_realz is not None)
    _set_metadata(sample, WITH_CROSS_KEY, xg is not None and xn_realz is not None)
    _set_array(sample, "cosmo", cosmo)

    if cls is not None:
        _set_array(sample, "cls", cls)
        _set_metadata(sample, N_NOISE_KEY, int(cls.shape[0]))
        _set_metadata(sample, N_CLS_KEY, int(cls.shape[1]))
        _set_metadata(sample, N_Z_CROSS_KEY, int(cls.shape[2]))

    if kg is not None and sn_realz is not None:
        assert kg.shape == sn_realz.shape[1:]
        _set_metadata(sample, N_PIX_KEY, int(kg.shape[0]))
        _set_metadata(sample, N_Z_WL_KEY, int(kg.shape[1]))
        for i, sn in enumerate(sn_realz):
            _set_array(sample, f"kg_{i}", kg + sn)

    if dg is not None and pn_realz is not None:
        assert dg.shape == pn_realz.shape[1:]
        if kg is None:
            _set_metadata(sample, N_PIX_KEY, int(dg.shape[0]))
        else:
            assert kg.shape[0] == dg.shape[0]
        _set_metadata(sample, N_Z_GC_KEY, int(dg.shape[1]))
        for i, pn in enumerate(pn_realz):
            _set_array(sample, f"dg_{i}", dg + pn)

    if xg is not None and xn_realz is not None:
        _set_metadata(sample, N_Z_CROSS_MAP_KEY, int(xg.shape[1]))
        for i, xn in enumerate(xn_realz):
            _set_array(sample, f"xg_{i}", xg + xn)

    return sample


def decode_grid_sample(
    sample,
    noise_indices,
    n_pix=None,
    n_z_WL=None,
    n_z_GC=None,
    n_z_cross_map=None,
    n_z_cross=None,
    n_params=None,
    n_noise=None,
    n_cls=None,
    with_lensing=True,
    with_clustering=True,
    with_cross=False,
    return_maps=True,
    return_cls=True,
    tensor_backend="tensorflow",
):
    """Decode a WebDataset grid sample into the same output schema as ``parse_inverse_grid``."""
    noise_indices = tuple(noise_indices)
    validate_grid_sample(
        sample,
        noise_indices,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross=with_cross,
        return_maps=return_maps,
        return_cls=return_cls,
    )
    output = {}
    cosmo = _to_tensor(_get_array(sample, "cosmo"), dtype=torch.float32)
    output["cosmo"] = _with_shape(cosmo, (_metadata(sample, "n_params"),) if n_params is None else (n_params,))

    for i in noise_indices:
        if return_maps:
            if with_lensing:
                _decode_data_vector(output, sample, f"kg_{i}", f"kg_{i}", n_pix, n_z_WL, "n_z_WL", tensor_backend)
            if with_clustering:
                _decode_data_vector(output, sample, f"dg_{i}", f"dg_{i}", n_pix, n_z_GC, "n_z_GC", tensor_backend)
            if with_cross:
                _decode_data_vector(
                    output, sample, f"xg_{i}", f"xg_{i}", n_pix, n_z_cross_map, "n_z_cross_map", tensor_backend
                )

        if return_cls:
            n_z_mc = _parse_none_value(sample, "n_z_WL", n_z_WL) if with_lensing else 0
            n_z_ml = _parse_none_value(sample, "n_z_GC", n_z_GC) if with_clustering else 0
            bin_indices, _ = cross_statistics.get_cross_bin_indices(
                n_z_mc,
                n_z_ml,
                with_lensing,
                with_clustering,
                with_cross_z=True,
                with_cross_probe=(with_lensing and with_clustering),
            )
            _decode_cls(
                output, sample, "cls", f"cl_{i}", n_noise, n_cls, n_z_cross, i, bin_indices, tensor_backend
            )

    output["i_sobol"] = _to_tensor(_metadata(sample, "i_sobol"), dtype=torch.int64)
    output["i_signal"] = _to_tensor(_metadata(sample, "i_signal"), dtype=torch.int64)
    return output


def decode_grid_cls_sample(sample, n_noise=None, n_cls=None, n_z_cross=None, n_params=None):
    """Decode only the grid power-spectrum fields, matching ``parse_inverse_grid_cls``."""
    validate_grid_sample(sample, return_maps=False, return_cls=True)
    output = {}
    cls = _to_tensor(_get_array(sample, "cls"), dtype=torch.float32)
    output["cls"] = _with_shape(
        cls,
        (_metadata(sample, "n_noise"), _metadata(sample, "n_cls"), _metadata(sample, "n_z_cross"))
        if n_noise is None and n_cls is None and n_z_cross is None
        else (n_noise, n_cls, n_z_cross),
    )
    cosmo = _to_tensor(_get_array(sample, "cosmo"), dtype=torch.float32)
    output["cosmo"] = _with_shape(cosmo, (_metadata(sample, "n_params"),) if n_params is None else (n_params,))
    output["i_sobol"] = _to_tensor(_metadata(sample, "i_sobol"), dtype=torch.int64)
    output["i_signal"] = _to_tensor(_metadata(sample, "i_signal"), dtype=torch.int64)
    return output


# Fiducial samples #####################################################################################################


def encode_fiducial_sample(
    cosmo_pert_labels,
    kg_perts,
    dg_perts,
    ia_pert_labels,
    ia_perts,
    sn_realz,
    bg_pert_labels,
    bg_perts,
    pn_realz,
    cl_perts,
    cl_ia_perts,
    cl_bg_perts,
    i_signal,
) -> Sample:
    """Encode a fiducial sample as a WebDataset sample dictionary."""
    assert len(kg_perts) == len(dg_perts) == len(cosmo_pert_labels)
    assert len(ia_pert_labels) == len(ia_perts)
    assert len(bg_pert_labels) == len(bg_perts)
    assert len(sn_realz) == len(pn_realz) == cl_perts.shape[1]

    sample: Sample = {}
    _set_metadata(sample, SCHEMA_VERSION_KEY, SCHEMA_VERSION)
    _set_metadata(sample, N_PIX_KEY, int(kg_perts[0].shape[0]))
    _set_metadata(sample, N_Z_WL_KEY, int(kg_perts[0].shape[1]))
    _set_metadata(sample, N_Z_GC_KEY, int(dg_perts[0].shape[1]))
    _set_metadata(sample, I_SIGNAL_KEY, i_signal)
    _set_metadata(sample, N_NOISE_KEY, int(cl_perts.shape[1]))
    _set_metadata(sample, N_CLS_KEY, int(cl_perts.shape[2]))
    _set_metadata(sample, N_Z_CROSS_KEY, int(cl_perts.shape[3]))
    _set_metadata(sample, WITH_LENSING_KEY, True)
    _set_metadata(sample, WITH_CLUSTERING_KEY, True)
    _set_metadata(sample, WITH_CROSS_KEY, False)
    _set_array(sample, "cls", cl_perts[0])

    for label, kg_pert, dg_pert, cl_pert in zip(cosmo_pert_labels, kg_perts, dg_perts, cl_perts):
        _set_array(sample, f"kg_{label}", kg_pert)
        _set_array(sample, f"dg_{label}", dg_pert)
        _set_array(sample, f"cl_{label}", cl_pert)
    for label, ia_pert, cl_ia_pert in zip(ia_pert_labels, ia_perts, cl_ia_perts):
        _set_array(sample, f"kg_{label}", ia_pert)
        _set_array(sample, f"cl_{label}", cl_ia_pert)
    for i, sn in enumerate(sn_realz):
        _set_array(sample, f"sn_{i}", sn)
    for label, bg_pert, cl_bg_pert in zip(bg_pert_labels, bg_perts, cl_bg_perts):
        _set_array(sample, f"dg_{label}", bg_pert)
        _set_array(sample, f"cl_{label}", cl_bg_pert)
    for i, pn in enumerate(pn_realz):
        _set_array(sample, f"pn_{i}", pn)
    return sample


def decode_fiducial_sample(
    sample,
    pert_labels,
    noise_indices,
    n_pix=None,
    n_z_WL=None,
    n_z_GC=None,
    n_noise=None,
    n_cls=None,
    n_z_cross=None,
    with_lensing=True,
    with_clustering=True,
    return_maps=True,
    return_cls=True,
):
    """Decode a WebDataset fiducial sample into the same output schema as ``parse_inverse_fiducial``."""
    pert_labels = tuple(pert_labels)
    noise_indices = tuple(noise_indices)
    validate_fiducial_sample(
        sample,
        pert_labels,
        noise_indices,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        return_maps=return_maps,
        return_cls=return_cls,
    )
    output = {}
    bin_indices, _ = cross_statistics.get_cross_bin_indices(
        _parse_none_value(sample, "n_z_WL", n_z_WL) if with_lensing else 0,
        _parse_none_value(sample, "n_z_GC", n_z_GC) if with_clustering else 0,
        with_lensing,
        with_clustering,
        with_cross_z=True,
        with_cross_probe=(with_lensing and with_clustering),
    )

    for label in pert_labels:
        if return_maps:
            if with_lensing and "bg" not in label:
                _decode_data_vector(output, sample, f"kg_{label}", f"kg_{label}", n_pix, n_z_WL, "n_z_WL")
            if with_clustering and "Aia" not in label:
                _decode_data_vector(output, sample, f"dg_{label}", f"dg_{label}", n_pix, n_z_GC, "n_z_GC")
        if return_cls:
            _decode_cls(
                output, sample, f"cl_{label}", f"cl_{label}", n_noise, n_cls, n_z_cross, noise_indices, bin_indices
            )

    if return_maps:
        for i in noise_indices:
            if with_lensing:
                _decode_data_vector(output, sample, f"sn_{i}", f"sn_{i}", n_pix, n_z_WL, "n_z_WL")
            if with_clustering:
                _decode_data_vector(output, sample, f"pn_{i}", f"pn_{i}", n_pix, n_z_GC, "n_z_GC")

    output["i_signal"] = _to_tensor(_metadata(sample, "i_signal"), dtype=torch.int64)
    return output


def decode_fiducial_cls_sample(sample, n_noise=None, n_cls=None, n_z_cross=None):
    """Decode only the fiducial power-spectrum fields, matching ``parse_inverse_fiducial_cls``."""
    validate_fiducial_sample(sample, return_maps=False, return_cls=False)
    _raise_invalid_sample("fiducial", (), _missing_array_keys(sample, (CLS_KEY,)))
    output = {}
    cls = _to_tensor(_get_array(sample, "cls"), dtype=torch.float32)
    output["cls"] = _with_shape(
        cls,
        (_metadata(sample, "n_noise"), _metadata(sample, "n_cls"), _metadata(sample, "n_z_cross"))
        if n_noise is None and n_cls is None and n_z_cross is None
        else (n_noise, n_cls, n_z_cross),
    )
    output["i_signal"] = _to_tensor(_metadata(sample, "i_signal"), dtype=torch.int64)
    return output


# Verification #########################################################################################################


def verify_grid_sample(
    sample, n_noise_per_signal, kg, sn_samples, dg, pn_samples, cosmo, i_sobol, i_signal, cls, xg=None, xn_samples=None
):
    """Assert that a grid WebDataset sample round-trips correctly."""
    with_cross_probe = xg is not None and xn_samples is not None
    with_lensing = kg is not None and sn_samples is not None
    with_clustering = dg is not None and pn_samples is not None
    inv = decode_grid_sample(
        sample,
        range(n_noise_per_signal),
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross=with_cross_probe,
        return_cls=cls is not None,
    )
    for i_noise in range(n_noise_per_signal):
        if with_lensing:
            assert np.allclose(inv[f"kg_{i_noise}"], kg + sn_samples[i_noise])
        if with_clustering:
            assert np.allclose(inv[f"dg_{i_noise}"], dg + pn_samples[i_noise])
        if cls is not None:
            assert np.allclose(inv[f"cl_{i_noise}"], cls[i_noise])
        if with_cross_probe:
            assert np.allclose(inv[f"xg_{i_noise}"], xg + xn_samples[i_noise])
    assert np.allclose(inv["cosmo"], cosmo)
    assert np.allclose(inv["i_sobol"], i_sobol)
    assert np.allclose(inv["i_signal"], i_signal)

    if cls is not None:
        inv_cls = decode_grid_cls_sample(sample)
        assert np.allclose(inv_cls["cls"], cls)
        assert np.allclose(inv_cls["cosmo"], cosmo)
        assert np.allclose(inv_cls["i_sobol"], i_sobol)
        assert np.allclose(inv_cls["i_signal"], i_signal)
    LOGGER.debug("Decoded the WebDataset grid sample successfully")
    return True


def verify_fiducial_sample(sample, pert_labels, noise_indices, expected):
    """Assert that selected decoded fiducial fields match expected arrays.

    Args:
        sample: WebDataset sample dictionary.
        pert_labels: Labels passed to :func:`decode_fiducial_sample`.
        noise_indices: Noise indices passed to :func:`decode_fiducial_sample`.
        expected: Mapping from decoded output keys to expected values.
    """
    decoded = decode_fiducial_sample(sample, pert_labels, noise_indices)
    for key, value in expected.items():
        assert key in decoded, f"decoded fiducial sample is missing {key!r}"
        assert np.allclose(decoded[key], value), key
    LOGGER.debug("Decoded the WebDataset fiducial sample successfully")
    return True
