import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
wds = pytest.importorskip("webdataset")
torch = pytest.importorskip("torch")

from msfm.utils import webdatasets

N_PIX = 5
N_Z_WL = 2
N_Z_GC = 1
N_Z_CROSS_MAP = N_Z_WL * N_Z_GC
N_NOISE = 3
N_CLS = 4
N_Z_CROSS = (N_Z_WL + N_Z_GC) * (N_Z_WL + N_Z_GC + 1) // 2
N_PARAMS = 2


def _grid_arrays(with_cross=True):
    kg = np.arange(N_PIX * N_Z_WL, dtype=np.float32).reshape(N_PIX, N_Z_WL)
    sn = (100 + np.arange(N_NOISE * N_PIX * N_Z_WL, dtype=np.float32)).reshape(N_NOISE, N_PIX, N_Z_WL)
    dg = (200 + np.arange(N_PIX * N_Z_GC, dtype=np.float32)).reshape(N_PIX, N_Z_GC)
    pn = (300 + np.arange(N_NOISE * N_PIX * N_Z_GC, dtype=np.float32)).reshape(N_NOISE, N_PIX, N_Z_GC)
    cls = (400 + np.arange(N_NOISE * N_CLS * N_Z_CROSS, dtype=np.float32)).reshape(N_NOISE, N_CLS, N_Z_CROSS)
    cosmo = np.array([0.125, 0.25], dtype=np.float32)
    xg = (500 + np.arange(N_PIX * N_Z_CROSS_MAP, dtype=np.float32)).reshape(N_PIX, N_Z_CROSS_MAP)
    xn = (600 + np.arange(N_NOISE * N_PIX * N_Z_CROSS_MAP, dtype=np.float32)).reshape(N_NOISE, N_PIX, N_Z_CROSS_MAP)
    if not with_cross:
        xg = None
        xn = None
    return kg, sn, dg, pn, cls, cosmo, xg, xn


def _write_one_grid_shard(path, *, with_cross=True):
    kg, sn, dg, pn, cls, cosmo, xg, xn = _grid_arrays(with_cross=with_cross)
    sample = webdatasets.encode_grid_sample(
        kg,
        sn,
        dg,
        pn,
        cls,
        cosmo,
        i_sobol=7,
        i_signal=11,
        xg=xg,
        xn_realz=xn,
    )
    sample["__key__"] = "synthetic-grid-000000"
    with wds.TarWriter(str(path), encoder=True) as sink:
        sink.write(sample)
    return sample, (kg, sn, dg, pn, cls, cosmo, xg, xn)


def _read_single_sample(path):
    return next(iter(wds.WebDataset([str(path)], shardshuffle=False)))


def _assert_tensor_matches(tensor, array):
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == tuple(array.shape)
    np.testing.assert_array_equal(tensor.numpy(), array)


def test_grid_webdataset_tar_roundtrips_arrays_dtypes_shapes_and_noise_indices(tmp_path):
    shard = tmp_path / "grid-000000.tar"
    _, (kg, sn, dg, pn, cls, cosmo, xg, xn) = _write_one_grid_shard(shard, with_cross=True)

    sample = _read_single_sample(shard)
    decoded = webdatasets.decode_grid_sample(
        sample,
        noise_indices=[2, 0],
        with_cross=True,
        return_maps=True,
        return_cls=True,
    )

    _assert_tensor_matches(decoded["cosmo"], cosmo)
    _assert_tensor_matches(decoded["kg_2"], kg + sn[2])
    _assert_tensor_matches(decoded["kg_0"], kg + sn[0])
    _assert_tensor_matches(decoded["dg_2"], dg + pn[2])
    _assert_tensor_matches(decoded["dg_0"], dg + pn[0])
    _assert_tensor_matches(decoded["xg_2"], xg + xn[2])
    _assert_tensor_matches(decoded["xg_0"], xg + xn[0])
    _assert_tensor_matches(decoded["cl_2"], cls[2])
    _assert_tensor_matches(decoded["cl_0"], cls[0])
    assert decoded["i_sobol"].dtype == torch.int64
    assert decoded["i_signal"].dtype == torch.int64
    assert decoded["i_sobol"].numpy() == 7
    assert decoded["i_signal"].numpy() == 11


def test_grid_webdataset_decode_honors_return_maps_and_return_cls(tmp_path):
    shard = tmp_path / "grid-000000.tar"
    _, (_, _, _, _, cls, cosmo, _, _) = _write_one_grid_shard(shard, with_cross=True)
    sample = _read_single_sample(shard)

    maps_only = webdatasets.decode_grid_sample(
        sample, noise_indices=[1], with_cross=True, return_maps=True, return_cls=False
    )
    assert "kg_1" in maps_only
    assert "dg_1" in maps_only
    assert "xg_1" in maps_only
    assert "cl_1" not in maps_only
    _assert_tensor_matches(maps_only["cosmo"], cosmo)

    cls_only = webdatasets.decode_grid_sample(
        sample, noise_indices=[1], with_cross=True, return_maps=False, return_cls=True
    )
    assert "kg_1" not in cls_only
    assert "dg_1" not in cls_only
    assert "xg_1" not in cls_only
    _assert_tensor_matches(cls_only["cl_1"], cls[1])


def test_grid_webdataset_optional_cross_maps_are_absent_when_disabled(tmp_path):
    shard = tmp_path / "grid-000000.tar"
    _, (kg, sn, dg, pn, cls, _, _, _) = _write_one_grid_shard(shard, with_cross=False)
    sample = _read_single_sample(shard)

    decoded = webdatasets.decode_grid_sample(
        sample, noise_indices=[1], with_cross=False, return_maps=True, return_cls=True
    )
    _assert_tensor_matches(decoded["kg_1"], kg + sn[1])
    _assert_tensor_matches(decoded["dg_1"], dg + pn[1])
    _assert_tensor_matches(decoded["cl_1"], cls[1])
    assert "xg_1" not in decoded


def _minimal_grid_pipeline(*, return_maps=True, return_cls=True, with_cross=False):
    tf = pytest.importorskip("tensorflow")
    from msfm.grid_pipeline import GridPipeline

    pipeline = GridPipeline.__new__(GridPipeline)
    pipeline.conf = {"analysis": {"grid": {"n_noise_per_signal": N_NOISE, "n_perms_per_cosmo": 1}, "n_patches": 1}}
    pipeline.params = ["Omega_m", "sigma8"]
    pipeline.all_params = ["Omega_m", "sigma8"]
    pipeline.n_all_params = N_PARAMS
    pipeline.n_noise_total = N_NOISE
    pipeline.n_signal_total = 1
    pipeline.n_dv_pix = N_PIX
    pipeline.n_z_WL = N_Z_WL
    pipeline.n_z_GC = N_Z_GC
    pipeline.n_z_cross = N_Z_CROSS_MAP if with_cross else N_Z_CROSS
    pipeline.n_cls = N_CLS
    pipeline.with_lensing = not with_cross
    pipeline.with_clustering = not with_cross
    pipeline.with_cross = with_cross
    pipeline.return_maps = return_maps
    pipeline.return_cls = return_cls
    pipeline.apply_norm = False
    pipeline.with_padding = True
    pipeline.z_bin_inds = None
    pipeline.masks_WL = torch.ones((N_PIX, N_Z_WL), dtype=torch.float32)
    pipeline.masks_GC = torch.ones((N_PIX, N_Z_GC), dtype=torch.float32)
    pipeline.normalize_lensing = lambda value: value
    pipeline.normalize_clustering = lambda value: value
    return pipeline


@pytest.mark.parametrize("return_maps,return_cls", [(True, True), (True, False), (False, True)])
def test_grid_pipeline_loader_returns_torch_tensors_for_maps_cls_and_indices(tmp_path, return_maps, return_cls):
    shard = tmp_path / "grid-000000.tar"
    _write_one_grid_shard(shard, with_cross=False)
    pipeline = _minimal_grid_pipeline(return_maps=return_maps, return_cls=return_cls, with_cross=False)

    batch = next(
        iter(
            pipeline.get_dset(
                pattern=str(shard),
                local_batch_size=1,
                noise_indices=[2],
                signal_indices=[0],
                n_workers=1,
                n_readers=1,
                n_prefetch=0,
                is_eval=True,
            )
        )
    )
    if return_maps:
        assert isinstance(map_tensor, torch.Tensor)
        assert tuple(map_tensor.shape) == (1, N_PIX, N_Z_WL + N_Z_GC)
    else:
        assert "maps" not in batch
    if return_cls:
        assert isinstance(cl_tensor, torch.Tensor)
        assert tuple(cl_tensor.shape) == (1, N_CLS, N_Z_CROSS)
    else:
        assert cl_tensor is None
    assert isinstance(cosmo_tensor, torch.Tensor)
    assert all(isinstance(value, torch.Tensor) for value in index)


def test_grid_pipeline_loader_returns_torch_tensor_for_optional_cross_maps(tmp_path):
    shard = tmp_path / "grid-000000.tar"
    _write_one_grid_shard(shard, with_cross=True)
    pipeline = _minimal_grid_pipeline(return_maps=True, return_cls=False, with_cross=True)

    batch = next(
        iter(
            pipeline.get_dset(
                pattern=str(shard),
                local_batch_size=1,
                noise_indices=[1],
                signal_indices=[0],
                n_workers=1,
                n_readers=1,
                n_prefetch=0,
                is_eval=True,
            )
        )
    )

    assert isinstance(map_tensor, torch.Tensor)
    assert tuple(map_tensor.shape) == (1, N_PIX, N_Z_CROSS_MAP)
    assert cl_tensor is None
    assert isinstance(cosmo_tensor, torch.Tensor)
    assert all(isinstance(value, torch.Tensor) for value in index)


def _fiducial_arrays():
    labels = ["fiducial", "Omega_m_p", "Aia_p", "bg_p"]
    cosmo_labels = labels[:2]
    ia_labels = labels[2:3]
    bg_labels = labels[3:]
    kg_perts = [
        (700 + offset + np.arange(N_PIX * N_Z_WL, dtype=np.float32)).reshape(N_PIX, N_Z_WL) for offset in (0, 20)
    ]
    dg_perts = [
        (800 + offset + np.arange(N_PIX * N_Z_GC, dtype=np.float32)).reshape(N_PIX, N_Z_GC) for offset in (0, 20)
    ]
    ia_perts = [(900 + np.arange(N_PIX * N_Z_WL, dtype=np.float32)).reshape(N_PIX, N_Z_WL)]
    bg_perts = [(1000 + np.arange(N_PIX * N_Z_GC, dtype=np.float32)).reshape(N_PIX, N_Z_GC)]
    sn = (1100 + np.arange(N_NOISE * N_PIX * N_Z_WL, dtype=np.float32)).reshape(N_NOISE, N_PIX, N_Z_WL)
    pn = (1200 + np.arange(N_NOISE * N_PIX * N_Z_GC, dtype=np.float32)).reshape(N_NOISE, N_PIX, N_Z_GC)
    cl_perts = (1300 + np.arange(2 * N_NOISE * N_CLS * N_Z_CROSS, dtype=np.float32)).reshape(
        2, N_NOISE, N_CLS, N_Z_CROSS
    )
    cl_ia = (1400 + np.arange(1 * N_NOISE * N_CLS * N_Z_CROSS, dtype=np.float32)).reshape(1, N_NOISE, N_CLS, N_Z_CROSS)
    cl_bg = (1500 + np.arange(1 * N_NOISE * N_CLS * N_Z_CROSS, dtype=np.float32)).reshape(1, N_NOISE, N_CLS, N_Z_CROSS)
    return (
        labels,
        cosmo_labels,
        kg_perts,
        dg_perts,
        ia_labels,
        ia_perts,
        sn,
        bg_labels,
        bg_perts,
        pn,
        cl_perts,
        cl_ia,
        cl_bg,
    )


def _write_one_fiducial_shard(path):
    arrays = _fiducial_arrays()
    (
        _,
        cosmo_labels,
        kg_perts,
        dg_perts,
        ia_labels,
        ia_perts,
        sn,
        bg_labels,
        bg_perts,
        pn,
        cl_perts,
        cl_ia,
        cl_bg,
    ) = arrays
    sample = webdatasets.encode_fiducial_sample(
        cosmo_labels,
        kg_perts,
        dg_perts,
        ia_labels,
        ia_perts,
        sn,
        bg_labels,
        bg_perts,
        pn,
        cl_perts,
        cl_ia,
        cl_bg,
        i_signal=13,
    )
    sample["__key__"] = "synthetic-fiducial-000000"
    with wds.TarWriter(str(path), encoder=True) as sink:
        sink.write(sample)
    return sample, arrays


def test_fiducial_webdataset_tar_roundtrips_arrays_dtypes_shapes_and_noise_indices(tmp_path):
    shard = tmp_path / "fiducial-000000.tar"
    _, arrays = _write_one_fiducial_shard(shard)
    (
        labels,
        _,
        kg_perts,
        dg_perts,
        ia_labels,
        ia_perts,
        sn,
        bg_labels,
        bg_perts,
        pn,
        cl_perts,
        cl_ia,
        cl_bg,
    ) = arrays

    sample = _read_single_sample(shard)
    decoded = webdatasets.decode_fiducial_sample(
        sample,
        pert_labels=labels,
        noise_indices=[2, 0],
        return_maps=True,
        return_cls=True,
    )

    _assert_tensor_matches(decoded["kg_fiducial"], kg_perts[0])
    _assert_tensor_matches(decoded["dg_fiducial"], dg_perts[0])
    _assert_tensor_matches(decoded["kg_Omega_m_p"], kg_perts[1])
    _assert_tensor_matches(decoded["dg_Omega_m_p"], dg_perts[1])
    _assert_tensor_matches(decoded[f"kg_{ia_labels[0]}"], ia_perts[0])
    assert f"dg_{ia_labels[0]}" not in decoded
    _assert_tensor_matches(decoded[f"dg_{bg_labels[0]}"], bg_perts[0])
    assert f"kg_{bg_labels[0]}" not in decoded
    _assert_tensor_matches(decoded["sn_2"], sn[2])
    _assert_tensor_matches(decoded["sn_0"], sn[0])
    _assert_tensor_matches(decoded["pn_2"], pn[2])
    _assert_tensor_matches(decoded["pn_0"], pn[0])
    _assert_tensor_matches(decoded["cl_fiducial"], cl_perts[0][[2, 0]])
    _assert_tensor_matches(decoded["cl_Omega_m_p"], cl_perts[1][[2, 0]])
    _assert_tensor_matches(decoded[f"cl_{ia_labels[0]}"], cl_ia[0][[2, 0]])
    _assert_tensor_matches(decoded[f"cl_{bg_labels[0]}"], cl_bg[0][[2, 0]])
    assert decoded["i_signal"].dtype == torch.int64
    assert decoded["i_signal"].numpy() == 13


def test_fiducial_webdataset_decode_honors_return_maps_and_return_cls(tmp_path):
    shard = tmp_path / "fiducial-000000.tar"
    _, arrays = _write_one_fiducial_shard(shard)
    labels = arrays[0]
    sample = _read_single_sample(shard)

    maps_only = webdatasets.decode_fiducial_sample(sample, labels, [1], return_maps=True, return_cls=False)
    assert "kg_fiducial" in maps_only
    assert "sn_1" in maps_only
    assert "cl_fiducial" not in maps_only

    cls_only = webdatasets.decode_fiducial_sample(sample, labels, [1], return_maps=False, return_cls=True)
    assert "kg_fiducial" not in cls_only
    assert "sn_1" not in cls_only
    assert "cl_fiducial" in cls_only


def test_grid_webdataset_decode_can_return_pytorch_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    shard = tmp_path / "grid-000000.tar"
    _, (kg, sn, dg, pn, cls, cosmo, xg, xn) = _write_one_grid_shard(shard, with_cross=True)
    sample = _read_single_sample(shard)

    decoded = webdatasets.decode_grid_sample(
        sample,
        noise_indices=[2, 0],
        with_cross=True,
        return_maps=True,
        return_cls=True,
        tensor_backend="torch",
    )

    expected = {
        "cosmo": cosmo,
        "kg_2": kg + sn[2],
        "kg_0": kg + sn[0],
        "dg_2": dg + pn[2],
        "dg_0": dg + pn[0],
        "xg_2": xg + xn[2],
        "xg_0": xg + xn[0],
        "cl_2": cls[2],
        "cl_0": cls[0],
    }
    for key, array in expected.items():
        assert isinstance(decoded[key], torch.Tensor)
        assert decoded[key].dtype == torch.float32
        torch.testing.assert_close(decoded[key], torch.from_numpy(array))
    assert decoded["i_sobol"].dtype == torch.int64
    assert decoded["i_signal"].dtype == torch.int64
    assert decoded["i_sobol"].item() == 7
    assert decoded["i_signal"].item() == 11


def test_grid_pipeline_masks_padding_and_z_bin_selection_exact_tensors():
    pipeline = _minimal_grid_pipeline(return_maps=True, return_cls=True, with_cross=False)
    pipeline.with_padding = False
    pipeline.mask_total = tf.constant([True, False, True, True, False])
    pipeline.z_bin_inds = [2, 0]
    pipeline.masks_WL = tf.constant(
        [[1.0, 0.0], [1.0, 1.0], [0.5, 1.0], [1.0, 1.0], [0.0, 1.0]], dtype=tf.float32
    )
    pipeline.masks_GC = tf.constant([[1.0], [0.0], [1.0], [0.5], [1.0]], dtype=tf.float32)

    kg = tf.constant(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        ],
        dtype=tf.float32,
    )
    dg = tf.constant([[[100.0], [200.0], [300.0], [400.0], [500.0]]], dtype=tf.float32)
    cl = tf.reshape(tf.range(N_CLS * N_Z_CROSS, dtype=tf.float32), (1, N_CLS, N_Z_CROSS))
    data_vectors = {
        "cosmo": tf.constant([[0.125, 0.25]], dtype=tf.float32),
        "kg": kg,
        "dg": dg,
        "cl": cl,
        "i_sobol": tf.constant([7], dtype=tf.int64),
        "i_signal": tf.constant([11], dtype=tf.int64),
        "i_noise": tf.constant([2], dtype=tf.int64),
    }

    map_tensor, cl_tensor, cosmo_tensor, index = pipeline._augmentations(dict(data_vectors))

    expected_full = tf.concat([kg * pipeline.masks_WL, dg * pipeline.masks_GC], axis=-1)
    expected = tf.gather(tf.boolean_mask(expected_full, pipeline.mask_total, axis=1), pipeline.z_bin_inds, axis=-1)
    tf.debugging.assert_equal(map_tensor, expected)
    tf.debugging.assert_equal(cl_tensor, cl)
    tf.debugging.assert_equal(cosmo_tensor, data_vectors["cosmo"])
    for got, want in zip(index, (data_vectors["i_sobol"], data_vectors["i_signal"], data_vectors["i_noise"])):
        tf.debugging.assert_equal(got, want)


def test_grid_downsampling_unsorted_segment_mean_parent_indices_hand_computed():
    dv = tf.constant(
        [
            [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]],
            [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0], [8.0, 80.0]],
        ],
        dtype=tf.float32,
    )
    parent_output_idx = tf.constant([0, 1, 0, 1], dtype=tf.int32)

    dv_t = tf.transpose(dv, perm=[1, 0, 2])
    downsampled = tf.transpose(tf.math.unsorted_segment_mean(dv_t, parent_output_idx, 2), perm=[1, 0, 2])

    expected = tf.constant(
        [
            [[3.0, 30.0], [5.0, 50.0]],
            [[4.0, 40.0], [6.0, 60.0]],
        ],
        dtype=tf.float32,
    )
    tf.debugging.assert_equal(downsampled, expected)
