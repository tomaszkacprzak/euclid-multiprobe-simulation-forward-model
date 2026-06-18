import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from msfm.utils import lensing


def test_noise_gen_in_place_seeded_torch_invariants_and_zero_pixels():
    gamma_abs = np.array([0.2, 0.4], dtype=np.float32)
    weights = np.array([1.0, 1.0], dtype=np.float32)
    pix = np.array([0, 2], dtype=np.int32)
    base_patch_pix = np.array([0, 1, 2], dtype=np.int32)

    generator = torch.Generator().manual_seed(123)
    gamma1, gamma2 = lensing.noise_gen_in_place(
        gamma_abs, weights, pix, base_patch_pix, n_pix=3, n_noise_per_signal=2, rng=generator
    )

    assert gamma1.shape == (3, 2)
    assert gamma2.shape == (3, 2)
    np.testing.assert_allclose(gamma1[1], np.zeros(2, dtype=np.float32), atol=0.0)
    np.testing.assert_allclose(gamma2[1], np.zeros(2, dtype=np.float32), atol=0.0)
    np.testing.assert_allclose(np.hypot(gamma1[0], gamma2[0]), np.full(2, 0.2, dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(np.hypot(gamma1[2], gamma2[2]), np.full(2, 0.4, dtype=np.float32), rtol=1e-6)
