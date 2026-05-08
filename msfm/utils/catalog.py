import os, h5py
import numpy as np
import healpy as hp

from msfm.utils import logger, files

LOGGER = logger.get_logger(__file__)


def _get_rot_x(ang):
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(ang), -np.sin(ang)], [0.0, np.sin(ang), np.cos(ang)]]
    ).T  # Inverse because of healpy


def _get_rot_y(ang):
    return np.array(
        [[np.cos(ang), 0.0, np.sin(ang)], [0.0, 1.0, 0.0], [-np.sin(ang), 0.0, np.cos(ang)]]
    ).T  # Inverse because of healpy


def _get_rot_z(ang):
    return np.array(
        [[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]]
    ).T  # Inverse because of healpy


def survey_angles_to_pix(conf, ra, dec, n_side):
    """Rotate to the position in Fig. 4 of https://arxiv.org/pdf/2511.04681"""

    conf = files.load_config(conf)

    # healpy convention in radian
    theta = -np.deg2rad(dec) + np.pi / 2
    phi = np.deg2rad(ra)
    vec = hp.ang2vec(theta=theta, phi=phi)

    # rotate footprint to allow for cut-outs
    # https://github.com/des-science/multiprobe-simulation-forward-model/blob/main/notebooks/pixel_file_catalog_level.ipynb
    y_rot = _get_rot_y(conf["analysis"]["footprint"]["rotation"]["y_rad"])
    z_rot = _get_rot_z(conf["analysis"]["footprint"]["rotation"]["z_rad"])
    vec = np.dot(np.dot(z_rot, y_rot), vec.T)

    # per-object pixel index
    pix = hp.vec2pix(n_side, vec[0], vec[1], vec[2])

    return pix


def build_metacal_map_from_cat(conf, debug=True, force_recompute=False):
    conf = files.load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    gamma_cache_dir = f"{repo_dir}/data/metacal_wl_gamma_map.npy"
    count_cache_dir = f"{repo_dir}/data/metacal_wl_count_map.npy"

    # load from cache if available and not forcing recompute
    if not force_recompute:
        try:
            wl_gamma_map = np.load(gamma_cache_dir)
            wl_count_map = np.load(count_cache_dir)
            LOGGER.info("Loaded metacal maps from cache")
            return wl_gamma_map, wl_count_map
        except FileNotFoundError:
            LOGGER.info("Cache not found, computing metacal maps")

    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)

    n_z = len(conf["survey"]["metacal"]["z_bins"])
    Aeff = conf["survey"]["Aeff"]
    R_gamma = conf["survey"]["metacal"]["R_gamma"]
    R_s = conf["survey"]["metacal"]["R_s"]

    cat_dir = conf["dirs"]["catalog"]

    index = h5py.File(f"{cat_dir}/DESY3_indexcat.h5", "r")
    gold = h5py.File(f"{cat_dir}/DESY3_GOLD_2_2.1.h5", "r")
    metacal = h5py.File(f"{cat_dir}/DESY3_metacal_v03-004.h5", "r")
    dnf = h5py.File(f"{cat_dir}/DESY3_GOLD_2_2.1_DNF.h5", "r")

    wl_gamma_map = np.zeros((n_pix, n_z, 2))
    wl_count_map = np.zeros((n_pix, n_z), dtype=np.int32)
    for i in range(n_z):
        metacal_bin = index[f"/index/select_bin{i+1}"][:]

        # positions
        dec = gold["/catalog/gold/dec"][:][metacal_bin]
        ra = gold["/catalog/gold/ra"][:][metacal_bin]

        pix = survey_angles_to_pix(conf, ra, dec, n_side)
        count_map = np.bincount(pix, minlength=n_pix)

        # properties
        e1 = metacal["/catalog/unsheared/e_1"][:][metacal_bin]
        e2 = metacal["/catalog/unsheared/e_2"][:][metacal_bin]
        w = metacal["/catalog/unsheared/weight"][:][metacal_bin]

        # following eq. (10) in https://arxiv.org/pdf/2403.02314
        gamma1_map = np.bincount(pix, weights=e1 * w, minlength=n_pix)
        gamma2_map = np.bincount(pix, weights=e2 * w, minlength=n_pix)
        w_map = np.bincount(pix, weights=w, minlength=n_pix)

        mask = w_map > 0
        # using eq. (4) in https://arxiv.org/pdf/2105.13543 for the total shear response
        gamma1_map[mask] /= w_map[mask] * (R_gamma[i] + R_s[i])
        gamma2_map[mask] /= w_map[mask] * (R_gamma[i] + R_s[i])

        wl_gamma_map[:, i, 0] = gamma1_map
        wl_gamma_map[:, i, 1] = gamma2_map
        wl_count_map[:, i] = count_map

        if debug:
            if i == 0:
                LOGGER.warning("Compare with Table 1 in https://arxiv.org/pdf/2105.13543")
            LOGGER.info(f"Metacalibration bin {i+1}")

            def w_mean(x):
                return np.sum(w * x) / np.sum(w)

            # eq. (12) in https://arxiv.org/pdf/2011.03408
            neff_deg = np.sum(w) ** 2 / np.sum(w**2) / Aeff
            neff_arcmin = neff_deg / 60**2
            LOGGER.info(f"N_gal = {len(metacal_bin)}, n_eff = {neff_arcmin:.3f} [arcmin^-2]")

            LOGGER.info(f"mean(e1) = {w_mean(e1):.2e}, mean(e2) = {w_mean(e2):.2e}")

            # eq. (13) in https://arxiv.org/pdf/2011.03408
            # adapted from https://github.com/des-science/multiprobe-simulation-inference/blob/main/dev/notebooks/des_y3/marco_metacal_snippet.ipynb
            sigma_e_H12 = np.sqrt(
                0.5
                * (np.sum((e1 * w) ** 2) + np.sum((e2 * w) ** 2))
                # the R_gamma factor here seems to be missing from the paper?
                / np.sum(w * R_gamma[i]) ** 2
                * (np.sum(w) ** 2 / np.sum(w**2))
            )
            # eq. (10) in https://arxiv.org/pdf/2011.03408 (approximately, since sigma_m is missing)
            sigma_e_C13 = np.sqrt(0.5 * np.sum(w**2 * (e1**2 + e2**2)) / np.sum(w**2))
            LOGGER.info(f"sigma_e (H12) = {sigma_e_H12:.3f}, sigma_e (C13) = {sigma_e_C13:.3f}")

            # compare the mean shear response from Table 1 in https://arxiv.org/pdf/2105.13543 with the catalog
            R11 = metacal["/catalog/unsheared/R11"][:][metacal_bin]
            R22 = metacal["/catalog/unsheared/R22"][:][metacal_bin]
            R_gamma_cat = w_mean((R11 + R22) / 2)
            assert np.isclose(
                R_gamma[i], R_gamma_cat, atol=1e-5, rtol=1e-3
            ), f"R_gamma from config ({R_gamma[i]}) does not match mean(R_gamma) from catalog ({R_gamma_cat})"
            LOGGER.info(f"mean(R_gamma) = {R_gamma_cat:.4f}")

            z_dnf = dnf["/catalog/unsheared/zmc_sof"][:][metacal_bin]
            LOGGER.info(f"z_mean (DNF) = {w_mean(z_dnf):.4f}")

    index.close()
    gold.close()
    metacal.close()
    dnf.close()

    np.save(gamma_cache_dir, wl_gamma_map)
    np.save(count_cache_dir, wl_count_map)
    LOGGER.info(f"Saved metacal maps to {gamma_cache_dir} and {count_cache_dir}")

    return wl_gamma_map, wl_count_map


def build_maglim_map_from_cat(conf, debug=True, force_recompute=False):
    conf = files.load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    cache_dir = f"{repo_dir}/data/maglim_gc_count_map.npy"

    # load from cache if available and not forcing recompute
    if not force_recompute:
        try:
            gc_count_map = np.load(cache_dir)
            LOGGER.info("Loaded maglim map from cache")
            return gc_count_map
        except FileNotFoundError:
            LOGGER.info("Cache not found, computing maglim map")

    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)

    n_z = len(conf["survey"]["maglim"]["z_bins"])
    z_lims = conf["survey"]["maglim"]["z_lims"]
    Aeff = conf["survey"]["Aeff"]

    cat_dir = conf["dirs"]["catalog"]

    index = h5py.File(f"{cat_dir}/DESY3_indexcat.h5", "r")
    dnf = h5py.File(f"{cat_dir}/DESY3_GOLD_2_2.1_DNF.h5", "r")
    maglim = h5py.File(f"{cat_dir}/DESY3_maglim_redmagic_v0.5.1.h5", "r")

    maglim_index = index["index/maglim/select"][:]
    dec = maglim["catalog/maglim/dec"][:][maglim_index]
    ra = maglim["catalog/maglim/ra"][:][maglim_index]
    z = dnf["catalog/unsheared/zmean_sof"][:][maglim_index]

    gc_count_map = np.zeros((n_pix, n_z))
    for i in range(n_z):
        z_mask = (z_lims[i][0] < z) & (z < z_lims[i][1])

        # positions
        dec_bin = dec[z_mask]
        ra_bin = ra[z_mask]
        pix = survey_angles_to_pix(conf, ra_bin, dec_bin, n_side)

        gc_count_map[:, i] = np.bincount(pix, minlength=n_pix)

        if debug:
            if i == 0:
                LOGGER.warning("Compare with Table 1 in https://arxiv.org/pdf/2105.13546")

            LOGGER.info(f"Maglim bin {i+1}")

            N_gal = np.sum(z_mask)
            neff_deg = N_gal / Aeff
            neff_arcmin = neff_deg / 60**2
            LOGGER.info(f"N_gal = {N_gal}, n_eff = {neff_arcmin:.3f} [arcmin^-2]")

    index.close()
    dnf.close()
    maglim.close()

    # save to cache
    np.save(cache_dir, gc_count_map)
    LOGGER.info(f"Saved maglim map to {cache_dir}")

    return gc_count_map


def get_shapes_from_cat(conf):
    conf = files.load_config(conf)

    n_side = conf["analysis"]["n_side"]
    R_gamma = conf["survey"]["metacal"]["R_gamma"]
    R_s = conf["survey"]["metacal"]["R_s"]

    cat_dir = conf["dirs"]["catalog"]

    metacal = h5py.File(f"{cat_dir}/DESY3_metacal_v03-004.h5", "r")
    index = h5py.File(f"{cat_dir}/DESY3_indexcat.h5", "r")
    gold = h5py.File(f"{cat_dir}/DESY3_GOLD_2_2.1.h5", "r")

    n_z = len(conf["survey"]["metacal"]["z_bins"])
    gamma_1 = []
    gamma_2 = []
    weight = []
    pixels = []
    for i in range(n_z):
        metacal_bin = index[f"/index/select_bin{i+1}"][:]
        LOGGER.info(f"Metacalibration bin {i+1}: N_gal = {len(metacal_bin)}")

        # positions
        dec = gold["/catalog/gold/dec"][:][metacal_bin]
        ra = gold["/catalog/gold/ra"][:][metacal_bin]

        pix = survey_angles_to_pix(conf, ra, dec, n_side)

        # properties
        e1 = metacal["/catalog/unsheared/e_1"][:][metacal_bin]
        e2 = metacal["/catalog/unsheared/e_2"][:][metacal_bin]
        w = metacal["/catalog/unsheared/weight"][:][metacal_bin]

        # we include the shear response factor from eq. (4) in https://arxiv.org/pdf/2105.13543 here for simplicity
        # since this is a per-bin (not per-object) quantity
        gamma_1.append(e1 / (R_gamma[i] + R_s[i]))
        gamma_2.append(e2 / (R_gamma[i] + R_s[i]))
        weight.append(w)
        pixels.append(pix)

    metacal.close()
    index.close()
    gold.close()

    return gamma_1, gamma_2, weight, pixels
