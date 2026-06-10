"""
Created on May 2023
Author: Arne Thomsen

Tools to handle the scale cuts/Gaussian smoothing.
"""

import numpy as np
import os

from msfm.utils import files, logger, imports

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


def rad_to_arcmin(theta):
    return theta / np.pi * (180 * 60)


def arcmin_to_rad(theta):
    return theta * np.pi / (60 * 180)


def ell_to_angle(ell, arcmin=False, method="naive"):
    if isinstance(ell, list):
        ell = np.array(ell)

    # method like 6.2 of https://academic.oup.com/mnras/article/505/4/5714/6296446
    if method == "naive":
        theta = np.pi / ell

    elif method == "physical":
        theta = 1 / ell

    if arcmin:
        theta = rad_to_arcmin(theta)

    return theta


def angle_to_ell(theta, arcmin=False, method="naive"):
    if isinstance(theta, list):
        theta = np.array(theta)

    # method like 6.2 of https://academic.oup.com/mnras/article/505/4/5714/6296446
    if arcmin:
        theta = arcmin_to_rad(theta)

    if method == "naive":
        ell = np.pi / theta

    elif method == "physical":
        ell = 1 / theta

    return ell


def gaussian_low_pass_factor_alm(
    l: np.ndarray,
    l_max: int = None,
    theta_fwhm: float = None,
    arcmin: bool = True,
    hard_cut: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """Remove small scales with a low pass filter (Gaussian smoothing)

    Args:
        l (np.ndarray): Array of ell values to evaluate the factor for.
        l_max (int, optional): Ell to set the smallest scale to consider. Defaults to None.
        theta_fwhm (float, optional): Alternatively, the smallest scale can also be defined in real space as the fwhm
            of a Gaussian. Defaults to None.
        arcmin (bool, optional): Regarding theta_fwhm. Defaults to True.

    Raises:
        ValueError: If both l_max and theta_fwhm are specified.

    Returns:
        np.ndarray: Array of coefficients in [0,1] to multiply with the alms. This has the shape of a half a Gaussian,
            going to zero for l -> + inf.
    """

    if l_max is None and (theta_fwhm is None or theta_fwhm == 0):
        return np.ones_like(l, dtype=dtype)
    elif l_max is not None and theta_fwhm is not None:
        raise ValueError("Either l_max or theta_fwhm must be specified, not both")

    if hard_cut:
        if l_max is None:
            l_max = angle_to_ell(theta_fwhm, arcmin, method="naive")

        low_pass_fac = np.zeros_like(l, dtype=dtype)
        low_pass_fac[l <= l_max] = 1
    else:
        if theta_fwhm is None:
            theta_fwhm = ell_to_angle(l_max, arcmin, method="naive")
        if arcmin:
            theta_fwhm = arcmin_to_rad(theta_fwhm)

        sigma = theta_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # https://github.com/healpy/healpy/blob/be5d47b0720d2de69d422f661d75cd3577327d5a/healpy/sphtfunc.py#L974C13-L975C1
        low_pass_fac = np.exp(-0.5 * l * (l + 1) * sigma**2)

    return low_pass_fac.astype(dtype)


def gaussian_high_pass_factor_alm(
    l: np.ndarray,
    l_min: int = None,
    theta_fwhm: float = None,
    arcmin: bool = True,
    hard_cut: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """Remove big scales with a high pass filter (1 - Gaussian smoothing)

    Args:
        l (np.ndarray): Array of ell values to evaluate the factor for.
        l_min (int, optional): Ell to set the smallest scale to consider. Defaults to None.
        theta_fwhm (float, optional): Alternatively, the smallest scale can also be defined in real space as the fwhm
            of a Gaussian. Defaults to None.
        arcmin (bool, optional): Regarding theta_fwhm. Defaults to True.

    Raises:
        ValueError: If both l_min and theta_fwhm are specified.

    Returns:
        np.ndarray: Array of coefficients in [0,1] to multiply with the alms. This has the shape of a asymmetric
            sigmoid (1 - Gaussian) and goes to one for l -> + inf.
    """

    if (l_min is None or l_min == 0) and theta_fwhm is None:
        return np.ones_like(l, dtype=dtype)
    elif l_min is not None and theta_fwhm is not None:
        raise ValueError("Either l_min or theta_fwhm must be specified, not both")

    if hard_cut:
        if l_min is None:
            l_min = angle_to_ell(theta_fwhm, arcmin, method="naive")

        high_pass_fac = np.zeros_like(l, dtype=dtype)
        high_pass_fac[l >= l_min] = 1
    else:
        high_pass_fac = 1 - gaussian_low_pass_factor_alm(l, l_min, theta_fwhm, arcmin)

    return high_pass_fac.astype(dtype)


def cls_to_smoothed_cls(
    cls: np.ndarray,
    l_min: int = None,
    l_max: int = None,
    theta_fwhm: float = None,
    arcmin: bool = True,
    hard_cut: bool = False,
) -> np.ndarray:
    """
    Apply high-pass and low-pass filters to the input power spectrum to obtain a smoothed power spectrum. This is
    written to be consistent with smoothing and smoothalm from healpy.
    https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.smoothalm.html
    Note that cls can't be tomographic.

    Args:
        cls (np.ndarray): Input power spectra of shape (n_examples, n_ell) or (n_ell,).
        l_min (int): Minimum multipole moment to remove large scales.
        l_max (int, optional): Maximum multipole moment to remove small scales. Defaults to None.
        theta_fwhm (float, optional): Full width at half maximum (FWHM) of the Gaussian smoothing kernel. Defaults to
            None, then l_max has to be provided.
        arcmin (bool, optional): If True, theta_fwhm is in arcminutes. If False, theta_fwhm is in radians. Defaults
            to True.

    Returns:
        ndarray: Smoothed power spectrum.
    """

    l = np.arange(cls.shape[-1])

    # extra square because we're smoothing Cls, not alms
    high_pass_fac = gaussian_high_pass_factor_alm(l, l_min, hard_cut=hard_cut) ** 2
    low_pass_fac = gaussian_low_pass_factor_alm(l, l_max, theta_fwhm, arcmin, hard_cut=hard_cut) ** 2

    if cls.ndim == 1:
        cls = cls * high_pass_fac * low_pass_fac
    elif cls.ndim == 2:
        cls = cls * high_pass_fac[np.newaxis, :] * low_pass_fac[np.newaxis, :]
    elif cls.ndim == 3:
        raise NotImplementedError("cls.ndim == 3 not implemented yet")

    return cls

def alm_to_smoothed_map(
    alm: np.ndarray,
    n_side: int,
    l_min: int = None,
    l_max: int = None,
    theta_fwhm: float = None,
    arcmin: bool = True,
    nest: bool = False,
    hard_cut: bool = False,
) -> np.ndarray:
    """Takes in alm coefficients and returns a map that has been smoothed according to l_min and l_max or theta_fwhm.

    Args:
        alm (np.array): Single spherical harmonics decomposition.
        n_side (int): Healpix nside of the output map.
        l_min (int, optional): Largest scale.
        l_max (int, optional): Smallest scale, specified as an ell.
        theta_fwhm (float, optional): Smallest scale, specified as an angle, which is used as the FWHM of a Gaussian.
        arcmin (bool, optional): Whether the smallest scale is specified as an angle in arcmin, otherwise it is in
            radian.
        nest (bool, optional): Whether the (full sky) output map should be returned in NEST ordering.

    Returns:
        np.array: Healpy map of shape (n_pix,)
    """

    # alm are computed for the standard l_max = 3 * n_side - 1
    l = hp.Alm.getlm(3 * n_side - 1)[0]

    # remove large scales (map - Gaussian smoothing)
    high_pass_fac = gaussian_high_pass_factor_alm(l, l_min, hard_cut=hard_cut)

    # remove small scales (Gaussian smoothing), this produces identical results as hp.smoothalm(fwhm=theta_fwhm)
    low_pass_fac = gaussian_low_pass_factor_alm(l, l_max, theta_fwhm, arcmin, hard_cut=hard_cut)

    alm = alm * high_pass_fac * low_pass_fac

    full_map = hp.alm2map(alm, nside=n_side, pol=False)

    if nest:
        full_map = hp.reorder(full_map, r2n=True)

    return full_map, alm

def map_to_smoothed_map(
    full_map: np.ndarray,
    n_side: int,
    l_min: int = None,
    l_max: int = None,
    theta_fwhm: float = None,
    arcmin: bool = True,
    nest: bool = False,
    hard_cut: bool = False,
    conf: dict = None,
) -> np.ndarray:
    """Takes in a (multiple) full sky healpy map(s) and returns a (multiple) map(s) that has (have) been smoothed
    according to l_min and l_max. The input can either be a single map, or a stack of multiple tomographic bins along
    the final axis.

    Args:
        full_map (np.array): Full sky healpy map(s) of the appropriate n_side and shape (n_pix,) or (n_pix, n_z_bins).
        n_side (int): Healpix nside of the output map.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        theta_fwhm (Union[float, list]): Smallest scale(s), specified as an angle, which is used as the FWHM of a
            Gaussian.
        arcmin (bool, optional): Whether the smallest scale is specified as an angle in arcmin, otherwise it is in
            radian.
        nest (bool, optional): Whether the (full sky) input map is in NEST ordering. The output map is returned in the
            same ordering as the input map.

    Returns:
        np.array: Smoothed (full sky) healpy map(s) of shape (n_pix,) or (n_pix, n_z_bins).
    """

    # healpy path
    conf = files.load_config(conf)
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # multiple tomographic bins along final axis
    if full_map.ndim == 2:
        n_z = full_map.shape[1]

        if l_min is None:
            l_min = [None] * n_z
        if l_max is None:
            l_max = [None] * n_z
        if theta_fwhm is None:
            theta_fwhm = [None] * n_z

        if isinstance(l_min, list) and isinstance(l_max, list):
            assert n_z == len(l_min) == len(l_max)
        elif isinstance(l_min, list) and isinstance(theta_fwhm, list):
            assert n_z == len(l_min) == len(theta_fwhm)
        else:
            raise ValueError(f"For tomographic inputs, l_min and l_max or theta_fwhm must be lists of length n_z_bins")

        alms = []
        for i_z in range(n_z):
            current_map = full_map[:, i_z]
            if nest:
                current_map = hp.reorder(current_map, n2r=True)

            alm = hp.map2alm(current_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)

            full_map[:, i_z], alm = alm_to_smoothed_map(
                alm,
                n_side,
                l_min[i_z],
                l_max[i_z],
                theta_fwhm[i_z],
                arcmin,
                nest,
                hard_cut=hard_cut,
            )

            alms.append(alm)

        alm = np.stack(alms, axis=1)

    # single map
    elif full_map.ndim == 1:
        if nest:
            full_map = hp.reorder(full_map, n2r=True)

        alm = hp.map2alm(full_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)

        full_map, alm = alm_to_smoothed_map(
            alm,
            n_side,
            l_min,
            l_max,
            theta_fwhm,
            arcmin,
            nest,
            hard_cut=hard_cut,
        )

    else:
        raise ValueError(f"Unknown full_map.ndim: {full_map.ndim}, must be 1 or 2")

    return full_map, alm

def data_vector_to_smoothed_data_vector(
    data_vector: np.ndarray,
    data_vec_pix: np.ndarray,
    n_side: int,
    l_min: int,
    l_max: int = None,
    theta_fwhm: float = None,
    arcmin: bool = True,
    mask: np.ndarray = None,
    hard_cut: bool = False,
    conf: dict = None,
):
    """Takes in a (multiple) padded data vector(s) and returns a (multiple) data vectors(s) that has (have) been
    smoothed according to l_min and l_max. The input can either be a single map, or a stack of multiple tomographic
    bins along axis = 1 = -1.

    Args:
        data_vector (np.array): Partial sky padded data vector(s) of shape (len(data_vec_pix),) or
            (len(data_vec_pix), n_z_bins).
        data_vec_pix (np.array): Indices of the (padded) data vector in NEST ordering.
        n_side (int): Healpix nside of the output map.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        theta_fwhm (Union[float, list]): Smallest scale(s), specified as an angle, which is used as the FWHM of a
            Gaussian.
        arcmin (bool, optional): Whether the smallest scale is specified as an angle in arcmin, otherwise it is in
            radian.

    Returns:
        np.array: Smoothed data vector(s) of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
    """

    n_pix = hp.nside2npix(n_side)

    if mask is not None:
        data_vector *= mask

    # multiple tomographic bins along final axis
    if data_vector.ndim == 2:
        n_z_bins = data_vector.shape[1]

        full_map = np.zeros((n_pix, n_z_bins), dtype=np.float32)
        full_map[data_vec_pix] = data_vector

    # single map
    elif data_vector.ndim == 1:
        full_map = np.zeros(n_pix, dtype=np.float32)
        full_map[data_vec_pix] = data_vector

    else:
        raise ValueError(f"Unknown data_vector.ndim: {data_vector.ndim}, must be 1 or 2")

    full_map, alm = map_to_smoothed_map(
        full_map, n_side, l_min, l_max, theta_fwhm, arcmin, nest=True, conf=conf, hard_cut=hard_cut
    )

    data_vector = full_map[data_vec_pix]
    if mask is not None:
        data_vector *= mask

    return data_vector, alm


# Gaussian Random Fields ##############################################################################################


def data_vector_to_grf_data_vector(
    np_seed: int,
    data_vector: np.ndarray,
    data_vec_pix: np.ndarray,
    n_side: int,
    l_min: int,
    l_max: int = None,
    theta_fwhm: float = None,
    arcmin: bool = True,
    mask: np.ndarray = None,
    hard_cut: bool = False,
):
    """Takes in a (multiple) padded data vector(s) and returns a (multiple) data vectors(s) that has (have) been
    smoothed according to l_min and l_max and transformed to a Gaussian Random Field. This destroys all non-Gaussian
    information and is meant for testing purposes only, to be compared with Cls. The input can either be a single map,
    or a stack of multiple tomographic bins along axis = 1 = -1.

    Args:
        np_seed (int): A numpy random seed used in the (intrinsically random) generation alm -> map. It's important
        data_vector (np.array): Partial sky padded data vector(s) of shape (len(data_vec_pix),) or
            (len(data_vec_pix), n_z_bins).
        data_vec_pix (np.array): Indices of the (padded) data vector in NEST ordering.
        n_side (int): Healpix nside of the output map.
            that this is the same for all tomographic bins and maps that are added later on the GRF level (like signal
            and noise maps, both for weak lensing and galaxy clustering).
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        theta_fwhm (Union[float, list]): Smallest scale(s), specified as an angle, which is used as the FWHM of a
            Gaussian.
        arcmin (bool, optional): Whether the smallest scale is specified as an angle in arcmin, otherwise it is in
            radian.

    Returns:
        np.array: Smoothed data vector(s) of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
    """

    n_pix = hp.nside2npix(n_side)

    if mask is not None:
        data_vector *= mask

    # alm are computed for the standard l_max = 3 * n_side - 1
    l = hp.Alm.getlm(3 * n_side - 1)[0]

    # healpy path
    conf = files.load_config()
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # multiple tomographic bins along final axis
    if data_vector.ndim == 2:
        n_z = data_vector.shape[1]

        if l_min is None:
            l_min = [None] * n_z
        if l_max is None:
            l_max = [None] * n_z
        if theta_fwhm is None:
            theta_fwhm = [None] * n_z

        if isinstance(l_min, list) and isinstance(l_max, list):
            assert n_z == len(l_min) == len(l_max)
        elif isinstance(l_min, list) and isinstance(theta_fwhm, list):
            assert n_z == len(l_min) == len(theta_fwhm)
        else:
            raise ValueError(f"For tomographic inputs, l_min and l_max or theta_fwhm must be lists of length n_z_bins")

        alms = []
        for i_z in range(n_z):
            full_map = np.zeros((n_pix), dtype=np.float32)
            full_map[data_vec_pix] = data_vector[:, i_z]
            full_map = hp.reorder(full_map, n2r=True)

            alm = hp.map2alm(full_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)

            # smoothing
            high_pass_fac = gaussian_high_pass_factor_alm(l, l_min[i_z], hard_cut=hard_cut)
            low_pass_fac = gaussian_low_pass_factor_alm(l, l_max[i_z], theta_fwhm[i_z], arcmin, hard_cut=hard_cut)
            alm = alm * high_pass_fac * low_pass_fac

            # make a Gaussian Random Field
            cl = hp.alm2cl(alm)
            np.random.seed(np_seed)
            grf = hp.synfast(cl, nside=n_side, pol=False).astype(np.float32)
            grf = hp.reorder(grf, r2n=True)

            # padding is populated too
            data_vector[:, i_z] = grf[data_vec_pix]

            alms.append(alm)

        alm = np.stack(alms, axis=1)

    # single map
    elif data_vector.ndim == 1:
        assert (isinstance(l_min, int) and isinstance(l_max, int)) or (
            isinstance(l_min, int) and isinstance(theta_fwhm, float)
        )

        full_map = np.zeros(n_pix, dtype=np.float32)
        full_map[data_vec_pix] = data_vector
        full_map = hp.reorder(full_map, n2r=True)

        alm = hp.map2alm(full_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)

        # smoothing
        high_pass_fac = gaussian_high_pass_factor_alm(l, l_min, hard_cut=hard_cut)
        low_pass_fac = gaussian_low_pass_factor_alm(l, l_max, theta_fwhm, arcmin, hard_cut=hard_cut)
        alm = alm * high_pass_fac * low_pass_fac

        # make a Gaussian Random Field
        cl = hp.alm2cl(alm)
        np.random.seed(np_seed)
        grf = hp.synfast(cl, nside=n_side, pol=False).astype(np.float32)
        grf = hp.reorder(grf, r2n=True)

        # padding is populated too
        data_vector = grf[data_vec_pix]

    else:
        raise ValueError(f"Unknown data_vector.ndim: {data_vector.ndim}, must be 1 or 2")

    if mask is not None:
        data_vector *= mask

    return data_vector, alm


def alm_to_grf_map(alm, n_side, l_min, l_max, np_seed):
    """NOTE this function has not been tested yet in conjunction with run_datavectors.py and is deprecated.

    Take in an alm vector and return a full sky Gaussian Random Field in ring ordering. This is for testing purposes
    only and enables a comparison of the networks to power spectra. Note that it's important that the different
    tomographic bins are all generated according to the same np.random.seed, otherwise the tomographic information
    gets lost too.

    Args:
        alm (np.ndarray): Vector of complex alm coefficients. Only a single tomographic bin at a time is supported
            by this function, unlike some of the above in this file.
        n_side (int): Healpix nside of the output map.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        np_seed (int): A numpy random seed used in the (intrinsically random) generation alm -> map.

    Returns:
        np.array: Smoothed full sky healpy map for a single tomographic bin of shape (n_pix,).
    """
    raise DeprecationWarning("This function has not been tested yet in conjunction with run_datavectors.py")

    cl = hp.alm2cl(alm)

    # remove large scales
    cl[np.arange(0, l_min)] = 0

    # remove small scales and make a Gaussian Random Field
    np.random.seed(np_seed)
    grf = hp.synfast(cl, nside=n_side, pol=False, fwhm=ell_to_angle(l_max)).astype(np.float32)

    return grf
