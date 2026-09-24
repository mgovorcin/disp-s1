from datetime import timedelta

import isce3
import numpy as np
from dolphin import baseline
from dolphin._types import Filename
from numpy.typing import ArrayLike
from opera_utils import (
    CslcParseError,
    get_cslc_orbit,
    get_radar_wavelength,
)
from opera_utils._cslc import get_s1_orbit
from pyproj import CRS, Transformer


def _get_grids(x: ArrayLike, y: ArrayLike, epsg: int) -> tuple:
    X, Y = np.meshgrid(x, y)
    xx = X.flatten()
    yy = Y.flatten()
    crs = CRS.from_epsg(epsg)
    utm_to_lonlat = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = utm_to_lonlat.transform(xx=xx, yy=yy, radians=False)
    lon = lon.reshape(X.shape)
    lat = lat.reshape(Y.shape)
    return lon, lat


def _get_orbit(h5file: Filename) -> isce3.core.Orbit:
    """Read a CSLC's orbit, tolerating filenames `opera_utils` cannot parse.

    `get_cslc_orbit` parses the filename for one purpose: to choose between the
    S1 and NISAR orbit readers. DISP-S1 is S1-only, so that dispatch can only
    ever pick S1 here -- but it raises `CslcParseError` first on any name
    outside the OPERA patterns.

    That bites on disp-s1's *own output*. A compressed SLC written from
    COMPASS-named inputs is called
    ``compressed_<burst_id>_<ref>_<start>_<end>.h5``, which matches neither
    `CSLC_S1_FILE_REGEX` nor `COMPRESSED_CSLC_S1_FILE_REGEX` (both want the
    full ``OPERA_L2_...`` form). The file itself is fine -- it carries a
    complete ``metadata/orbit`` group in the standard S1 layout -- so the only
    thing standing between it and a baseline is the name. Ministack 1 of a
    chain takes ministack 0's compressed SLC as its reference, so this fails
    every run after the first.

    Fall back to the S1 reader, which is the only one this PGE could have used.
    """
    try:
        return get_cslc_orbit(h5file)
    except CslcParseError:
        times, positions, velocities, reference_epoch = get_s1_orbit(h5file)
        state_vectors = [
            isce3.core.StateVector(
                isce3.core.DateTime(reference_epoch + timedelta(seconds=t)), x, v
            )
            for t, x, v in zip(times, positions, velocities)
        ]
        return isce3.core.Orbit(state_vectors)


def compute_baselines(
    h5file_ref: Filename,
    h5file_sec: Filename,
    x: ArrayLike,
    y: ArrayLike,
    epsg: int,
    height: float = 0.0,
    threshold: float = 1e-08,
    maxiter: int = 50,
    delta_range: float = 10.0,
):
    """Compute the perpendicular baseline at a subsampled grid for two CSLCs.

    Parameters.
    ----------
    h5file_ref : Filename
        Path to reference OPERA S1 CSLC HDF5 file.
    h5file_sec : Filename
        Path to secondary OPERA S1 CSLC HDF5 file.
    height: float
        Target height to use for baseline computation.
        Default = 0.0
    latlon_subsample: int
        Factor by which to subsample the CSLC latitude/longitude grids.
        Default = 30
    threshold : float
        isce3 geo2rdr: azimuth time convergence threshold in meters
        Default = 1e-8
    maxiter : int
        isce3 geo2rdr: Maximum number of Newton-Raphson iterations
        Default = 50
    delta_range : float
        isce3 geo2rdr: Step size used for computing derivative of doppler
        Default = 10.0

    Returns
    -------
    baselines : np.ndarray
        2D array of perpendicular baselines

    """
    lon_grid, lat_grid = _get_grids(x=x, y=y, epsg=epsg)
    lon_arr = lon_grid.ravel()
    lat_arr = lat_grid.ravel()

    ellipsoid = isce3.core.Ellipsoid()
    zero_doppler = isce3.core.LUT2d()
    wavelength = get_radar_wavelength(h5file_ref)
    side = isce3.core.LookSide.Right

    orbit_ref = _get_orbit(h5file_ref)
    orbit_sec = _get_orbit(h5file_sec)

    baselines = []
    for lon, lat in zip(lon_arr, lat_arr):
        llh_rad = np.deg2rad([lon, lat, height]).reshape((3, 1))
        az_time_ref, range_ref = isce3.geometry.geo2rdr(
            llh_rad,
            ellipsoid,
            orbit_ref,
            zero_doppler,
            wavelength,
            side,
            threshold=threshold,
            maxiter=maxiter,
            delta_range=delta_range,
        )
        az_time_sec, range_sec = isce3.geometry.geo2rdr(
            llh_rad,
            ellipsoid,
            orbit_sec,
            zero_doppler,
            wavelength,
            side,
            threshold=threshold,
            maxiter=maxiter,
            delta_range=delta_range,
        )

        pos_ref, velocity = orbit_ref.interpolate(az_time_ref)
        pos_sec, _ = orbit_sec.interpolate(az_time_sec)

        # Accounting for along-track shift
        rvelunit = velocity / np.linalg.norm(velocity)
        pos_sec = pos_sec - np.dot(pos_sec - pos_ref, rvelunit) * rvelunit

        b = baseline.compute(
            llh_rad, pos_ref, pos_sec, range_ref, range_sec, velocity, ellipsoid
        )

        baselines.append(b)

    return np.array(baselines).reshape(lon_grid.shape)


def _interpolate_data(
    data: np.ndarray, shape: tuple[int, int], method="linear"
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    # Create coordinate arrays for the original data
    orig_coords = [np.linspace(0, 1, s) for s in data.shape]

    # Create coordinate arrays for the desired output shape
    new_coords = [np.linspace(0, 1, s) for s in shape]

    # Create the interpolator
    interp = RegularGridInterpolator(orig_coords, data, method=method)

    # Create a mesh grid for the new coordinates
    mesh = np.meshgrid(*new_coords, indexing="xy")

    # Perform the interpolation
    return interp(np.array(mesh).T.astype("float32"))
