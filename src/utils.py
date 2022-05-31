"Utility functions"

import numpy as np

import xarray as xr


def detrend(ds, dim="time", ensemble_dim="member"):
    """
    Remove the trend from a dataset.

    Parameters
    ----------
    ds : xarray object
        The data to detrend
    dim : str
        The dimension along which to detrend
    ensemble_dim : str
        The name of the ensemble dimension if it exists. If this
        dimension exists, the ensemble mean trend is removed
    """

    if ensemble_dim in ds.dims:
        fit = ds.copy().mean(ensemble_dim)
    else:
        fit = ds.copy()

    trend = fit.polyfit(dim=dim, deg=1)
    if isinstance(ds, xr.Dataset):
        for v in ds.data_vars:
            fit[v] = xr.polyval(ds[dim], trend[f"{v}_polyfit_coefficients"])
    else:
        fit = xr.polyval(ds[dim], trend["polyfit_coefficients"])

    return ds - fit


def extract_lon_lat_box(
    ds, box, weighted_average, lon_dim="lon", lat_dim="lat", area_name="area"
):
    """
    Return a region specified by a range of longitudes and latitudes.

    Parameters
    ----------
    ds : xarray Dataset or DataArray
        The data to subset and average. Assumed to include an "area" Variable
    box : iterable
        Iterable with the following elements in this order:
        [lon_lower, lon_upper, lat_lower, lat_upper]
        where longitudes are specified between 0 and 360 deg E and latitudes
        are specified between -90 and 90 deg N
    weighted_average : boolean
        If True, reture the area weighted average over the region, otherwise
        return the region
    lon_dim : str, optional
        The name of the longitude dimension
    lat_dim : str, optional
        The name of the latitude dimension
    area_name : str, optional
        The name of the area variable
    """

    # Force longitudues to range from 0-360
    ds = ds.assign_coords({lon_dim: (ds[lon_dim] + 360) % 360})

    if (lat_dim in ds.dims) and (lon_dim in ds.dims):
        # Can extract region using indexing
        average_dims = [lat_dim, lon_dim]

        # Allow for regions that cross 360 deg
        if box[0] > box[1]:
            lon_logic_func = np.logical_or
        else:
            lon_logic_func = np.logical_and
        lon_inds = np.where(
            lon_logic_func(
                ds[lon_dim].values >= box[0],
                ds[lon_dim].values <= box[1],
            )
        )[0]
        lat_inds = np.where(
            np.logical_and(
                ds[lat_dim].values >= box[2],
                ds[lat_dim].values <= box[3],
            )
        )[0]
        region = ds.isel({lon_dim: lon_inds, lat_dim: lat_inds})
    else:
        # Use `where` to extract region
        if (lat_dim in ds.dims) and (lon_dim not in ds.dims):
            average_dims = set([lat_dim, *ds[lon_dim].dims])
        elif (lat_dim not in ds.dims) and (lon_dim in ds.dims):
            average_dims = set([*ds[lat_dim].dims, lon_dim])
        else:
            average_dims = set([*ds[lat_dim].dims, *ds[lon_dim].dims])

        # Allow for regions that cross 360 deg
        if box[0] > box[1]:
            lon_region = (ds[lon_dim] >= box[0]) | (ds[lon_dim] <= box[1])
        else:
            lon_region = (ds[lon_dim] >= box[0]) & (ds[lon_dim] <= box[1])
        lat_region = (ds[lat_dim] >= box[2]) & (ds[lat_dim] <= box[3])
        region = ds.where(lon_region & lat_region, drop=True)

    if weighted_average:
        return region.weighted(ds[area_name].fillna(0)).mean(
            dim=average_dims, keep_attrs=True
        )
    else:
        return region


def extract_subpolar_gyre_region(
    ds,
    lon_dim="lon",
    lat_dim="lat",
    area_name="area",
):
    """
    Return the average over the subpolar gyre region

    Parameters
    ----------
    ds : xarray object
        The data to extract the region from
    lon_dim : str, optional
        The name of the longitude dimension
    lat_dim : str, optional
        The name of the latitude dimension
    area_name : str, optional
        The name of the area variable
    """
    return extract_lon_lat_box(
        ds,
        box=[310, 350, 45, 60],
        weighted_average=True,
        lon_dim=lon_dim,
        lat_dim=lat_dim,
        area_name=area_name,
    )


def calculate_NAO_index(ds):
    """
    Return the North Atlantic Oscillation index used by Smith et al. (2020)

    Parameters
    ----------
    ds : xarray Dataset or DataArray
        array containing sea level pressure data
    """
    Azores_box = extract_lon_lat_box(
        ds,
        box=[332, 340, 36, 40],
        weighted_average=True,
    )
    Iceland_box = extract_lon_lat_box(
        ds,
        box=[335, 344, 63, 70],
        weighted_average=True,
    )
    nao = Azores_box - Iceland_box
    return nao - nao.mean("time")


def calculate_AMV_index(ds):
    """
    Return the Atlantic multidecadal variability index used by Smith et al. (2020)

    Parameters
    ----------
    ds : xarray Dataset or DataArray
        array containing near-surface temperature or sea surface temperature
    """
    NA_box = extract_lon_lat_box(
        ds,
        box=[280, 360, 0, 60],
        weighted_average=True,
    )
    global_box = extract_lon_lat_box(
        ds,
        box=[0, 360, -60, 60],
        weighted_average=True,
    )
    amv = NA_box - global_box
    return amv - amv.mean("time")
