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


def calculate_NAO_index(ds, time_dim="time"):
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
    return nao - nao.mean(time_dim)


def _get_consecutive_month_avg(ds, months, time_dim):
    """
    Return the average over a set of consecutive months

    Parameters
    ----------
    ds : xarray Dataset or DataArray
        The data to average
    months : list of int
        The set of consecutive months to average. E.g. [12, 1, 2, 3] for
        Boreal Winter period used in Smith et al. (2020)
    """
    first_full_period_index = (
        (ds.time.dt.month.compute() == months[0]).argmax(time_dim).item(0)
    )
    period = (
        ds.isel({time_dim: slice(first_full_period_index, None)})
        .time.dt.month.isin(months)
        .compute()
    )
    return coarsen(ds.where(period, drop=True), len(months), time_dim)


def calculate_period_NAO_index(ds, period_months):
    """
    Given monthly data, return the average North Atlantic Oscillation index
    used by Smith et al. (2020) over a specified set of consecutive months

    Parameters
    ----------
    ds : xarray Dataset or DataArray
        Array containing sea level pressure data
    period_months : list of int
        The set of consecutive months to calculate the NAO index for.
        E.g. [12, 1, 2, 3] for Boreal Winter index used in Smith et al. (2020)
    """
    if "lead" in ds.dims:
        time_dim = "lead"
        mean_dim = "init"
    else:
        time_dim = mean_dim = "time"

    ds_period = _get_consecutive_month_avg(ds, period_months, time_dim)

    return calculate_NAO_index(ds_period, mean_dim)


def calculate_AMV_index(ds, time_dim="time"):
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
    return amv - amv.mean(time_dim)


def calculate_period_AMV_index(ds, period_months):
    """
    Given monthly data, return the average Atlantic multidecadal variability
    index used by Smith et al. (2020) over a specified set of consecutive
    months

    Parameters
    ----------
    ds : xarray Dataset or DataArray
        Array containing sea level pressure data
    period_months : list of int
        The set of consecutive months to calculate the NAO index for.
        E.g. [12, 1, 2, 3] for Boreal Winter index used in Smith et al. (2020)
    """
    if "lead" in ds.dims:
        time_dim = "lead"
        mean_dim = "init"
    else:
        time_dim = mean_dim = "time"

    ds_period = _get_consecutive_month_avg(ds, period_months, time_dim)

    return calculate_AMV_index(ds_period, mean_dim)


def round_to_start_of_month(ds, dim):
    """
    Return provided array with specified time dimension rounded to the start of
    the month

    Parameters
    ----------
    ds : xarray Dataset
        The dataset with a dimension(s) to round
    dim : str
        The name of the dimensions to round
    """
    from xarray.coding.cftime_offsets import MonthBegin

    if isinstance(dim, str):
        dim = [dim]
    for d in dim:
        ds = ds.copy().assign_coords({d: ds[d].compute().dt.floor("D") - MonthBegin()})
    return ds


def coarsen(ds, window_size, dim, start_points=None):
    """
    Coarsen data, applying 'max' to all relevant coords and optionally starting
    at a particular time point in the array

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to coarsen
    start_points : list
        Value(s) of coordinate `dim` to start the coarsening from. If these fall
        outside the range of the coordinate, coarsening starts at the beginning
        of the array
    dim : str, optional
        The name of the dimension to coarsen along
    """
    if start_points is None:
        start_points = [None]

    aux_coords = [c for c in ds.coords if dim in ds[c].dims]
    dss = []
    for start_point in start_points:
        dss.append(
            ds.sel({dim: slice(start_point, None)})
            .coarsen(
                {dim: window_size},
                boundary="trim",
                coord_func={d: "max" for d in aux_coords},
            )
            .mean()
        )
    return xr.concat(dss, dim=dim).sortby(dim)


def get_hindcast_temporal_mean(hcst, mean_lead_ranges=[(0, 1)]):
    """
    Given annual hindcasts, return the hindcast over a specified averaging
    period. Averages are taken over the lead dimension. The time assigned
    to the output is the time at the last lead in the averaging period

    hcst : xarray object
        The annual hindcast data
    mean_lead_range : list of tuple
        A list of lead year ranges to average over, specified as indexers. not
        coordinate values. Lead year ranges include the first element in the
        tuple, but not the last. E.g. (0,1) averages over lead year 0, (1,5)
        averages over years 1-4 etc.
    """

    res = []
    for r in mean_lead_ranges:
        hcst_mean = hcst.isel(lead=slice(*r))
        hcst_mean = hcst_mean.assign_coords(
            {"time": hcst_mean.time.isel(lead=-1, drop=True)}
        ).mean("lead")
        hcst_mean = hcst_mean.swap_dims({"init": "time"})
        res.append(hcst_mean.assign_coords({"temporal_mean": r[1] - r[0]}))
    res = xr.concat(res, dim="temporal_mean")
    if res.sizes["temporal_mean"] == 1:
        return res.squeeze("temporal_mean")
    else:
        return res


def get_observation_temporal_mean(ds, temporal_means):
    """
    Return the specified rolling means along the time dimension

    ds : xarray object
        The data to rolling average
    window : list of int
        A list of window lengths
    """
    res = []
    for av in temporal_means:
        rm = (
            ds.rolling({"time": av}, min_periods=av, center=False).mean().dropna("time")
        )
        rm = rm.assign_coords({"temporal_mean": av})
        res.append(rm)
    res = xr.concat(res, dim="temporal_mean")
    if res.sizes["temporal_mean"] == 1:
        return res.squeeze("temporal_mean")
    else:
        return res
