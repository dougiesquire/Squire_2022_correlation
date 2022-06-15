""" Functions for preparing data """

import glob

import dask

import numpy as np
import numpy.testing as npt

import xarray as xr

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data/raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data/processed"

def convert_time_to_lead(
    ds, time_dim="time", time_freq=None, init_dim="init", lead_dim="lead"
):
    """
    Return provided array with time dimension converted to lead time dimension
    and time added as additional coordinate

    Parameters
    ----------
    ds : xarray Dataset
        A dataset with a time dimension
    time_dim : str, optional
        The name of the time dimension
    time_freq : str, optional
        The frequency of the time dimension. If not provided, will try to use
        xr.infer_freq to determine the frequency. This is only used to add a
        freq attr to the lead time coordinate
    init_dim : str, optional
        The name of the initial date dimension in the output
    lead_dim : str, optional
        The name of the lead time dimension in the output
    """
    init_date = ds[time_dim][0].item()
    if time_freq is None:
        time_freq = xr.infer_freq(ds[time_dim])
    lead_time = range(len(ds[time_dim]))
    time_coord = (
        ds[time_dim]
        .rename({time_dim: lead_dim})
        .assign_coords({lead_dim: lead_time})
        .expand_dims({init_dim: [init_date]})
    ).compute()
    dataset = ds.rename({time_dim: lead_dim}).assign_coords(
        {lead_dim: lead_time, init_dim: [init_date]}
    )
    dataset = dataset.assign_coords({time_dim: time_coord})
    dataset[lead_dim].attrs["units"] = time_freq
    return dataset


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


def gridarea_cdo(ds):
    """
    Returns the area weights computed using cdo's gridarea function
    Note, this function writes ds to disk, so strip back ds to only what is needed

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to passed to cdo gridarea
    """
    import os
    import uuid
    from cdo import Cdo

    infile = uuid.uuid4().hex
    outfile = uuid.uuid4().hex
    ds.to_netcdf(f"./{infile}.nc")

    Cdo().gridarea(input=f"./{infile}.nc", output=f"./{outfile}.nc")

    weights = xr.open_dataset(f"./{outfile}.nc").load()
    os.remove(f"./{infile}.nc")
    os.remove(f"./{outfile}.nc")
    return weights["cell_area"]


def _cmip6_dcpp(
    model, experiment, variant_id, grid, variables, realm, years, members, version
):
    """Open CMIP6 dcpp variables from specified monthly realm"""

    def _dcpp_file(y, m, v):
        path = f"{RAW_DATA_DIR}/{model}_{experiment}/s{y}-r{m}{variant_id}/{realm}/{v}/{grid}"
        if version == "latest":
            versions = sorted(glob.glob(f"{path}/v????????"))
            if len(versions) == 0:
                raise ValueError(f"No versions found for {path}")
            else:
                path = versions[-1]
        else:
            path = f"{path}/{version}"

        file_pattern = (
            f"{v}_{realm}_{model}_{experiment}_s{y}-r{m}{variant_id}_{grid}_*.nc"
        )
        files = sorted(glob.glob(f"{path}/{file_pattern}"))
        if len(files) == 0:
            raise ValueError(f"No files found for {path}/{file_pattern}")
        else:
            return files

    def _open_dcpp(y, m, v):
        files = _dcpp_file(y, m, v)
        return xr.concat(
            [xr.open_dataset(f, chunks={}, use_cftime=True) for f in files],
            dim="time",
        )[v]

    def _open_dcpp_delayed(y, m, v, d0):
        var_data = dask.delayed(_open_dcpp)(y, m, v).data
        return dask.array.from_delayed(var_data, d0.shape, d0.dtype)

    # Get init and lead coodinates (assumed same for all members and variables)
    time = []
    for idx, y in enumerate(years):
        d0 = _open_dcpp(y, members[0], variables[0])
         # Ensure calendars are all the same
        if idx == 0:
            calendar = d0.time.dt.calendar
        time.append(d0.convert_calendar(calendar, use_cftime=True)["time"])
    init = [t[0].item() for t in time]
    d0 = convert_time_to_lead(d0.to_dataset(), time_freq="months")[variables[0]]
        
    ds = []
    for v in variables:
        delayed = []
        for y in years:
            delayed.append(
                dask.array.stack([_open_dcpp_delayed(y, m, v, d0) for m in members], axis=0)
            )
        delayed = dask.array.stack(delayed, axis=0)

        ds.append(
            xr.DataArray(
                delayed,
                dims=["init", "member", *d0.dims],
                coords={
                    "member": members,
                    "init": init,
                    **d0.coords,
                    "time": (["init", "lead"], time),
                },
                attrs=d0.attrs,
            ).to_dataset(name=v)
        )
    return xr.merge(ds).compute()


def extract_HadGEM3_variable(variable, realm, save_as):
    """
    Open HadGEM3-GC31-MM dcpp variable from specified monthly realm and save as a zarr collection
    
    Parameters
    ----------
    variable : str
        The name of the variable to extract
    realm : str
        The name of the realm containing the variable
    """
    model = "HadGEM3-GC31-MM"
    variant_id = "i1p1f2"
    grid = "gn"
    hindcast_years = range(1960, 2018 + 1)
    forecast_years = range(2019, 2020 + 1)
    members = range(1, 10 + 1)
    version = "v20200???"
    dcppA = _cmip6_dcpp(
        model, "dcppA-hindcast", variant_id, grid, [variable], realm, hindcast_years, members, version
    )
    dcppB = _cmip6_dcpp(
        model, "dcppB-forecast", variant_id, grid, [variable], realm, forecast_years, members, version
    )
    ds = xr.concat([dcppA, dcppB], dim="init")
    
    ### Add cell area
    if realm == "Omon":
        file = (
            f"{RAW_DATA_DIR}/{model}_piControl/r1i1p1f1/Ofx/areacello/{grid}/v20200108/"
            f"areacello_Ofx_{model}_piControl_r1i1p1f1_{grid}.nc"
        )
    elif realm == "Amon":
        file = (
            f"{RAW_DATA_DIR}/{model}_piControl/r1i1p1f1/fx/areacella/{grid}/v20200108/"
            f"areacella_fx_{model}_piControl_r1i1p1f1_{grid}.nc"
        )
    else:
        raise ValueError(f"I don't know where to find the area for realm: {realm}")
    area = xr.open_dataset(file, chunks={})
    ds = ds.assign_coords(area)

    chunks = {"init": -1, "member": -1, "lead": -1, "lat": 27, "lon": 36}
    ds.chunk(chunks).to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
    
    return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")


def extract_EC_Earth3_variable(variable, realm, save_as):
    """Open EC-Earth3 dcppA-hindcast variable from specified monthly realm"""
    def _fix_lat_lon(ds1, ds2):
        """
        Lat and lon values are not exactly the same to numerical precision 
        for different experiments
        """
        # Lat and lon values are not exactly the same to numerical precision for ds and area
        for c in ["lat", "lon"]:
            if c in ds2.coords:
                if np.array_equiv(ds2[c].values, ds1[c].values):
                    pass
                else:
                    npt.assert_allclose(ds2[c].values, ds1[c].values, rtol=1e-06)
                    ds1 = ds1.assign_coords({c: ds2[c]})
        return ds1
        
    model = "EC-Earth3"
    variant_id = "i1p1f1"
    grid = "gn" if realm == "Omon" else "gr"
    hindcast_years = range(1960, 2018 + 1)
    forecast_years = range(2019, 2020 + 1)
    members = range(1, 10 + 1)
    version = "v2020121?"
    ds = _cmip6_dcpp(
        model, "dcppA-hindcast", variant_id, grid, [variable], realm, hindcast_years, members, version
    )
    # dcppB-forecast are on a different longitude grid

    ### Add cell area
    if realm == "Omon":
        file = (
            f"{RAW_DATA_DIR}/{model}_historical/r1{variant_id}/Ofx/areacello/{grid}/v20200918/"
            f"areacello_Ofx_{model}_historical_r1{variant_id}_{grid}.nc"
        )
    elif realm == "Amon":
        file = (
            f"{RAW_DATA_DIR}/{model}_historical/r1{variant_id}/fx/areacella/{grid}/v20210324/"
            f"areacella_fx_{model}_historical_r1{variant_id}_{grid}.nc"
        )
    else:
        raise ValueError(f"I don't know where to find the area for realm: {realm}")
    area = xr.open_dataset(file, chunks={})
    area = _fix_lat_lon(area, ds)

    ds = ds.assign_coords(area)
    
    chunks = {"init": -1, "member": -1, "lead": -1, "lat": 32, "lon": 32}
    ds.chunk(chunks).to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")

    return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")


def prepare_HadSLP2r():
    ds = xr.open_dataset(f"{RAW_DATA_DIR}/slp.mnmean.real.nc", use_cftime=True)[["slp"]]
    ds = ds.assign_coords({"area": gridarea_cdo(ds)})
    
    return ds
#     chunks = {"time": -1, "lat": 32, "lon": 32}
#     ds.chunk(chunks).to_zarr(f"{PROCESSED_DATA_DIR}/HadSLP2r.zarr", mode="w")

#     return xr.open_zarr(f"{PROCESSED_DATA_DIR}/HadSLP2r.zarr")