"Functions for preparing data"

import os

import glob

import dask

import itertools

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


def interpolate_to_regular_grid(ds, resolution, add_area=True, ignore_degenerate=True):
    import xesmf

    """
    Interpolate to a regular grid with a specified resolution using xesmf

    Note, xESMF puts zeros where there is no data to interpolate. Here we
    add an offset to ensure no zeros, mask zeros, and then remove offset
    This hack will potentially do funny things for interpolation methods
    more complicated than bilinear.
    See https://github.com/JiaweiZhuang/xESMF/issues/15

    Parameters
    ----------
    ds : xarray Dataset
        The data to interpolate
    resolution : float
        The longitude and latitude resolution of the grid to interpolate to in degrees
    add_area : bool, optional
        If True (default) add a coordinate for the cell areas
    ignore_degenerate : bool, optional
        If True ESMF will ignore degenerate cells when carrying out
        the interpolation
    """
    lat_bounds = np.arange(-90, 91, resolution)
    lon_bounds = np.arange(0, 361, resolution)
    lats = (lat_bounds[:-1] + lat_bounds[1:]) / 2
    lons = (lon_bounds[:-1] + lon_bounds[1:]) / 2
    ds_out = xr.DataArray(
        data=np.ones((len(lats), len(lons))),
        coords={
            "lat": lats,
            "lon": lons,
        },
    ).to_dataset(name="dummy")
    ds_out["lat"].attrs = {
        "cartesian_axis": "Y",
        "long_name": "latitude",
        "units": "degrees_N",
    }
    ds_out["lon"].attrs = {
        "cartesian_axis": "X",
        "long_name": "longitude",
        "units": "degrees_E",
    }

    C = 1
    ds_rg = ds.copy() + C
    regridder = xesmf.Regridder(
        ds_rg,
        ds_out,
        "bilinear",
        ignore_degenerate=ignore_degenerate,
    )
    ds_rg = regridder(ds_rg, keep_attrs=True)
    ds_rg = ds_rg.where(ds_rg != 0.0) - C

    # Add back in attributes:
    for v in ds_rg.data_vars:
        ds_rg[v].attrs = ds[v].attrs

    if add_area:
        if "area" in ds_out:
            area = ds_out["area"]
        else:
            area = gridarea_cdo(ds_out)
        return ds_rg.assign_coords({"area": area})
    else:
        return ds_rg


def _open_cmip6(
    model, experiment, variant_id, grid, variables, realm, members, version, dcpp_start_years=None
):
    """
    Open CMIP6 dcpp variable(s) from specified monthly realm
    This works for experiment="historical", "piControl", "dcppA-hindcast" and "dcppB-forecast"
    I haven't tried other experiments
    """
    
    def _get_files(member, variable, dcpp_start_year=None):
        """ Returns a list of the available cmip6 files """
        if dcpp_start_year is None:
            sub_experiment = f"r{member}{variant_id}"
        else:
            sub_experiment = f"s{dcpp_start_year}-r{member}{variant_id}"
            
        path = (
            f"{RAW_DATA_DIR}/{model}_{experiment}/{sub_experiment}/{realm}/{variable}/{grid}"
        )
        
        if version == "latest":
            versions = sorted(glob.glob(f"{path}/v????????"))
            if len(versions) == 0:
                raise ValueError(f"No versions found for {path}")
            else:
                path = versions[-1]
        else:
            path = f"{path}/{version}"
            
        file_pattern = (
            f"{variable}_{realm}_{model}_{experiment}_{sub_experiment}_{grid}_*.nc"
        )
        files = sorted(glob.glob(f"{path}/{file_pattern}"))
        if len(files) == 0:
            raise ValueError(f"No files found for {path}/{file_pattern}")
        else:
            return files
        
    def _open(member, variable, dcpp_start_year=None):
        """ Open the available cmip6 files """
        files = _get_files(member, variable, dcpp_start_year)
        return xr.concat(
            [xr.open_dataset(f, chunks={}, use_cftime=True) for f in files],
            dim="time",
        )[variable]
    
    def _open_delayed(member, variable, data_template, dcpp_start_year=None):
        """ Dask delayed version of _open """
        data = dask.delayed(_open)(member, variable, dcpp_start_year).data
        return dask.array.from_delayed(data, data_template.shape, data_template.dtype)
    
    # Get a template for the data to be opened
    if "dcpp" in experiment:
        if dcpp_start_years is None:
            raise ValueError(f"dcpp_start_years must be provided with experiment={experiment}")
        years = dcpp_start_years
        time = []
        for idx, year in enumerate(years):
            template = _open(members[0], variables[0], year)
            # Force all calendars to be the same
            if idx == 0:
                calendar = template.time.dt.calendar
            time.append(template.convert_calendar(calendar, use_cftime=True)["time"])
        init = [t[0].item() for t in time]
        template = convert_time_to_lead(template.to_dataset(), time_freq="months")[variables[0]]
        
        dims=["init", "member", *template.dims]
        coords={
            "member": members,
            "init": init,
            **template.coords,
            "time": (["init", "lead"], time),
        }
    else:
        years = [None]
        template = _open(members[0], variables[0])
        
        dims=["member", *template.dims]
        coords={
            "member": members,
            **template.coords,
        }
        
    # Open all arrays
    delayed = []
    for variable, year in itertools.product(variables, years):
        delayed.append(
            dask.array.stack([_open_delayed(member, variable, template, year) for member in members], axis=0)
        )
        
    # Split delayed into variables and concatenate into xarray DataArray
    ds = []
    for n, variable in enumerate(variables):
        delayed_variable = delayed[n*len(years):(n+1)*len(years)]
        if "dcpp" in experiment:
            delayed_variable = dask.array.stack(delayed_variable, axis=0) # Stack along init dimension
        else:
            delayed_variable = delayed_variable[0]
        
        ds.append(
            xr.DataArray(
                delayed_variable,
                dims=dims,
                coords=coords,
                attrs=template.attrs,
            ).to_dataset(name=variable)
        )
        
    return xr.merge(ds).compute()


def _prepare_cmip(specs, area_file=None):
    """ Prepare some CMIP data from a list of spec dictionaries """
    
    def _fix_lat_lon(ds1, ds2):
        """
        Lat and lon values are not exactly the same to numerical precision
        for different experiments for some models (e.g. EC-Earth3)
        """
        # Lat and lon values are not exactly the same to numerical precision
        # for ds and area
        for c in ["lat", "lon"]:
            if c in ds2.coords:
                if np.array_equiv(ds2[c].values, ds1[c].values):
                    pass
                else:
                    npt.assert_allclose(ds2[c].values, ds1[c].values, rtol=1e-06)
                    ds1 = ds1.assign_coords({c: ds2[c]})
        return ds1
    
    ds = []
    for spec in specs:
        ds.append(_open_cmip6(**spec))
    ds = xr.combine_by_coords(ds, combine_attrs="drop_conflicts")
    
    ### Add cell area from hard-coded paths
    if area_file is not None:
        area = xr.open_dataset(area_file, chunks={})
        name = os.path.basename(area_file).split("_")[0]
        area = _fix_lat_lon(area.rename({name: "area"}), ds)
        ds = ds.assign_coords(area)

    # Interpolate to regular grid
    ds = interpolate_to_regular_grid(ds, resolution=1.0)
    
    if "init" in ds.dims:
        chunks = {"init": -1, "member": -1, "lead": -1, "lat": 20, "lon": 20}
    else:
        chunks = {"time": -1, "member": -1, "lat": 20, "lon": 20}
        
    return ds.chunk(chunks)


def prepare_HadGEM3_GC31_MM(experiments, realm, variables, save_as=None):
    """
    Open/save HadGEM3-GC31-MM from specified experiment(s) and monthly realm, 
    regrid to a 1deg x 1deg regular grid and save as a zarr collection

    Parameters
    ----------
    experiments : list of str
        The name(s) of the experiment(s) to open
    realm : str
        The name of the realm containing the variable(s)
    variables : list of str
        The name(s) of the variable(s) to extract
    save_as : str, optional
        Filename to use to save the prepared data. If None, return the lazy
        data
    """
    
    # Hard-coded paths to take area from
    if realm == "Omon":
        area_file = (
            f"{RAW_DATA_DIR}/HadGEM3-GC31-MM_piControl/r1i1p1f1/Ofx/areacello/gn/v20200108/"
            f"areacello_Ofx_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn.nc"
        )
    elif realm == "Amon":
        area_file = (
            f"{RAW_DATA_DIR}/HadGEM3-GC31-MM_piControl/r1i1p1f1/fx/areacella/gn/v20200108/"
            f"areacella_fx_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn.nc"
        )
    else:
        raise ValueError("I don't (yet) support this realm")
        
    # Generate specs for different experiments
    specs = []
    for exp in experiments:
        # Shared specs across experiments
        exp_spec = dict(
            model = "HadGEM3-GC31-MM",
            grid = "gn",
            version = "latest",
            realm = realm,
            variables = variables,
            experiment = exp,
        )
        if exp == "dcppA-hindcast":
            exp_spec["variant_id"] = "i1p1f2"
            exp_spec["dcpp_start_years"] = range(1960, 2018 + 1)
            exp_spec["members"] = range(1, 10 + 1)
        elif exp == "dcppB-forecast":
            exp_spec["variant_id"] = "i1p1f2"
            # Some of 2021 is available but some is missing (e.g s2021-r34i1p2f1/Omon)
            exp_spec["dcpp_start_years"] = range(2019, 2020 + 1)
            exp_spec["members"] = range(1, 10 + 1)
        elif exp == "historical":
            exp_spec["variant_id"] = "i1p1f3"
            exp_spec["dcpp_start_years"] = None
            exp_spec["members"] = range(1, 4 + 1)
        else:
            raise ValueError("Please set up specs dict for this experiement")
        specs.append(exp_spec)
    
    ds = _prepare_cmip(specs, area_file)
    
    if save_as is not None:
        ds.to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
        return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")
    else:
        return ds


def prepare_EC_Earth3(experiments, realm, variables, save_as=None):
    """
    Open/save EC-Earth3 variable(s) from specified experiment(s) and monthly realm, 
    regrid to a 1deg x 1deg regular grid and save as a zarr collection

    Parameters
    ----------
    experiments : list of str
        The name(s) of the experiment(s) to open
    realm : str
        The name of the realm containing the variable(s)
    variables : list of str
        The name(s) of the variable(s) to extract
    save_as : str, optional
        Filename to use to save the prepared data. If None, return the lazy
        data
    """
    
    # Hard-coded paths to take area from
    if realm == "Omon":
        grid = "gn"
        area_file = (
            f"{RAW_DATA_DIR}/EC-Earth3_historical/r1i1p1f1/"
            f"Ofx/areacello/gn/v20200918/"
            f"areacello_Ofx_EC-Earth3_historical_r1i1p1f1_gn.nc"
        )
    elif realm == "Amon":
        grid = "gr"
        area_file = (
            f"{RAW_DATA_DIR}/EC-Earth3_historical/r1i1p1f1/"
            f"fx/areacella/gr/v20210324/"
            f"areacella_fx_EC-Earth3_historical_r1i1p1f1_gr.nc"
        )
    else:
        raise ValueError("I don't (yet) support this realm")

    # Generate specs for different experiments
    specs = []
    for exp in experiments:
        # Shared specs across experiments
        exp_spec = dict(
            model = "EC-Earth3",
            variant_id = "i1p1f1",
            grid = grid,
            realm = realm,
            variables = variables,
            experiment = exp,
        )
        if exp == "dcppA-hindcast":
            exp_spec["dcpp_start_years"] = range(1960, 2018 + 1)
            exp_spec["version"] = "v2020121?"
            exp_spec["members"] = range(1, 10 + 1)
        elif exp == "historical":
            exp_spec["dcpp_start_years"] = None
            exp_spec["version"] = "latest"
            # Member 3 has screwy lats that can't be readily concatenated
            # Members 11, 13, 15 start in 1849
            # Member 19 has very few variables replicated for Omon
            # Members 101-150 only span 197001-201412
            exp_spec["members"] = [1, 2, 4, 6, 9, 10, 12, 14, 16, 17]
        else:
            raise ValueError("Please set up specs dict for this experiement")
        specs.append(exp_spec)
    
    ds = _prepare_cmip(specs, area_file)
    
    if save_as is not None:
        ds.to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
        return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")
    else:
        return ds


def prepare_CanESM5(experiments, realm, variables, save_as=None):
    """
    Open/save CanESM5 variable(s) from specified experiment(s) and monthly realm, 
    regrid to a 1deg x 1deg regular grid and save as a zarr collection

    Parameters
    ----------
    experiments : list of str
        The name(s) of the experiment(s) to open
    realm : str
        The name of the realm containing the variable(s)
    variables : list of str
        The name(s) of the variable(s) to extract
    save_as : str, optional
        Filename to use to save the prepared data. If None, return the lazy
        data
    """
    
    # Hard-coded paths to take area from
    if realm == "Omon":
        area_file = (
            f"{RAW_DATA_DIR}/CanESM5_historical/r1i1p2f1/Ofx/areacello/gn/"
            f"v20190429/areacello_Ofx_CanESM5_historical_r1i1p2f1_gn.nc"
        )
    elif realm == "Amon":
        area_file = (
            f"{RAW_DATA_DIR}/CanESM5_historical/r1i1p2f1/fx/areacella/gn/"
            f"v20190429/areacella_fx_CanESM5_historical_r1i1p2f1_gn.nc"
        )
    else:
        raise ValueError("I don't (yet) support this realm")
        
    # Generate specs for different experiments
    specs = []
    for exp in experiments:
        # Shared specs across experiments
        exp_spec = dict(
            model = "CanESM5",
            grid = "gn",
            version = "v20190429",
            variant_id = "i1p2f1",
            realm = realm,
            variables = variables,
            experiment = exp,
        )
        if exp == "dcppA-hindcast":
            exp_spec["dcpp_start_years"] = range(1960, 2019 + 1)
              # Only 20 psl members available for 1960 - 2016
            exp_spec["members"] = range(1, 20 + 1)
        elif exp == "dcppB-forecast":
            exp_spec["dcpp_start_years"] = range(2020, 2020 + 1)
              # Reduce members to match dcppA-hindcast
            exp_spec["members"] = range(1, 20 + 1)
        elif exp == "historical":
            exp_spec["dcpp_start_years"] = None
            exp_spec["members"] = range(1, 40 + 1)
        else:
            raise ValueError("Please set up specs dict for this experiement")
        specs.append(exp_spec)
    
    ds = _prepare_cmip(specs, area_file)
    
    if save_as is not None:
        ds.to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
        return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")
    else:
        return ds


def prepare_CESM1_1_CAM5_CMIP5(experiments, realm, variables, save_as=None):
    """
    Open/save CESM1_1_CAM5_CMIP5 variable(s) from specified experiment(s)
    and monthly realm, regrid to a 1deg x 1deg regular grid and save as a
    zarr collection

    Parameters
    ----------
    experiments : list of str
        The name(s) of the experiment(s) to open
    realm : str
        The name of the realm containing the variable(s)
    variables : list of str
        The name(s) of the variable(s) to extract
    save_as : str, optional
        Filename to use to save the prepared data. If None, return the lazy
        data
    """
    
    # Hard-coded paths to take area from
    if realm == "Omon":
        area_file = (
             f"{RAW_DATA_DIR}/CESM2_historical/r1i1p1f1/Ofx/areacello/gn/"
            f"v20190308/areacello_Ofx_CESM2_historical_r1i1p1f1_gn.nc"
        )
    elif realm == "Amon":
        area_file = (
            f"{RAW_DATA_DIR}/CESM2_historical/r1i1p1f1/fx/areacella/gn/"
            f"v20190308/areacella_fx_CESM2_historical_r1i1p1f1_gn.nc"
        )
    else:
        raise ValueError("I don't (yet) support this realm")

    # Generate specs for different experiments
    specs = []
    for exp in experiments:
        # Shared specs across experiments
        exp_spec = dict(
            model = "CESM1-1-CAM5-CMIP5",
            grid = "gn",
            variant_id = "i1p1f1",
            version = "v201910??",
            realm = realm,
            variables = variables,
            experiment = exp,
        )
        if exp == "dcppA-hindcast":
            exp_spec["dcpp_start_years"] = range(1960, 2017 + 1)
            exp_spec["members"] = range(1, 40 + 1)
        else:
            raise ValueError("Please set up specs dict for this experiement")
        specs.append(exp_spec)
    
    ds = _prepare_cmip(specs, area_file)
    
    if save_as is not None:
        ds.to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
        return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")
    else:
        return ds


def prepare_MIROC6(experiments, realm, variables, save_as=None):
    """
    Open/save MIROC6 variable(s) from specified experiment(s)
    and monthly realm, regrid to a 1deg x 1deg regular grid and save as a
    zarr collection

    Parameters
    ----------
    experiments : list of str
        The name(s) of the experiment(s) to open
    realm : str
        The name of the realm containing the variable(s)
    variables : list of str
        The name(s) of the variable(s) to extract
    save_as : str, optional
        Filename to use to save the prepared data. If None, return the lazy
        data
    """
    
    # Hard-coded paths to take area from
    if realm == "Omon":
        area_file = (
            f"{RAW_DATA_DIR}/MIROC6_historical/r1i1p1f1/Ofx/areacello/gn/"
            f"v20190311/areacello_Ofx_MIROC6_historical_r1i1p1f1_gn.nc"
        )
    elif realm == "Amon":
        area_file = (
            f"{RAW_DATA_DIR}/MIROC6_historical/r1i1p1f1/fx/areacella/gn/"
            f"v20190311/areacella_fx_MIROC6_historical_r1i1p1f1_gn.nc"
        )
    else:
        raise ValueError("I don't (yet) support this realm")
        

    # Generate specs for different experiments
    specs = []
    for exp in experiments:
        # Shared specs across experiments
        exp_spec = dict(
            model = "MIROC6",
            grid = "gn",
            version = "v20??????",
            variant_id = "i1p1f1",
            members = range(1, 10 + 1),
            realm = realm,
            variables = variables,
            experiment = exp,
        )
        if exp == "dcppA-hindcast":
            exp_spec["dcpp_start_years"] = range(1960, 2021 + 1)
        elif exp == "historical":
            exp_spec["dcpp_start_years"] = None
        else:
            raise ValueError("Please set up specs dict for this experiement")
        specs.append(exp_spec)
    
    ds = _prepare_cmip(specs, area_file)
    
    if save_as is not None:
        ds.to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
        return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")
    else:
        return ds


def prepare_MPI_ESM1_2_HR(experiments, realm, variables, save_as=None):
    """
    Open/save MPI-ESM1-2-HR variable(s) from specified experiment(s)
    and monthly realm, regrid to a 1deg x 1deg regular grid and save as a
    zarr collection

    Parameters
    ----------
    experiments : list of str
        The name(s) of the experiment(s) to open
    realm : str
        The name of the realm containing the variable(s)
    variables : list of str
        The name(s) of the variable(s) to extract
    save_as : str, optional
        Filename to use to save the prepared data. If None, return the lazy
        data
    """
    
    # Hard-coded paths to take area from
    if realm == "Omon":
        area_file = (
            f"{RAW_DATA_DIR}/MPI-ESM1-2-HR_historical/r1i1p1f1/Ofx/areacello/gn/"
            f"v20190710/areacello_Ofx_MPI-ESM1-2-HR_historical_r1i1p1f1_gn.nc"
        )
    elif realm == "Amon":
        area_file = (
            f"{RAW_DATA_DIR}/MPI-ESM1-2-HR_historical/r1i1p1f1/fx/areacella/gn/"
            f"v20190710/areacella_fx_MPI-ESM1-2-HR_historical_r1i1p1f1_gn.nc"
        )
    else:
        raise ValueError("I don't (yet) support this realm")
        
    # Generate specs for different experiments
    specs = []
    for exp in experiments:
        # Shared specs across experiments
        exp_spec = dict(
            model = "MPI-ESM1-2-HR",
            grid = "gn",
            version = "v20??????",
            variant_id = "i1p1f1",
            
            realm = realm,
            variables = variables,
            experiment = exp,
        )
        if exp == "dcppA-hindcast":
            exp_spec["dcpp_start_years"] = range(1960, 2018 + 1)
            exp_spec["members"] = range(1, 5 + 1)
        elif exp == "historical":
            exp_spec["dcpp_start_years"] = None
            exp_spec["members"] = range(1, 10 + 1)
        else:
            raise ValueError("Please set up specs dict for this experiement")
        specs.append(exp_spec)
    
    ds = _prepare_cmip(specs, area_file)
    
    if save_as is not None:
        ds.to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
        return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")
    else:
        return ds


def prepare_IPSL_CM6A_LR(experiments, realm, variables, save_as=None):
    """
    Open/save IPSL-CM6A-LR variable(s) from specified experiment(s)
    and monthly realm, regrid to a 1deg x 1deg regular grid and save as a
    zarr collection

    Parameters
    ----------
    experiments : list of str
        The name(s) of the experiment(s) to open
    realm : str
        The name of the realm containing the variable(s)
    variables : list of str
        The name(s) of the variable(s) to extract
    save_as : str, optional
        Filename to use to save the prepared data. If None, return the lazy
        data
    """
    
    # Hard-coded paths to take area (area already present for Omon)
    if realm == "Omon":
        grid = "gn"
        area_file = None
    elif realm == "Amon":
        grid = "gr"
        area_file = (
            f"{RAW_DATA_DIR}/IPSL-CM6A-LR_historical/r1i1p1f1/fx/areacella/{grid}/"
            f"v20180803/areacella_fx_IPSL-CM6A-LR_historical_r1i1p1f1_{grid}.nc"
        )
    else:
        raise ValueError("I don't (yet) support this realm")
        
    # Generate specs for different experiments
    specs = []
    for exp in experiments:
        # Shared specs across experiments
        exp_spec = dict(
            model = "IPSL-CM6A-LR",
            variant_id = "i1p1f1",
            members = range(1, 10 + 1),
            grid = grid,
            realm = realm,
            variables = variables,
            experiment = exp,
        )
        if exp == "dcppA-hindcast":
            exp_spec["dcpp_start_years"] = range(1960, 2016 + 1)
            exp_spec["version"] = "v20200108"
        elif exp == "historical":
            exp_spec["dcpp_start_years"] = None
            exp_spec["version"] = "v20180803"
        else:
            raise ValueError("Please set up specs dict for this experiement")
        specs.append(exp_spec)
    
    ds = _prepare_cmip(specs, area_file)
    
    if save_as is not None:
        ds.to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
        return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")
    else:
        return ds


def prepare_NorCPM1(experiments, realm, variables, save_as=None):
    """
    Open/save NorCPM1 from specified experiment(s) and monthly realm, 
    regrid to a 1deg x 1deg regular grid and save as a zarr collection

    Parameters
    ----------
    experiments : list of str
        The name(s) of the experiment(s) to open
    realm : str
        The name of the realm containing the variable(s)
    variables : list of str
        The name(s) of the variable(s) to extract
    save_as : str, optional
        Filename to use to save the prepared data. If None, return the lazy
        data
    """
    
    # Hard-coded paths to take area from
    if realm == "Omon":
        area_file = (
            f"{RAW_DATA_DIR}/NorCPM1_historical/r1i1p1f1/Ofx/areacello/gn/"
            f"v20200724/areacello_Ofx_NorCPM1_historical_r1i1p1f1_gn.nc"
        )
    elif realm == "Amon":
        area_file = (
            f"{RAW_DATA_DIR}/NorCPM1_historical/r1i1p1f1/fx/areacella/gn/"
            f"v20200724/areacella_fx_NorCPM1_historical_r1i1p1f1_gn.nc"
        )
    else:
        raise ValueError("I don't (yet) support this realm")
        
    # Generate specs for different experiments
    specs = []
    for exp in experiments:
        # Shared specs across experiments
        exp_spec = dict(
            model = "NorCPM1",
            grid = "gn",
            version = "latest",
            realm = realm,
            variables = variables,
            experiment = exp,
        )
        if exp == "dcppA-hindcast":
            # 10 members of i2p1f1 also exist
            exp_spec["variant_id"] = "i1p1f1"
            exp_spec["dcpp_start_years"] = range(1960, 2018 + 1)
            exp_spec["members"] = range(1, 10 + 1)
        elif exp == "historical":
            exp_spec["variant_id"] = "i1p1f1"
            exp_spec["dcpp_start_years"] = None
            exp_spec["members"] = range(1, 30 + 1)
        else:
            raise ValueError("Please set up specs dict for this experiement")
        specs.append(exp_spec)
    
    ds = _prepare_cmip(specs, area_file)
    
    if save_as is not None:
        ds.to_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr", mode="w")
        return xr.open_zarr(f"{PROCESSED_DATA_DIR}/{save_as}.zarr")
    else:
        return ds


def prepare_HadSLP2r():
    """
    Add a cell area coordinate to the HadSLP2r dataset and save as a zarr collection
    """
    ds = xr.open_dataset(f"{RAW_DATA_DIR}/slp.mnmean.real.nc", use_cftime=True)[["slp"]]
    ds = ds.assign_coords({"area": gridarea_cdo(ds)})

    chunks = {"time": -1, "lat": -1, "lon": -1}
    ds.chunk(chunks).to_zarr(f"{PROCESSED_DATA_DIR}/psl_HadSLP2r.zarr", mode="w")

    return xr.open_zarr(f"{PROCESSED_DATA_DIR}/psl_HadSLP2r.zarr")


def prepare_HadISST():
    """
    Add a cell area coordinate to the HadISST dataset and save as a zarr collection
    """
    ds = xr.open_dataset(f"{RAW_DATA_DIR}/HadISST_sst.nc", chunks={}, use_cftime=True)[
        ["sst"]
    ]
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.assign_coords({"area": gridarea_cdo(ds)})

    # Mask sea ice
    ds = ds.where(ds > -1000)

    chunks = {"time": -1, "lat": 90, "lon": 180}
    ds.chunk(chunks).to_zarr(f"{PROCESSED_DATA_DIR}/tos_HadISST.zarr", mode="w")

    return xr.open_zarr(f"{PROCESSED_DATA_DIR}/tos_HadISST.zarr")
