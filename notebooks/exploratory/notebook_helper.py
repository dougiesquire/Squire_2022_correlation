"""Ad hoc helper functions to tidy up notebooks for sharing """

import xarray as xr

from src import utils


DATA_DIR = "../../data/processed/"

def load_dcpp_hindcasts():
    models = [
        "CanESM5",
        "CESM1-1-CAM5-CMIP5",
        "EC-Earth3",
        "HadGEM3-GC31-MM",
        "IPSL-CM6A-LR",
        "MIROC6",
        "MPI-ESM1-2-HR",
        "NorCPM1",
    ]

    n_init = 58
    winter_months = [12, 1, 2, 3]

    hindcast = []
    prev_member = 0
    for model in models:
        tos = xr.open_zarr(f"{DATA_DIR}/tos_Omon_{model}_dcpp.zarr", use_cftime=True)
        AMV = utils.calculate_period_AMV_index(tos["tos"], winter_months).to_dataset(
            name="AMV"
        )
        AMV = AMV.assign_coords({"init": range(1960, 1960 + AMV.sizes["init"])})

        psl = xr.open_zarr(f"{DATA_DIR}/psl_Amon_{model}_dcpp.zarr", use_cftime=True) / 100
        NAO = utils.calculate_period_NAO_index(psl["psl"], winter_months).to_dataset(
            name="NAO"
        )
        NAO = NAO.assign_coords({"init": range(1960, 1960 + NAO.sizes["init"])})

        ds = xr.merge((AMV.compute(), NAO.compute()))
        ds = ds.sel(lead=slice(14, 120)).assign_coords({"lead": range(1, 10)})
        ds = ds.assign_coords({"member": ds.member + prev_member})
        ds = ds.assign_coords({"model": ("member", ds.sizes["member"] * [model])})
        ds = utils.round_to_start_of_month(ds, "time")

        prev_member = ds.member.values[-1]

        hindcast.append(ds)

    return xr.concat(
        hindcast, dim="member", coords="minimal", compat="override", join="inner"
    )


def load_historical():
    
    models = ["HadGEM3-GC31-MM"]
    
    winter_months = [12, 1, 2, 3]

    historical = []
    prev_member = 0
    for model in models:
        tos = xr.open_zarr(f"{DATA_DIR}/tos_Omon_{model}_historical.zarr", use_cftime=True)
        AMV = utils.calculate_period_AMV_index(tos["tos"], winter_months).to_dataset(
            name="AMV"
        )

        psl = (
            xr.open_zarr(f"{DATA_DIR}/psl_Amon_{model}_historical.zarr", use_cftime=True)
            / 100
        )
        NAO = utils.calculate_period_NAO_index(psl["psl"], winter_months).to_dataset(
            name="NAO"
        )

        ds = xr.merge((AMV.compute(), NAO.compute()))
        ds = ds.assign_coords({"member": ds.member + prev_member})
        ds = ds.assign_coords({"model": ("member", ds.sizes["member"] * [model])})

        prev_member = ds.member.values[-1]

        historical.append(ds)

    return xr.concat(historical, dim="member")


def load_reanalysis():
    winter_months = [12, 1, 2, 3]

    tos = xr.open_zarr(f"{DATA_DIR}/tos_HadISST.zarr", use_cftime=True)["sst"]
    AMV = utils.calculate_period_AMV_index(tos, winter_months).to_dataset(name="AMV")
    AMV = utils.round_to_start_of_month(AMV, dim="time")

    psl = xr.open_zarr(f"{DATA_DIR}/psl_HadSLP2r.zarr", use_cftime=True)["slp"]
    NAO = utils.calculate_period_NAO_index(psl, winter_months).to_dataset(name="NAO")

    return xr.merge((AMV.compute(), NAO.compute()), join="inner")