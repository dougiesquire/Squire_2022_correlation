"Some helper function to tidy up notebooks"

import xarray as xr


def get_hindcast_rolling_mean(hcst, rolling_means):
    """
    Given annual hindcasts, return the hindcast over a specified averaging
    period. Averages are taken over the lead dimension, from index 0 to
    rolling_mean - 1. The time assigned to the output is the time at the
    last lead in the averaging period
    """

    res = [
        hcst.isel(lead=0, drop=True)
        .swap_dims({"init": "time"})
        .assign_coords({"rolling_mean": 1})
    ]
    for av in rolling_means:
        hcst_mean = hcst.isel(lead=slice(av))
        hcst_mean = hcst_mean.assign_coords(
            {"time": hcst_mean.time.isel(lead=-1, drop=True)}
        ).mean("lead")
        hcst_mean = hcst_mean.swap_dims({"init": "time"})
        res.append(hcst_mean.assign_coords({"rolling_mean": av}))

    return xr.concat(res, dim="rolling_mean")