"Tools for working with Autoregressive models"


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import xarray as xr

from itertools import cycle

import matplotlib.pyplot as plt

from src import stats


def yule_walker(ds, order, dim="time", kwargs={}):
    """
    Vectorized xarray wrapper on
    statsmodels.regression.linear_model.yule_walker

    Parameters
    ----------
    ds : xarray object
        The data to use to estimate the AR(n) parameters
    order : int
        The order of the AR(n) process
    dim : str
        The dimension along which to apply the Yule-Walker equations
    kwargs: dict, optional
        Additional kwargs to pass to
        statsmodels.regression.linear_model.yule_walker
    """
    from statsmodels.regression.linear_model import yule_walker

    def _yule_walker(data, order, kwargs):
        rho, sigma = yule_walker(data, order=order, demean=True, **kwargs)
        return np.array((sigma, *rho))

    return xr.apply_ufunc(
        _yule_walker,
        ds,
        kwargs=dict(order=order, kwargs=kwargs),
        input_core_dims=[[dim]],
        output_core_dims=[["coeff"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs=dict(output_sizes={"coeff": order + 1}),
    ).assign_coords({"coeff": range(order + 1)})


def fit(
    ds,
    order,
    dim="time",
    ar_kwargs=dict(trend="n"),
    select_order_kwargs=dict(maxlag=10, ic="bic", glob=False, trend="n"),
):
    """
    Fit an AR model(s)

    Parameters
    ----------
    ds : xarray object
        The data to fit the AR model to
    order : int or str
        The order of the AR(n) process to fit. Alternatively, users can pass
        the string "select_order" in order to use
        statsmodels.tsa.ar_model.ar_select_order to determine the order.
    dim : str
        The dimension along which to fit the AR model(s)
    ar_kwargs: dict, optional
        kwargs to pass to statsmodels.tsa.ar_model.AutoReg. Only used if order
        is an integer
    select_order_kwargs: dict, optional
        kwargs to pass to statsmodels.tsa.ar_model.ar_select_order. Only used
        if order="select_order".
    """
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order

    def _ar_select_order(data, maxlag, kwargs):
        res = ar_select_order(data, maxlag, **kwargs).model.fit()
        params = np.empty(maxlag + 1)
        params[:] = np.nan
        params[: len(res.params)] = res.params
        params[-1] = np.sqrt(res.sigma2)
        return params

    def _ar(data, order, kwargs):
        res = AutoReg(data, lags=order, **kwargs).fit()
        return np.concatenate((res.params, [np.sqrt(res.sigma2)]))

    if order == "select_order":
        assert "maxlag" in select_order_kwargs, (
            "Must provide maxlag parameter to select_order_kwargs when using "
            "order='select_order'"
        )

        func = _ar_select_order
        kwargs = select_order_kwargs.copy()
        maxlag = kwargs.pop("maxlag")
        kwargs = dict(maxlag=maxlag, kwargs=kwargs)
        n_params = maxlag + 1
    else:
        func = _ar
        kwargs = dict(order=order, kwargs=ar_kwargs)
        n_params = order + 1

    res = xr.apply_ufunc(
        func,
        ds,
        kwargs=kwargs,
        input_core_dims=[[dim]],
        output_core_dims=[["params"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs=dict(output_sizes={"params": n_params}),
    )
    res = res.assign_coords(
        {"params": [f"phi_{lag}" for lag in range(1, n_params)] + ["sigma_e"]}
    ).dropna("params")
    res = res.assign_coords({"order": res.count("params").values - 1})
    return res


def generate_samples(
    params, scale, n_times, n_samples, n_members=None, rolling_means=None
):
    """
    Generate random samples from an AR process

    Parameters
    ----------
    params : numpy array
        The AR parameters
    scale : float
        The standard deviation of noise.
    n_times : int
        The number of timesteps per sample
    n_samples : int
        The number of samples to generate
    n_members : int, optional
        The number of ensemble members to generate. N ensemble members are
        generated from N predictions initialised from samples of the provided
        process. When provided with rolling_mean, rolling means of length L
        are computed by averaging prediction times 1->L.
    rolling_means : list, optional
        A list of lengths of rolling means to compute
    """
    from statsmodels.tsa.arima_process import ArmaProcess

    if n_members is not None:
        extend = len(params) - 1
    elif rolling_means is None:
        extend = 0
    else:
        extend = max(rolling_means) - 1

    # Generate some AR series
    process = ArmaProcess(np.concatenate(([1], -params)))
    s = process.generate_sample(
        nsample=(n_times + extend, n_samples), scale=scale, axis=0
    )

    if n_members is None:
        s = xr.DataArray(
            s,
            coords={
                "time": range(1, n_times + extend + 1),
                "sample": range(n_samples),
            },
        )
    else:
        n_leads = 1 if rolling_means is None else max(rolling_means)
        s = predict(
            process.arcoefs,
            s,
            n_leads,
            n_members=n_members,
            scale=scale,
        )
        s = s.rename({"init": "time"})

    if rolling_means is not None:
        if n_members is not None:
            res = [s.sel(lead=1, drop=True).assign_coords({"rolling_mean": 1})]
            for av in rolling_means:
                rm = s.sel(lead=slice(1, av)).mean("lead")
                rm = rm.assign_coords({"rolling_mean": av})
                res.append(rm)
        else:
            res = [s.assign_coords({"rolling_mean": 1})]
            for av in rolling_means:
                rm = (
                    s.rolling({"time": av}, min_periods=av, center=False)
                    .mean()
                    .dropna("time")
                )
                rm = rm.assign_coords({"rolling_mean": av})
                res.append(rm)

        s = xr.concat(res, dim="rolling_mean", join="inner")
        s = s.assign_coords({"time": range(1, s.sizes["time"] + 1)})

    return s.squeeze(drop=True)


def predict(params, inits, n_steps, n_members=1, scale=None):
    """
    Advance an Autoregressive model forward in time from initial conditions
    by n_steps

    Parameters
    ----------
    params : numpy array
        The AR(n) model coefficients of the form [param_lag_1, param_lag_2,
        ... param_lag_n]
    inits : numpy array
        Array containing the initial conditions. Can be 1D or 2D. If the
        latter, the second axis should contain different samples of initial
        conditions.
    n_steps : int
        The number of timesteps to step forward from each initial condition
    scale : float
        The standard deviation of the noise term in the AR(n) model. If None,
        no noise term is included in the predictive model
    """

    def _epsilon(scale, size):
        return np.random.normal(scale=scale, size=size)

    order = len(params)
    params = np.flip(params)

    # Some quick checks
    assert len(inits) >= len(params), (
        f"At least {order} initial conditions must be provided for an "
        f"AR({order}) model"
    )

    if inits.ndim == 1:
        inits = np.expand_dims(inits, axis=-1)

    inits_stacked = sliding_window_view(inits, window_shape=order, axis=0)

    # res = [member, init, sample, lead]
    res = np.empty((n_members, *inits_stacked.shape[:-1], n_steps + order))
    res[:, :, :, :order] = inits_stacked
    for step in range(order, n_steps + order):
        fwd = np.sum(params * res[:, :, :, step - order : step], axis=-1)

        if scale is not None:
            fwd += _epsilon(scale, fwd.shape)
        res[:, :, :, step] = fwd

    # Bundle into xarray DataArray for convenience
    return xr.DataArray(
        res[:, :, :, order:],
        coords={
            "member": range(n_members),
            "init": range(order - 1, len(inits)),
            "sample": range(inits.shape[1]),
            "lead": range(1, n_steps + 1),
        },
    ).squeeze(drop=True)


def generate_samples_like(
    ts,
    order,
    n_times,
    n_samples,
    n_members=None,
    rolling_means=None,
    plot_diagnostics=True,
):
    """
    Convenience function for generating samples from a AR process fitted to
    the provided timeseries.

    Parameters
    ----------
    ts : xarray object
        The timeseries to fit the AR model to
    order : int or str
        The order of the AR(n) process to fit. Alternatively, users can pass
        the string "select_order" in order to use
        statsmodels.tsa.ar_model.ar_select_order to determine the order.
    n_times : int
        The number of timesteps per sample
    n_samples : int
        The number of samples to generate
    n_members : int, optional
        The number of ensemble members to generate. N ensemble members are
        generated from N predictions initialised from samples of the provided
        process. When provided with rolling_mean, rolling means of length L
        are computed by averaging prediction times 1->L.
    rolling_means : list, optional
        A list of lengths of rolling means to compute
    plot_diagnostics : boolean, optional
        If True, plot some diagnostic plots
    """

    p = fit(ts, order=order)
    order = p.order.values
    params = p.isel(params=slice(0, -1)).values
    scale = p.isel(params=-1).values

    samples = generate_samples(
        params,
        scale,
        n_times=n_times,
        n_samples=n_samples,
        n_members=n_members,
        rolling_means=rolling_means,
    )

    if plot_diagnostics:
        fig = plt.figure(figsize=(12, 14))

        # Example timeseries
        ax = fig.add_subplot(4, 1, 1)
        n_samp = 4
        expl = generate_samples(
            params,
            scale,
            n_times=len(ts),
            n_samples=n_samp,
        )
        expl.plot.line(ax=ax, x="time", label=f"AR({order}) model")
        ax.plot(ts, label="piControl", color="k")
        ax.grid()
        ax.set_xlabel("Time")
        ax.set_title("Example fitted timeseries")
        hand, lab = ax.get_legend_handles_labels()
        ax.legend(handles=[hand[0], hand[-1]], labels=[lab[0], lab[-1]])

        # Partial ACFs
        ax = fig.add_subplot(4, 2, 3)
        stats.acf(expl, partial=True).plot.line(
            ax=ax, x="lag", label=f"AR({order}) model", add_legend=False
        )
        stats.acf(ts, partial=True).plot(ax=ax, label="piControl", color="k")
        ax.grid()
        ax.set_xlabel("Lag")
        ax.set_title("pACF")
        hand, lab = ax.get_legend_handles_labels()
        ax.legend(handles=[hand[0], hand[-1]], labels=[lab[0], lab[-1]])

        # PDFs
        ax = fig.add_subplot(4, 2, 4)
        h, bin_edges = np.histogram(ts, bins=30, density=True)
        for s in expl.sample:
            ax.hist(
                expl.sel(sample=s),
                30,
                alpha=0.6,
                density=True,
                label=f"AR({order}) model",
            )
        ax.plot(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            h,
            label="piControl",
            color="k",
        )
        ax.grid()
        ax.set_title("pdf")
        hand, lab = ax.get_legend_handles_labels()
        ax.legend(handles=[hand[0], hand[-1]], labels=[lab[0], lab[-1]])

        # Example forecasts
        ax = fig.add_subplot(4, 1, 3)
        n_leads = max(rolling_means) if rolling_means is not None else 10
        n_mem = n_members if n_members is not None else 50
        expl = predict(
            params,
            ts,
            n_leads,
            n_members=n_mem,
            scale=scale,
        )
        inits = expl.init[::n_leads]
        colors = [f"C{i}" for i in range(0, 10)]
        colorcycler = cycle(colors)
        label = True
        for init in inits.values:
            color = next(colorcycler)

            if label:
                lab = f"AR({order}) forecast"
            else:
                lab = "__nolabel__"

            q1 = expl.sel(init=init).quantile(0.05, dim="member")
            q2 = expl.sel(init=init).quantile(0.95, dim="member")
            ax.fill_between(
                range(init + 1, init + n_leads + 1),
                q1,
                q2,
                color=color,
                alpha=0.4,
                label=lab,
            )

            label = False

        ax.plot(ts, label="piControl", color="k")
        ax.set_xlim(len(ts) - 200, len(ts))
        ax.set_xlabel("Time")
        ax.set_title("Example forecasts")
        ax.legend()
        ax.grid()

        # Example outputs
        ax = fig.add_subplot(4, 1, 4)
        samples_plot = samples
        if "sample" in samples.dims:
            samples_plot = samples_plot.isel(
                sample=np.random.randint(0, high=len(samples_plot.sample))
            )
        if "rolling_mean" in samples.dims:
            samples_plot = samples_plot.isel(rolling_mean=-1)

        if "member" in samples.dims:
            samples_plot.plot.line(
                x="time",
                color=[0.7, 0.7, 0.7],
                add_legend=False,
            )
            samples_plot.sel(member=1).plot(label="Simulated member 1")
            samples_plot.mean("member").plot(label="Simulated ensemble mean")
            ax.legend()
        else:
            samples_plot.plot()
        ax.set_xlabel("Time")
        ax.set_title(f"Example sample: {ax.get_title()}")
        ax.grid()

        fig.tight_layout()

    return samples
