"Tools for working with (Vector) Autoregressive models"


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
    n_lags,
    dim="time",
    kwargs={},
):
    """
    Fit a (Vector) Autoregressive model(s)

    Parameters
    ----------
    ds : xarray Dataset
        The data to fit the (V)AR model to. If multiple variables are available
        in ds, a VAR model is fitted
    n_lags : int or str
        The order of the (V)AR(n) process to fit. For single variable (AR)
        models, users can alternatively pass the string "select_order" to use
        statsmodels.tsa.ar_model.ar_select_order to determine the order.
    dim : str
        The dimension along which to fit the AR model(s)
    kwargs: dict, optional
        kwargs to pass to the relevant statsmodels (sm) method
        - For AR model with n_lags specified: sm.tsa.ar_model.AutoReg
        - For AR model with n_lags="select_order": sm.tsa.ar_model.ar_select_order
        - For VAR model with n_lags specified: sm.tsa.api.VAR

    Returns
    -------
    params : xarray Dataset
        The fitted (V)AR model parameters. For each variable, the first
        n_vars*n_lags parameters along the "params" dimension correspond to the
        (V)AR coefficients in the order output by
        statsmodels.tsa.api.VAR(data).fit(n_lags, trend="n").params, i.e.:
        [ϕ_var1_lag1, ..., ϕ_varN_lag1, ..., ϕ_var1_lagM, ..., ϕ_varN_lagM]
        and the last n_vars parameters correspond to the noise (co)variances:
        [sigma2_var1, ..., sigma2_varN]
    """
    def _ar_select_order(data, maxlag, kwargs):
        "Wrapper for statsmodels.tsa.ar_model.ar_select_order"
        res = ar_select_order(data, maxlag, **kwargs).model.fit()
        params = np.empty(maxlag + 1)
        params[:] = np.nan
        if res.ar_lags is not None:
            params[[l - 1 for l in res.ar_lags]] = res.params
        params[-1] = res.sigma2
        return params

    def _ar(data, n_lags, kwargs):
        "Wrapper for statsmodels.tsa.ar_model.AutoReg"
        res = AutoReg(data, lags=n_lags, **kwargs).fit()
        return np.concatenate((res.params, [res.sigma2]))

    def _var(*data, n_lags, kwargs):
        "Wrapper for statsmodels.tsa.api.VAR"
        res = VAR(np.column_stack(data)).fit(n_lags, **kwargs)
        params = np.vstack((res.params, res.sigma_u))
        return tuple([params[:, i] for i in range(params.shape[1])])

    if "trend" in kwargs:
        if kwargs["trend"] != "n":
            raise ValueError("The function does not support fitting with a trend")
    else:
        kwargs["trend"] = "n"

    variables = list(ds.data_vars)
    if len(variables) > 1:
        from statsmodels.tsa.api import VAR
        
        if n_lags == "select_order":
            raise ValueError("Cannot use 'select_order' with a VAR model")
        func = _var
        kwargs = dict(n_lags=n_lags, kwargs=kwargs)
        n_params = len(variables) * n_lags + len(variables)
    else:
        from statsmodels.tsa.ar_model import AutoReg, ar_select_order
        
        if n_lags == "select_order":
            assert (
                "maxlag" in kwargs
            ), "Must provide maxlag parameter to kwargs when using n_lags='select_order'"

            func = _ar_select_order
            maxlag = kwargs.pop("maxlag")
            kwargs = dict(maxlag=maxlag, kwargs=kwargs)
            n_params = kwargs["maxlag"] + 1
            n_lags = n_params - 1
        else:
            func = _ar
            kwargs = dict(n_lags=n_lags, kwargs=kwargs)
            n_params = n_lags + 1

    data = [ds[v] for v in ds]
    input_core_dims = len(variables) * [[dim]]
    output_core_dims = len(variables) * [["params"]]
    output_dtypes = len(variables) * [float]

    res = xr.apply_ufunc(
        func,
        *data,
        kwargs=kwargs,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        vectorize=True,
        dask="parallelized",
        output_dtypes=output_dtypes,
        dask_gufunc_kwargs=dict(output_sizes={"params": n_params}),
    )

    if len(variables) > 1:
        res = xr.merge([r.to_dataset(name=v) for v, r in zip(variables, res)])
    else:
        res = res.to_dataset()

    param_labels = [f"{v}.lag{l}" for l in range(1, n_lags + 1) for v in variables] + [
        f"{v}.noise_var" for v in variables
    ]
    res = res.assign_coords({"params": param_labels}).dropna("params", how="all")
    return res


def generate_samples(params, n_times, n_samples, n_members=None, rolling_means=None):
    """
    Generate random samples from a (Vector) Autoregressive process.

    Parameters
    ----------
    params : xarray Dataset
        The (V)AR parameters as output by ar_model.fit()
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

    Returns
    -------
    samples : xarray Dataset
        Simulations of the (V)AR process
    """

    variables = list(params.data_vars)
    n_vars = len(variables)
    n_coefs = int(params.sizes["params"] - n_vars)
    n_lags = int(n_coefs / n_vars)

    # Extract coefs and noise variance
    sigma2 = np.column_stack(
        [params[v].isel(params=slice(-len(variables), None)) for v in variables]
    )
    coefs = np.column_stack(
        [params[v].isel(params=slice(-len(variables))) for v in variables]
    )

    if n_members is not None:
        extend = n_coefs - 1
    elif rolling_means is None:
        extend = 0
    else:
        extend = max(rolling_means) - 1

    dims = ["time", "sample"]
    coords = {
        "time": range(1, n_times + extend + 1),
        "sample": range(n_samples),
    }

    if len(variables) > 1:
        # VAR process
        from statsmodels.tsa.vector_ar.var_model import VARProcess

        # Restack coefs to format expected by VARProcess.simulate_var
        coefs = coefs.reshape((n_lags, n_vars, n_vars)).swapaxes(1, 2)

        # For trend = "n" (see statsmodels/tsa/vector_ar/var_model.py#L1356):
        coefs_exog = coefs[:0].T
        _params_info = {
            "k_trend": 0,
            "k_exog_user": 0,
            "k_ar": n_lags,
        }

        process = VARProcess(coefs, coefs_exog, sigma2, _params_info=_params_info)
        s = np.empty(shape=(n_times + extend, n_vars, n_samples))
        extend += n_lags
        for i in range(n_samples):
            s[..., i] = process.simulate_var(steps=n_times + extend).astype("float32")[
                n_lags:
            ]
        s = [d.squeeze(axis=1) for d in np.hsplit(s, n_vars)]
    else:
        # AR process
        from statsmodels.tsa.arima_process import ArmaProcess

        process = ArmaProcess(np.concatenate(([1], -coefs.flatten())))
        s = [
            process.generate_sample(
                nsample=(n_times + extend, n_samples),
                scale=np.sqrt(sigma2),
                axis=0,
            ).astype("float32")
        ]

    s = xr.Dataset(
        data_vars={v: (dims, d) for v, d in zip(variables, s)},
        coords=coords,
    )

    if n_members is not None:
        n_leads = 1 if rolling_means is None else max(rolling_means)
        s = predict(
            params,
            s,
            n_leads,
            n_members=n_members,
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


def predict(params, inits, n_steps, n_members=1):
    """
    Advance a (Vector) Autoregressive model forward in time from initial conditions
    by n_steps

    Parameters
    ----------
    params : xarray Dataset
        The (V)AR parameters as output by ar_model.fit().
    inits : xarray Dataset
        Dataset containing the initial conditions for the predictions. Must have
        the same variables as params and a "time" dimension
    n_steps : int
        The number of timesteps to step forward from each initial condition
    n_members : int
        The number of ensemble members to run from each initial condition. Members
        differ only in their realisation of the (V)AR noise component

    Returns
    -------
    prediction : xarray Dataset
        The predictions made from each initial condition
    """

    variables = list(params.data_vars)
    n_vars = len(variables)
    n_coefs = int(params.sizes["params"] - n_vars)
    n_lags = int(n_coefs / n_vars)

    # Reorder the coefs so that they can be easily matmul by the predictors and
    # then appended to predictors to make the next prediction. I.e. reorder from
    # the input order:
    # var1_lag1, ..., varN_lag1, ..., var1_lagM, ..., varN_lagM
    # to:
    # var1_lagM, ..., varN_lagM, ..., var1_lag1, ..., varN_lag1
    sigma2 = np.column_stack(
        [params[v].isel(params=slice(-len(variables), None)) for v in variables]
    )
    sort_coefs = [f"{v}.lag{l}" for l in range(n_lags, 0, -1) for v in variables]
    coefs = np.column_stack([params.sel(params=sort_coefs)[v] for v in variables])

    # Some quick checks
    assert inits.sizes["time"] >= n_lags, (
        f"At least {order} initial conditions must be provided for an "
        f"AR({order}) model"
    )

    def _predict(*inits, coefs, sigma2, n_steps, n_members):
        """Advance a (V)AR model from initial conditions"""

        # Stack the inits as predictors, sorted so that they are ordered
        # consistently with the order of coefs
        inits_lagged = [
            sliding_window_view(init, window_shape=n_lags, axis=-1) for init in inits
        ]
        inits_lagged = np.stack(inits_lagged, axis=-1).reshape(
            inits_lagged[0].shape[:-1] + (-1,), order="C"
        )

        res = np.empty(
            (n_members, *inits_lagged.shape[:-1], n_vars * n_steps + n_coefs),
            dtype="float32",
        )
        res[..., :n_coefs] = inits_lagged
        for step in range(n_coefs, n_vars * n_steps + n_coefs, n_vars):
            fwd = np.matmul(res[..., step - n_coefs : step], coefs)

            # Add noise
            if n_vars > 1:
                noise = np.random.RandomState().multivariate_normal(
                    np.zeros(len(sigma2)), sigma2, size=fwd.shape[:-1]
                )
            else:
                noise = np.random.normal(scale=np.sqrt(sigma2), size=fwd.shape)
            fwd += noise
            res[..., step : step + n_vars] = fwd

        # Drop the first n_coefs steps, which correspond to the initial conditions
        res = res[..., n_coefs:]

        # Split into variables
        if n_vars > 1:
            res = [res[..., i::n_vars] for i in range(n_vars)]
            return tuple(res)
        else:
            return res

    pred = xr.apply_ufunc(
        _predict,
        *[inits[v] for v in variables],
        kwargs={
            "coefs": coefs,
            "sigma2": sigma2,
            "n_steps": n_steps,
            "n_members": n_members,
        },
        input_core_dims=len(variables) * [["time"]],
        output_core_dims=len(variables) * [["member", "time", "lead"]],
        exclude_dims=set(["time"]),
        vectorize=True,
        dask="parallelized",
    )

    if len(variables) > 1:
        pred = xr.merge([p.to_dataset(name=v) for v, p in zip(variables, pred)])
    else:
        pred = pred.to_dataset()
    pred = pred.rename({"time": "init"})
    coords = {
        "member": range(n_members),
        "init": inits["time"].values[n_lags - 1 :],
        "lead": range(1, n_steps + 1),
        **inits.drop("time").coords,
    }
    return pred.assign_coords(coords)


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
        The timeseries to fit the AR model to. If a "member" dimension exists,
        AR params are calculated for each member separately and the ensemble
        mean parameters are used to generate the synthetic signals.
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
    if "member" in p.dims:
        # Use the most common order and average params across members
        order = stats.mode(p.order, dim="member").compute().item()
        p = fit(ts, order=order).mean("member").assign_coords({"order": order})
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

        alpha = 0.2
        q = (0.05, 0.95)

        if "member" in ts.dims:
            ts_m = ts.mean("member")
            ts_r = (ts.quantile(q[0], "member"), ts.quantile(q[1], "member"))
        else:
            ts_m = ts

        # Example timeseries
        ax = fig.add_subplot(4, 1, 1)
        n_samp = 3
        expl = generate_samples(
            params,
            scale,
            n_times=len(ts),
            n_samples=n_samp,
        )
        expl.plot.line(ax=ax, x="time", label=f"AR({order}) model series")
        if "member" in ts.dims:
            ax.fill_between(
                range(1, len(ts_m) + 1),
                ts_r[0],
                ts_r[1],
                label="_nolabel_",
                color="k",
                edgecolor="none",
                alpha=alpha,
            )
        ax.plot(range(1, len(ts_m) + 1), ts_m, label="Input timeseries", color="k")
        ax.grid()
        ax.set_xlabel("Time")
        ax.set_title("Example fitted timeseries")
        hand, lab = ax.get_legend_handles_labels()
        ax.legend(handles=[hand[0], hand[-1]], labels=[lab[0], lab[-1]])

        # Partial ACFs
        ax = fig.add_subplot(4, 2, 3)
        stats.acf(expl, partial=True).plot.line(
            ax=ax, x="lag", label=f"AR({order}) model series", add_legend=False
        )
        ts_acf = stats.acf(ts, partial=True)
        if "member" in ts_acf.dims:
            ts_acf_r = (
                ts_acf.quantile(q[0], "member"),
                ts_acf.quantile(q[1], "member"),
            )
            ax.fill_between(
                ts_acf.lag,
                ts_acf_r[0],
                ts_acf_r[1],
                label="_nolabel_",
                color="k",
                edgecolor="none",
                alpha=alpha,
            )
            ts_acf.mean("member").plot(ax=ax, label="Input timeseries", color="k")
        else:
            ts_acf.plot(ax=ax, label="Input timeseries", color="k")
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
                label=f"AR({order}) model series",
            )
        ax.plot(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            h,
            label="Input timeseries",
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
        init = expl.isel(sample=0)
        fcst = predict(
            params,
            init,
            n_leads,
            n_members=n_mem,
            scale=scale,
        )
        inits = fcst.init[::n_leads]
        colors = [f"C{i}" for i in range(0, 10)]
        colorcycler = cycle(colors)
        label = True
        for i in inits.values:
            color = next(colorcycler)

            if label:
                lab = f"AR({order}) forecast"
            else:
                lab = "__nolabel__"

            q1 = fcst.sel(init=i).quantile(0.05, dim="member")
            q2 = fcst.sel(init=i).quantile(0.95, dim="member")
            ax.fill_between(
                range(i + 1, i + n_leads + 1),
                q1,
                q2,
                color=color,
                alpha=0.4,
                label=lab,
            )
            ax.plot(
                range(i + 1, i + n_leads + 1),
                fcst.sel(init=i).mean("member"),
                color=color,
            )

            label = False
        ax.plot(init, label=f"AR({order}) model series", color="k")
        ax.set_xlim(len(ts) - min(len(ts), 200), len(ts))
        ax.set_xlabel("Time")
        ax.set_title("Example forecasts")
        ax.legend()
        ax.grid()

        # Example outputs
        ax = fig.add_subplot(4, 1, 4)
        samples_plot = samples
        ts_plot = ts_m
        if "sample" in samples.dims:
            samples_plot = samples_plot.isel(
                sample=np.random.randint(0, high=len(samples_plot.sample))
            )

        if "rolling_mean" in samples.dims:
            av = samples_plot.rolling_mean.values[-1]
            samples_plot = samples_plot.sel(rolling_mean=av)
            ts_plot = ts_plot.rolling({"time": av}, min_periods=av, center=False).mean(
                "time"
            )

        if "member" in samples.dims:
            ax.fill_between(
                samples_plot.time.values,
                samples_plot.quantile(0.05, dim="member"),
                samples_plot.quantile(0.95, dim="member"),
                color="C0",
                edgecolor="none",
                alpha=0.4,
            )
            samples_plot.sel(member=1).plot(
                color="C0", linestyle="--", label="Simulated member 1"
            )
            samples_plot.mean("member").plot(
                color="C0", label="Simulated ensemble mean"
            )
        else:
            samples_plot.plot()
        if "member" in ts.dims:
            lab = "Input timeseries, ensemble mean"
        else:
            lab = "Input timeseries"
        ax.plot(
            range(1, min(ts_plot.sizes["time"]+1, samples_plot.sizes["time"] + 1)),
            ts_plot.isel(time=slice(-samples_plot.sizes["time"], None)),
            label=lab,
            color="k",
        )
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_title(f"Example sample: {ax.get_title()}")
        ax.grid()

        fig.tight_layout()

    return samples
