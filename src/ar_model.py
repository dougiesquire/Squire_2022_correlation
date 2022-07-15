"Tools for working with (Vector) Autoregressive models"


import os

import sys

import jax

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
            params[[lag - 1 for lag in res.ar_lags]] = res.params
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

    params = xr.apply_ufunc(
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
        params = xr.merge([r.to_dataset(name=v) for v, r in zip(variables, params)])
    else:
        params = params.to_dataset()

    param_labels = [
        f"{v}.lag{lag}" for lag in range(1, n_lags + 1) for v in variables
    ] + [f"{v}.noise_var" for v in variables]
    model_order = (
        (params[list(params.data_vars)[0]].count("params") - len(variables))
        / len(variables)
    ).astype(int)
    params = params.assign_coords({"model_order": model_order})
    params = params.assign_coords({"params": param_labels}).dropna("params", how="all")
    return params


def generate_samples(params, n_times, n_samples, n_members=None, rolling_means=None, seed=None):
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
    seed : int, optional
        Seed for the generation of random noise. If seed is None, then will 
        try to generate seed from /dev/urandom (or the Windows analogue) if 
        available

    Returns
    -------
    samples : xarray Dataset
        Simulations of the (V)AR process
    """
    def _generate_samples(*params, extend_time, seed):
        """Generate samples of (V)AR process with specified params"""
        n_vars = len(params)

        # Drop NaNs
        params = [p[~np.isnan(p)] for p in params]

        # Extract coefs and noise variance
        sigma2 = np.column_stack([p[-len(variables) :] for p in params])
        coefs = np.column_stack([p[: -len(variables)] for p in params])

        n_coefs = coefs.shape[0]
        n_lags = int(n_coefs / n_vars)

        if n_vars > 1:
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
            samples = np.empty(shape=(n_samples, n_times + extend_time, n_vars))
            extend_time += n_lags
            samples = process.simulate_var(steps=n_times + extend_time, seed=seed, nsimulations=n_samples).astype(
                    "float32"
                )
            
            # Move samples dimension (0) to last place
            samples = np.moveaxis(samples, 0, -1)
            samples = samples[n_lags:]
            return tuple([d.squeeze(axis=1) for d in np.hsplit(samples, n_vars)])
        else:
            # AR process
            from statsmodels.tsa.arima_process import ArmaProcess

            process = ArmaProcess(np.concatenate(([1], -coefs.flatten())))
            state = np.random.RandomState(seed)
            return process.generate_sample(
                nsample=(n_times + extend_time, n_samples),
                scale=np.sqrt(sigma2),
                axis=0,
                distrvs=state.standard_normal
            ).astype("float32")

    variables = list(params.data_vars)
    
    n_lags_max = params["model_order"].max().item()
    if n_members is not None:
        extend_time = n_lags_max - 1 if n_lags_max > 0 else 0
    elif rolling_means is None:
        extend_time = 0
    else:
        extend_time = max(rolling_means) - 1
    
    samples = xr.apply_ufunc(
        _generate_samples,
        *[params[v] for v in variables],
        kwargs=dict(extend_time=extend_time, seed=seed),
        input_core_dims=[["params"]] * len(variables),
        output_core_dims=[["time", "sample"]] * len(variables),
        vectorize=True
    )
    if len(variables) > 1:
        samples = xr.merge([samp.to_dataset(name=var) for var, samp in zip(variables, samples)])
    else:
        samples = samples.to_dataset()  
    
    if n_members is not None:
        n_leads = 1 if rolling_means is None else max(rolling_means)
        samples = predict(
            params,
            samples,
            n_steps=n_leads,
            n_members=n_members,
        )
        samples = samples.rename({"init": "time"})

    if rolling_means is not None:
        if n_members is not None:
            res = []
            for av in rolling_means:
                rm = samples.sel(lead=slice(1, av)).mean("lead")
                rm = rm.assign_coords({"rolling_mean": av})
                res.append(rm)
        else:
            res = []
            for av in rolling_means:
                rm = (
                    samples.rolling({"time": av}, min_periods=av, center=False)
                    .mean()
                    .dropna("time")
                )
                rm = rm.assign_coords({"rolling_mean": av})
                res.append(rm)

        samples = xr.concat(res, dim="rolling_mean", join="inner")
    samples = samples.assign_coords(
        {"time": range(1, samples.sizes["time"] + 1), "sample": range(n_samples)}
    )
    samples = samples.assign_coords({"model_order": params["model_order"]})

    return samples.squeeze(drop=True)


def predict_old(params, inits, n_steps, n_members=1):
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
    sort_coefs = [f"{v}.lag{lag}" for lag in range(n_lags, 0, -1) for v in variables]
    coefs = np.column_stack([params.sel(params=sort_coefs)[v] for v in variables])
    print(coefs.shape)
    
    # Some quick checks
    assert inits.sizes["time"] >= n_lags, (
        f"At least {n_lags} initial conditions must be provided for an "
        f"AR({n_lags}) model"
    )

    def _predict(*inits, coefs, sigma2, n_steps, n_members):
        """Advance a (V)AR model from initial conditions"""

        def _get_noise(sigma2, size):
            if len(sigma2) > 1:
                return np.random.RandomState().multivariate_normal(
                    np.zeros(len(sigma2)), sigma2, size=size
                )
            else:
                return np.expand_dims(
                    np.random.normal(scale=np.sqrt(sigma2), size=size), -1
                )   
        
        if n_lags != 0:
            # Stack the inits as predictors, sorted so that they are ordered
            # consistently with the order of coefs
            inits_lagged = [
                sliding_window_view(init, window_shape=n_lags, axis=-1)
                for init in inits
            ]
            inits_lagged = np.stack(inits_lagged, axis=-1).reshape(
                inits_lagged[0].shape[:-1] + (-1,), order="C"
            )

            shape = (n_members, *inits_lagged.shape[:-1], n_vars * n_steps + n_coefs)
            res = np.empty(shape, dtype="float32")
            res[..., :n_coefs] = inits_lagged
            for step in range(n_coefs, n_vars * n_steps + n_coefs, n_vars):
                print(res[..., step - n_coefs : step].shape)
                print(coefs.shape)
                print(res[..., step - n_coefs : step] @ coefs)
                fwd = np.matmul(res[..., step - n_coefs : step], coefs)
                print(fwd.shape)
                # Add noise
                fwd += _get_noise(sigma2, fwd.shape[:-1])
                res[..., step : step + n_vars] = fwd

            # Drop the first n_coefs steps, which correspond to the initial conditions
            res = res[..., n_coefs:]
        else:
            shape = (n_members, *inits[0].shape)
            res = np.concatenate(
                [_get_noise(sigma2, shape) for step in range(n_steps)], axis=-1
            )

        # Split into variables and move member axis to -2 position
        if n_vars > 1:
            res = [np.moveaxis(res[..., i::n_vars], 0, -2) for i in range(n_vars)]
            return tuple(res)
        else:
            return np.moveaxis(res, 0, -2)

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
        output_core_dims=len(variables) * [["time", "member", "lead"]],
        exclude_dims=set(["time"]),
    )

    if len(variables) > 1:
        pred = xr.merge([p.to_dataset(name=v) for v, p in zip(variables, pred)])
    else:
        pred = pred.to_dataset()
    pred = pred.rename({"time": "init"})
    first_init_idx = 0 if n_lags == 0 else n_lags - 1
    coords = {
        "member": range(n_members),
        "init": inits["time"].values[first_init_idx:],
        "lead": range(1, n_steps + 1),
        **inits.drop("time").coords,
    }
    return pred.assign_coords(coords)


def predict(params, inits, n_steps, n_members=1, seed=None):
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
    n_members : int. optional
        The number of ensemble members to run from each initial condition. Members
        differ only in their realisation of the (V)AR noise component
    seed : int, optional
        Seed for the generation of random noise. Note that setting this value will
        mean that the same seed (hence noise) is used for every prediction step.
        If seed is None, then will try to generate seed from /dev/urandom (or the 
        Windows analogue) if available.
    Returns
    -------
    prediction : xarray Dataset
        The predictions made from each initial condition
    """

    variables = list(params.data_vars)
    n_vars = len(variables)
    n_coefs = int(params.sizes["params"] - n_vars)
    n_lags = int(n_coefs / n_vars)

    # Fill NaNs with zeros (Nans come about if there are multiple models with different
    # orders)
    params = params.fillna(0)

    # Ensure params is the last dimension, and shared dimensions are in the right
    # place for broadcasting during matmul
    shared_dims = [d for d in inits.dims if d in params.dims]
    params = params.transpose(*shared_dims, "params")
    inits = inits.transpose(..., *shared_dims, "time")

    # Reorder the coefs so that they can be easily matmul'd by the predictors and
    # then appended to predictors to make the next prediction. I.e. reorder from
    # the input order:
    # var1_lag1, ..., varN_lag1, ..., var1_lagM, ..., varN_lagM
    # to:
    # var1_lagM, ..., varN_lagM, ..., var1_lag1, ..., varN_lag1
    sigma2 = np.stack(
        [params[v].isel(params=slice(-len(variables), None)) for v in variables],
        axis=-1,
    )
    sort_coefs = [f"{v}.lag{lag}" for lag in range(n_lags, 0, -1) for v in variables]
    coefs = np.stack(
        [params[v].sel(params=sort_coefs) for v in variables],
        axis=-1,
    )

    # Some quick checks
    assert inits.sizes["time"] >= n_lags, (
        f"At least {n_lags} initial conditions must be provided for an "
        f"AR({n_lags}) model"
    )

    def _predict(*inits, coefs, sigma2, n_steps, n_members, seed):
        """Advance a (V)AR model from initial conditions"""

        def _get_noise(sigma2, size, seed):
            if seed is None:
                # Generate seed from /dev/urandom/
                seed = int.from_bytes(os.urandom(4), sys.byteorder)
                    
            if sigma2.shape[-1] > 1:
                # Use jax because it allows cov to have shape (..., N,N)
                # Note, I also tried mxnp.random.multivariate_normal but this
                # produced unexpected behaviour (see 
                # https://github.com/apache/incubator-mxnet/issues/21095)
                key = jax.random.PRNGKey(seed)
                mean = np.zeros(sigma2.shape[-1])
                cov = sigma2
                noise = np.array(
                    jax.random.multivariate_normal(key, mean, cov, shape=size)
                )

                # state = np.random.RandomState(seed)
                # mean = np.zeros(sigma2.shape[-1])
                # cov = sigma2
                # noise = state.multivariate_normal(
                #     mean, cov, size=size
                # )

                return noise
            else:
                state = np.random.RandomState(seed)
                return np.expand_dims(
                    state.normal(scale=np.sqrt(sigma2), size=size), -1
                )

        if n_lags != 0:
            # Stack the inits as predictors, sorted so that they are ordered
            # consistently with the order of coefs
            inits_lagged = [
                sliding_window_view(init, window_shape=n_lags, axis=-1)
                for init in inits
            ]
            inits_lagged = np.stack(inits_lagged, axis=-1).reshape(
                inits_lagged[0].shape[:-1] + (-1,), order="C"
            )
            
            # Move init axis to 0 to make broadcasting easier
            inits_lagged = np.moveaxis(inits_lagged, -2, 0)

            shape = (n_members, *inits_lagged.shape[:-1], n_vars * n_steps + n_coefs)
            res = np.empty(shape, dtype="float32")
            res[..., :n_coefs] = inits_lagged
            for step in range(n_coefs, n_vars * n_steps + n_coefs, n_vars):
                # Expand then squeeze dummy axes for matrix multiplication
                predictors = np.expand_dims(res[..., step - n_coefs : step], -2)
                predicted = np.squeeze(np.matmul(predictors, coefs), -2)
                
                # Add noise
                size = predicted.shape[:-(sigma2.ndim - 1)]
                noise = _get_noise(sigma2, size, seed)
                predicted += noise
                res[..., step : step + n_vars] = predicted

            # Drop the first n_coefs steps, which correspond to the initial conditions
            res = res[..., n_coefs:]
        else:
            shape = (n_members, *inits[0].shape)
            res = np.concatenate(
                [_get_noise(sigma2, shape) for step in range(n_steps)], axis=-1
            )
            
        # Split into variables and move member (0) and time (1) axis to -3 and -2
        if n_vars > 1:
            res = [np.moveaxis(res[..., i::n_vars], [0,1], [-3,-2]) for i in range(n_vars)]
            return tuple(res)
        else:
            return np.moveaxis(res, [0,1], [-3,-2])

    pred = xr.apply_ufunc(
        _predict,
        *[inits[v] for v in variables],
        kwargs={
            "coefs": coefs,
            "sigma2": sigma2,
            "n_steps": n_steps,
            "n_members": n_members,
            "seed": seed,
        },
        input_core_dims=len(variables) * [["time"]],
        output_core_dims=len(variables) * [["member", "time", "lead"]],
        exclude_dims=set(["time"]),
    )

    if len(variables) > 1:
        pred = xr.merge([p.to_dataset(name=v) for v, p in zip(variables, pred)])
    else:
        pred = pred.to_dataset()
    pred = pred.rename({"time": "init"})
    first_init_idx = 0 if n_lags == 0 else n_lags - 1
    coords = {
        "member": range(n_members),
        "init": inits["time"].values[first_init_idx:],
        "lead": range(1, n_steps + 1),
        **inits.drop("time").coords,
    }
    return pred.assign_coords(coords)


def generate_samples_like(
    ds,
    n_lags,
    n_times,
    n_samples,
    n_members=None,
    rolling_means=None,
    fit_kwargs={},
    plot_diagnostics=True,
):
    """
    Convenience function for generating samples from a AR process fitted to
    the provided timeseries.

    Parameters
    ----------
    ds : xarray Dataset
        The data to fit the (V)AR model to. If multiple variables are available
        in ds, a VAR model is fitted. If a "member" dimension exists, (V)AR
        params are calculated for each member separately and the ensemble mean
        parameters are used to generate the synthetic signals.
    n_lags : int or str
        The order of the (V)AR(n) process to fit. For single variable (AR)
        models, users can alternatively pass the string "select_order" to use
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
    fit_kwargs : dict, optional
        kwargs to pass to ar_model.fit()
    plot_diagnostics : boolean, optional
        If True, plot some diagnostic plots
    """

    params = fit(ds, n_lags=n_lags, kwargs=fit_kwargs)
    if "member" in params.dims:
        # Use the most common order and average params across members
        n_lags = stats.mode(params.model_order, dim="member").compute().item()
        params = fit(ds, n_lags=n_lags).mean("member")
    samples = generate_samples(
        params,
        n_times=n_times,
        n_samples=n_samples,
        n_members=n_members,
        rolling_means=rolling_means,
    )

    if plot_diagnostics:
        alpha = 0.2
        q = (0.05, 0.95)
        variables = list(ds.data_vars)
        n_rows = len(variables) * 4

        if n_lags == "select_order":
            n_lags = int((params.sizes["params"] - len(variables)) / len(variables))
        legend_str = "VAR" if len(variables) > 1 else "AR"
        legend_str += f"({n_lags}) model series"

        fig = plt.figure(figsize=(12, len(variables) * 14))

        if "member" in ds.dims:
            ds_m = ds.mean("member")
            ds_r = (ds.quantile(q[0], "member"), ds.quantile(q[1], "member"))
        else:
            ds_m = ds

        # Example timeseries
        idx = 1
        n_samp = 3
        expl = generate_samples(
            params,
            n_times=ds.sizes["time"],
            n_samples=n_samp,
        )
        for idx, v in enumerate(variables):
            ax = fig.add_subplot(n_rows, 1, idx + 1)

            expl[v].plot.line(
                ax=ax,
                x="time",
                label=legend_str,
                add_legend=False,
            )
            if "member" in ds.dims:
                ax.fill_between(
                    range(1, ds_m[v].sizes["time"] + 1),
                    ds_r[0][v],
                    ds_r[1][v],
                    label="_nolabel_",
                    color="k",
                    edgecolor="none",
                    alpha=alpha,
                )
                ax.plot(
                    range(1, ds_m[v].sizes["time"] + 1),
                    ds[v].isel(member=0),
                    label="Input _nolabel_",
                    linestyle="--",
                    color="k",
                )
            ax.plot(
                range(1, ds_m[v].sizes["time"] + 1),
                ds_m[v],
                label="Input timeseries",
                color="k",
            )
            ax.grid(True)
            ax.set_xlabel("Time")
            if idx == 0:
                ax.set_title("Example fitted timeseries")
                ax.set_xlabel("")
                hand, lab = ax.get_legend_handles_labels()
                ax.legend(handles=[hand[0], hand[-1]], labels=[lab[0], lab[-1]])
            else:
                ax.set_title("")

        # Partial ACFs
        idx_offset = 2 * len(variables)
        for idx, v in enumerate(variables):
            ax = fig.add_subplot(n_rows, 2, (len(variables) * idx) + idx_offset + 1)
            stats.acf(expl[v], partial=True).plot.line(
                ax=ax, x="lag", label=legend_str, add_legend=False
            )
            ds_acf = stats.acf(ds[v], partial=True)
            if "member" in ds_acf.dims:
                ds_acf_r = (
                    ds_acf.quantile(q[0], "member"),
                    ds_acf.quantile(q[1], "member"),
                )
                ax.fill_between(
                    ds_acf.lag,
                    ds_acf_r[0],
                    ds_acf_r[1],
                    label="_nolabel_",
                    color="k",
                    edgecolor="none",
                    alpha=alpha,
                )
                ds_acf.mean("member").plot(ax=ax, label="Input timeseries", color="k")
            else:
                ds_acf.plot(ax=ax, label="Input timeseries", color="k")
            ax.grid(True)
            if idx == 0:
                ax.set_title("pACF")
                ax.set_xlabel("")
                hand, lab = ax.get_legend_handles_labels()
                ax.legend(handles=[hand[0], hand[-1]], labels=[lab[0], lab[-1]])
            else:
                ax.set_title("")
                ax.set_xlabel("Lag")

        # PDFs
        for idx, v in enumerate(variables):
            ax = fig.add_subplot(n_rows, 2, (len(variables) * idx) + idx_offset + 2)
            h, bin_edges = np.histogram(ds[v], bins=30, density=True)
            for s in expl.sample:
                ax.hist(
                    expl[v].sel(sample=s),
                    30,
                    alpha=0.6,
                    density=True,
                    label=legend_str,
                )
            ax.plot(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                h,
                label="Input timeseries",
                color="k",
            )
            ax.grid(True)
            ax.set_ylabel(f"pdf({v})")
            if idx == 0:
                ax.set_title("pdf")
                hand, lab = ax.get_legend_handles_labels()
                ax.legend(handles=[hand[0], hand[-1]], labels=[lab[0], lab[-1]])
            else:
                ax.set_title("")

        # Example forecasts
        n_leads = max(rolling_means) if rolling_means is not None else 10
        n_mem = n_members if n_members is not None else 50
        init = expl.isel(sample=0)
        fcst = predict(
            params,
            init,
            n_leads,
            n_members=n_mem,
        )
        for idx, v in enumerate(variables):
            ax = fig.add_subplot(n_rows, 1, idx + idx_offset + 1)

            inits = fcst[v].init[::n_leads]
            colors = [f"C{i}" for i in range(0, 10)]
            colorcycler = cycle(colors)
            label = True
            for i in inits.values:
                color = next(colorcycler)

                if label:
                    lab = legend_str
                else:
                    lab = "__nolabel__"

                q1 = fcst[v].sel(init=i).quantile(0.05, dim="member")
                q2 = fcst[v].sel(init=i).quantile(0.95, dim="member")
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
                    fcst[v].sel(init=i).mean("member"),
                    color=color,
                )

                label = False
            ax.plot(init[v], label=legend_str, color="k")
            ax.set_xlim(ds.sizes["time"] - min(ds.sizes["time"], 200), ds.sizes["time"])
            ax.grid(True)
            ax.set_ylabel(v)
            if idx == 0:
                ax.set_title("Example forecasts")
                ax.set_xlabel("")
                ax.legend()
            else:
                ax.set_title("")
                ax.set_xlabel("Time")

        # Example outputs
        idx_offset = 3 * len(variables)
        for idx, v in enumerate(variables):
            ax = fig.add_subplot(n_rows, 1, idx + idx_offset + 1)
            samples_plot = samples[v]
            ds_plot = ds_m[v]
            if "sample" in samples.dims:
                samples_plot = samples_plot.isel(
                    sample=np.random.randint(0, high=len(samples_plot.sample))
                )

            if "rolling_mean" in samples.dims:
                av = samples_plot.rolling_mean.values[-1]
                samples_plot = samples_plot.sel(rolling_mean=av)
                ds_plot = ds_plot.rolling(
                    {"time": av}, min_periods=av, center=False
                ).mean("time")

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
            if "member" in ds.dims:
                lab = "Input timeseries, ensemble mean"
            else:
                lab = "Input timeseries"
            ax.plot(
                range(
                    1, min(ds_plot.sizes["time"] + 1, samples_plot.sizes["time"] + 1)
                ),
                ds_plot.isel(time=slice(-samples_plot.sizes["time"], None)),
                label=lab,
                color="k",
            )
            ax.grid(True)
            if idx == 0:
                ax.legend()
                ax.set_title(f"Example sample: {ax.get_title()}")
                ax.set_xlabel("")
            else:
                ax.set_title("")
                ax.set_xlabel("Time")

        fig.tight_layout()

    return samples
