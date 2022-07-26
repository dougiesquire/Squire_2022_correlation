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

jax.config.update("jax_platform_name", "cpu")

BYSTANDER_DIM = "bystander"
STACK_DIM = "stack"
PARAM_DIM = "params"
DUMMY_DIM = "dummy"


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


def _prep_for_fit(ds, dim):
    """
    Prep a dataset for fitting. This function stacks data into a
    3D array with dimensions `BYSTANDER_DIM`, `STACK_DIM` and `dim`

    Also returns a list of dummy dimensions to be squeezed
    """

    ds_prepped = ds.copy()

    stack_dims = []
    dummy_dim = []

    if dim == "lead":
        stack_dims.append("init")

    if "member" in ds.dims:
        stack_dims.append("member")

    bystander_dims = list(set(ds.dims) - set(stack_dims + [dim]))
    to_stack = {BYSTANDER_DIM: bystander_dims}

    # Ensure that there are always bystander and stack dims
    if not stack_dims:
        ds_prepped = ds_prepped.expand_dims(STACK_DIM)
    else:
        to_stack.update({STACK_DIM: stack_dims})

    if not bystander_dims:
        dummy_dim.append(DUMMY_DIM)
        bystander_dims.append(DUMMY_DIM)
        ds_prepped = ds_prepped.expand_dims(DUMMY_DIM)

    dim_order = [BYSTANDER_DIM, STACK_DIM, dim]
    ds_prepped = ds_prepped.stack(to_stack).transpose(*dim_order)

    return ds_prepped, dummy_dim


def fit(ds, n_lags, dim="time"):
    """
    Fit a (Vector) Autoregressive model(s)

    Parameters
    ----------
    ds : xarray Dataset
        The data to fit the (V)AR model to. If multiple variables are available
        in ds, a VAR model is fitted. If a "member" dimension is present, then
        all elements along this dimension are fitted together
    n_lags : int or str
        The order of the (V)AR(n) process to fit. For single variable (AR)
        models, users can alternatively pass the string "select_order" to use
        statsmodels.tsa.ar_model.ar_select_order to determine the order.
    dim : str
        The dimension along which to fit the AR model(s). If `dim="lead"`, then
        all elements along the dimension "init" (assumed to be present) are
        fitted together

    Returns
    -------
    params : xarray Dataset
        The fitted (V)AR model parameters.
    """

    def _get_predictor_lags(array, n_lags):
        """
        Expand new axis containing stacked lags along `dim` (last axis)
        in the following order [lagM, ..., lag2, lag1] and then flatten
        the `STACK_DIM` and `dim` axes
        """
        lags = sliding_window_view(array, window_shape=n_lags, axis=-1)[..., :-1, :]
        return lags.reshape((lags.shape[0], lags.shape[1] * lags.shape[2], n_lags))

    def _get_predictands(array, n_lags):
        """
        Extract the predictands corresponding to the predictor lags from
        `_get_predictor_lags` and then flatten the `STACK_DIM` and `dim`
        axes
        """
        resp = array[..., n_lags:]
        return resp.reshape(resp.shape[0], resp.shape[1] * resp.shape[2])

    def _var_lstsq(*data, n_lags):
        """
        Fit a (V)AR model to the input data

        Each `data` array has dimensions `BYSTANDER_DIM`, `STACK_DIM` and `dim`
        """

        neqs = len(data)

        # Stack array(s) like
        # [var1_lagM, ..., var1_lag1, ..., varN_lagM, ..., varN_lag1]
        predictor = np.concatenate(
            [_get_predictor_lags(array, n_lags) for array in data], axis=-1
        )
        predictand = np.stack(
            [_get_predictands(array, n_lags) for array in data], axis=-1
        )

        # Estimate the least squares coefs using the pseudo-inverse
        pinv = np.linalg.pinv(predictor)
        coefs = np.matmul(pinv, predictand)

        # Calculate the variance of the residuals
        resid = predictand - np.matmul(predictor, coefs)
        sum_squared_error = np.matmul(np.swapaxes(resid, -1, -2), resid)
        # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        # process $u$, see Lütkepohl p.75
        df_correction = neqs * n_lags if neqs > 1 else 0
        df = predictor.shape[1] - df_correction
        noise_var = sum_squared_error / df

        params = np.concatenate((coefs, noise_var), -2)

        # Unpack into variables
        params = [params[..., i] for i in range(params.shape[-1])]

        if len(params) > 1:
            return tuple(params)
        else:
            return params[0]

    variables = list(ds.data_vars)
    n_params = len(variables) * n_lags + len(variables)

    ds_prepped, to_squeeze = _prep_for_fit(ds, dim)

    data = [ds_prepped[v] for v in ds_prepped]
    input_core_dims = len(variables) * [[STACK_DIM, dim]]
    output_core_dims = len(variables) * [["params"]]
    output_dtypes = len(variables) * [float]
    param_labels = [
        f"{v}.lag{lag}" for v in variables for lag in range(n_lags, 0, -1)
    ] + [f"{v}.noise_var" for v in variables]

    params = xr.apply_ufunc(
        _var_lstsq,
        *data,
        kwargs=dict(n_lags=n_lags),
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=output_dtypes,
        dask_gufunc_kwargs=dict(output_sizes={PARAM_DIM: n_params}),
    )

    # Bundle back into a Dataset
    if len(variables) > 1:
        params = xr.merge(
            [
                p.unstack(BYSTANDER_DIM).to_dataset(name=v)
                for v, p in zip(variables, params)
            ]
        )
    else:
        params = params.unstack(BYSTANDER_DIM).to_dataset()

    params = params.squeeze(to_squeeze, drop=True)

    model_order = (
        (params[list(params.data_vars)[0]].count("params") - len(variables))
        / len(variables)
    ).astype(int)
    params = params.assign_coords({"model_order": model_order})
    params = params.assign_coords({PARAM_DIM: param_labels})

    return params


def select_order(ds, dim="time", maxlag=10, kwargs={"ic": "aic"}):
    """
    xarray wrapper for sm.tsa.ar_model.ar_select_order to select
    and fit AR model base on specified information criterion

    Parameters
    ----------
    ds : xarray Dataset
        The data to fit the AR model(s) to.
    dim : str
        The dimension along which to fit the AR model(s)
    kwargs: dict, optional
        kwargs to pass to sm.tsa.ar_model.ar_select_order. Note,
        {"trend": "n"} is always added to kwargs
    """
    from statsmodels.tsa.ar_model import ar_select_order

    def _ar_select_order(data, maxlag, kwargs):
        "Wrapper for statsmodels.tsa.ar_model.ar_select_order"
        res = ar_select_order(data, maxlag, **kwargs).model.fit()
        params = np.empty(maxlag + 1)
        params[:] = np.nan
        if res.ar_lags is not None:
            params[[lag - 1 for lag in res.ar_lags]] = res.params
        params[-1] = res.sigma2
        return params

    if "trend" in kwargs:
        if kwargs["trend"] != "n":
            raise ValueError("This function does not support fitting with a trend")
    else:
        kwargs["trend"] = "n"

    variable = list(ds.data_vars)[0]

    kwargs = dict(maxlag=maxlag, kwargs=kwargs)
    n_params = kwargs["maxlag"] + 1
    n_lags = n_params - 1

    params = xr.apply_ufunc(
        _ar_select_order,
        ds,
        kwargs=kwargs,
        input_core_dims=[[dim]],
        output_core_dims=[["params"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs=dict(output_sizes={"params": n_params}),
    )

    param_labels = [f"{variable}.lag{lag}" for lag in range(1, n_lags + 1)] + [
        "noise_var"
    ]
    model_order = (params[variable].count("params") - 1).astype(int)
    params = params.assign_coords({"model_order": model_order})
    params = params.assign_coords({"params": param_labels}).dropna("params", how="all")
    return params


def generate_samples(
    params, n_times, n_samples, n_members=None, temporal_means=None, seed=None
):
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
        generated from N predictions initialised from a sample of the provided
        process. When provided with temporal_means, means of length L
        are computed by averaging prediction times 1->L.
    temporal_means : list, optional
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
            samples = process.simulate_var(
                steps=n_times + extend_time, seed=seed, nsimulations=n_samples
            ).astype("float32")

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
                distrvs=state.standard_normal,
            ).astype("float32")

    variables = list(params.data_vars)

    n_lags_max = int((params.sizes["params"] - len(variables)) / len(variables))
    if n_members is not None:
        extend_time = n_lags_max - 1 if n_lags_max > 0 else 0
    elif temporal_means is None:
        extend_time = 0
    else:
        extend_time = max(temporal_means) - 1

    samples = xr.apply_ufunc(
        _generate_samples,
        *[params[v] for v in variables],
        kwargs=dict(extend_time=extend_time, seed=seed),
        input_core_dims=[["params"]] * len(variables),
        output_core_dims=[["time", "sample"]] * len(variables),
        vectorize=True,
    )
    if len(variables) > 1:
        samples = xr.merge(
            [samp.to_dataset(name=var) for var, samp in zip(variables, samples)]
        )
    else:
        samples = samples.to_dataset()
        
    samples = samples.assign_coords({
        "time": range(1, samples.sizes["time"]+1),
        "sample": range(n_samples)
    })

    if n_members is not None:
        n_leads = 1 if temporal_means is None else max(temporal_means)
        samples = predict(
            params,
            samples,
            n_steps=n_leads,
            n_members=n_members,
        )
        samples = samples.rename({"init": "time"})

    if temporal_means is not None:
        if n_members is not None:
            res = []
            for av in temporal_means:
                rm = samples.sel(lead=slice(1, av)).mean("lead")
                rm = rm.assign_coords({"temporal_mean": av})
                res.append(rm)
        else:
            res = []
            for av in temporal_means:
                rm = (
                    samples.rolling({"time": av}, min_periods=av, center=False)
                    .mean()
                    .dropna("time")
                )
                rm = rm.assign_coords({"temporal_mean": av})
                res.append(rm)

        samples = xr.concat(res, dim="temporal_mean", join="inner")
    samples = samples.assign_coords(
        {"time": range(1, samples.sizes["time"] + 1), "sample": range(n_samples)}
    )
    if "model_order" in params:
        samples = samples.assign_coords({"model_order": params["model_order"]})

    return samples.squeeze(drop=True)


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
    # then appended to predictors to make the next prediction. I.e. reorder to:
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
                mean = np.zeros(sigma2.shape[:-1])
                cov = sigma2
                return np.array(
                    jax.random.multivariate_normal(key, mean, cov, shape=size[:-1])
                )

                # state = np.random.RandomState(seed)
                # mean = np.zeros(sigma2.shape[-1])
                # cov = sigma2
                # return state.multivariate_normal(
                #     mean, cov, size=size
                # )
            else:
                state = np.random.RandomState(seed)
                return state.normal(scale=np.sqrt(sigma2.squeeze(-1)), size=size)

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
                noise = _get_noise(sigma2, size=predicted.shape, seed=seed)
                predicted += noise
                res[..., step : step + n_vars] = predicted

            # Drop the first n_coefs steps, which correspond to the initial conditions
            res = res[..., n_coefs:]
        else:
            # Move init axis to 0 to match what is done when n_lags != 0
            inits_shape = list(inits[0].shape)
            inits_shape.insert(0, inits_shape.pop())
            shape = (n_members, *inits_shape, n_vars)
            res = np.concatenate(
                [_get_noise(sigma2, shape, seed) for step in range(n_steps)], axis=-1
            )

        # Split into variables and move member (0) and time (1) axis to -3 and -2
        if n_vars > 1:
            res = [
                np.moveaxis(res[..., i::n_vars], [0, 1], [-3, -2])
                for i in range(n_vars)
            ]
            return tuple(res)
        else:
            return np.moveaxis(res, [0, 1], [-3, -2])

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
    temporal_means=None,
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
        generated from N predictions initialised from a sample of the provided
        process. When provided with temporal_mean, rolling means of length L
        are computed by averaging prediction times 1->L.
    temporal_means : list, optional
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
        temporal_means=temporal_means,
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
        n_leads = max(temporal_means) if temporal_means is not None else 10
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

            if "temporal_mean" in samples.dims:
                av = samples_plot.temporal_mean.values[-1]
                samples_plot = samples_plot.sel(temporal_mean=av)
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
