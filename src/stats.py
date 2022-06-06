"Functions loosely in the realm of statistics"

import sys

import math

import scipy

import numpy as np

import xarray as xr

import xskillscore as xs

import statsmodels.api as sm

from collections import OrderedDict
from itertools import chain, islice, cycle


def acf(ds, dim="time", partial=False, nlags=10, kwargs={}):
    """
    Vectorized xarray wrapper on statsmodels.tsa.acf and .pacf

    Parameters
    ----------
    ds : xarray object
        The data to use to compute the ACF
    dim : str
        The dimension along which to compute the ACF
    partial: bool, optional
        If True, return the partial ACF
    nlags: int, optional
        The number of lags to compute
    kwargs: dict, optional
        Additional kwargs to pass to the statsmodel acf function
    """

    def _acf(data, nlags, partial):
        if partial:
            return sm.tsa.pacf(data, nlags=nlags, method="ywm")
        else:
            return sm.tsa.acf(data, nlags=nlags)

    return xr.apply_ufunc(
        _acf,
        ds,
        kwargs=dict(nlags=nlags, partial=partial),
        input_core_dims=[[dim]],
        output_core_dims=[["lag"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs=dict(output_sizes={"lag": nlags + 1}),
    ).assign_coords({"lag": range(nlags + 1)})


def student_t(N, r):
    """
    Return the Student-t distribution for null correlation

    Parameters
    ----------
    N : int
        The number of correlation samples
    r : numpy array
        The correlation values at which to return the pdf(r)
    """
    from scipy.stats import beta

    a = N / 2 - 1
    b = N / 2 - 1
    return beta(a, b, loc=-1, scale=2).pdf(r)


def mode(ds, dim):
    """
    xarray wrapper on scipy.stats.mode

    Parameters
    ----------
    ds : xarray object
        The input data
    dim : str
        The dimension along which to compute the mode
    """
    def _mode(data):
        m, _ = scipy.stats.mode(data, axis=-1, nan_policy='omit')
        return m.squeeze()

    return xr.apply_ufunc(
        _mode, 
        ds,
        input_core_dims=[[dim]],
        exclude_dims=set([dim]),
        dask="parallelized")


# Metrics
# ===============================================


def get_Type_I_error_rates(
    fcst, obsv, n_times, n_members, metric, method, method_kwargs, alpha=0.05
):
    """
    Returns the Type I error rates for infering a specified metric using a specified method

    Parameters
    ----------
    fcst : xarray object
        The simulated forecasts
    obsv : xarray object
        The simulated observations
    n_times : list
        List of sample lengths in time to calculate the error rates for
    n_members : list
        List of number of ensemble members to calculate the error rates for
    alpha : float, optional
        The alpha level to calculate the error rates at
    """
    import itertools

    res = []
    for Nt in n_times:
        res_mem = []
        for Nm in n_members:
            _, pval = infer_metric(
                fcst.sel(time=slice(0, Nt), member=slice(0, Nm)),
                obsv.sel(time=slice(0, Nt)),
                metric=metric,
                method=method,
                method_kwargs=method_kwargs,
            )
            res_mem.append(pval.assign_coords({"n_members": Nm, "n_times": Nt}))
        res.append(xr.concat(res_mem, dim="n_members"))
    return (xr.concat(res, dim="n_times") < alpha).mean("sample")


def infer_metric(a, b, metric, method, method_kwargs=None, dim="time"):
    """
    Return skill metric(s) between two series and corresponding p-value(s)

    Parameters
    ----------
    a : xarray object
        Input array
    b : xarray object
        Input array
    metric : str
        The metric to infer. Must be the name of a method in src.stats
    method : str
        The method to use to determine the p-value(s). Options are "bootstrap"
    method_kwargs : dict
        kwargs to pass to method
    dim : str
        The dimension along which to compute the metric
    """

    if method == "bootstrap":
        samp, p = bootstrap_pvalue(a, b, metric, dim, **method_kwargs)

    return samp, p


def pearson_r(a, b, dim="time"):
    """
    Return the Pearson correlation coefficient between two timeseries

    Parameters
    ----------
    a : xarray object
        Input array
    b : xarray Dataset
        Input object
    """
    if "member" in a.dims:
        a = a.mean("member")
    if "member" in b.dims:
        b = b.mean("member")

    return xs.pearson_r(a, b, dim)


def Fisher_z(ds):
    """
    Return the Fisher-z transformation of ds

    Parameters
    ----------
    ds : xarray Dataset
        The data to apply the Fisher-z transformation to
    """
    return np.arctanh(ds)


def effective_sample_size(fcst, obsv, N=None, dim="time"):
    """
    Return the effective sample size for temporally correlated data.

           1 - pf * po
    Neff = ----------- * N = P * N
           1 + pf * po

    where pf and po are the lag-1 autocorrelation coefficients of the
    forecast and observations

    Parameters
    ----------
    fcst :  xarray object
        The forecast data
    obsv : xarray object
        The observed data
    N : int, optional
        Provide this if you wish to estimate the coefficient P using
        fcst and obsv, but return Neff for a different sample size N
        than was available in fcst and obsv
    dim : str, optional
        The name of the time dimension
    """

    Neff = xs.effective_sample_size(fcst, obsv)

    if N is not None:
        P = Neff / fcst.sizes[dim]
        return P * N
    else:
        return Neff


# Bootstrapping
# ===============================================


def blocklength_Wilks(N, N_eff):
    """
    Return the block size to use for block boostrapping according to Wilks, pg 191

    Parameters
    ----------
    N : int
        The sample size
    N_eff : float
        The effective sample size
    """
    from scipy.optimize import fsolve

    def f(L, N, N_eff):
        return (N - L + 1) ** ((2 / 3) * (1 - (N_eff / N))) - L

    return round(fsolve(f, 5, args=(N, N_eff))[0])


def _get_pval_from_bootstrap(sample, iterations, null, transform):
    """ Calculate p-value(s) from bootstrapped iterations """
    if transform:
        transform = getattr(sys.modules[__name__], transform)
        null = transform(null)
        iterations = transform(iterations)

    left_tail_p = xr.where(iterations < null, 1, 0).mean("iteration") * 2
    right_tail_p = xr.where(iterations > null, 1, 0).mean("iteration") * 2
    return xr.where(sample >= 0, left_tail_p, right_tail_p)


def bootstrap_pvalue(
    a,
    b,
    metric,
    metric_dim,
    blocks,
    n_iteration,
    transform,
    return_iterations=False
):
    """
    Determine the two-tail p-value(s) for a given metric by block-
    bootstrapping as in Goddard et al. (2013)

    Parameters
    ----------
    a : xarray object
        Input array
    b : xarray object
        Input array
    metric : function
        The metric to infer.
    metric_dim : str
        The dimension along which to compute the metric
    blocks : dict
        Dictionary of the dimension(s) to bootstrap and the block sizes to use
        along each dimension: {dim: blocksize}.
    n_iteration : int
        The number of times to repeat the bootstrapping
    transform : str
        Transform to apply prior to estimating significant points. Must be the
        name of a method in src.stats
    """

    metric = getattr(sys.modules[__name__], metric)
    
    samp = metric(a, b, metric_dim)
    iterations = metric(
        *block_bootstrap(
            a,
            b,
            blocks=blocks,
            n_iteration=n_iteration,
        ),
        metric_dim,
    )

    if return_iterations:
        return samp, iterations
    else:
        pval = _get_pval_from_bootstrap(
            samp, 
            iterations, 
            null=0, 
            transform=transform
        )
        return samp, pval


def _get_blocked_random_indices(
    shape,
    block_axis,
    block_size,
    prev_block_sizes,
):
    """
    Return indices to randomly sample an axis of an array in consecutive
    (cyclic) blocks
    """

    def _random_blocks(length, block):
        """
        Indices to randomly sample blocks in a cyclic manner along an axis of a
        specified length
        """
        if block == length:
            return list(range(length))
        else:
            repeats = math.ceil(length / block)
            return list(
                chain.from_iterable(
                    islice(cycle(range(length)), s, s + block)
                    for s in np.random.randint(0, length, repeats)
                )
            )[:length]

    # Don't randomize within an outer block
    if len(prev_block_sizes) > 0:
        orig_shape = shape.copy()
        for i, b in enumerate(prev_block_sizes[::-1]):
            prev_ax = block_axis - (i + 1)
            shape[prev_ax] = math.ceil(shape[prev_ax] / b)

    if block_size == 1:
        indices = np.random.randint(
            0,
            shape[block_axis],
            shape,
        )
    else:
        non_block_shapes = [s for i, s in enumerate(shape) if i != block_axis]
        indices = np.moveaxis(
            np.stack(
                [
                    _random_blocks(shape[block_axis], block_size)
                    for _ in range(np.prod(non_block_shapes))
                ],
                axis=-1,
            ).reshape([shape[block_axis]] + non_block_shapes),
            0,
            block_axis,
        )

    if len(prev_block_sizes) > 0:
        for i, b in enumerate(prev_block_sizes[::-1]):
            prev_ax = block_axis - (i + 1)
            indices = np.repeat(indices, b, axis=prev_ax).take(
                range(orig_shape[prev_ax]), axis=prev_ax
            )
        return indices
    else:
        return indices


def _n_nested_blocked_random_indices(sizes, n_iteration):
    """
    Returns indices to randomly resample blocks of an array (with replacement)
    in a nested manner many times. Here, "nested" resampling means to randomly
    resample the first dimension, then for each randomly sampled element along
    that dimension, randomly resample the second dimension, then for each
    randomly sampled element along that dimension, randomly resample the third
    dimension etc.

    Parameters
    ----------
    sizes : OrderedDict
        Dictionary with {names: (sizes, blocks)} of the dimensions to resample
    n_iteration : int
        The number of times to repeat the random resampling
    """

    shape = [s[0] for s in sizes.values()]
    indices = OrderedDict()
    prev_blocks = []
    for ax, (key, (_, block)) in enumerate(sizes.items()):
        indices[key] = _get_blocked_random_indices(
            shape[: ax + 1] + [n_iteration], ax, block, prev_blocks
        )
        prev_blocks.append(block)
    return indices


def _expand_n_nested_random_indices(indices):
    """
    Expand the dimensions of the nested input arrays so that they can be
    broadcast and return a tuple that can be directly indexed

    Parameters
    ----------
    indices : list of numpy arrays
        List of numpy arrays of sequentially increasing dimension as output by
        the function `_n_nested_blocked_random_indices`. The last axis on all
        inputs is assumed to correspond to the iteration axis
    """
    broadcast_ndim = indices[-1].ndim
    broadcast_indices = []
    for i, ind in enumerate(indices):
        expand_axes = list(range(i + 1, broadcast_ndim - 1))
        broadcast_indices.append(np.expand_dims(ind, axis=expand_axes))
    return (..., *tuple(broadcast_indices))


def _block_bootstrap(*objects, blocks, n_iteration, exclude_dims=None):
    """
    Repeatedly bootstrap the provided arrays across the specified dimension(s)
    and stack the new arrays along a new "iteration" dimension. The
    boostrapping is done in a nested manner. I.e. bootstrap the first provided
    dimension, then for each bootstrapped sample along that dimenion, bootstrap
    the second provided dimension, then for each bootstrapped sample along that
    dimenion...

    Note, this function expands out the iteration dimension inside a
    universal function. However, this can generate very large chunks (it
    multiplies chunk size by the number of iterations) and it falls over for
    large numbers of iterations for reasons I don't understand. It is thus
    best to apply this function in blocks using `block_bootstrap`

    Parameters
    ----------
    objects : xarray Dataset(s)
        The data to bootstrap. Multiple datasets can be passed to be
        bootstrapped in the same way. Where multiple datasets are passed, all
        datasets need not contain all bootstrapped dimensions. However, because
        of the bootstrapping is applied in a nested manner, the dimensions in
        all input objects must also be nested. E.g., for `blocks.keys=['d1',
        'd2','d3']` an object with dimensions 'd1' and 'd2' is valid but an
        object with only dimension 'd2' is not.
    blocks : dict
        Dictionary of the dimension(s) to bootstrap and the block sizes to use
        along each dimension: {dim: blocksize}.
    n_iteration : int
        The number of times to repeat the bootstrapping
    exclude_dims : list of list
        List of the same length as the number of objects giving a list of
        dimensions specifed in `blocks` to exclude from each object. Default is
        to assume that no dimensions are excluded.
    """

    def _bootstrap(*arrays, indices):
        """Bootstrap the array(s) using the provided indices"""
        bootstrapped = [array[ind] for array, ind in zip(arrays, indices)]
        if len(bootstrapped) == 1:
            return bootstrapped[0]
        else:
            return tuple(bootstrapped)

    objects = list(objects)

    # Rename exclude_dims so they are not bootstrapped
    if exclude_dims is None:
        exclude_dims = [[] for _ in range(len(objects))]
    msg = (
        "exclude_dims should be a list of the same length as the number of "
        "objects containing lists of dimensions to exclude for each object"
    )
    assert isinstance(exclude_dims, list), msg
    assert len(exclude_dims) == len(objects), msg
    assert all(isinstance(x, list) for x in exclude_dims), msg
    renames = []
    for i, (obj, exclude) in enumerate(zip(objects, exclude_dims)):
        objects[i] = obj.rename(
            {d: f"dim{ii}" for ii, d in enumerate(exclude)},
        )
        renames.append({f"dim{ii}": d for ii, d in enumerate(exclude)})

    dim = list(blocks.keys())
    if isinstance(dim, str):
        dim = [dim]

    # Check that boostrapped dimensions are the same size on all objects
    for d in blocks.keys():
        dim_sizes = [o.sizes[d] for o in objects if d in o.dims]
        assert all(
            s == dim_sizes[0] for s in dim_sizes
        ), f"Block dimension {d} is not the same size on all input objects"

    # Get the sizes of the bootstrap dimensions
    sizes = None
    for obj in objects:
        try:
            sizes = OrderedDict(
                {d: (obj.sizes[d], b) for d, b in blocks.items()},
            )
            break
        except KeyError:
            pass
    if sizes is None:
        raise ValueError(
            "At least one input object must contain all dimensions in dim",
        )

    # Generate the random indices first so that we can be sure that each
    # dask chunk uses the same indices. Note, I tried using random.seed()
    # to achieve this but it was flaky. These are the indices to bootstrap
    # all objects.
    nested_indices = _n_nested_blocked_random_indices(sizes, n_iteration)

    # Need to expand the indices for broadcasting for each object separately
    # as each object may have different dimensions
    indices = []
    input_core_dims = []
    for obj in objects:
        available_dims = [d for d in dim if d in obj.dims]
        indices_to_expand = [nested_indices[key] for key in available_dims]

        # Check that dimensions are nested
        ndims = [i.ndim for i in indices_to_expand]
        # Start at 2 due to iteration dim
        if ndims != list(range(2, len(ndims) + 2)):
            raise ValueError("The dimensions of all inputs must be nested")

        indices.append(_expand_n_nested_random_indices(indices_to_expand))
        input_core_dims.append(available_dims)

    # Loop over objects because they may have non-matching dimensions and
    # we don't want to broadcast them as this will unnecessarily increase
    # chunk size for dask arrays
    result = []
    for obj, ind, core_dims in zip(objects, indices, input_core_dims):
        if isinstance(obj, xr.Dataset):
            # Assume all variables have the same dtype
            output_dtype = obj[list(obj.data_vars)[0]].dtype
        else:
            output_dtype = obj.dtype

        result.append(
            xr.apply_ufunc(
                _bootstrap,
                obj,
                kwargs=dict(
                    indices=[ind],
                ),
                input_core_dims=[core_dims],
                output_core_dims=[core_dims + ["iteration"]],
                dask="parallelized",
                dask_gufunc_kwargs=dict(
                    output_sizes={"iteration": n_iteration},
                ),
                output_dtypes=[output_dtype],
            )
        )

    # Rename excluded dimensions
    return tuple(res.rename(rename) for res, rename in zip(result, renames))


def block_bootstrap(*objects, blocks, n_iteration, exclude_dims=None):
    """
    Repeatedly bootstrap the provided arrays across the specified
    dimension(s) and stack the new arrays along a new "iteration"
    dimension. The boostrapping is done in a nested manner. I.e. bootstrap
    the first provided dimension, then for each bootstrapped sample along
    that dimenion, bootstrap the second provided dimension, then for each
    bootstrapped sample along that dimenion...

    Parameters
    ----------
    objects : xarray Dataset(s)
        The data to bootstrap. Multiple datasets can be passed to be
        bootstrapped in the same way. Where multiple datasets are passed, all
        datasets need not contain all bootstrapped dimensions. However, because
        of the bootstrapping is applied in a nested manner, the dimensions in
        all input objects must also be nested. E.g., for `blocks.keys=['d1',
        'd2','d3']` an object with dimensions 'd1' and 'd2' is valid but an
        object with only dimension 'd2' is not.
    blocks : dict
        Dictionary of the dimension(s) to bootstrap and the block sizes to use
        along each dimension: {dim: blocksize}.
    n_iteration : int
        The number of times to repeat the bootstrapping
    exclude_dims : list of list
        List of the same length as the number of objects giving a list of
        dimensions specifed in `blocks` to exclude from each object. Default
        is to assume that no dimensions are excluded.
    """
    # The fastest way to perform the iterations is to expand out the
    # iteration dimension inside the universal function (see
    # _iterative_bootstrap). However, this can generate very large chunks (it
    # multiplies chunk size by the number of iterations) and it falls over
    # for large numbers of iterations for reasons I don't understand. Thus
    # here we loop over blocks of iterations to generate the total number
    # of iterations.

    def _max_chunk_size_MB(ds):
        """
        Get the max chunk size in a dataset
        """

        def size_of_chunk(chunks, itemsize):
            """
            Returns size of chunk in MB given dictionary of chunk sizes
            """
            N = 1
            for value in chunks:
                if not isinstance(value, int):
                    value = max(value)
                N = N * value
            return itemsize * N / 1024**2

        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name="ds")

        chunks = []
        for var in ds.data_vars:
            da = ds[var]
            chunk = da.chunks
            itemsize = da.data.itemsize
            if chunk is None:
                # numpy array
                chunks.append((da.data.size * itemsize) / 1024**2)
            else:
                chunks.append(size_of_chunk(chunk, itemsize))
        return max(chunks)

    # Choose iteration blocks to limit chunk size on dask arrays
    if objects[
        0
    ].chunks:  # TO DO: this is not a very good check that input is dask array
        MAX_CHUNK_SIZE_MB = 200
        ds_max_chunk_size_MB = max(
            [_max_chunk_size_MB(obj) for obj in objects],
        )
        blocksize = int(MAX_CHUNK_SIZE_MB / ds_max_chunk_size_MB)
        if blocksize > n_iteration:
            blocksize = n_iteration
        if blocksize < 1:
            blocksize = 1
    else:
        blocksize = n_iteration

    bootstraps = []
    for _ in range(blocksize, n_iteration + 1, blocksize):
        bootstraps.append(
            _block_bootstrap(
                *objects,
                blocks=blocks,
                n_iteration=blocksize,
                exclude_dims=exclude_dims,
            )
        )

    leftover = n_iteration % blocksize
    if leftover:
        bootstraps.append(
            _block_bootstrap(
                *objects,
                blocks=blocks,
                n_iteration=leftover,
                exclude_dims=exclude_dims,
            )
        )

    return tuple(
        [
            xr.concat(
                b,
                dim="iteration",
                coords="minimal",
                compat="override",
            )
            for b in zip(*bootstraps)
        ]
    )
