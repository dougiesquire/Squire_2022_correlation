"Some convenience plotting functions"

import numpy as np

from src import stats

from itertools import cycle

import matplotlib.pyplot as plt


def pearson_r_distributions(fcst, obsv, sample, block):
    """
    Plot the various distributions used for inference of the Pearson correlation
    coefficient

    Parameters
    ----------
    fcst : xarray object
        Array containing samples of hindcasts
    obsv : xarray object
        Array containing samples of observations
    sample : int
        The sample to plot the distributions for
    block : int
        The block size for the bootstrapped distribution
    """

    def _get_bin_centres(bins):
        return (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    alpha = 0.4
    colors = [f"C{i}" for i in range(10)]
    colorcycler = cycle(colors)

    r, r_bs = stats.bootstrap_pvalue(
        fcst.sel(sample=sample),
        obsv.sel(sample=sample),
        metric="pearson_r",
        metric_dim="time",
        blocks={"time": block, "member": 1},
        n_iteration=10_000,
        transform="Fisher_z",
        return_iterations=True,
    )
    pval_bs = stats._get_pval_from_bootstrap(r, r_bs, null=0, transform="Fisher_z")

    # t-test null distribution
    color = next(colorcycler)
    x = np.linspace(-1, 1, 100)
    t_dist = stats.student_t(fcst.sizes["time"], x)
    ax.fill_between(
        x, t_dist, color=color, alpha=alpha, label="t-test null distribution"
    )
    hatch = (np.sign(x) == np.sign(r.values)) & (abs(x) > abs(r.values))
    ax.fill_between(
        x[hatch],
        t_dist[hatch],
        color=color,
        facecolor="none",
        hatch="\\\\\\",
    )

    # Monte-Carlo null distribution
    color = next(colorcycler)
    r_mc = stats.pearson_r(fcst, obsv)
    pval_mc = 2 * ((np.sign(r_mc) == np.sign(r)) & (abs(r_mc) > abs(r))).mean("sample")
    h, b = np.histogram(
        r_mc,
        30,
        density=True,
    )
    bins, width = _get_bin_centres(b)
    ax.bar(
        bins,
        h,
        width=width,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        label="Monte Carlo null distribution",
    )
    hatch = (np.sign(bins) == np.sign(r.values)) & (abs(bins) > abs(r.values))
    ax.bar(
        bins[hatch],
        h[hatch],
        width=width,
        facecolor="none",
        edgecolor=color,
        hatch="///",
    )

    # Bootstrapped distribution
    color = next(colorcycler)
    h, b = np.histogram(
        r_bs,
        30,
        density=True,
    )
    bins, width = _get_bin_centres(b)
    ax.bar(
        bins,
        h,
        width=width,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        label=f"{block}-block bootstrapped distribution",
    )
    hatch = (np.sign(bins) != np.sign(r.values)) & (abs(bins) > 0)
    ax.bar(
        bins[hatch],
        h[hatch],
        width=width,
        facecolor="none",
        edgecolor=color,
        hatch="++",
    )

    ylim = ax.get_ylim()
    ax.plot(
        [r, r],
        ylim,
        color="k",
        linestyle="--",
        label="Sample correlation",
    )
    ax.set_ylim(ylim)

    _ = ax.text(
        0.01,
        0.68,
        (
            f"p-values\n"
            f"- {block}-block bootstrapped: {pval_bs.values:.4f}\n"
            f"- Monte Carlo: {pval_mc.values:.4f}"
        ),
        va="top",
        transform=ax.transAxes,
        fontsize=12,
    )

    ax.grid()
    ax.legend(loc="upper left")


def acf(*objects, headings, partial=False, panel_dim="rolling_mean", nlags=20):
    """
    Plot the autocorrelation function for multiple objects

    Parameters
    ----------
    objects : xarray objects
        The data to use to calculate and plot the autocorrelation function(s)
    headings : list of str
        Headings to use for each object in objects
    partial : boolean, optional
        If True, plot the partial autocorrelation function(s)
    panel_dim : str, optional
        If provided, different value along this dim will be plotted in different
        panels
    nlags : int, optional
        The number of lags to plot
    """
    assert len(objects) == len(headings)
    nrows = objects[0].sizes[panel_dim]
    ncols = 2 if any(["member" in ds.dims for ds in objects]) else 1

    fig = plt.figure(figsize=(14, 4 * nrows))
    axs = fig.subplots(nrows, ncols, sharex=True, sharey=True)
    if ncols == 1:
        axs = np.reshape(axs, (nrows, 1))

    alpha = 0.3
    quantiles = (0.05, 0.95)

    ACF_str = "pACF" if partial else "ACF"

    colorcycler = cycle([f"C{i}" for i in range(10)])
    for heading, obj in zip(headings, objects):
        color = next(colorcycler)
        for idx, val in enumerate(obj[panel_dim].values):
            if "member" in obj.dims:
                # Plot the mean acf per ensemble in one panel
                p = stats.acf(
                    obj.sel({panel_dim: val}).dropna("time"),
                    partial=partial,
                    nlags=nlags,
                )
                if "sample" in obj.dims:
                    pm = p.mean(["member", "sample"])
                    pr = (
                        p.mean("member").quantile(quantiles[0], "sample"),
                        p.mean("member").quantile(quantiles[1], "sample"),
                    )

                    axs[idx, 0].fill_between(
                        p.lag,
                        pr[0],
                        pr[1],
                        color=color,
                        edgecolor="none",
                        alpha=alpha,
                        label="_nolabel_",
                    )
                else:
                    pm = p.mean("member")

                axs[idx, 0].plot(
                    p.lag,
                    pm,
                    color=color,
                    label=f"{heading}",
                )
                axs[idx, 0].set_title(
                    f"Mean {ACF_str} per ensemble, {panel_dim} = {val}"
                )
                axs[idx, 0].set_ylabel("pACF" if partial else "ACF")

                # Plot the ensemble mean acf in the other panel
                p = stats.acf(
                    obj.sel({panel_dim: val}).dropna("time").mean("member"),
                    partial=partial,
                    nlags=nlags,
                )
                if "sample" in obj.dims:
                    pm = p.mean("sample")
                    pr = (
                        p.quantile(quantiles[0], "sample"),
                        p.quantile(quantiles[1], "sample"),
                    )

                    axs[idx, 1].fill_between(
                        p.lag,
                        pr[0],
                        pr[1],
                        color=color,
                        edgecolor="none",
                        alpha=alpha,
                        label="_nolabel_",
                    )
                else:
                    pm = p

                axs[idx, 1].plot(
                    p.lag,
                    pm,
                    color=color,
                    label=f"{heading}",
                )
                axs[idx, 1].set_title(f"Ensemble mean {ACF_str}, {panel_dim} = {val}")
                axs[idx, 1].grid(True)

            else:
                p = stats.acf(
                    obj.sel({panel_dim: val}).dropna("time"),
                    partial=partial,
                    nlags=nlags,
                )

                if "sample" in obj.dims:
                    pm = p.mean("sample")
                    pr = (
                        p.quantile(quantiles[0], "sample"),
                        p.quantile(quantiles[1], "sample"),
                    )

                    axs[idx, 0].fill_between(
                        p.lag,
                        pr[0],
                        pr[1],
                        color=color,
                        edgecolor="none",
                        alpha=alpha,
                        label="_nolabel_",
                    )
                else:
                    pm = p

                axs[idx, 0].plot(
                    p.lag,
                    pm,
                    color=color,
                    label=f"{heading}",
                )

            axs[idx, 0].set_ylabel(ACF_str)
            axs[idx, 0].grid(True)

    axs[0, 0].legend()

    fig.tight_layout()
