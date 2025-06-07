from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import norm
import numpy as np
import pandas as pd

def ci_error(series, alpha=0.05):
    z = norm.ppf(1 - alpha/2)
    return z * series.std(ddof=1) / np.sqrt(series.count())


def heatmap(df, metric, stat, cmap="YlOrRd", transformation=100, fig_dir=None, title=None, unit=None, ylim=None, dpi: int = 300, font_sizes: dict = None, normalize_by_baseline=False):
    val_df   = df.loc[:, (metric, stat)]
    heat_df  = val_df.unstack(level="model")
    
    # Normalize by baseline if requested
    if normalize_by_baseline:
        if "Baseline" in heat_df.index:
            baseline_values = heat_df.loc["Baseline"]
            heat_df = heat_df.div(baseline_values, axis=1)
            transformation = 1  # No additional transformation needed
            unit = "× baseline"
        else:
            print(f"Warning: 'Baseline' method not found in data for {metric}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heat_df * transformation,
        annot=True, fmt=".2f" if not normalize_by_baseline else ".2f",
        cmap=cmap,
        cbar_kws={"label": f"{unit}"},
        vmin=ylim[0] if not normalize_by_baseline else None,
        vmax=ylim[1] if not normalize_by_baseline else None,
        annot_kws={"size": 14}  # Increase annotation text size
    )
    plt.title(f"{title}", fontsize=font_sizes["title"])
    plt.ylabel("Method", fontsize=font_sizes["ylabel"])  
    plt.xlabel("Model", fontsize=font_sizes["xlabel"])    
    plt.xticks(fontsize=font_sizes["xtick"], rotation=90, ha='right')             
    plt.yticks(fontsize=font_sizes["ytick"])             
    plt.tight_layout()
    if fig_dir is not None:
        suffix = "_normalized" if normalize_by_baseline else ""
        plt.savefig(fig_dir / f"heatmap_{metric}{suffix}.png",
                    dpi=dpi, bbox_inches="tight")
    plt.show()

# def heatmap(df, metric, stat, cmap="YlOrRd", transformation=100, fig_dir=None, title=None, unit=None, ylim=None, dpi: int = 300, font_sizes: dict = None):
#     val_df   = df.loc[:, (metric, stat)]
#     heat_df  = val_df.unstack(level="model")

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(
#         heat_df * transformation,
#         annot=True, fmt=".1f",
#         cmap=cmap,
#         cbar_kws={"label": f"{unit}"},
#         vmin=ylim[0],
#         vmax=ylim[1],
#         annot_kws={"size": 14}  # Increase annotation text size
#     )
#     plt.title(f"{title}", fontsize=font_sizes["title"])
#     plt.ylabel("Method", fontsize=font_sizes["ylabel"])  
#     plt.xlabel("Model", fontsize=font_sizes["xlabel"])    
#     plt.xticks(fontsize=font_sizes["xtick"])             
#     plt.yticks(fontsize=font_sizes["ytick"])             
#     plt.tight_layout()
#     if fig_dir is not None:
#         plt.savefig(fig_dir / f"heatmap_{metric}.png",
#                     dpi=dpi, bbox_inches="tight")
#     plt.show()
    





def plot_method_deltas(df, metrics, alpha=0.05, baseline_name="Baseline", spec_colors=None, dpi: int = 300, fig_dir: Path = None, metric_map: dict = None, title: str = None, font_sizes: dict = None):
    z = norm.ppf(1 - alpha / 2)
    baseline = df.loc[baseline_name]

    delta_rows = []
    for method in df.index:
        if baseline_name == method:
            continue
        row = {"method": method}
        for metric in metrics:
            mean_diff = df.loc[method, (metric, 'mean')] - baseline[(metric, 'mean')]
            se1 = df.loc[method, (metric, 'std')] / np.sqrt(df.loc[method, (metric, 'count')])
            se2 = baseline[(metric, 'std')] / np.sqrt(baseline[(metric, 'count')])
            ci = z * np.sqrt(se1**2 + se2**2)
            row[f"delta_{metric}"] = mean_diff * 100
            row[f"ci_error_{metric}"] = ci * 100
        delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)

    # Plot the deltas
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(delta_df["method"]))
    bar_width = 0.12

    colors = plt.get_cmap("Paired").colors
    for i, metric in enumerate(metrics):
        if spec_colors is not None and metric in spec_colors:
            color = spec_colors[metric]
        else:
            color = colors[i]
        deltas = delta_df[f"delta_{metric}"]
        errors = delta_df[f"ci_error_{metric}"]
        ax.bar(x + i * bar_width, deltas, bar_width, yerr=errors, capsize=5,
            label=metric_map[metric]["name"], color=color)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(delta_df["method"], rotation=45, fontsize=font_sizes["xtick"])
    ax.set_ylabel("Delta vs. Baseline (%)", fontsize=font_sizes["ylabel"])
    ax.set_title(f"{title}", fontsize=font_sizes["title"])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_sizes["legend"])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    if fig_dir is not None:
        plt.savefig(fig_dir / f"method_deltas.png",
                    dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_model_metrics(df, method, metrics, alpha=0.05, spec_colors=None, dpi: int = 300, fig_dir: Path = None, metric_map: dict = None, font_sizes: dict = None):
    z = norm.ppf(1 - alpha / 2)

    method_df = df.loc[method]
    delta_rows = []

    for model in method_df.index:
        row = {"model": model}
        for metric in metrics:
            mean = method_df.loc[model, (metric, 'mean')]
            std = method_df.loc[model, (metric, 'std')]
            count = method_df.loc[model, (metric, 'count')]
            ci = z * std / np.sqrt(count)
            row[f"mean_{metric}"] = mean * 100
            row[f"ci_{metric}"] = ci * 100
        delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(delta_df["model"]))
    bar_width = 0.12
    colors = plt.get_cmap("Paired").colors

    for i, metric in enumerate(metrics):
        color = spec_colors[metric] if spec_colors and metric in spec_colors else colors[i]
        means = delta_df[f"mean_{metric}"]
        errors = delta_df[f"ci_{metric}"]
        ax.bar(x + i * bar_width, means, bar_width, yerr=errors, capsize=5,
               label=metric_map[metric]["name"], color=color)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(delta_df["model"], rotation=45, fontsize=font_sizes["xtick"])
    ax.set_ylabel("Score (%)", fontsize=font_sizes["ylabel"])
    ax.set_title(f"Metrics for '{method}'", fontsize=font_sizes["title"])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_sizes["legend"])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', labelsize=font_sizes["ytick"])

    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(fig_dir / f"model_metrics_{method}.png",
                    dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_model_deltas_vs_baseline(df, method, metrics, alpha=0.05, baseline_name="Baseline", spec_colors=None, ylim=None, dpi: int = 300, fig_dir: Path = None, metric_map: dict = None, font_sizes: dict = None):
    z = norm.ppf(1 - alpha / 2)

    method_df = df.loc[method]
    baseline_df = df.loc[baseline_name]

    delta_rows = []

    for model in method_df.index:
        row = {"model": model}
        for metric in metrics:
            mean_diff = method_df.loc[model, (metric, 'mean')] - baseline_df.loc[model, (metric, 'mean')]
            se1 = method_df.loc[model, (metric, 'std')] / np.sqrt(method_df.loc[model, (metric, 'count')])
            se2 = baseline_df.loc[model, (metric, 'std')] / np.sqrt(baseline_df.loc[model, (metric, 'count')])
            ci = z * np.sqrt(se1**2 + se2**2)
            row[f"delta_{metric}"] = mean_diff * 100
            row[f"ci_{metric}"] = ci * 100
        delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(delta_df["model"]))
    bar_width = 0.08
    colors = plt.get_cmap("Paired").colors

    for i, metric in enumerate(metrics):
        color = spec_colors[metric] if spec_colors and metric in spec_colors else colors[i]
        means = delta_df[f"delta_{metric}"]
        errors = delta_df[f"ci_{metric}"]
        ax.bar(x + i * bar_width, means, bar_width, yerr=errors, capsize=5,
               label=metric_map[metric]["name"], color=color)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(delta_df["model"], rotation=45, fontsize=font_sizes["xtick"])
    ax.set_ylabel("Delta vs. Baseline (%)", fontsize=font_sizes["ylabel"])
    ax.set_title(f"'{method}' Compared to Baseline", fontsize=font_sizes["title"])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_sizes["legend"])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', labelsize=font_sizes["ytick"])
    
    if ylim is None:
        lo = (means - errors).min().min()
        hi = (means + errors).max().max()
        bound = 1.1 * max(abs(lo), abs(hi))
        ylim = (-bound, bound)
    ax.set_ylim(*ylim)

    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(fig_dir / f"model_deltas_vs_baseline_{method}.png",
                    dpi=dpi, bbox_inches="tight")
    plt.show()

    
def mean_ci_table(
    df: pd.DataFrame,
    *,
    precision: int = 1,
    percent_metrics: set[str] | None = None,
    metric_map: dict[str, dict[str, str]] | None = None
) -> pd.DataFrame:
    percent_metrics = set(percent_metrics or [])
    out = pd.DataFrame(index=df.index)

    for metric in df.columns.get_level_values(0).unique():
        if ('mean' not in df[metric]) or ('ci_error' not in df[metric]):
            continue                                # skip incomplete metrics

        mean = df[(metric, 'mean')].copy()
        ci   = df[(metric, 'ci_error')].copy()

        if metric in percent_metrics:
            mean *= 100
            ci   *= 100

        out[metric] = (
            mean.round(precision).map(lambda x: f"{x:.{precision}f}")
            + " ± "
            + ci.round(precision).map(lambda x: f"{x:.{precision}f}")
        )
    if metric_map is not None:
        rename_dict = {
            metric: metric_map[metric]["name"] 
            for metric in out.columns 
            if metric in metric_map
        }
        out = out.rename(columns=rename_dict)

    # put the columns in alphabetical order by metric for prettier output
    out = out.reindex(sorted(out.columns), axis=1)
    return out

def multi_mean_ci_table(
    # multi benchmarks
    df: pd.DataFrame,
    *,
    precision: int = 1,
    percent_metrics: set[str] | None = None,
    metric_map: dict[str, dict[str, str]] | None = None
) -> pd.DataFrame:
    percent_metrics = set(percent_metrics or [])
    out = pd.DataFrame(index=df.index)

    # Iterate over all (benchmark, metric) pairs in columns
    col_tuples = df.columns.to_flat_index()
    pairs = sorted(set((bench, metric) for bench, metric, stat in col_tuples))

    for bench, metric in pairs:
        # Check if both 'mean' and 'ci_error' exist for this (bench, metric)
        if ((bench, metric, 'mean') not in df.columns or
            (bench, metric, 'ci_error') not in df.columns):
            continue

        mean = df[(bench, metric, 'mean')].copy()
        ci   = df[(bench, metric, 'ci_error')].copy()

        if metric in percent_metrics:
            mean *= 100
            ci   *= 100

        # Use a tuple as the column name to preserve benchmark info
        out[(bench, metric)] = (
            mean.round(precision).map(lambda x: f"{x:.{precision}f}")
            + " ± "
            + ci.round(precision).map(lambda x: f"{x:.{precision}f}")
        )
    if metric_map is not None:
        # Only rename the metric part, keep benchmark
        out.columns = pd.MultiIndex.from_tuples([
            (bench, metric_map.get(metric, {"name": metric})["name"])
            if metric in metric_map else (bench, metric)
            for bench, metric in out.columns
        ])

    # Sort columns for pretty output
    out = out.reindex(sorted(out.columns), axis=1)
    return out
    
# def plot_method_deltas(df, metrics, alpha=0.05, baseline_name="Baseline", spec_colors=None, dpi: int = 300, fig_dir: Path = None, metric_map: dict = None, font_sizes: dict = None):
#         z = norm.ppf(1 - alpha / 2)
#         baseline = df.loc[baseline_name]

#         delta_rows = []
#         for method in df.index:
#             if baseline_name == method:
#                 continue
#             row = {"method": method}
#             for metric in metrics:
#                 mean_diff = df.loc[method, (metric, 'mean')] - baseline[(metric, 'mean')]
#                 se1 = df.loc[method, (metric, 'std')] / np.sqrt(df.loc[method, (metric, 'count')])
#                 se2 = baseline[(metric, 'std')] / np.sqrt(baseline[(metric, 'count')])
#                 ci = z * np.sqrt(se1**2 + se2**2)
#                 row[f"delta_{metric}"] = mean_diff * 100
#                 row[f"ci_error_{metric}"] = ci * 100
#             delta_rows.append(row)

#         delta_df = pd.DataFrame(delta_rows)

#         # Plot the deltas
#         fig, ax = plt.subplots(figsize=(14, 6))
#         x = np.arange(len(delta_df["method"]))
#         bar_width = 0.12

#         colors = plt.get_cmap("Paired").colors
#         for i, metric in enumerate(metrics):
#             if spec_colors is not None and metric in spec_colors:
#                 color = spec_colors[metric]
#             else:
#                 color = colors[i]
#             deltas = delta_df[f"delta_{metric}"]
#             errors = delta_df[f"ci_error_{metric}"]
#             ax.bar(x + i * bar_width, deltas, bar_width, yerr=errors, capsize=5,
#                 label=metric_map[metric]["name"], color=color)

#         ax.axhline(0, color='black', linewidth=0.8)
#         ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
#         ax.set_xticklabels(delta_df["method"], rotation=45, fontsize=font_sizes["xtick"])
#         ax.set_ylabel("Delta vs. Baseline (%)", fontsize=font_sizes["ylabel"])
#         ax.set_title(f"Safety Metrics by Method Compared to Baseline", fontsize=font_sizes["title"])
#         ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_sizes["legend"])
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
#         ax.tick_params(axis='y', labelsize=font_sizes["ytick"])

#         plt.tight_layout()
        
#         if fig_dir is not None:
#             plt.savefig(fig_dir / f"method_deltas.png",
#                         dpi=dpi, bbox_inches="tight")
#         plt.show()