# coding: utf-8
import os
import json
import logging
import re
import gemmi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy
import pandas
from .tools import filename_replace_char, json_numpy_converter


matplotlib.use("Agg")


def match_sigfigs(value, ref):
    """Format `value` to have the same number of significant figures as `ref`."""
    if ref == 0:
        return f"{value:.2f}"
    decimals = max(0, -int(numpy.floor(numpy.log10(abs(ref)))) + 1)
    return f"{value:.{decimals}f}"


def df_scatter_plot(
    df, x_cols, y_cols_groups, filename="scatter_plot.png", per_element=False
):
    """
    Create multiple scatter subplots from a DataFrame and save as PNG.

    Args:
        df (pandas.DataFrame or str or file stream): DataFrame or a CSV file.
        x_cols (list of str): Column names for x-axis for each subplot.
                              If per_element=True,
                              x_cols should be 'atomic_number' or 'atom_id'.
        y_cols_groups (list of list of str): Each sublist is a group of
                                             y columns for one subplot.
        filename (str): Output PNG filename.
        per_element (bool): If True, create separate plots for each atom.
                            Hydrogens will be excluded.
    """

    def extract_atomic_numbers(df, atom_id_col):
        # atom_id can look like "A/00 1/H19A", I want to extract "H"
        atom_labels = df[atom_id_col]
        proton_numbers = []
        for atom_id in atom_labels:
            atom_id_part = atom_id.split("/")[-1]
            match = re.match(r"^([A-Z][a-z]?)", atom_id_part)
            element = match.group(1)
            proton_number = gemmi.Element(element).atomic_number
            proton_numbers.append(proton_number)
        return numpy.array(proton_numbers)

    if isinstance(df, str):
        df = pandas.read_csv(df)

    n_subplots = len(y_cols_groups)
    ncols = min(n_subplots, 2)
    nrows = int(numpy.ceil(n_subplots / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    axes = axes.flatten()
    max_y = 0
    max_y_b = 0
    if not per_element:
        assert len(x_cols) == len(
            y_cols_groups
        ), "x and y_groups must have the same length"

        # Create a combined mask: True only where all x columns are not null
        # for finding x_max and y_max
        combined_mask = numpy.logical_and.reduce([df[col].notnull() for col in x_cols])
        df_filt = df[combined_mask]

        max_x = 0
        max_x_b = 0
        for data in x_cols:
            if "sigma_b" not in data:
                max_x = max(max_x, df_filt[data].max())
        for i, group in enumerate(y_cols_groups):
            for data in group:
                if "sigma_b" not in data:
                    max_y = max(max_y, df_filt[data].max())
    else:  # per_element
        if x_cols[0] != "atomic_number":  # typically x_cols[0] == "atom_id":
            # get atomic numbers from atom_id and save it as a new column
            df["atomic_number"] = extract_atomic_numbers(df, x_cols[0])
        df = df[df["atomic_number"] > 1]  # exclude hydrogens
        for i, group in enumerate(y_cols_groups):
            for data in group:
                if "sigma_b" not in data:
                    max_y = max(max_y, df[data].max())

    for i, y_group in enumerate(y_cols_groups):
        ax = axes[i]
        ax.set_axisbelow(True)
        for j, y_col in enumerate(y_group):
            if not per_element:
                ax.scatter(df[x_cols[i]], df[y_col], marker=".", label=y_col, alpha=0.5)
                ax.set_xlabel(x_cols[i])
                ax.set_title(x_cols[i] + " vs " + ", ".join(y_group))
                ax.legend(loc="upper left")
            else:  # per_element
                ax.scatter(
                    df["atomic_number"] + j * 0.2,
                    df[y_col],
                    marker=".",
                    label=y_col,
                    alpha=0.5,
                )
                ax.set_title(", ".join(y_group) + " per element")
                ax.legend()
            if not per_element:
                if "sigma_b" not in y_col:
                    ax.set_xlim(0, max_x * 1.05)
                    ax.set_ylim(0, max_y * 1.05)
                else:  # not per_element, "sigma_b"
                    if "sigma_b" in x_cols[i]:
                        max_x_b_candidate = df_filt[x_cols[i]].max()
                        max_x_b = max(max_x_b, max_x_b_candidate)
                        ax.set_xlim(0, max_x_b * 1.05)
                    max_y_b_candidate = df_filt[y_col].max()
                    max_y_b = max(max_y_b, max_y_b_candidate)
                    ax.set_ylim(0, max_y_b * 1.05)
            else:  # per_element  # do not touch xlim
                if "sigma_b" not in y_col:
                    ax.set_ylim(0, max_y * 1.05)
                else:  # per_element, "sigma_b"
                    max_y_b_candidate = df[y_col].max()
                    max_y_b = max(max_y_b, max_y_b_candidate)
                    ax.set_ylim(0, max_y_b * 1.05)
        if len(y_group) == 1:
            ax.set_ylabel(y_group[0])
        ax.grid()

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def analyse_distribution(
    values, xlabel, outlier_factor=3.0, idx=0, prefix="", filtered=False, save=True
):
    """
    Analyse a distribution of values and return summary statistics.

    Args:
        values (list or numpy array): The values to be analysed.
        xlabel (str): The label for the x-axis of the histogram.
        outlier_factor (float): The factor to determine outliers based on IQR.
        idx (int): Index for naming the output files (applies if not set to 0).
        prefix (str): Prefix for the output filenames.
        filtered (bool): Whether the values are already filtered,
                          which will be reflected in the output filenames.
        save (bool): Whether to save the histogram and outliers to CSV files.
    Returns:
        dict: A dictionary containing summary statistics,
              including histogram bins and counts.
        dict: A dictionary containing histogram bins and counts.
        pandas.DataFrame or None: A DataFrame containing outliers with their
                                  indices and values, or None if there are no outliers.
    """
    if len(values) == 0:
        return {}, {}, None
    mean = numpy.mean(values)
    stdev = numpy.std(values, ddof=1)
    median = numpy.median(values)
    q1 = numpy.percentile(values, 25)
    q3 = numpy.percentile(values, 75)
    iqr = q3 - q1
    # MAD to be consistent with stdev for normal distribution
    mad = numpy.median(numpy.abs(values - median)) * 1.4826
    min_val = numpy.min(values)
    max_val = numpy.max(values)
    n_values = len(values)
    distr_analysis_dict = {
        "mean": mean,
        "stdev": stdev,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "mad": mad,
        "min": min_val,
        "max": max_val,
        "n": n_values,
    }

    if not save:
        return distr_analysis_dict, {}, None

    counts, bins = numpy.histogram(values, bins="auto")

    values_arr = numpy.asarray(values)

    # IQR-based criterion
    threshold_low = q1 - outlier_factor * iqr
    threshold_high = q3 + outlier_factor * iqr
    mask_iqr = (values_arr < threshold_low) | (values_arr > threshold_high)

    # Additional median-relative criterion only for R1 ?
    # is_r1 = str(xlabel).strip().upper() == "R1"
    # if is_r1 and median > 0:
    #     rel_low = 0.7 * median
    #     rel_high = 1.3 * median
    #     mask_rel = (values_arr < rel_low) | (values_arr > rel_high)

    #     # Outlier must violate BOTH criteria
    #     outlier_mask = mask_iqr & mask_rel
    # else:
    #     outlier_mask = mask_iqr
    outlier_mask = mask_iqr

    outlier_idx = numpy.where(outlier_mask)[0]
    outliers = (
        None
        if outlier_idx.size == 0
        else pandas.DataFrame({"index": outlier_idx, "value": values_arr[outlier_idx]})
    )

    csv_values_filename = filename_replace_char(f"histogram_{xlabel}_values.csv")
    csv_histogram_filename = filename_replace_char(f"histogram_{xlabel}.csv")
    if filtered:
        csv_values_filename = csv_values_filename.replace(
            "histogram", "histogram_filtered"
        )
        csv_histogram_filename = csv_histogram_filename.replace(
            "histogram", "histogram_filtered"
        )
    if idx:
        csv_values_filename = f"{prefix}group{idx}_bootstrap_{csv_values_filename}"
        csv_histogram_filename = (
            f"{prefix}group{idx}_bootstrap_{csv_histogram_filename}"
        )
    else:
        csv_values_filename = f"{prefix}{csv_values_filename}"
        csv_histogram_filename = f"{prefix}{csv_histogram_filename}"

    df_values = pandas.DataFrame(values)
    df_values.to_csv(csv_values_filename, index=False, header=False)
    logging.info(f"Saved raw values to {csv_values_filename}")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    df_histogram = pandas.DataFrame(
        {
            "bin_start": bins[:-1],
            "bin_center": bin_centers,
            "bin_end": bins[1:],
            "count": counts,
        }
    )
    df_histogram.to_csv(csv_histogram_filename, index=False, header=False)
    logging.info(f"Saved histogram bins and counts to {csv_histogram_filename}")

    if outliers is not None and not filtered:
        csv_outliers_filename = filename_replace_char(
            f"histogram_{xlabel}_outliers.csv"
        )
        if idx:
            csv_outliers_filename = (
                f"{prefix}group{idx}_bootstrap_{csv_outliers_filename}"
            )
        else:
            csv_outliers_filename = f"{prefix}{csv_outliers_filename}"
        outliers.to_csv(csv_outliers_filename, index=False, header=False)
        logging.info(
            f"Saved {len(outliers)} outliers outside the interval"
            f" [{threshold_low:.4f}, {threshold_high:.4f}] to {csv_outliers_filename}"
        )

    return (
        distr_analysis_dict,
        {
            "bins": bins,
            "counts": counts.astype(int),
        },
        outliers,
    )


def plot_histogram(values, xlabel, ref={}, idx=0, prefix="", outlier_factor=3.0):
    """
    Plot a histogram of the data and save as PNG.

    Args:
        data (list or numpy array): Data to plot.
        xlabel (str): Label for the x-axis and the output file.
        ref (dict): Reference values for the plot {label: value}.
        idx (int): Index for naming the output file (applies if not set to 0).
        prefix (str): Prefix for the output filename.
        outlier_factor (float): Factor for determining outliers.
    """
    distr, hist, outliers = analyse_distribution(
        values, xlabel, idx=idx, prefix=prefix, outlier_factor=outlier_factor
    )
    min_val = distr["min"]
    max_val = distr["max"]
    if ref:
        for ref_value in ref.values():
            min_val = min(min_val, ref_value)
            max_val = max(max_val, ref_value)

    plt.figure(figsize=(8, 6))
    plt.bar(
        hist["bins"][:-1],
        hist["counts"],
        width=numpy.diff(hist["bins"]),
        align="edge",
        alpha=0.7,
    )
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.gca().yaxis.set_major_locator(
        ticker.MaxNLocator(integer=True)
    )  # Ensure integer y-axis labels
    buffer = (max_val - min_val) * 0.05  # 5% buffer around the data range
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.axvline(
        distr["mean"],
        color="blue",
        linestyle="--",
        label=(
            "Mean ± St.Dev. ="
            f" {match_sigfigs(distr['mean'], distr['stdev'])} ± {distr['stdev']:.2g}"
        ),
    )
    plt.axvline(
        distr["median"],
        color="green",
        linestyle="--",
        label=(
            "Median ± MAD ="
            f" {match_sigfigs(distr['median'], distr['mad'])} ± {distr['mad']:.2g}"
        ),
    )
    for stat_ref_label, stat_value in ref.items():
        plt.axvline(
            stat_value,
            color="orange",
            linestyle="--",
            label=(
                "From single refinement:"
                f" {stat_ref_label} = {match_sigfigs(stat_value, distr['stdev'])}"
            ),
        )
    plt.grid(axis="y", alpha=0.7)
    plt.tight_layout()
    plt.legend(title=f"n = {len(values)}")
    png_filename = filename_replace_char(f"histogram_{xlabel}.png")
    if idx:
        png_filename = f"{prefix}group{idx}_bootstrap_{png_filename}"
    else:
        png_filename = f"{prefix}{png_filename}"
    plt.savefig(png_filename)
    plt.close()
    logging.info(f"Saved histogram to {png_filename}")

    return distr, hist, outliers


def scatter_plot_histogram(
    x, y, label, stat_ref={}, idx=0, prefix="", filtered=True, outlier_factor=3.0
):
    """
    Plot a scatter plot of x vs y including histograms and save as PNG.

    Args:
        x (list or numpy array): Data for the x-axis.
        y (list or numpy array): Data for the y-axis.
        stat_ref (dict): Reference statistic for the plot {label: value}.
        label (str): Label for the axes and the output file.
        idx (int): Index for naming the output file (applies if not set to 0).
        prefix (str): Prefix for the output filename.
        filtered (bool): Whether the values are already filtered.
        outlier_factor (float): Factor for determining outliers.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    distr, hist, outliers = analyse_distribution(
        x,
        label,
        idx=idx,
        prefix=prefix,
        filtered=filtered,
        outlier_factor=outlier_factor,
    )
    distr_init, _, _ = analyse_distribution(
        y,
        label,
        idx=idx,
        prefix=prefix,
        filtered=filtered,
        save=False,
        outlier_factor=outlier_factor,
    )
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    if stat_ref:
        for ref_value in stat_ref.values():
            min_val = min(min_val, ref_value)
            max_val = max(max_val, ref_value)
    buffer = (max_val - min_val) * 0.05  # 5% buffer around the data range

    fig, axs = plt.subplot_mosaic(
        [["histx"], ["scatter"]],
        sharex=True,
        figsize=(7, 9),
        height_ratios=(1, 2),
        layout="constrained",
    )

    ax = axs["scatter"]
    ax_histx = axs["histx"]

    """
    binwidth = 0.25
    xymax = max(numpy.max(numpy.abs(x)), numpy.max(numpy.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = numpy.arange(-lim, lim + binwidth, binwidth)
    """

    ax.scatter(x, y, alpha=0.1)
    ax.plot(  # line y=x
        [min_val - buffer, max_val + buffer],
        [min_val - buffer, max_val + buffer],
        color="gray",
        linestyle="--",
        alpha=0.7,
    )
    ax.axvline(
        distr["mean"],
        color="blue",
        linestyle="--",
        label=(
            "Mean ± St.Dev. ="
            f" {match_sigfigs(distr['mean'], distr['stdev'])} ± {distr['stdev']:.2g}"
        ),
    )
    ax.axvline(
        distr["median"],
        color="green",
        linestyle="--",
        label=(
            "Median ± MAD ="
            f" {match_sigfigs(distr['median'], distr['mad'])} ± {distr['mad']:.2g}"
        ),
    )
    for stat_ref_label, stat_value in stat_ref.items():
        ax.axvline(
            stat_value,
            color="orange",
            linestyle="--",
            label=(
                "From single refinement:"
                f" {stat_ref_label} = {match_sigfigs(stat_value, distr['stdev'])}"
            ),
        )
    ax.set_xlabel(f"Refined {label}")
    ax.set_ylabel(f"Initial {label}")
    ax.set_xlim(min_val - buffer, max_val + buffer)
    ax.set_ylim(min_val - buffer, max_val + buffer)
    ax.grid(True, alpha=0.7)
    ax.legend(title=f"n = {len(x)}")

    ax_histx.bar(
        hist["bins"][:-1],
        hist["counts"],
        width=numpy.diff(hist["bins"]),
        align="edge",
        alpha=0.7,
    )
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.set_ylabel("Frequency")
    # plt.xlabel(xlabel)
    ax_histx.yaxis.set_major_locator(
        ticker.MaxNLocator(integer=True)
    )  # Ensure integer y-axis labels
    ax_histx.axvline(
        distr["mean"],
        color="blue",
        linestyle="--",
        label=(
            "Mean ± St.Dev. ="
            f" {match_sigfigs(distr['mean'], distr['stdev'])} ± {distr['stdev']:.2g}"
        ),
    )
    ax_histx.axvline(
        distr["median"],
        color="green",
        linestyle="--",
        label=(
            "Median ± MAD ="
            f" {match_sigfigs(distr['median'], distr['mad'])} ± {distr['mad']:.2g}"
        ),
    )
    for stat_ref_label, stat_value in stat_ref.items():
        ax_histx.axvline(
            stat_value,
            color="orange",
            linestyle="--",
            label=(
                "From single refinement:"
                f" {stat_ref_label} = {match_sigfigs(stat_value, distr['stdev'])}"
            ),
        )
    ax_histx.grid(axis="y", alpha=0.7)

    png_filename_base = filename_replace_char(f"scatter_histogram_{label}")
    if idx:
        png_filename_base = f"{prefix}group{idx}_bootstrap_{png_filename_base}"
    else:
        png_filename_base = f"{prefix}{png_filename_base}"
    if filtered:
        png_filename_base = f"{png_filename_base}_filtered"
    png_filename = f"{png_filename_base}.png"
    fig.savefig(png_filename)
    logging.info(f"Saved scatter plot with histogram to {png_filename}")
    plt.close(fig)

    return distr, distr_init, outliers


def bootstrap_analyse_stats(jsons, json_ref, idx=0, prefix=""):
    logging.info(f"Loading {len(jsons)} json files with statistics...")
    with open(jsons[0]) as f:
        data_first = json.load(f)
    stats_avail = data_first[-1]["data"]["summary"].keys()
    data_overall_dict = {stat: [] for stat in stats_avail}
    data_overall_init_dict = {stat: [] for stat in stats_avail}
    """
    stats_additional = []
    for stat in stats_avail:
        if "CC" in stat:
            data_overall_dict[f"R2_{stat}"] = []
            data_overall_init_dict[f"R2_{stat}"] = []
            stats_additional.append(f"R2_{stat}")
    """

    for json_file in jsons:
        with open(json_file) as f:
            data_loaded = json.load(f)
        for stat in stats_avail:
            data_overall_init_dict[stat].append(
                data_loaded[0]["data"]["summary"].get(stat, 0)
            )
            data_overall_dict[stat].append(
                data_loaded[-1]["data"]["summary"].get(stat, 0)
            )

            """
            if "CC" in stat:
                # also calculate R = sqrt(1 - CC^2)
                # per resolution bin and average it with weights
                if "work" in stat:
                    n_label = "n_work"
                elif "free" in stat:
                    n_label = "n_free"
                else:
                    n_label = "n_obs"
                stat_binned = stat.replace("avg", "")

                CCvalues_init = []
                n_obs_values_init = []
                for bin_data in data_loaded[0]["data"]["binned"]:
                    CCvalues_init.append(bin_data.get(stat_binned, 0))
                    n_obs_values_init.append(bin_data.get(n_label, 0))
                CCvalues_init = numpy.array(CCvalues_init)
                n_obs_values_init = numpy.array(n_obs_values_init)
                R2values_init = numpy.sqrt(1 - numpy.power(CCvalues_init, 2))
                R2value_init = (
                    numpy.average(R2values_init, weights=n_obs_values_init)
                    if n_obs_values_init.size > 0 and n_obs_values_init.sum() > 0
                    else 0
                )
                data_overall_init_dict[f"R2_{stat}"].append(R2value_init)

                CCvalues = []
                n_obs_values = []
                for bin_data in data_loaded[-1]["data"]["binned"]:
                    CCvalues.append(bin_data.get(stat_binned, 0))
                    n_obs_values.append(bin_data.get(n_label, 0))
                CCvalues = numpy.array(CCvalues)
                n_obs_values = numpy.array(n_obs_values)
                R2values = numpy.sqrt(1 - numpy.power(CCvalues, 2))
                R2value = (
                    numpy.average(R2values, weights=n_obs_values)
                    if n_obs_values.size > 0 and n_obs_values.sum() > 0
                    else 0
                )
                data_overall_dict[f"R2_{stat}"].append(R2value)
            """

    if json_ref and os.path.isfile(json_ref):
        # Find reference statistics for a structure refined in a classic way
        stats_ref_equivalents = {
            "R": "R",
            "R1": "R1",
            "CCFavg": "CCFavg",
            "CCIavg": "CCIavg",
            "R_llw>0": "Rwork",
            "R_llw=0": "Rfree",
            "CCF_llw>0_avg": "CCFworkavg",
            "CCF_llw=0_avg": "CCFfreeavg",
            "R1_llw>0": "R1work",
            "R1_llw=0": "R1free",
            "CCI_llw>0_avg": "CCIworkavg",
            "CCI_llw=0_avg": "CCIfreeavg",
        }
        data_ref = {}

        with open(json_ref) as f:
            data_ref_loaded = json.load(f)
            data_ref = data_ref_loaded[-1]["data"]["summary"]
            if data_ref:
                logging.info(f"Loaded reference statistics from {json_ref}")
            else:
                logging.warning(f"No summary statistics found in reference {json_ref}")
    else:
        logging.warning(
            f"Reference file with statistics not found {json_ref if json_ref else ''}"
        )

    distrs = {"init": {}, "final": {}}
    llweight_R1_outliers = []
    # for stat in list(stats_avail) + stats_additional:
    for stat in stats_avail:
        stat_ref = {}
        if (
            json_ref
            and os.path.isfile(json_ref)
            and data_ref
            and stat in stats_ref_equivalents.keys()
            and data_ref.get(stats_ref_equivalents[stat], None) is not None
        ):
            stat_ref = {
                stats_ref_equivalents[stat]: data_ref.get(
                    stats_ref_equivalents[stat], None
                )
            }
        distr, distr_init, llweight_outliers = scatter_plot_histogram(
            data_overall_dict[stat],
            data_overall_init_dict[stat],
            stat,
            stat_ref,
            idx,
            prefix,
            filtered=False,
        )
        distrs["final"][stat] = distr
        distrs["init"][stat] = distr_init
        if stat == "R1":
            if llweight_outliers is not None:
                llweight_R1_outliers = llweight_outliers["index"].tolist()
            else:
                llweight_R1_outliers = []

    json_filename = (
        f"{prefix}group{idx}_bootstrap_statistics.json"
        if idx
        else f"{prefix}bootstrap_statistics.json"
    )
    with open(json_filename, "w") as f:
        json.dump(distrs, f, indent=2, default=json_numpy_converter)
    logging.info(f"Saved statistics to {json_filename}")

    if llweight_R1_outliers:
        distrs_filtered = {"init": {}, "final": {}}
        data_overall_filtered_dict = {}
        data_overall_init_filtered_dict = {}
        logging.info(
            f"\nIdentified {len(llweight_R1_outliers) if llweight_R1_outliers else 0}"
            " R1 value outliers which will be exluded from the following analysis\n"
            f" {llweight_R1_outliers if llweight_R1_outliers else ''}"
        )
        for stat in stats_avail:
            data_overall_filtered_dict[stat] = [
                val
                for i, val in enumerate(data_overall_dict[stat])
                if i not in llweight_R1_outliers
            ]
            data_overall_init_filtered_dict[stat] = [
                val
                for i, val in enumerate(data_overall_init_dict[stat])
                if i not in llweight_R1_outliers
            ]
            if (
                json_ref
                and os.path.isfile(json_ref)
                and data_ref
                and stat in stats_ref_equivalents.keys()
                and data_ref.get(stats_ref_equivalents[stat], None) is not None
            ):
                stat_ref = {
                    stats_ref_equivalents[stat]: data_ref.get(
                        stats_ref_equivalents[stat], None
                    )
                }
            distr_filtered, distr_init_filtered, _ = scatter_plot_histogram(
                data_overall_filtered_dict[stat],
                data_overall_init_filtered_dict[stat],
                stat,
                stat_ref,
                idx,
                prefix,
                filtered=True,
            )
            distrs_filtered["final"][stat] = distr_filtered
            distrs_filtered["init"][stat] = distr_init_filtered

        json_filename_filtered = (
            f"{prefix}group{idx}_bootstrap_statistics_filtered.json"
            if idx
            else f"{prefix}bootstrap_statistics_filtered.json"
        )
        with open(json_filename_filtered, "w") as f:
            json.dump(distrs_filtered, f, indent=2, default=json_numpy_converter)
        logging.info(f"Saved filtered statistics to {json_filename_filtered}")

    logging.info("")
    return llweight_R1_outliers
