# coding: utf-8
import os
import numpy
import pandas
import gemmi
import logging
import warnings
import re
import json
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import concurrent.futures
from .tools import (
    CID2RefmacRestraint,
    write_bin_stats,
    write_mtz_from_df,
    makeAddressStr,
    filename_replace_char,
    json_numpy_converter,
    scale_reflections,
    select_CIDs_of_residues,
    CRA2CID,
)


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


def bootstrap_dataset(mtz_file, binner, seeds=[1001, 1002, 1003], labin=""):
    """
    Bootstrap the dataset from an MTZ file and save the results in new MTZ files.

    Args:
        mtz_file (str): Path to the input MTZ file.
        binner (gemmi.Binner): gemmi.Binner object for resolution binning.
        seeds (list of int): List of random seeds for bootstrapping.
        labin (str): Column label (e.g. `IMEAN,SIGIMEAN`)
            to apply `df.dropna(subset=[labin.split(",")[0]])`.
    Returns:
        list of str: List of output MTZ filenames created during bootstrapping.
    """

    def resample(n, seed=1001, column_name="llweight"):
        """
        Create a DataFrame`llweight` column for resampling.

        Args:
            n (int): Number of items to resample.
            seed (int): Random seed for reproducibility.
            column_name (str): Name of the column to create in the DataFrame.
        Returns:
            pandas.Series: Series with the bootstrap weights for each reflection.
        """
        rng = numpy.random.default_rng(seed)
        df_random = pandas.DataFrame(
            rng.integers(1, n + 1, size=n), columns=["index_resample"]
        )
        df_weight = (
            df_random.groupby(["index_resample"])
            .size()
            .reindex(range(1, n + 1), fill_value=0)
        )
        return df_weight.rename(column_name)

    logging.info(f"\nBootstrapping dataset {mtz_file}")
    mtzs_out = []
    mtz = gemmi.read_mtz_file(mtz_file)
    df = pandas.DataFrame(data=mtz.array, columns=mtz.column_labels())
    df = df.astype({name: "int32" for name in ["H", "K", "L"]})
    columns_dict = {
        col.label: col.type for col in mtz.columns if col.label not in ["H", "K", "L"]
    }

    # i_col = "IMEAN"  # can be just "I" after servalcat fw or sigmaa, or IMEAN?
    # dropping reflections can cause problems, let's save the filtered dataset as MTZ
    if labin and labin.split(",")[0] in df.columns:
        df = df.dropna(subset=[labin.split(",")[0]])
        # Save the filtered dataset as MTZ, preserving all original columns
        mtz_filtered_name = (
            f"{os.path.splitext(os.path.basename(mtz_file))[0]}_filtered.mtz"
        )
        write_mtz_from_df(df, mtz, columns=columns_dict, filename=mtz_filtered_name)
    else:
        warnings.warn(
            f"Column {labin} not found in MTZ file {mtz_file}. "
            f"Using all reflections for bootstrapping."
        )

    hkl_array = numpy.array(df[["H", "K", "L"]].values, numpy.int32)
    hkl_array = numpy.ascontiguousarray(hkl_array, dtype=numpy.int32)
    df["bin"] = binner.get_bins(hkl_array)
    # print("No. unique reflections:", len(df))
    # print(df.head(10))
    # print(df.describe())

    # df_bootstrap1_weight_master = pandas.DataFrame()
    completeness_list = []
    for i, seed in enumerate(seeds):
        df_bootstrap1_weight = pandas.concat(
            [resample(len(group), seed) for _, group in df.groupby("bin")],
            ignore_index=True,
        )
        # Merge columns H, K, L from df and llweight from df_bootstrap1_weight
        df_bootstrap1_weight_hkl = df[["H", "K", "L"]].copy()
        df_bootstrap1_weight_hkl = df_bootstrap1_weight_hkl.merge(
            df_bootstrap1_weight, left_index=True, right_index=True
        )
        weight_sum = df_bootstrap1_weight.sum()
        if weight_sum != len(df):
            logging.warning(
                f"Sum of weight coefficients {weight_sum} does not match the"
                f" number of reflections {len(df)}."
            )

        # TODO: FreeR_flag
        # Save the llweights in the MTZ file
        mtz_out_name = (
            f"{os.path.splitext(os.path.basename(mtz_file))[0]}_llweight{i}.mtz"
        )
        write_mtz_from_df(
            df_bootstrap1_weight_hkl,
            mtz,
            columns={"llweight": "I"},
            filename=mtz_out_name,
        )
        mtzs_out.append(mtz_out_name)

        # Compute completeness
        n_unique = len(
            df_bootstrap1_weight_hkl[df_bootstrap1_weight_hkl["llweight"] > 0]
        )
        n_unique_expected = gemmi.count_reflections(
            mtz.cell,
            mtz.spacegroup,
            mtz.resolution_high(),
            mtz.resolution_low(),
            unique=True,
        )
        completeness = n_unique / n_unique_expected
        completeness_list.append(completeness)

        """if i == 0:
            df_bootstrap1_weight_master = df_bootstrap1_weight.copy()
            df_bootstrap1_weight_master = df_bootstrap1_weight_master.rename(
                "llweight_0"
            ).to_frame()
        else:
            df_bootstrap1_weight_master[f"llweight_{i}"] = (
                df_bootstrap1_weight.values
            )

        # TODO: FreeR_flag
        columns = {i_col: "J", "SIG" + i_col: "Q", "llweight": "I"}
        if "FreeR_flag" in df.columns:
            columns["FreeR_flag"] = "I"
        df_bootstrap1_data = df.merge(
            df_bootstrap1_weight.copy(), left_index=True, right_index=True, how="left"
        )
        df_bootstrap1_data = df_bootstrap1_data[
            df_bootstrap1_data["llweight"] > 0
        ]
        mtz_out_name = f"{os.path.splitext(mtz_file)[0]}_bootstrap_data_{i}.mtz"
        write_mtz_from_df(df_bootstrap1_data, mtz, columns, filename=mtz_out_name)
        mtzs_out.append(mtz_out_name)"""

    """df_bootstrap1 = df.merge(
        df_bootstrap1_weight_master, left_index=True, right_index=True, how="left"
    )
    # print(df_bootstrap1.head(10))
    # print(df_bootstrap1.describe())
    # TODO: other columns than I and FreeRflag?
    columns = {i_col: "J", "SIG" + i_col: "Q"}
    if "FreeR_flag" in df.columns:
        columns["FreeR_flag"] = "I"
    columns.update({f"llweight_{i}": "I" for i in range(len(seeds))})
    write_mtz_from_df(
        df_bootstrap1,
        mtz,
        columns,
        filename=f"{os.path.splitext(mtz_file)[0]}_bootstrap.mtz",
    )"""

    completeness_mean = numpy.mean(completeness_list)
    completeness_std = numpy.std(completeness_list, ddof=1, mean=completeness_mean)
    logging.info(
        f"Completeness of bootstrap datasets:"
        f" {completeness_mean:.2%} ± {completeness_std:.2%}"
    )

    return mtzs_out


def unrestrain(geometry_objects_ref, structure_file):
    """
    Create a restraints file with unrestrained geometry for bootstrapping.

    Args:
        geometry_objects_ref (list of dict): Geometry objects from reference structure.
        structure_file (str): Path to the structure file (PDB or mmCIF).

    Returns:
        str: Filename of the created restraints file.
    """
    from itertools import combinations

    logging.info("Creating unrestrained geometry for Servalcat...")
    geometry_objects_ref_unre = [
        g for g in geometry_objects_ref if g["type"] != "occupancy"
    ]
    CIDs_of_residue_pairs_master = set()  # set of lists of 2or3or4 residues
    for geometry_object in geometry_objects_ref_unre:
        assert len(geometry_object["values"]) == 1
        CIDs_of_residues = select_CIDs_of_residues(geometry_object)
        assert len(CIDs_of_residues) >= 2, f"Invalid geometry object: {geometry_object}"
        # Generate all unique pairs of different entries from a set/list.
        CIDs_of_residue_pairs = list(combinations(list(CIDs_of_residues), 2))
        CIDs_of_residue_pairs = [
            pair for pair in CIDs_of_residue_pairs if pair[0] != pair[1]
        ]
        CIDs_of_residue_pairs_master.update(CIDs_of_residue_pairs)

    # for each residue pair, create gemmi selections
    # to get pair of lists of atom CRAs
    # and then list of pairs of atoms, exclude covalently bonded atoms
    st = gemmi.read_structure(structure_file)
    restraints_lines = []
    unrestrained_distances = []
    covalently_bonded = dict()
    try:
        logging.info("Preparing topology for covalent bond detection...")
        monlib = gemmi.MonLib()
        topology = gemmi.prepare_topology(
            st, monlib, h_change=gemmi.HydrogenChange.Remove
        )
        for bond in topology.bonds:
            atom1, atom2 = bond.atoms
            covalently_bonded.setdefault(str(atom1), set()).add(str(atom2))
            covalently_bonded.setdefault(str(atom2), set()).add(str(atom1))
    except Exception as e:
        logging.warning(
            f"Could not prepare topology for {structure_file}. So during unrestraining,"
            " covalently bonded atoms could be included which is not optimal."
            f"\n{e}"
        )
        covalently_bonded = dict()
    for CID_residues_pair in CIDs_of_residue_pairs_master:
        CID_residue1_raw, CID_residue2_raw = CID_residues_pair
        CID_residue1 = CID_residue1_raw.split("@")[0]
        if len(CID_residue1_raw.split("@")) > 1:
            CID_residue1_symm = CID_residue1_raw.split("@")[1]
        else:
            CID_residue1_symm = False
        CID_residue2 = CID_residue2_raw.split("@")[0]
        if len(CID_residue2_raw.split("@")) > 1:
            CID_residue2_symm = CID_residue2_raw.split("@")[1]
        else:
            CID_residue2_symm = False
        sel1 = gemmi.Selection(CID_residue1)
        sel1_model = sel1.copy_model_selection(st[0])
        cra_atoms1 = [cra for cra in sel1_model.all() if not cra.atom.is_hydrogen()]
        sel2 = gemmi.Selection(CID_residue2)
        sel2_model = sel2.copy_model_selection(st[0])
        cra_atoms2 = [cra for cra in sel2_model.all() if not cra.atom.is_hydrogen()]
        for cra1 in cra_atoms1:
            for cra2 in cra_atoms2:
                atom1_CID = CRA2CID(cra1)
                atom2_CID = CRA2CID(cra2)
                if covalently_bonded:
                    if str(cra2.atom) in covalently_bonded.get(
                        str(cra1.atom), set()
                    ) or str(cra1.atom) in covalently_bonded.get(str(cra2.atom), set()):
                        restraints_lines.append(
                            "# Skipping covalent bond between"
                            f" {atom1_CID} and {atom2_CID}"
                        )
                        continue  # skip covalently bonded atoms
                if CID_residue1_symm or CID_residue2_symm:
                    if len(CID_residue1_raw.split("@")) > 1:
                        atom2_CID += f"@{CID_residue1_symm}"
                    elif len(CID_residue2_raw.split("@")) > 1:
                        atom2_CID += f"@{CID_residue2_symm}"
                unrestrained_distance = {
                    "type": "distance",
                    "atom1": atom1_CID,
                    "atom2": atom2_CID,
                    "atom3": "",
                    "atom4": "",
                }
                unrestrained_distances.append(unrestrained_distance)

    for unrestrained_distance in unrestrained_distances:
        restraint = CID2RefmacRestraint(unrestrained_distance)
        restraints_lines.append(restraint)

    for geometry_object in geometry_objects_ref_unre:
        assert len(geometry_object["values"]) == 1
        # if geometry_object["type"] == "distance":
        #     continue  # already processed above except of covalent bonds and hydrogens
        restraint = CID2RefmacRestraint(geometry_object)
        restraints_lines.append(restraint)

    restraints_filename = "restraints.txt"
    with open(restraints_filename, "w") as f:
        for line in restraints_lines:
            f.write(line + "\n")
    logging.info(f"Saved unrestrained geometry to {restraints_filename}")
    return restraints_filename


def analyse_distribution(
    values, xlabel, outlier_factor=2.0, idx=0, prefix="", filtered=False, save=True
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

    # Outliers are defined as values outside of outlier_factor*IQR from the quartiles
    threshold_low = q1 - outlier_factor * iqr
    outliers_low_dict = {i: values[i] for i in numpy.where(values < threshold_low)[0]}
    outliers_low = pandas.DataFrame(
        list(outliers_low_dict.items()), columns=["index", "value"]
    )
    threshold_high = q3 + outlier_factor * iqr
    outliers_high_dict = {i: values[i] for i in numpy.where(values > threshold_high)[0]}
    outliers_high = pandas.DataFrame(
        list(outliers_high_dict.items()), columns=["index", "value"]
    )

    # Combine outliers, handling empty DataFrames to avoid FutureWarning
    if outliers_low.empty and outliers_high.empty:
        outliers = None
    elif outliers_low.empty:
        outliers = outliers_high
    elif outliers_high.empty:
        outliers = outliers_low
    else:
        outliers = pandas.concat([outliers_low, outliers_high], ignore_index=True)

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


def plot_histogram(values, xlabel, ref={}, idx=0, prefix=""):
    """
    Plot a histogram of the data and save as PNG.

    Args:
        data (list or numpy array): Data to plot.
        xlabel (str): Label for the x-axis and the output file.
        ref (dict): Reference values for the plot {label: value}.
        idx (int): Index for naming the output file (applies if not set to 0).
        prefix (str): Prefix for the output filename.
    """
    distr, hist, _ = analyse_distribution(values, xlabel, idx=idx, prefix=prefix)
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


def scatter_plot_histogram(x, y, label, stat_ref={}, idx=0, prefix="", filtered=True):
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
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    distr, hist, outliers = analyse_distribution(
        x, label, idx=idx, prefix=prefix, filtered=filtered
    )
    distr_init, _, _ = analyse_distribution(
        y, label, idx=idx, prefix=prefix, filtered=filtered, save=False
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


def calculate_angle(atom1_pos, atom2_pos, atom3_pos, degrees=True):
    """
    Calculate the angle ABC (at atom2) from gemmi.Position objects.
    """
    A = numpy.array(atom1_pos.tolist())
    B = numpy.array(atom2_pos.tolist())
    C = numpy.array(atom3_pos.tolist())
    BA = A - B
    BC = C - B
    BA_norm = numpy.linalg.norm(BA)
    BC_norm = numpy.linalg.norm(BC)

    if BA_norm == 0 or BC_norm == 0:
        raise ValueError("One of the vectors has zero length.")

    cos_theta = numpy.dot(BA, BC) / (BA_norm * BC_norm)
    cos_theta = numpy.clip(cos_theta, -1.0, 1.0)  # Numerical stability
    theta_rad = numpy.arccos(cos_theta)

    return numpy.degrees(theta_rad) if degrees else theta_rad


def calculate_torsion_angle(atom1_pos, atom2_pos, atom3_pos, atom4_pos, degrees=True):
    """
    Calculates the torsion (dihedral) angle between four gemmi.Position objects.
    Angle is defined by A–B–C–D.
    """
    A = numpy.array(atom1_pos.tolist())
    B = numpy.array(atom2_pos.tolist())
    C = numpy.array(atom3_pos.tolist())
    D = numpy.array(atom4_pos.tolist())
    b1 = B - A
    b2 = C - B
    b3 = D - C

    # Normalize b2 to avoid length effects
    b2_norm = numpy.linalg.norm(b2)
    if b2_norm == 0:
        raise ValueError("Vector B–C has zero length.")
    b2_unit = b2 / b2_norm

    # Normal vectors to the planes
    n1 = numpy.cross(b1, b2)  # normal to plane A–B–C
    n2 = numpy.cross(b2, b3)  # normal to plane B–C–D
    n1_norm = numpy.linalg.norm(n1)
    n2_norm = numpy.linalg.norm(n2)

    if n1_norm == 0 or n2_norm == 0:
        raise ValueError("Colinear atoms - torsion angle undefined.")

    n1_unit = n1 / n1_norm
    n2_unit = n2 / n2_norm

    m1 = numpy.cross(n1_unit, b2_unit)
    x = numpy.dot(n1_unit, n2_unit)
    y = numpy.dot(m1, n2_unit)

    angle_rad = numpy.arctan2(y, x)
    angle_rad = -angle_rad  # why?
    return numpy.degrees(angle_rad) if degrees else angle_rad


def circular_mean_deg(angles_deg):
    angles_rad = numpy.deg2rad(angles_deg)
    mean_angle_rad = numpy.arctan2(
        numpy.mean(numpy.sin(angles_rad)), numpy.mean(numpy.cos(angles_rad))
    )
    mean_angle_deg = numpy.rad2deg(mean_angle_rad)
    # Ensure result is in [0, 360)
    if mean_angle_deg >= 180:
        mean_angle_deg -= 360
    elif mean_angle_deg < -180:
        mean_angle_deg += 360
    return mean_angle_deg


def circular_std_deg(angles_deg):
    angles_rad = numpy.deg2rad(angles_deg)
    R = numpy.sqrt(
        numpy.mean(numpy.sin(angles_rad)) ** 2 + numpy.mean(numpy.cos(angles_rad)) ** 2
    )
    # Circular standard deviation in degrees
    return numpy.rad2deg(numpy.sqrt(-2 * numpy.log(R)))


def apply_symmetry_and_translation(st, op, t, pos_cart):
    # Convert Cartesian to fractional
    pos_frac = st.cell.fractionalize(pos_cart)
    # Apply symmetry op (returns list), then wrap as Fractional
    sym_applied = gemmi.Fractional(*op.apply_to_xyz(pos_frac))
    new_frac = sym_applied + gemmi.Fractional(t[0], t[1], t[2])
    # Convert back to Cartesian
    pos_cart_new = st.cell.orthogonalize(new_frac)
    return pos_cart_new


def select_cids_for_geometry_analysis(geometry_cids_file):
    """
    Read a file with atom CIDs for geometry analysis.

    Example file content:
    //AAA/401/O3
    //AAA/401/O3 //AAA/228/NE2
    //AAA/401/N2 //AAA/228/OE1
    //AAA/401/O2 //AAA/57/OG1@2665
    //AAA/401/O1 //AAA/401/O2 //AAA/254/ND2
    //AAA/176/NE //AAA/176/CZ //AAA/176/NH2 //AAA/401/O4

    Each line has 1, 2, 3, or 4 atom CIDs for
    occupancy, bond, angle, or torsion analysis.
    The CID format is extended to allow specifying symmetry mates (e.g. @2665)

    Returns:
        list of dict: Each dict has keys: atom1, atom2, atom3, atom4, type, values
    """
    objects_geom = []
    if geometry_cids_file and os.path.isfile(geometry_cids_file):
        with open(geometry_cids_file) as f:
            for line in f:
                if not line.strip():
                    continue
                cids = line.split()
                assert len(cids) in [
                    1,
                    2,
                    3,
                    4,
                ], f"Invalid line in {geometry_cids_file}: {line}"
                object_geom = {}
                object_geom["values"] = []
                object_geom["atom1"] = cids[0]
                object_geom["atom2"] = ""
                object_geom["atom3"] = ""
                object_geom["atom4"] = ""
                if len(cids) == 1:
                    object_geom["type"] = "occupancy"
                elif len(cids) == 2:
                    object_geom["atom2"] = cids[1]
                    object_geom["type"] = "distance"
                elif len(cids) == 3:
                    object_geom["atom2"] = cids[1]
                    object_geom["atom3"] = cids[2]
                    object_geom["atom4"] = ""
                    object_geom["type"] = "angle"
                elif len(cids) == 4:
                    object_geom["atom2"] = cids[1]
                    object_geom["atom3"] = cids[2]
                    object_geom["atom4"] = cids[3]
                    object_geom["type"] = "torsion"
                objects_geom.append(object_geom)

    return objects_geom


def geometry_analysis_load(st, objects_cids):
    """
    For a given structure, calculate bond lengths, angles, and torsions
    for the specified atom CIDs.
    `objects_cids` is a list of dicts with keys:
    atom1, atom2, atom3, atom4, type, values
    The results are appended to the 'values' list in each dict.
    """

    for object_geom in objects_cids:

        if object_geom["type"] == "occupancy":
            sel = gemmi.Selection(f"{object_geom['atom1'].split('@')[0]}")
            sel_model = sel.copy_model_selection(st[0])
            assert sel_model.count_atom_sites() == 1, (
                f"{object_geom['atom1']} does not select exactly one atom but"
                f" {sel_model.count_atom_sites()} atoms."
            )
            occ = sel.first(st)[1].atom.occ
            object_geom["values"].append(occ)
            continue

        pos1 = get_pos_from_cid(st, object_geom["atom1"])
        pos2 = get_pos_from_cid(st, object_geom["atom2"], pos1)

        if object_geom["type"] == "distance":
            dist = pos1.dist(pos2)
            object_geom["values"].append(dist)

        elif object_geom["type"] in ["angle", "torsion"]:
            pos3 = get_pos_from_cid(st, object_geom["atom3"], pos2)

            if object_geom["type"] == "angle":
                angle = calculate_angle(pos1, pos2, pos3)
                object_geom["values"].append(angle)

            elif object_geom["type"] == "torsion":
                pos4 = get_pos_from_cid(st, object_geom["atom4"], pos3)
                torsion = calculate_torsion_angle(pos1, pos2, pos3, pos4)
                object_geom["values"].append(torsion)

    return objects_cids


def get_pos_from_cid(st, cid, pos_reference=None):
    """
    Get Cartesian position of an atom from its CID in a structure.
    Extended CID format: /model/chain/residue/atom[@symop]
    where symop is e.g. 2665 for space group operation No. 2
    and translation (+1,+1,0), i.e. -x+1, -y+1, z in I222.
    """
    sel = gemmi.Selection(f"{cid.split('@')[0]}")
    sel_model = sel.copy_model_selection(st[0])
    assert sel_model.count_atom_sites() == 1, (
        f"{cid} does not select exactly one atom but"
        f" {sel_model.count_atom_sites()} atoms."
    )
    if pos_reference and "@" not in cid:
        # get position corresponding to a symmetry mate with
        # the shortest distance
        pos_candidates = []
        for i_symm_op in range(len(list(st.find_spacegroup().operations()))):
            pos = st.cell.find_nearest_pbc_position(
                pos_reference, sel.first(st)[1].atom.pos, i_symm_op
            )
            pos_candidates.append(pos)
        pos = min(pos_candidates, key=lambda p: pos_reference.dist(p))
    elif "@" in cid:
        # get the distance of an explicitly given symmetry mate
        symop_str = cid.split("@")[-1]
        assert len(symop_str) == 4, (
            "Symmetry operation format must have 4 digits (e.g. 2665 or 1555),"
            f" this is invalid: {symop_str}"
        )
        try:
            symop_no = int(symop_str[0])
            op = list(st.find_spacegroup().operations())[symop_no - 1]
            t = (
                int(symop_str[1]) - 5,
                int(symop_str[2]) - 5,
                int(symop_str[3]) - 5,
            )
            pos = sel.first(st)[1].atom.pos
            pos = gemmi.Position(pos.x, pos.y, pos.z)  # Make a copy
            pos = apply_symmetry_and_translation(st, op, t, pos)
        except Exception as e:
            raise ValueError(
                "Symmetry operation format must have 4 digits (e.g. 2665 or 1555),"
                f" this is invalid: {cid}.\n{e}"
            )
    else:
        pos = sel.first(st)[1].atom.pos
    return pos


def get_smcif_tables(smcif_block):
    """Extract relevant tables from a small molecule CIF block and their columns."""

    def get_table_and_columns(smcif_block, col_names):
        table = smcif_block.find(col_names)
        if not table:
            logging.warning(f"Table not found in small molecule CIF block. {col_names}")
            return None, []
        columns = []
        for col in col_names:
            col = col.strip("?")  # Remove '?' prefix if present
            try:
                column = table.find_column(col)
                columns.append(column)
            except (RuntimeError, IndexError):
                if "symmetry" in col:
                    columns.append([None] * len(table))
                else:
                    logging.warning(
                        f"Column not found in small" f" molecule CIF block: {col}"
                    )
        return table, columns

    coords_cols = [
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_U_iso_or_equiv",
        "?_atom_site_type_symbol",
        "?_atom_site_occupancy",
    ]
    u_aniso_cols = [
        "_atom_site_aniso_label",
        "_atom_site_aniso_U_11",
        "_atom_site_aniso_U_22",
        "_atom_site_aniso_U_33",
        "_atom_site_aniso_U_12",
        "_atom_site_aniso_U_13",
        "_atom_site_aniso_U_23",
    ]
    bond_cols = [
        "_geom_bond_atom_site_label_1",
        "_geom_bond_atom_site_label_2",
        "_geom_bond_distance",
        "?_geom_bond_site_symmetry_2",
    ]
    angle_cols = [
        "_geom_angle_atom_site_label_1",
        "_geom_angle_atom_site_label_2",
        "_geom_angle_atom_site_label_3",
        "_geom_angle",
        "?_geom_angle_site_symmetry_1",
        "?_geom_angle_site_symmetry_3",
    ]
    torsion_cols = [
        "_geom_torsion_atom_site_label_1",
        "_geom_torsion_atom_site_label_2",
        "_geom_torsion_atom_site_label_3",
        "_geom_torsion_atom_site_label_4",
        "_geom_torsion",
        "?_geom_torsion_site_symmetry_1",
        "?_geom_torsion_site_symmetry_2",
        "?_geom_torsion_site_symmetry_3",
        "?_geom_torsion_site_symmetry_4",
    ]

    coords_table, coords_columns = get_table_and_columns(smcif_block, coords_cols)
    u_aniso_table, u_aniso_columns = get_table_and_columns(smcif_block, u_aniso_cols)
    bond_table, bond_columns = get_table_and_columns(smcif_block, bond_cols)
    angle_table, angle_columns = get_table_and_columns(smcif_block, angle_cols)
    torsion_table, torsion_columns = get_table_and_columns(smcif_block, torsion_cols)

    return (
        (coords_table, coords_columns),
        (u_aniso_table, u_aniso_columns),
        (bond_table, bond_columns),
        (angle_table, angle_columns),
        (torsion_table, torsion_columns),
    )


def extract_value_and_stdev(value):
    """
    Extract base value and standard deviation from
    e.g. '-0.1234(5)' -> (-0.1234, 0.0005)
    """
    match = re.match(r"(-?[0-9.]+)\((\d+)\)", value)
    if match:
        base, sigma_digits = match.groups()
        base_value = float(base)

        # Calculate decimal places for scaling of sigma
        base_parts = base.split(".")
        decimal_places = len(base_parts[1]) if len(base_parts) > 1 else 0
        stdev = float(sigma_digits) * (10**-decimal_places)

        return base_value, stdev
    else:
        return float(re.sub(r"\(.*\)", "", value)), None


def collect_geometry_lists(
    table,
    atom_cols,
    symmetry_cols=[],
    value_sigma_cols=[],
    value_sigma_cols_names=[],
    elem_col="",
):
    """Collect atom lists (for bonds, angles, torsions).
    Do not include atoms from symmetry-related molecules.
    If elem_col is provided, hydrogen atoms will be excluded from the analysis.

    Args:
        table: CIF table to process.
        atom_cols: List of columns with atom labels.
        symmetry_cols: List of columns with symmetry information (optional).
        value_sigma_cols: List of columns with values and standard deviations (optional)
        value_sigma_cols_names: List of base names for value/sigma columns (optional).
        elem_col: Column with chmemical element (optional).
    Returns:
        List of dicts with geometry information.
    """
    j_idx_filtered = [
        j
        for j in range(len(table))
        if (
            (not symmetry_cols or all(col[j] in [".", None] for col in symmetry_cols))
            and (not elem_col or elem_col[j] != "H")
        )
    ]
    geom_list = [
        {f"atom{i + 1}": atom_cols[i][j_idx] for i in range(len(atom_cols))}
        for j_idx in j_idx_filtered
    ]

    if value_sigma_cols:
        for entry, j_idx in zip(geom_list, j_idx_filtered):
            for i in range(len(value_sigma_cols)):
                value, sigma = extract_value_and_stdev(value_sigma_cols[i][j_idx])
                entry[f"{value_sigma_cols_names[i]}_deposit"] = value
                entry[f"sigma_{value_sigma_cols_names[i]}_deposit"] = sigma

    return geom_list


def collect_values_smcif(smcif, skip_hydrogen=True):
    """
    Collect values about geometry from a small molecule CIF file from SHELX.

    Args:
        smcif (str): Path to the small molecule CIF file.

    Returns:
        tuple: (atoms_list, occ_list,
               u_aniso_list, bonds_list, angles_list, torsions_list)
               where each list contains dicts with geometry information.
    """
    smcif_block = gemmi.cif.read(smcif).sole_block()
    value_shelx_res_file = smcif_block.find_value("_shelx_res_file")
    value_computing_structure_refinement = smcif_block.find_value(
        "_computing_structure_refinement"
    )
    atoms_list = occ_list = bonds_list = u_aniso_list = angles_list = torsions_list = []

    if value_shelx_res_file or (
        value_computing_structure_refinement
        and "shelx" in value_computing_structure_refinement.lower()
    ):
        (
            (table_coords, coords_cols),
            (table_u_aniso, u_aniso_cols),
            (table_bond, bond_columns),
            (table_angle, angle_columns),
            (table_torsion, torsion_columns),
        ) = get_smcif_tables(smcif_block)

        (
            atom_col,
            x_fract_col,
            y_fract_col,
            z_fract_col,
            u_iso_col,
            elem_col,
            occ_col,
        ) = coords_cols
        if not skip_hydrogen:
            elem_col = ""
        atoms_list = collect_geometry_lists(
            table_coords,
            [atom_col],
            [],
            [x_fract_col, y_fract_col, z_fract_col, u_iso_col],
            ["x_frac", "y_frac", "z_frac", "u_iso"],
            elem_col=elem_col,  # exclude hydrogens
        )

        st = gemmi.read_small_structure(smcif)
        st_smcif_cras = [
            cra for cra in st[0].all() if not (skip_hydrogen and cra.atom.is_hydrogen())
        ]
        assert len(atoms_list) == len(st_smcif_cras), (
            f"Number of atoms in coordinates table ({len(atoms_list)}) does not match"
            f" number of atoms in structure {smcif} ({len(st_smcif_cras)})"
            f" while hydrogens are {'excluded' if skip_hydrogen else 'included'}."
        )
        for i in range(len(atoms_list)):
            # Convert x y z to Cartesian coordinates
            frac = gemmi.Fractional(
                atoms_list[i]["x_frac_deposit"],
                atoms_list[i]["y_frac_deposit"],
                atoms_list[i]["z_frac_deposit"],
            )
            cart = st.cell.orthogonalize(frac)
            atoms_list[i]["x_deposit"] = cart.x
            atoms_list[i]["y_deposit"] = cart.y
            atoms_list[i]["z_deposit"] = cart.z
            atoms_list[i]["sigma_x_deposit"] = (
                st.cell.a * atoms_list[i]["sigma_x_frac_deposit"]
                if atoms_list[i]["sigma_x_frac_deposit"] is not None
                else None
            )
            atoms_list[i]["sigma_y_deposit"] = (
                st.cell.b * atoms_list[i]["sigma_y_frac_deposit"]
                if atoms_list[i]["sigma_y_frac_deposit"] is not None
                else None
            )
            atoms_list[i]["sigma_z_deposit"] = (
                st.cell.c * atoms_list[i]["sigma_z_frac_deposit"]
                if atoms_list[i]["sigma_z_frac_deposit"] is not None
                else None
            )

            # Compute B_iso and sigma_B_iso:  B = 8 pi^2 U
            atoms_list[i]["b_iso_deposit"] = (
                8 * numpy.pi**2 * atoms_list[i]["u_iso_deposit"]
            )
            atoms_list[i]["sigma_b_iso_deposit"] = (
                8 * numpy.pi**2 * atoms_list[i]["sigma_u_iso_deposit"]
                if atoms_list[i]["sigma_u_iso_deposit"] is not None
                else None
            )

        if occ_col:
            occ_list = collect_geometry_lists(
                table_coords, [occ_col], [], [], ["occupancy"]
            )

        if u_aniso_cols:
            (
                u_aniso_atom_col,
                u11_col,
                u22_col,
                u33_col,
                u12_col,
                u13_col,
                u23_col,
            ) = u_aniso_cols
            u_aniso_list = collect_geometry_lists(
                table_u_aniso,
                [u_aniso_atom_col],
                [],
                [u11_col, u22_col, u33_col, u12_col, u13_col, u23_col],
                ["u11", "u22", "u33", "u12", "u13", "u23"],
            )
            for i in range(len(u_aniso_list)):
                u_aniso_list[i]["u_aniso_atom"] = u_aniso_atom_col[i]
        else:
            u_aniso_list = []

        atom1_col, atom2_col, value_sigma_col, symmetry2_col = bond_columns
        bonds_list = collect_geometry_lists(
            table_bond,
            [atom1_col, atom2_col],
            [symmetry2_col],
            [value_sigma_col],
            ["bond"],
        )

        if angle_columns:
            (
                atom1_col,
                atom2_col,
                atom3_col,
                value_sigma_col,
                symmetry1_col,
                symmetry3_col,
            ) = angle_columns
            angles_list = collect_geometry_lists(
                table_angle,
                [atom1_col, atom2_col, atom3_col],
                [symmetry1_col, symmetry3_col],
                [value_sigma_col],
                ["angle"],
            )

        if torsion_columns:
            (
                atom1_col,
                atom2_col,
                atom3_col,
                atom4_col,
                value_sigma_col,
                symmetry1_col,
                symmetry2_col,
                symmetry3_col,
                symmetry4_col,
            ) = torsion_columns
            torsions_list = collect_geometry_lists(
                table_torsion,
                [atom1_col, atom2_col, atom3_col, atom4_col],
                [symmetry1_col, symmetry2_col, symmetry3_col, symmetry4_col],
                [value_sigma_col],
                ["torsion"],
            )

    return atoms_list, occ_list, u_aniso_list, bonds_list, angles_list, torsions_list


def bootstrap_analyse_structures(
    refined_mmcifs,
    mmcif_ref,
    idx=0,
    prefix="",
    skip_hydrogen=True,
    smcif="",
    geometry_cids_file="",
    geometry_objects_ref=[],
):
    """
    Analyse structure models (mmCIF files) to compute mean coordinates and B-factors.
    The structure models are expected to be after refinement against a bootstrapped
    data set. They must have the same number of atoms and the same atom identifiers.

    Args:
        refined_mmcifs (list of str): List of mmCIF filenames.
        mmcif_ref: Reference mmCIF (refined in a standard way)
        idx (int): Index for naming the output files (applies if not set to 0).
        prefix (str): Prefix for the output filenames.
        skip_hydrogen (bool): If True, skip hydrogen atoms in the analysis.
        smcif (str): Path to a corresponding small molecule CIF file.
        geometry_atoms (str): Path to a corresponding file with list of atoms.

    Returns:
        None: Writes the statistics in '{prefix}group{idx}_bootstrap_mean_stats.csv' and
              the mean structure to '{prefix}group{idx}_bootstrap_mean_structure.mmcif'
              where 1000 * sigma_coordinate is saved as B-value.
    """

    # numpy.set_printoptions(threshold=numpy.inf)
    st_first = gemmi.read_structure(refined_mmcifs[0])
    st_first_cras = [
        cra
        for cra in st_first[0].all()
        if not (skip_hydrogen and cra.atom.is_hydrogen())
    ]
    logging.info(
        f"{len(st_first_cras)} atoms in the first structure {refined_mmcifs[0]}"
        " will be analysed."
    )
    if skip_hydrogen:
        logging.info("(Not taking into account hydrogen atoms)")

    atom_addresses = [makeAddressStr(cra) for cra in st_first_cras]
    atomic_numbers = [cra.atom.element.atomic_number for cra in st_first_cras]
    coords = numpy.zeros(
        (len(st_first_cras), 3, len(refined_mmcifs)), dtype=numpy.float32
    )
    b_values = numpy.zeros(
        (len(st_first_cras), len(refined_mmcifs)), dtype=numpy.float32
    )
    u_aniso = numpy.zeros(
        (len(st_first_cras), 6, len(refined_mmcifs)), dtype=numpy.float32
    )
    occupancies = numpy.zeros(
        (len(st_first_cras), len(refined_mmcifs)), dtype=numpy.float32
    )

    ref_b_value = {}
    if mmcif_ref and os.path.isfile(mmcif_ref):
        st_ref = gemmi.read_structure(mmcif_ref)
        st_ref_cras = [
            cra
            for cra in st_ref[0].all()
            if not (skip_hydrogen and cra.atom.is_hydrogen())
        ]
        logging.info(
            f"{len(st_ref_cras)} atoms in the reference structure {mmcif_ref}"
            " will be analysed."
        )
        if skip_hydrogen:
            logging.info("(Not taking into account hydrogen atoms)")

        ref_b_values = []
        if len(st_first_cras) != len(st_ref_cras):
            logging.warning(
                f"Inconsistent reference structure {mmcif_ref}"
                f" ({len(st_ref_cras)} atoms) with structure models after bootstrapping"
                f" ({len(st_first_cras)} atoms)"
            )
        for cra_ref in st_ref_cras:
            ref_b_values.append(cra_ref.atom.b_iso)
        ref_b_values = numpy.array(ref_b_values)
        ref_b_value = {"median B-value": numpy.median(ref_b_values)}

    if geometry_cids_file:
        geometry_objects = select_cids_for_geometry_analysis(geometry_cids_file)

    if smcif:
        atoms_list, occ_list, u_aniso_list, bonds_list, angles_list, torsions_list = (
            collect_values_smcif(smcif)
        )
        bonds = numpy.full(
            (len(bonds_list), len(refined_mmcifs)), numpy.nan, dtype=numpy.float32
        )
        angles = numpy.full(
            (len(angles_list), len(refined_mmcifs)), numpy.nan, dtype=numpy.float32
        )
        torsions = numpy.full(
            (len(torsions_list), len(refined_mmcifs)), numpy.nan, dtype=numpy.float32
        )

    logging.info(f"Loading {len(refined_mmcifs)} structure models...")
    # Collect coordinates and B-values
    for s, mmcif in enumerate(refined_mmcifs):
        st = gemmi.read_structure(mmcif)

        if geometry_cids_file and geometry_objects:
            geometry_objects = geometry_analysis_load(st, geometry_objects)

        st_cras = [
            cra for cra in st[0].all() if not (skip_hydrogen and cra.atom.is_hydrogen())
        ]
        assert len(st_first_cras) == len(st_cras), (
            f"Different number of atoms in structure model after bootstrapping: {mmcif}"
            f". Expected {len(st_first_cras)} atoms, got {len(st_cras)}."
        )
        for a, (cra_first, cra) in enumerate(zip(st_first_cras, st_cras)):
            assert (
                cra_first.atom.name == cra.atom.name
                and cra_first.atom.altloc == cra.atom.altloc
                and cra_first.residue.name == cra.residue.name
                and cra_first.residue.seqid == cra.residue.seqid
                and cra_first.chain.name == cra.chain.name
            ), f"Inconsistent structure models after bootstrapping: {mmcif}."
            coords[a, :, s] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
            b_values[a, s] = cra.atom.b_iso
            u_aniso[a, :, s] = [
                cra.atom.aniso.u11,
                cra.atom.aniso.u22,
                cra.atom.aniso.u33,
                cra.atom.aniso.u12,
                cra.atom.aniso.u13,
                cra.atom.aniso.u23,
            ]
            occupancies[a, s] = cra.atom.occ

        if smcif:
            # Calculate geometry
            if bonds_list:
                st_cras_atom_names = {cra.atom.name: cra for cra in st_cras}
                for b, bond in enumerate(bonds_list):
                    cra1 = st_cras_atom_names.get(bond["atom1"])
                    cra2 = st_cras_atom_names.get(bond["atom2"])
                    if cra1 and cra2:
                        bonds[b, s] = cra1.atom.pos.dist(cra2.atom.pos)
            if angles_list:
                for a, angle in enumerate(angles_list):
                    cra1 = st_cras_atom_names.get(angle["atom1"])
                    cra2 = st_cras_atom_names.get(angle["atom2"])
                    cra3 = st_cras_atom_names.get(angle["atom3"])
                    if cra1 and cra2 and cra3:
                        angles[a, s] = calculate_angle(
                            cra1.atom.pos, cra2.atom.pos, cra3.atom.pos
                        )
            if torsions_list:
                for t, torsion in enumerate(torsions_list):
                    cra1 = st_cras_atom_names.get(torsion["atom1"])
                    cra2 = st_cras_atom_names.get(torsion["atom2"])
                    cra3 = st_cras_atom_names.get(torsion["atom3"])
                    cra4 = st_cras_atom_names.get(torsion["atom4"])
                    if cra1 and cra2 and cra3 and cra4:
                        torsions[t, s] = calculate_torsion_angle(
                            cra1.atom.pos, cra2.atom.pos, cra3.atom.pos, cra4.atom.pos
                        )

    if geometry_cids_file and geometry_objects:
        geometry_analysis_occs = [
            obj for obj in geometry_objects if obj["type"] == "occupancy"
        ]
        geometry_analysis_bonds = [
            obj for obj in geometry_objects if obj["type"] == "distance"
        ]
        geometry_analysis_angles_torsions = [
            obj for obj in geometry_objects if obj["type"] in ["angle", "torsion"]
        ]
        obj_occ_refs = [{}] * len(geometry_analysis_occs)
        obj_bond_refs = [{}] * len(geometry_analysis_bonds)
        obj_angle_torsion_refs = [{}] * len(geometry_analysis_angles_torsions)
        if mmcif_ref and os.path.isfile(mmcif_ref):
            geometry_analysis_occs_ref = [
                obj for obj in geometry_objects_ref if obj["type"] == "occupancy"
            ]
            assert len(geometry_analysis_occs) == len(geometry_analysis_occs_ref)
            for i, obj in enumerate(geometry_analysis_occs_ref):
                assert len(obj["values"]) == 1
                obj_occ_refs[i] = {"occupancy": obj["values"][0]}
            geometry_analysis_bonds_ref = [
                obj for obj in geometry_objects_ref if obj["type"] == "distance"
            ]
            assert len(geometry_analysis_bonds) == len(geometry_analysis_bonds_ref)
            for i, obj in enumerate(geometry_analysis_bonds_ref):
                assert len(obj["values"]) == 1
                obj_bond_refs[i] = {"distance": obj["values"][0]}
            geometry_analysis_angles_torsions_ref = [
                obj
                for obj in geometry_objects_ref
                if obj["type"] in ["angle", "torsion"]
            ]
            assert len(geometry_analysis_angles_torsions) == len(
                geometry_analysis_angles_torsions_ref
            )
            for i, obj in enumerate(geometry_analysis_angles_torsions_ref):
                assert len(obj["values"]) == 1
                obj_angle_torsion_refs[i] = {obj["type"]: obj["values"][0]}

        for i, obj in enumerate(geometry_analysis_occs):
            obj["values"] = numpy.array(obj["values"])
            obj["mean"] = numpy.nanmean(obj["values"])
            obj["std"] = numpy.nanstd(obj["values"], ddof=1)
            plot_histogram(
                obj["values"],
                f"occupancy {obj['atom1']}",
                obj_occ_refs[i],
                idx,
                prefix,
            )
            del obj["values"]

        for i, obj in enumerate(geometry_analysis_bonds):
            obj["values"] = numpy.array(obj["values"])
            obj["mean"] = numpy.nanmean(obj["values"])
            obj["std"] = numpy.nanstd(obj["values"], ddof=1)
            plot_histogram(
                obj["values"],
                f"distance {obj['atom1']} {obj['atom2']}",
                obj_bond_refs[i],
                idx,
                prefix,
            )
            del obj["values"]

        for i, obj in enumerate(geometry_analysis_angles_torsions):
            obj["values"] = numpy.array(obj["values"])
            obj["mean"] = circular_mean_deg(obj["values"])
            obj["std"] = circular_std_deg(obj["values"])
            if obj["type"] == "angle":
                plot_histogram(
                    obj["values"],
                    f"angle {obj['atom1']} {obj['atom2']} {obj['atom3']}",
                    obj_angle_torsion_refs[i],
                    idx,
                    prefix,
                )
            else:  # torsion
                plot_histogram(
                    obj["values"],
                    f"torsion angle"
                    f" {obj['atom1']} {obj['atom2']} {obj['atom3']} {obj['atom4']}",
                    obj_angle_torsion_refs[i],
                    idx,
                    prefix,
                )
            del obj["values"]
        df = pandas.DataFrame(
            geometry_analysis_occs
            + geometry_analysis_bonds
            + geometry_analysis_angles_torsions
        )
        filename = (
            f"{prefix}group{idx}_mean_geometry_stats.txt"
            if idx
            else f"{prefix}mean_geometry_stats.txt"
        )
        df.to_string(filename, index=False, na_rep="")
        logging.info(f"Saved geometry statistics to {filename}")

    if smcif:
        if bonds_list:
            mean_bonds = numpy.mean(bonds, axis=1)
            std_bonds = numpy.std(bonds, axis=1, ddof=1)
            for b, bond in enumerate(bonds_list):
                bond["mean_bond"] = mean_bonds[b]
                bond["sigma_bond"] = std_bonds[b]
            df_bonds = pandas.DataFrame(bonds_list)
            csv_filename_bonds = (
                f"{prefix}group{idx}_mean_bonds_stats.csv"
                if idx
                else f"{prefix}mean_bonds_stats.csv"
            )
            df_bonds = df_bonds.round(6)
            df_bonds.to_csv(csv_filename_bonds, index=False)
            logging.info(f"Mean bond statistics written to {csv_filename_bonds}.")

        if angles_list:
            # mean_angles = numpy.nanmean(angles, axis=1)
            # std_angles = numpy.nanstd(angles, axis=1, ddof=1)
            mean_angles = numpy.array([circular_mean_deg(row) for row in angles])
            std_angles = numpy.array([circular_std_deg(row) for row in angles])
            for a, angle in enumerate(angles_list):
                angle["mean_angle"] = mean_angles[a]
                angle["sigma_angle"] = std_angles[a]
            df_angles = pandas.DataFrame(angles_list)
            csv_filename_angles = (
                f"{prefix}group{idx}_mean_angles_stats.csv"
                if idx
                else f"{prefix}mean_angles_stats.csv"
            )
            df_angles = df_angles.round(6)
            df_angles.to_csv(csv_filename_angles, index=False)
            logging.info(f"Mean angle statistics written to {csv_filename_angles}.")

        if torsions_list:
            # mean_torsions = numpy.nanmean(torsions, axis=1)
            # std_torsions = numpy.nanstd(torsions, axis=1, ddof=1)
            mean_torsions = numpy.array([circular_mean_deg(row) for row in torsions])
            std_torsions = numpy.array([circular_std_deg(row) for row in torsions])
            for t, torsion in enumerate(torsions_list):
                torsion["mean_torsion"] = mean_torsions[t]
                torsion["sigma_torsion"] = std_torsions[t]
            df_torsions = pandas.DataFrame(torsions_list)
            csv_filename_torsions = (
                f"{prefix}group{idx}_mean_torsions_stats.csv"
                if idx
                else f"{prefix}mean_torsions_stats.csv"
            )
            df_torsions = df_torsions.round(6)
            df_torsions.to_csv(csv_filename_torsions, index=False)
            logging.info(
                f"Mean torsion angle statistics written to {csv_filename_torsions}."
            )

    # Compute mean and standard deviation per atom
    mean_coords = numpy.mean(coords, axis=2)  # shape: (n_atoms, 3)
    std_coords = numpy.std(coords, ddof=1, axis=2)  # shape: (n_atoms, 3)
    # std_coords_norm = sqrt(σ_x² + σ_y² + σ_z²)
    #  (when assuming no correlation which is not the case)
    # std_coords_norm = numpy.linalg.norm(std_coords, axis=1)  # shape: (n_atoms,)
    #
    # std_coords_norm = sqrt(σ_x² + σ_y² + σ_z² + 2 * (σ_xy + σ_xz + σ_yz))
    # Calculate joint sigma of coordinates, assuming correlation between x, y, z
    std_coords_norm = numpy.zeros(len(st_first_cras))
    cov_coords = numpy.zeros((len(st_first_cras), 3))  # shape: (n_atoms, 3)
    for i in range(len(st_first_cras)):
        cov = numpy.cov(coords[i, :, :])
        cov_coords[i] = (cov[0, 1], cov[0, 2], cov[1, 2])  # σ_xy, σ_xz, σ_yz
        std_coords_norm[i] = numpy.sqrt(
            numpy.trace(cov) + 2 * (cov[0, 1] + cov[0, 2] + cov[1, 2])
        )
    mean_b_values = numpy.mean(b_values, axis=1)  # shape: (n_atoms,)
    std_b_values = numpy.std(b_values, ddof=1, axis=1)  # shape: (n_atoms,)
    mean_u_aniso = numpy.mean(u_aniso, axis=2)  # shape: (n_atoms, 6)
    std_u_aniso = numpy.std(u_aniso, ddof=1, axis=2)  # shape: (n_atoms, 6)
    mean_occupancies = numpy.mean(occupancies, axis=1)  # shape: (n_atoms,)
    std_occupancies = numpy.std(occupancies, ddof=1, axis=1)  # shape: (n_atoms,)

    mean_b_values_per_structure = numpy.mean(b_values, axis=0)  # shape: (n_structures,)
    plot_histogram(
        mean_b_values_per_structure, "Average B-value", ref_b_value, idx, prefix
    )

    keys_u_aniso = [
        "u11",
        "sigma_u11",
        "u22",
        "sigma_u22",
        "u33",
        "sigma_u33",
        "u12",
        "sigma_u12",
        "u13",
        "sigma_u13",
        "u23",
        "sigma_u23",
    ]
    # Write calculated data as a CSV file
    csv_data = []
    for i, atom_address in enumerate(atom_addresses):
        csv_data.append(
            {
                "atom_id": atom_address,
                "atomic_number": atomic_numbers[i],
                "mean_x": mean_coords[i][0],
                "mean_y": mean_coords[i][1],
                "mean_z": mean_coords[i][2],
                "sigma_x": std_coords[i][0],
                "sigma_y": std_coords[i][1],
                "sigma_z": std_coords[i][2],
                "sigma_xy": cov_coords[i][0],
                "sigma_xz": cov_coords[i][1],
                "sigma_yz": cov_coords[i][2],
                "sigma_coord": std_coords_norm[i],
                "mean_b": mean_b_values[i],
                "sigma_b": std_b_values[i],
                "mean_u11": mean_u_aniso[i][0],
                "mean_u22": mean_u_aniso[i][1],
                "mean_u33": mean_u_aniso[i][2],
                "mean_u12": mean_u_aniso[i][3],
                "mean_u13": mean_u_aniso[i][4],
                "mean_u23": mean_u_aniso[i][5],
                "sigma_u11": std_u_aniso[i][0],
                "sigma_u22": std_u_aniso[i][1],
                "sigma_u33": std_u_aniso[i][2],
                "sigma_u12": std_u_aniso[i][3],
                "sigma_u13": std_u_aniso[i][4],
                "sigma_u23": std_u_aniso[i][5],
                "mean_occupancy": mean_occupancies[i],
                "sigma_occupancy": std_occupancies[i],
            }
        )
        if smcif and atoms_list:
            # Add the deposited values
            for key in [
                "x",
                "sigma_x",
                "y",
                "sigma_y",
                "z",
                "sigma_z",
                "b_iso",
                "sigma_b_iso",
            ]:
                csv_data[i][f"{key}_deposit"] = atoms_list[i][f"{key}_deposit"]
            if occ_list:
                csv_data[i]["occupancy_deposit"] = occ_list[i]["occupancy_deposit"]
            for key in keys_u_aniso:
                csv_data[i][f"{key}_deposit"] = None
            for i_aniso in range(len(u_aniso_list)):
                if u_aniso_list[i_aniso]["u_aniso_atom"] == st_first_cras[i].atom.name:
                    for key in keys_u_aniso:
                        csv_data[i][f"{key}_deposit"] = u_aniso_list[i_aniso][
                            f"{key}_deposit"
                        ]
                    break
    df_csv = pandas.DataFrame(csv_data)
    df_csv = df_csv.round(6)
    csv_filename = (
        f"{prefix}group{idx}_bootstrap_mean_stats.csv"
        if idx
        else f"{prefix}bootstrap_mean_stats.csv"
    )
    df_csv.to_csv(csv_filename, index=False)
    logging.info(f"Mean structure statistics written to {csv_filename}.")

    if smcif and atoms_list:
        # df_csv_noH = df_csv[~df_csv["atom_id"].str.contains("/H")]
        png_filename = (
            f"{prefix}group{idx}_bootstrap_mean_stats_plot_xyzb.png"
            if idx
            else f"{prefix}bootstrap_mean_stats_plot_xyzb.png"
        )
        df_scatter_plot(
            df_csv,
            [
                "sigma_x_deposit",
                "sigma_y_deposit",
                "sigma_z_deposit",
                "sigma_b_iso_deposit",
            ],
            [["sigma_x"], ["sigma_y"], ["sigma_z"], ["sigma_b"]],
            filename=png_filename,
        )
        png_filename_per_element = (
            f"{prefix}group{idx}_bootstrap_mean_stats_plot_xyzb_per_element.png"
            if idx
            else f"{prefix}bootstrap_mean_stats_plot_xyzb_per_element.png"
        )
        df_scatter_plot(
            df_csv,
            ["atom_id"],
            [
                ["sigma_x", "sigma_x_deposit"],
                ["sigma_y", "sigma_y_deposit"],
                ["sigma_z", "sigma_z_deposit"],
                ["sigma_b", "sigma_b_iso_deposit"],
            ],
            filename=png_filename_per_element,
            per_element=True,
        )

    # Write mean structure as mmCIF
    for i, cra in enumerate(st_first_cras):
        # Replace position with mean coordinates
        cra.atom.pos = gemmi.Position(*mean_coords[i])
        # Replace B-factor with norm of std deviation (or square it if desired)
        cra.atom.b_iso = std_coords_norm[i]  # or (8π²/3)*σ² ???
        cra.atom.occ = mean_occupancies[i]
    mean_structure_prefix = (
        f"{prefix}group{idx}_bootstrap_mean_structure"
        if idx
        else f"{prefix}bootstrap_mean_structure"
    )
    st_first.make_mmcif_document().write_file(f"{mean_structure_prefix}.mmcif")
    logging.info(f"Mean structure saved as {mean_structure_prefix}.mmcif")
    try:
        st_first.write_pdb(f"{mean_structure_prefix}.pdb")
        logging.info(f"Mean structure saved as {mean_structure_prefix}.pdb")
    except:  # noqa: E722
        logging.warning(
            "Saving the mean structure in the PDB format was not successful."
        )
    return


def bootstrap_mean_map(
    refined_mtzs, idx=0, prefix="", binner=None, mtz_ref="", n_proc=4
):
    """
    Calculate the mean 2Fo-Fc and Fo-Fc maps from refined MTZ files after bootstrapping.
    The maps are expected to be after refinement against a bootstrapped
    data set.

    Args:
        refined_mtzs (list of str): List of MTZ filenames.
        idx (int): Index for naming the output file (applies if not set to 0).
        prefix (str): Prefix for the output filename.
        binner (gemmi.Binner): Binner object for resolution bins (optional).
        mtz_ref (str): Reference MTZ file for scaling (optional).

    Returns:
        None: Writes the mean maps in
            '{prefix}group{idx}_bootstrap_mean_map_all.mtz'
            '{prefix}group{idx}_bootstrap_mean_map_llweight0.mtz'
            '{prefix}group{idx}_bootstrap_mean_map_llweightpos.mtz'
            '{prefix}group{idx}_bootstrap_mean_map_llweightposw.mtz'
    """

    def merge_reflections_bootstrap(
        df_master,
        mtz_first=None,
        prefix="",
        suffix="",
        idx=0,
        binner=None,
        mtz_ref="",
        do_llweighting=False,
    ):
        """
        Merge reflections from the master DataFrame and calculate mean maps.

        Args:
            df_master (pandas.DataFrame): DataFrame containing reflections.
                It must contain columns "H", "K", "L", "F_complex", "DEL_F_complex",
            mtz_first (gemmi.Mtz): Reference MTZ object for cell and spacegroup.
            prefix (str): Prefix for the output filename.
            suffix (str): Suffix for the output filename.
            idx (int): Index for naming the output file.

        Returns:
            pandas.DataFrame: DataFrame with mean maps.
        """

        """# noqa: E741
        def is_centric_vectorized(h, k, l):  # noqa: E741
            return mtz_first.spacegroup.operations().is_reflection_centric(
                (int(h), int(k), int(l))  # noqa: E741
            )"""

        def calculate_mean_std_count(df, do_llweighting=False):
            """Calculate mean and standard deviation and number of structure factors."""

            def stats_func(miller_index_df, column_name, do_llweighting=do_llweighting):
                """
                Compute weighted or unweighted mean, std, and count
                for one Miller index.

                Args:
                    miller_index_df (pandas.DataFrame): DataFrame for
                        a specific Miller index.
                    column_name (str): Column name to compute stats on.
                    do_llweighting (bool): Whether to apply llweighting.

                Returns:
                    pandas.Series: Series containing mean, std, and count.
                """
                x = miller_index_df[column_name].values

                if len(x) <= 1:
                    return pandas.Series([numpy.mean(x), 0.0, len(x)])

                if do_llweighting and "llweight" in miller_index_df.columns:
                    # Weighted mean and variance
                    w = miller_index_df["llweight"].values
                    w = w / numpy.sum(w)  # normalize weights
                    mean_val = numpy.sum(w * x)

                    real_mean = numpy.real(mean_val)
                    imag_mean = numpy.imag(mean_val)
                    real_part = numpy.real(x)
                    imag_part = numpy.imag(x)
                    real_var = numpy.sum(w * (real_part - real_mean) ** 2)
                    imag_var = numpy.sum(w * (imag_part - imag_mean) ** 2)
                else:
                    # Unweighted mean and variance
                    mean_val = numpy.mean(x)
                    real_mean = numpy.real(mean_val)
                    imag_mean = numpy.imag(mean_val)
                    real_part = numpy.real(x)
                    imag_part = numpy.imag(x)
                    real_var = numpy.var(real_part, ddof=1, mean=real_mean)
                    imag_var = numpy.var(imag_part, ddof=1, mean=imag_mean)

                std_val = numpy.sqrt(real_var + imag_var)
                return pandas.Series([mean_val, std_val, len(x)])

            # F_complex: apply stats_func to each Miller index
            df_mean_fwt = df.groupby(["H", "K", "L"], as_index=False).apply(
                lambda d: stats_func(d, "F_complex", do_llweighting=do_llweighting),
                include_groups=False,
            )
            # This converts Series to DataFrame
            # df_mean_fwt = df_mean_fwt.unstack(level=-1)
            df_mean_fwt.columns = [
                "H",
                "K",
                "L",
                "F_complex_mean",
                "SIGFWT",
                "FWTcount",
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
                df_mean_fwt["SIGFWT"] = df_mean_fwt["SIGFWT"].astype(numpy.float32)
                df_mean_fwt["FWTcount"] = df_mean_fwt["FWTcount"].astype(numpy.int32)
            df_mean_fwt = df_mean_fwt.reset_index()

            # DEL_F_complex: apply stats_func to each Miller index
            df_mean_delfwt = df.groupby(["H", "K", "L"], as_index=False).apply(
                lambda d: stats_func(d, "DEL_F_complex", do_llweighting=do_llweighting),
                include_groups=False,
            )
            # This converts Series to DataFrame
            # df_mean_delfwt = df_mean_delfwt.unstack(level=-1)
            df_mean_delfwt.columns = [
                "H",
                "K",
                "L",
                "DEL_F_complex_mean",
                "SIGDELFWT",
                "DELFWTcount",
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
                df_mean_delfwt["SIGDELFWT"] = df_mean_delfwt["SIGDELFWT"].astype(
                    numpy.float32
                )
                df_mean_delfwt["DELFWTcount"] = df_mean_delfwt["DELFWTcount"].astype(
                    numpy.int32
                )
            df_mean_delfwt = df_mean_delfwt.reset_index()

            df_mean_fwt_delfwt = df_mean_fwt.merge(
                df_mean_delfwt, on=["H", "K", "L"], how="outer"
            )
            return df_mean_fwt_delfwt

            """
            # old code which treated centric and acentric reflections differently
            # and variance of acentric reflections was divided by two
            if is_centric:
                sigma_func = lambda x: (  # noqa: E731
                    numpy.sqrt(numpy.var(numpy.abs(x), ddof=1)) if len(x) > 1 else 0
                )
            else:
                sigma_func = lambda x: (  # noqa: E731
                    numpy.sqrt(
                        (numpy.var(numpy.real(x), ddof=1)
                        + numpy.var(numpy.imag(x), ddof=1)) / 2
                    )
                    if len(x) > 1
                    else 0
                )

            df_mean_f = (
                df.groupby(["H", "K", "L"])["F_complex"]
                .agg(
                    [
                        ("F_complex_mean", lambda x: numpy.mean(x)),
                        ("SIGFWT", sigma_func),
                        ("FWTcount", "count"),
                    ]
                )
                .reset_index()
            )
            df_mean_delf = (
                df.groupby(["H", "K", "L"])["DEL_F_complex"]
                .agg(
                    [
                        ("DEL_F_complex_mean", lambda x: numpy.mean(x)),
                        ("SIGDELFWT", sigma_func),
                        ("DELFWTcount", "count"),
                    ]
                )
                .reset_index()
            )

            return df_mean_f.merge(df_mean_delf, on=["H", "K", "L"], how="outer")"""

        """
        is_centric_vec = numpy.vectorize(is_centric_vectorized)
        centric_mask = is_centric_vec(df_master["H"], df_master["K"], df_master["L"])
        acentric = df_master[~centric_mask]
        centric = df_master[centric_mask]
        stats_acentric = calculate_mean_std_count(acentric, is_centric=False)
        stats_centric = calculate_mean_std_count(centric, is_centric=True)
        df_mean = pandas.concat([stats_acentric, stats_centric], ignore_index=True)"""

        df_master = df_master.astype({col: "int32" for col in ["H", "K", "L"]})
        df_mean = calculate_mean_std_count(df_master, do_llweighting=do_llweighting)

        # Convert to amplitude and phase
        df_mean["FWT"] = numpy.abs(df_mean["F_complex_mean"])
        df_mean["PHWT"] = numpy.rad2deg(numpy.angle(df_mean["F_complex_mean"]))
        df_mean["DELFWT"] = numpy.abs(df_mean["DEL_F_complex_mean"])
        df_mean["PHDELWT"] = numpy.rad2deg(numpy.angle(df_mean["DEL_F_complex_mean"]))

        if mtz_first and prefix and suffix and idx:
            # Save the mean maps as an MTZ file
            columns = {
                "FWT": "F",
                "PHWT": "P",
                "SIGFWT": "Q",
                "FWTcount": "I",
                "DELFWT": "F",
                "PHDELWT": "P",
                "SIGDELFWT": "Q",
                "DELFWTcount": "I",
            }
            mtz_filename = (
                f"{prefix}group{idx}_bootstrap_mean_map{suffix}.mtz"
                if idx
                else f"{prefix}bootstrap_mean_map{suffix}.mtz"
            )
            write_mtz_from_df(
                df_mean[
                    [
                        "H",
                        "K",
                        "L",
                        "FWT",
                        "PHWT",
                        "SIGFWT",
                        "FWTcount",
                        "DELFWT",
                        "PHDELWT",
                        "SIGDELFWT",
                        "DELFWTcount",
                    ]
                ],
                mtz_first,
                columns,
                filename=mtz_filename,
            )

        # Calculate statistics per bin
        if binner and mtz_first:
            hkl_array = numpy.array(df_mean[["H", "K", "L"]].values, numpy.int32)
            hkl_array = numpy.ascontiguousarray(hkl_array, dtype=numpy.int32)
            df_mean["bin"] = binner.get_bins(hkl_array)
            bin_stats = []
            for b in range(binner.size):
                df_bin = df_mean[df_mean["bin"] == b]
                if not df_bin.empty:
                    mean_fwt = df_bin["FWT"].mean()
                    mean_sigfwt = df_bin["SIGFWT"].mean()
                    mean_fwt_sigfwt = mean_fwt / mean_sigfwt if mean_sigfwt else 0.0
                    fwt_count = df_bin["FWTcount"].sum()
                    mean_delfwt = df_bin["DELFWT"].mean()
                    mean_sigdelfwt = df_bin["SIGDELFWT"].mean()
                    mean_delfwt_sigdelfwt = (
                        mean_delfwt / mean_sigdelfwt if mean_sigdelfwt else 0.0
                    )
                    delfwt_count = df_bin["DELFWTcount"].sum()
                    bin_n_unique = len(df_bin)
                    bin_n_unique_expected = gemmi.count_reflections(
                        mtz_first.cell,
                        mtz_first.spacegroup,
                        binner.dmin_of_bin(b),
                        binner.dmax_of_bin(b),
                        unique=True,
                    )
                    completeness = bin_n_unique / bin_n_unique_expected
                    bin_stats.append(
                        {
                            "bin": b + 1,
                            "dmax": binner.dmax_of_bin(b),
                            "dmin": binner.dmin_of_bin(b),
                            "mean_FWT": mean_fwt,
                            "mean_SIGFWT": mean_sigfwt,
                            "mean_FWT_SIGFWT": mean_fwt_sigfwt,
                            "FWTcount": fwt_count,
                            "mean_DELFWT": mean_delfwt,
                            "mean_SIGDELFWT": mean_sigdelfwt,
                            "mean_DELFWT_SIGDELFWT": mean_delfwt_sigdelfwt,
                            "DELFWTcount": delfwt_count,
                            "count": bin_n_unique,
                            "completeness": completeness,
                        }
                    )
                else:
                    bin_stats.append(
                        {
                            "bin": b + 1,
                            "dmax": binner.dmax_of_bin(b),
                            "dmin": binner.dmin_of_bin(b),
                            "mean_FWT": 0.0,
                            "mean_SIGFWT": 0.0,
                            "mean_FWT_SIGFWT": 0.0,
                            "FWTcount": 0,
                            "mean_DELFWT": 0.0,
                            "mean_SIGDELFWT": 0.0,
                            "mean_DELFWT_SIGDELFWT": 0.0,
                            "DELFWTcount": 0,
                            "count": 0,
                            "completeness": 0.0,
                        }
                    )
            if prefix and suffix:
                stats_filename = (
                    f"{prefix}group{idx}_bootstrap_mean_map{suffix}.txt"
                    if idx
                    else f"{prefix}bootstrap_mean_map{suffix}.txt"
                )
                write_bin_stats(bin_stats, stats_filename)

        if mtz_ref and binner and prefix and suffix:
            mtz_scaled_prefix = (
                f"{prefix}group{idx}_bootstrap_mean_map{suffix}"
                if idx
                else f"{prefix}bootstrap_mean_map{suffix}"
            )
            df_scaled, bin_stats_scaled = scale_reflections(
                mtz_ref,
                df_mean.copy(),
                binner,
                output_mtz2_prefix=mtz_scaled_prefix,
            )
            """if mtz_ref and prefix and suffix and idx:
                # Save the scaled mean maps as an MTZ file
                columns_scaled = {
                    "FWT": "F",
                    "PHWT": "P",
                    "DELFWT": "F",
                    "PHDELWT": "P",
                }
                write_mtz_from_df(
                    df_scaled[["H", "K", "L"] + list(columns_scaled.keys())],
                    mtz_ref,
                    columns_scaled,
                    mtz_scaled_filename,
                )"""

            # Calculate statistics per bin
            # if binner and mtz_ref:
            hkl_array = numpy.array(df_scaled[["H", "K", "L"]].values, numpy.int32)
            hkl_array = numpy.ascontiguousarray(hkl_array, dtype=numpy.int32)
            df_scaled["bin"] = binner.get_bins(hkl_array)
            for b in range(binner.size):
                df_bin = df_scaled[df_scaled["bin"] == b]
                if not df_bin.empty:
                    mean_fwt = df_bin["FWT"].mean()
                    # mean_sigfwt = df_bin["SIGFWT"].mean()
                    # mean_fwt_sigfwt = mean_fwt / mean_sigfwt if mean_sigfwt else 0.0
                    # fwt_count = df_bin["FWTcount"].sum()
                    mean_delfwt = df_bin["DELFWT"].mean()
                    # mean_sigdelfwt = df_bin["SIGDELFWT"].mean()
                    # mean_delfwt_sigdelfwt = (
                    # #     mean_delfwt / mean_sigdelfwt if mean_sigdelfwt else 0.0
                    # )
                    # delfwt_count = df_bin["DELFWTcount"].sum()
                    bin_n_unique = len(df_bin)
                    """bin_n_unique_expected = gemmi.count_reflections(
                        mtz_first.cell,
                        mtz_first.spacegroup,
                        binner.dmin_of_bin(b),
                        binner.dmax_of_bin(b),
                        unique=True,
                    )
                    completeness = bin_n_unique / bin_n_unique_expected"""
                    bin_stats_scaled[b].update(
                        {
                            "bin": b + 1,
                            "dmax": binner.dmax_of_bin(b),
                            "dmin": binner.dmin_of_bin(b),
                            "mean_FWT": mean_fwt,
                            # "mean_SIGFWT": mean_sigfwt,
                            # "mean_FWT_SIGFWT": mean_fwt_sigfwt,
                            # "FWTcount": fwt_count,
                            "mean_DELFWT": mean_delfwt,
                            # "mean_SIGDELFWT": mean_sigdelfwt,
                            # "mean_DELFWT_SIGDELFWT": mean_delfwt_sigdelfwt,
                            # "DELFWTcount": delfwt_count,
                            "count": bin_n_unique,
                            # "completeness": completeness,
                        }
                    )
                else:
                    bin_stats_scaled.update(
                        {
                            "bin": b + 1,
                            "dmax": binner.dmax_of_bin(b),
                            "dmin": binner.dmin_of_bin(b),
                            "mean_FWT": 0.0,
                            # "mean_SIGFWT": 0.0,
                            # "mean_FWT_SIGFWT": 0.0,
                            # "FWTcount": 0,
                            "mean_DELFWT": 0.0,
                            # "mean_SIGDELFWT": 0.0,
                            # "mean_DELFWT_SIGDELFWT": 0.0,
                            # "DELFWTcount": 0,
                            "count": 0,
                            # "completeness": 0.0,
                        }
                    )
            stats_scaled_filename = mtz_scaled_prefix + "_scaled_stats.txt"
            write_bin_stats(bin_stats_scaled, stats_scaled_filename)

        return df_scaled

    logging.info(f"Loading {len(refined_mtzs)} density maps...")
    columns_selected = ["H", "K", "L", "FWT", "PHWT", "DELFWT", "PHDELWT", "llweight"]
    if binner:
        if mtz_ref:
            logging.info(f"Scaling reflections to {mtz_ref}")
        else:
            mtz_first = gemmi.read_mtz_file(refined_mtzs[0])
            col_labels_first = mtz_first.column_labels()
            df_first = pandas.DataFrame(data=mtz_first.array, columns=col_labels_first)
            df_first = df_first[columns_selected]
            logging.info(f"Scaling reflections to {refined_mtzs[0]}")

    # Process MTZ files in parallel
    def _add_reflections(worker_args):
        """
        Worker function to scale one MTZ file for bootstrap mean map calculation.
        Returns (df, bin_stats) or (None, None) on failure.
        """
        mtz_file, columns_selected, binner, mtz_ref = worker_args
        try:
            mtz = gemmi.read_mtz_file(mtz_file)
            col_labels = mtz.column_labels()
            df = pandas.DataFrame(data=mtz.array, columns=col_labels)
            df = df[columns_selected]
            if df.empty:
                logging.warning(
                    f"No reflections in {mtz_file} for FWT/PHWT/DELFWT/PHDELWT."
                )
                return None, None

            bin_stats = None
            if binner and mtz_ref:
                # scale per resolution bin
                mtz_file_base = os.path.splitext(os.path.basename(mtz_file))[0]
                df, bin_stats = scale_reflections(
                    mtz_ref, df, binner, output_mtz2_prefix=mtz_file_base
                )

            df = df.astype({name: "int32" for name in ["H", "K", "L"]})
            return df, bin_stats

        except Exception as e:
            logging.error(f"Error processing {mtz_file}: {e}")
            import traceback

            traceback.print_exc()
            return None, None

    worker_args_list = [
        (mtz_file, columns_selected, binner, mtz_ref) for mtz_file in refined_mtzs
    ]
    df_list = []
    bin_stats_bootstrap_scale = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as executor:
        futures = [
            executor.submit(_add_reflections, worker_args)
            for worker_args in worker_args_list
        ]
        for future in concurrent.futures.as_completed(futures):
            df, bin_stats = future.result()
            if df is not None:
                df_list.append(df)
                if bin_stats is not None:
                    bin_stats_bootstrap_scale.append(bin_stats)
    if df_list:
        df_master = pandas.concat(df_list, ignore_index=True)
    else:
        logging.error("No valid MTZ files processed.")
        return

    if bin_stats_bootstrap_scale:
        try:
            json_filename = (
                f"{prefix}group{idx}_bootstrap_map_scaling_stats.json"
                if idx
                else f"{prefix}bootstrap_map_scaling_stats.json"
            )
            with open(json_filename, "w") as f_json:
                json.dump(
                    bin_stats_bootstrap_scale,
                    f_json,
                    indent=4,
                    default=json_numpy_converter,
                )
            logging.info(f"Saved bootstrap map scaling stats to {json_filename}")
        except Exception as e:
            logging.warning(
                f"Could not write bootstrap map scaling stats JSON file: {e}"
            )

    # Convert FWT & PHWT and DELFWT & PHDELWT to complex numbers and calculate mean
    df_master["F_complex"] = df_master["FWT"] * numpy.exp(
        1j * numpy.deg2rad(df_master["PHWT"])
    )
    df_master["DEL_F_complex"] = df_master["DELFWT"] * numpy.exp(
        1j * numpy.deg2rad(df_master["PHDELWT"])
    )
    # print(df_master.head(10))
    # print(df_master[["H", "K", "L", "FWT", "PHWT"]].describe())
    df_master_llweight_0 = df_master[df_master["llweight"] == 0].copy()
    df_master_llweight_pos = df_master[df_master["llweight"] > 0].copy()

    mtz_first = gemmi.read_mtz_file(refined_mtzs[0])
    # save 4 mean maps: all reflections, llweight == 0,
    # llweight > 0 and llweight > 0 weighted average
    merge_reflections_bootstrap(
        df_master, mtz_first, prefix, "_all", idx, binner, mtz_ref
    )
    merge_reflections_bootstrap(
        df_master_llweight_0, mtz_first, prefix, "_llweight0", idx, binner, mtz_ref
    )
    merge_reflections_bootstrap(
        df_master_llweight_pos, mtz_first, prefix, "_llweightpos", idx, binner, mtz_ref
    )
    merge_reflections_bootstrap(
        df_master_llweight_pos,
        mtz_first,
        prefix,
        "_llweightposw",
        idx,
        binner,
        mtz_ref,
        do_llweighting=True,
    )

    return
