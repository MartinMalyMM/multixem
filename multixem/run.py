# coding: utf-8
import os
import sys
import argparse
import subprocess
import pprint
import numpy
import pandas
import gemmi
import matplotlib.pyplot as plt
import warnings
from collections import Counter
import concurrent.futures
from . import __version__


def create_parser():
    """
    Create the argument parser for the command-line interface.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """

    def positive_int(value):
        if isinstance(value, list):
            positive_int(v for v in value)
        else:
            try:
                ivalue = int(value)
            except ValueError:
                raise argparse.ArgumentTypeError(f"{value} is not an integer.")
            if ivalue <= 0:
                raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
            return ivalue

    def existing_file(path):
        abs_norm_path = os.path.abspath(os.path.normpath(path))
        if not os.path.isfile(abs_norm_path):
            raise argparse.ArgumentTypeError(f"File does not exist: {abs_norm_path}")
        return abs_norm_path

    class ArgumentDefaultsHelpFormatterCustom(argparse.ArgumentDefaultsHelpFormatter):
        def _get_help_string(self, action):
            help_str = action.help
            # Only show default if it's not None and not suppressed
            if action.default is not None and action.default != argparse.SUPPRESS:
                if "%(default)" not in help_str:
                    help_str += f" (default: {action.default})"
            return help_str

    parser = argparse.ArgumentParser(
        prog="multixem",
        description="Refinement pipeline for multiple data sets in structure biology.",
        formatter_class=ArgumentDefaultsHelpFormatterCustom,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
        help="show version and exit",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Main pipeline subcommand (default behavior)
    main_parser = subparsers.add_parser(
        "pipeline",
        help="Run the main refinement pipeline",
        formatter_class=ArgumentDefaultsHelpFormatterCustom,
    )

    # Add all existing arguments to the main parser
    main_parser.add_argument(
        "-p", "--prefix", type=str, help="Prefix for the output files."
    )
    main_parser.add_argument(
        "-u",
        "--hklin_unmerged",
        type=existing_file,
        nargs="+",
        help="Input unmerged diffraction data file(s).",
    )
    # TODO more files
    main_parser.add_argument(
        "--hklin_free", type=existing_file, help="Input MTZ file for test flags."
    )
    main_parser.add_argument(
        "--hklin",
        type=existing_file,
        nargs="+",
        help="Input merged diffraction data file(s).",
    )
    main_parser.add_argument(
        "--model", type=existing_file, help="Input atomic structure model file."
    )
    main_parser.add_argument(
        "--n_batches",
        type=positive_int,
        nargs="+",
        default=60,
        help="Number of batches per merging group, or list of batch edges"
        + " where to split the data."
        + " Must be a positive integer or space-separated list of positive integers.",
    )
    main_parser.add_argument(
        "--n_bins",
        type=positive_int,
        default=20,
        help="Number of resolution bins. Must be a positive integer.",
    )
    main_parser.add_argument(
        "--servalcat_args",
        type=str,
        default=[],
        help="Command line arguments for Servalcat, recommend to put them"
        + " between apostrophes.",
    )
    main_parser.add_argument(
        "--n_proc",
        type=positive_int,
        default=4,
        help="Number of processes to use for parallelisation."
        + " Must be a positive integer.",
    )
    main_parser.add_argument(
        "--amplitude",
        action="store_true",
        help="Use amplitude rather than intensities (not recommended).",
    )
    main_parser.add_argument(
        "--molrep",
        action="store_true",
        help="Run MolRep for molecular replacement before structure refinement.",
    )
    main_parser.add_argument(
        "--bootstrap",
        type=positive_int,
        default=0,
        help="No. of bootstrapped sub data sets to be created and used for refinement."
        + " Must be a positive integer.",
    )
    main_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run (only for development).",
    )
    # TODO: if input has Friedel pairs but a user wants to merge them

    # Bootstrap mean map subcommand
    mean_parser = subparsers.add_parser(
        "mean",
        help="Calculate mean maps from bootstrapped refinement results",
        formatter_class=ArgumentDefaultsHelpFormatterCustom,
    )
    mean_parser.add_argument(
        "file_name_template",
        type=str,
        help=(
            "Template name for input files, e.g. put `dataset_llweight`"
            " for `dataset_llweight*_refine.mtz` and `dataset_llweight*_refine.mmcif`."
        ),
    )
    mean_parser.add_argument(
        "--prefix", type=str, help="Prefix for the output filename"
    )

    def validate_args(args):
        if args.n_batches and not args.hklin_unmerged:
            parser.error("--n_batches requires --hklin_unmerged to be provided.")

    main_parser.set_defaults(func=validate_args)

    return parser


def write_bin_stats(bin_stats_list, filename):
    """
    Save bin statistics to a fixed-width text file.

    Args:
        bin_stats_list (list of dict): A list where each dictionary represents
            statistics for a resolution bin. Each dictionary can have the
            following keys:
            - "bin" (int or str): Bin number or "overall" for overall statistics.
            - "dmax" (float): Maximum resolution (Å).
            - "dmin" (float): Minimum resolution (Å).
            - "dmin_star2" (float): Minimum inverse resolution squared (Å⁻²).
            - Additional keys may be included for other statistics.
        filename (str): Path to the output file.
    """
    # Convert to DataFrame and save
    stats_df = pandas.DataFrame(bin_stats_list)
    # stats_df.to_csv(stats_filename, index=False, sep="\t", float_format="%.4f")
    # Round float columns to 4 decimal places
    float_cols = stats_df.select_dtypes(include=["float"]).columns
    stats_df[float_cols] = stats_df[float_cols].round(4)
    # Save as fixed-width file
    stats_df.to_string(
        buf=open(filename, "w"),
        index=False,
        justify="right",
    )
    print(f"Saved statistics to {filename}")


def check_reflection_file_columns(hklin, unmerged=False):
    """
    Check the input reflection file for the presence of intensities,
    amplitudes and Friedel pairs.

    Args:
        hklin (str): gemmi.Mtz object or path to the input reflection file.

    Returns:
        tuple: A tuple containing three boolean values:
            - intensities_found: True if intensity columns are found.
            - amplitudes_found: True if amplitude columns are found.
            - anom: True if Friedel pairs are present.
    """
    if not isinstance(hklin, gemmi.Mtz):
        m = gemmi.read_mtz_file(hklin)
    else:
        m = hklin
    # TODO: CIF
    # dmax = m.resolution_low()
    # dmin = m.resolution_high()
    unexpected_column_warning = (
        "This is quite unusual for unmerged data file, are you sure about the file?"
    )

    anom = False
    intensities_found = False
    amplitudes_found = False
    for column in m.columns:
        if column.type == "J":
            print(
                "Column with intensity (type J, no Friedel pairs) found:", column.label
            )
            intensities_found = True
        elif column.type == "Q":
            print(
                "Column with standard deviation associated to intensity/amplitude"
                " column (type Q, no Friedel pairs) found:",
                column.label,
            )
        elif column.type == "K":
            print(
                "Column with intensity (type K, Friedel pairs)" " found:", column.label
            )
            print("Friedel pairs will be kept separately.")
            anom = True
            intensities_found = True
        elif column.type == "M":
            print(
                "Column with standard deviation associated to intensity column"
                " (type M, Friedel pairs) found:",
                column.label,
            )
        elif column.type == "G":
            print(
                "Column with amplitude (type G, Friedel pairs)" " found:", column.label
            )
            anom = True
            amplitudes_found = True
            if unmerged:
                warnings.warn(unexpected_column_warning)
        elif column.type == "L":
            print(
                "Column with standard deviation associated to amplitude"
                " (type L, Friedel pairs) found:",
                column.label,
            )
        elif column.type == "F":
            print(
                "Column with amplitude (type F, no Friedel pairs)" " found:",
                column.label,
            )
            amplitudes_found = True
            if unmerged:
                warnings.warn(unexpected_column_warning)
    return intensities_found, amplitudes_found, anom


def merge_in_groups(
    unmerged, n_bins, prefix, n_batches_per_group=60, batches_edges=[], i_group_prefix=0
):

    def merge_group(
        df_groups,
        i_group,
        cell,
        spacegroup,
        binner,
        n_expected,
        wavelength=0,
        n_groups=0,
        anom=True,
        prefix="",
        i_group_prefix=0,
    ):
        intensities = gemmi.Intensities()
        intensities.set_data(
            cell,
            spacegroup,
            df_groups[i_group][["H", "K", "L"]].values,
            df_groups[i_group]["I"].values,
            df_groups[i_group]["SIGI"].values,
        )
        if anom:
            intensities.prepare_for_merging(gemmi.DataType.Anomalous)
        else:
            intensities.prepare_for_merging(gemmi.DataType.Mean)
        bin_stats = intensities.calculate_merging_stats(binner)
        # Collect bin statistics into a list of dictionaries
        bin_stats_list = []
        for n, stats in enumerate(bin_stats):
            bin_stats_list.append(
                {
                    "bin": n + 1,
                    "dmax": binner.dmax_of_bin(n),
                    "dmin": binner.dmin_of_bin(n),
                    "CC1/2": stats.cc_half(),
                    "CC*": stats.cc_star(),
                    "Rmeas": stats.r_meas(),
                    "Rpim": stats.r_pim(),
                }
            )
        # Add overall stats as a separate entry
        overall_stats = intensities.calculate_merging_stats(None)
        bin_stats_list.append(
            {
                "bin": "overall",
                "dmax": intensities.resolution_range()[0],
                "dmin": intensities.resolution_range()[1],
                "CC1/2": overall_stats[0].cc_half(),
                "CC*": overall_stats[0].cc_star(),
                "Rmeas": overall_stats[0].r_meas(),
                "Rpim": overall_stats[0].r_pim(),
            }
        )

        if anom:
            intensities.merge_in_place(gemmi.DataType.Anomalous)
        else:
            intensities.merge_in_place(gemmi.DataType.Mean)
        # SIGI from merging:  1/sqrt(∑w), where w=1/sigma^2
        mtz_group_merged = intensities.prepare_merged_mtz(with_nobs=True)
        if wavelength:
            mtz_group_merged.dataset(0).wavelength = wavelength
        if n_groups:
            g_with_leading_zeros = str(i_group_prefix + i_group + 1).zfill(
                len(str(n_groups))
            )
        else:
            g_with_leading_zeros = i_group_prefix + i_group + 1
        # After merging, add <I> <I/sigI> n_unique n_obs completeness multiplicity
        df = pandas.DataFrame(
            data=mtz_group_merged.array, columns=mtz_group_merged.column_labels()
        )
        df["BIN"] = binner.get_bins(mtz_group_merged)
        for b in range(binner.size):
            df_bin = df[df["BIN"] == b]
            bin_n_unique_expected = gemmi.count_reflections(
                cell,
                spacegroup,
                binner.dmin_of_bin(b),
                binner.dmax_of_bin(b),
                unique=True,
            )
            bin_stats_list[b]["<I>"] = df_bin["IMEAN"].mean()
            bin_stats_list[b]["<I/sigI>"] = (
                df_bin["IMEAN"] / df_bin["SIGIMEAN"]
            ).mean()
            bin_stats_list[b]["n_unique"] = len(df_bin)
            bin_stats_list[b]["completeness"] = len(df_bin) / bin_n_unique_expected
            bin_stats_list[b]["n_obs"] = int(df_bin["NOBS"].sum())
            bin_stats_list[b]["multiplicity"] = df_bin["NOBS"].mean()
        n_unique_expected = gemmi.count_reflections(
            cell,
            spacegroup,
            intensities.resolution_range()[1],
            intensities.resolution_range()[0],
            unique=True,
        )
        bin_stats_list[-1]["<I>"] = df["IMEAN"].mean()
        bin_stats_list[-1]["<I/sigI>"] = (df["IMEAN"] / df["SIGIMEAN"]).mean()
        bin_stats_list[-1]["n_unique"] = len(df)
        # TODO: overall completeness is not correct
        bin_stats_list[-1]["completeness"] = len(df) / n_unique_expected
        bin_stats_list[-1]["n_obs"] = int(df["NOBS"].sum())
        bin_stats_list[-1]["multiplicity"] = df["NOBS"].mean()
        bin_stats_list[-1]["n_unique"] = len(intensities.miller_array)
        stats_filename = (
            f"{prefix}group{i_group_prefix + i_group + 1}_merging_stats.txt"
        )
        write_bin_stats(bin_stats_list, stats_filename)

        ## n_expected = gemmi.count_reflections(m.cell, m.spacegroup, dmin, dmax)
        completeness = len(intensities.miller_array) / n_expected
        print(
            f"Merged group {i_group_prefix + i_group + 1} of batches: #reflections:",
            len(intensities.miller_array),
            " => completeness:",
            f"{completeness:.3f}",
        )

        mtz_group_merged_filename = f"{prefix}group{g_with_leading_zeros}_I.mtz"
        mtz_group_merged.write_to_file(mtz_group_merged_filename)

        return mtz_group_merged_filename, bin_stats_list

    if unmerged.lower().endswith(".hkl"):
        xds_ascii = gemmi.read_xds_ascii(unmerged)
        m = xds_ascii.to_mtz()
        # Calculate resolution range from the unmerged file
        d_array = m.cell.calculate_d_array(m.make_miller_array())
        dmax = max(d_array)
        dmin = min(d_array)
    if unmerged.lower().endswith(".cif") or unmerged.lower().endswith(".ent"):
        doc = gemmi.cif.read(unmerged)
        rblocks = gemmi.as_refln_blocks(doc)
        for rblock in rblocks:
            if not rblock.is_merged():
                Convert = gemmi.CifToMtz()
                m = Convert.convert_block_to_mtz(rblock)
                d_array = m.cell.calculate_d_array(m.make_miller_array())
                dmax = max(d_array)
                dmin = min(d_array)
                break
    else:
        m = gemmi.read_mtz_file(unmerged)
        dmax = m.resolution_low()
        dmin = m.resolution_high()

    print(f"Resolution limits: {dmax:.3f} - {dmin:.3f} A")
    print(f"Space group: {m.spacegroup.hm} (No. {m.spacegroup.number})")
    print(
        f"Unit cell: {m.cell.a:.3f} {m.cell.b:.3f} {m.cell.c:.3f}"
        f" {m.cell.alpha:.3f} {m.cell.beta:.3f} {m.cell.gamma:.3f}"
    )
    # Scan the columns of the input unmerged MTZ file
    # and check if Friedel pairs are present or not
    anom = False
    intensities_found, amplitudes_found, anom = check_reflection_file_columns(
        m, unmerged=True
    )
    # print(m.dataset(0).wavelength) == 0.0
    # print(m.dataset(1).wavelength) OK
    # print(m.datasets[0].wavelength) == 0.0
    if m.datasets[-1].wavelength:
        wavelength = m.datasets[-1].wavelength
        print("Wavelength from input file:", m.datasets[-1].wavelength)
    else:
        wavelength = 0.0
        print("No wavelength found in input file.")
    print(
        "Setting up resolution bins according to the file",
        unmerged,
        f"with resolution limits {dmax:.3f} - {dmin:.3f} A",
    )
    binner_master = gemmi.Binner()
    binner_master.setup_from_1_d2(
        n_bins, gemmi.Binner.Method.Dstar2, m.make_1_d2_array(), m.get_cell()
    )
    # n_expected = len(gemmi.make_miller_array(m.cell, m.spacegroup, 2.2, float('inf')))
    n_expected = gemmi.count_reflections(m.cell, m.spacegroup, dmin, dmax)
    print(
        f"Expected number of reflections for resolution range ({dmax:.3f} - {dmin:.3f}"
        f" A), cell and symmetry from the input file {unmerged}:",
        n_expected,
    )

    print(f"No. batches: {len(m.batches)}")
    batch = m.batches[0]
    # print(batch)
    # print(batch.number) (0 + 1 = 1)
    # print(batch.title)
    # print(batch.axes)
    # print(batch.dataset_id)
    # print(list(batch.ints))
    # print(list(batch.floats))
    try:
        if len(batch.floats) > 37 and batch.floats[36] and batch.floats[37]:
            print(
                "Start/end phi of the first batch found:"
                f" {batch.floats[36]}, {batch.floats[37]}"
            )
        else:
            print("Batch start/end phi of the first batch not found.")
    except (IndexError, AttributeError) as e:
        print(f"Batch start and end of phi not found. Error: {e}")

    # hkl = m.make_miller_array()
    # intensity = m.column_with_label('I')
    # sigma = m.column_with_label('SIGI')
    # batch_col = m.column_with_label('BATCH')
    # print(batch_col[0])
    # print(batch.__dir__())

    df_groups = []
    df = pandas.DataFrame(data=m.array, columns=m.column_labels())
    df = df.astype({name: "int32" for name in ["H", "K", "L"]})

    # TODO: a function that converts any selection criteria in lists of batches.
    # Note that batch numberring is 1, 2, ..., 2000
    # but Python numberring is 0, 1, ..., 1999
    if batches_edges:
        batches_split = [0]
        for i in range(len(batches_edges)):
            batches_split.append(batches_split[-1] + batches_edges[i])
    elif n_batches_per_group:
        batches_split = list(range(0, len(m.batches), n_batches_per_group))
        batches_split.append(len(m.batches))
    print(batches_split)

    mtz_groups = []
    bin_stats_lists = []
    for i_group in range(len(batches_split) - 1):
        # print(batches_split[i_group], batches_split[i_group+1])
        df_group = df.loc[
            (df["BATCH"] >= batches_split[i_group])
            & (df["BATCH"] < batches_split[i_group + 1])
        ]
        df_groups.append(df_group)
        mtz_group, bin_stats_list = merge_group(
            df_groups,
            i_group,
            m.cell,
            m.spacegroup,
            binner_master,
            n_expected,
            wavelength,
            n_groups=len(batches_split),
            anom=anom,
            prefix=prefix,
            i_group_prefix=i_group_prefix,
        )
        mtz_groups.append(mtz_group)
        bin_stats_lists.append(bin_stats_list)
    n_expected_list = [n_expected] * len(mtz_groups)
    print("Merged MTZ files:", mtz_groups)
    return mtz_groups, bin_stats_lists, n_expected_list, binner_master


def run_molrep(model, mtz):
    import shutil

    prefix_local = f"{os.path.splitext(os.path.basename(mtz))[0]}_molrep"
    log_filename = prefix_local + ".log"
    pdb_filename = prefix_local + ".pdb"
    cmd = [
        "molrep",
        "-f",
        mtz,
        "-m",
        model,
    ]
    print("Running command:", " ".join(cmd))
    try:
        with open(log_filename, "w") as log_file:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
    shutil.copy2(
        os.path.join(os.getcwd(), "molrep.pdb"), os.path.join(os.getcwd(), pdb_filename)
    )
    return pdb_filename


def run_servalcat_fwt(mtz_groups_i, prefix="", n_proc=1):
    """
    Run `servalcat fw` to perform French Wilson conversion of intensities
       to structure factor amplitudes.

    Args:
        mtz_groups_i (list): List of input MTZ files.
        prefix (str): Prefix for the output files.

    Returns:
        list: List of output MTZ files, columns
              H K L F SIGF I SIG I + if Friedel pairs also F(+) SIGF(+) F(-) SIGF(-).
              servalcat fw drops NOBS, NOBS(+) and NOBS(-) columns.
    """
    print(
        "Running servalcat fw to convert intensities to structure factor amplitudes..."
    )
    mtz_groups_fi = []

    def run_fw_one(args):
        i_group, mtz_group_i = args
        group_fi_prefix = os.path.splitext(os.path.basename(mtz_group_i))[0] + "F"
        log_group_fi = f"{group_fi_prefix}.log"
        mtz_group_fi = f"{group_fi_prefix}.mtz"
        cmd = ["servalcat", "fw", "--hklin", mtz_group_i, "-o", group_fi_prefix]
        print("Running command:", " ".join(cmd))
        try:
            with open(log_group_fi, "w") as log_file:
                subprocess.run(
                    cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
            return mtz_group_fi
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as executor:
        results = list(executor.map(run_fw_one, enumerate(mtz_groups_i)))
    mtz_groups_fi.extend([r for r in results if r])
    return mtz_groups_fi


def run_servalcat_refine(
    mtzs_fi,
    models,
    mtzs_free=[],
    source="xray",
    arguments=[],
    sigmaa=True,
    quick=False,
    n_proc=1,
):  # , prefix=""):
    # TODO: source -s
    # TODO: --keyword_file, --config
    refined_mmcifs = []
    refined_mtzs = []

    def refine_one(params):
        i_mtz, (mtz_fi, mtz_free, model) = params
        local_refined_mmcifs = []
        local_refined_mtzs = []
        if mtzs_free and "--labin_llweight" in arguments:
            prefix_local = (
                f"{os.path.splitext(os.path.basename(mtz_fi))[0]}_"
                f"llweight{i_mtz}_refine"
            )
        else:
            prefix_local = f"{os.path.splitext(os.path.basename(mtz_fi))[0]}_refine"
        log_filename = prefix_local + ".log"
        cmd = [
            "servalcat",
            "refine_xtal_norefmac",
            "--hklin",
            mtz_fi,
            "--model",
            model,
            "-s",
            source,
            "--hout",
            "-o",
            prefix_local,
        ]
        if mtz_free:
            cmd.extend(["--hklin_free", mtz_free])
        if arguments:
            cmd.extend(arguments)
        if quick:
            cmd.extend(["--ncycle", "1"])
        print("Running command:", " ".join(cmd))
        try:
            with open(log_filename, "w") as log_file:
                subprocess.run(
                    cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {e}")
        if sigmaa:
            log_filename_sigmaa = prefix_local + "_sigmaa.log"
            cmd_sigmaa = [
                "servalcat",
                "sigmaa",
                "--hklin",
                mtz_fi,
                "--model",
                prefix_local + ".mmcif",
                "-s",
                source,
                "-o",
                prefix_local + "_sigmaa",
            ]
            if mtz_free:
                cmd_sigmaa.extend(["--hklin_free", mtz_free])
            if arguments:
                cmd_sigmaa.extend(arguments)
            print("Running command:", " ".join(cmd_sigmaa))
            try:
                with open(log_filename_sigmaa, "w") as log_file_sigmaa:
                    subprocess.run(
                        cmd_sigmaa,
                        check=True,
                        stdout=log_file_sigmaa,
                        stderr=subprocess.STDOUT,
                    )
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running command: {e}")
            local_refined_mtzs.append(prefix_local + "_sigmaa.mtz")
        else:
            local_refined_mtzs.append(prefix_local + ".mtz")
        local_refined_mmcifs.append(prefix_local + ".mmcif")
        return local_refined_mmcifs[0], local_refined_mtzs[0]

    if len(mtzs_fi) == len(models) >= 2:
        models_list = models
    else:
        models_list = [models[0]] * len(mtzs_fi)

    if mtzs_free and len(mtzs_free) >= 2 and len(mtzs_fi) == 1:
        # refinement after bootstrapping
        params = zip(mtzs_fi * len(mtzs_free), mtzs_free, models_list)
    elif not mtzs_free:
        # refinement after merging, no free set provided
        params = zip(mtzs_fi, [None] * len(mtzs_fi), models_list)
    elif len(mtzs_free) == 1:
        # refinement after merging, single free set provided
        params = zip(mtzs_fi, mtzs_free * len(mtzs_fi), models_list)
    else:
        # unexpected case, should not happen
        raise ValueError(
            "Unexpected case: both mtzs_fi and mtzs_free have" " more than one element."
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as executor:
        results = list(executor.map(refine_one, enumerate(params)))
    for mmcif, mtz in results:
        refined_mmcifs.append(mmcif)
        refined_mtzs.append(mtz)
    return refined_mmcifs, refined_mtzs


def calc_scale_real(df, column="FP", b=0, dmax=0.0, dmin=0.0):
    """
    Assumes that df contains columns F1 and F2.
    scale_real = sum_hkl (F1 * F2) / sum_hkl F2**2.
    If denominator is zero, returns 1.

    Args:
        df (pandas.DataFrame): DataFrame with columns for F1 and F2.
        column (str): Base name of the columns for F1 and F2.
        b (int): Bin number, used for warnings.
        dmax (float): Maximum resolution for the bin, used in warnings.
        dmin (float): Minimum resolution for the bin, used in warnings.
    Returns:
        float: Scale factor for the bin.
    """
    nomin = (df[column + "1"] * df[column + "2"]).sum()
    denomin = (df[column + "2"] ** 2).sum()
    if not numpy.isclose(denomin, 0):
        return nomin / denomin
    else:
        if b and dmax and dmin:
            warnings.warn(
                f"Scale denominator for bin {b + 1} is zero"
                f" ({dmax} - {dmin} A),"
                " setting scale for this bin to 1."
            )
        return 1.0


def calc_scale_complex(df, column="F_est", column_denom="", b=0, dmax=0.0, dmin=0.0):
    """
    Assumes that df contains columns `F_est`1RE, `F_est`1IM,
    `F_est`2RE, `F_est`2IM and `F_est`2.
    scale_complex = sum_hkl (F1RE * F2RE + F1IM * F2IM) / sum_hkl F2**2
    If denominator is zero, returns 1.

    Args:
        df (pandas.DataFrame): DataFrame with columns for F1 and F2.
        column (str): Base name of the columns for F1 and F2.
        column_denom (str): Column name for the denominator, if different from F2.
        b (int): Bin number, used for warnings.
        dmax (float): Maximum resolution for the bin, used in warnings.
        dmin (float): Minimum resolution for the bin, used in warnings.
    Returns:
        float: Scale factor for the bin.
    """
    scale_complex_numer = (
        (df[column + "1RE"] * df[column + "2RE"])
        + (df[column + "1IM"] * df[column + "2IM"])
    ).sum()
    if column_denom:
        scale_complex_denomin = (df[column_denom] ** 2).sum()
    else:
        scale_complex_denomin = (df[column + "2"] ** 2).sum()
    # equivalent to the previous line:
    # scale_complex_denomin = (df[column + '2RE']**2 + df[column + '2IM']**2).sum()
    if not numpy.isclose(scale_complex_denomin, 0):
        return scale_complex_numer / scale_complex_denomin
    else:
        warnings.warn(
            f"Scale denominator for bin {b + 1} is zero"
            f" ({dmax} - {dmin} A),"
            " setting scale for this bin to 1."
        )
        return 1.0


def compare_mtzs_fi(mtzs_fi, binner, bin_stats_matrix=[], n_expected=[]):

    # noqa: E501
    def compare_mtz_fi_pair(
        mtz_fi1, mtz_fi2, binner, bin_stats_list1=[], bin_stats_list2=[]
    ):
        # f_col = "F"
        i_col = "IMEAN"  # can be just "I" after servalcat fw
        column_label_dropna = i_col  # or F?
        mtz1 = gemmi.read_mtz_file(mtz_fi1)
        mtz2 = gemmi.read_mtz_file(mtz_fi2)
        print("")
        print(f"{str(mtz1.cell)} in file {mtz_fi1}")
        print(f"{str(mtz2.cell)} in file {mtz_fi2}")
        if mtz1.cell != mtz2.cell:
            print("WARNING: Unit cell parameters are different.")
        mtz_df1 = pandas.DataFrame(data=mtz1.array, columns=mtz1.column_labels())
        mtz_df2 = pandas.DataFrame(data=mtz2.array, columns=mtz2.column_labels())
        mtz_df1 = mtz_df1.astype({name: "int32" for name in ["H", "K", "L"]})
        mtz_df2 = mtz_df2.astype({name: "int32" for name in ["H", "K", "L"]})

        column_labels = mtz1.column_labels()
        column_labels.remove("H")
        column_labels.remove("K")
        column_labels.remove("L")
        # afterwards, rename to FP1, SIGFP1, ..., FP2, SIGFP2, ...
        # columns1 = [col + '1' for col in columns]
        column_labels_dict1 = {col: col + "1" for col in column_labels}
        # columns2 = [col + '2' for col in columns]
        column_labels_dict2 = {col: col + "2" for col in column_labels}

        # mtz_df1 = mtz_df1[['H', 'K', 'L'] + columns]  # Select only relevant columns
        mtz_df1 = mtz_df1.dropna(
            subset=[column_label_dropna]
        )  # Select only reflections with F
        mtz_df1 = mtz_df1.rename(columns=column_labels_dict1)  # Rename
        # print("")
        # print(mtz_df1.head(10))
        n_refl1 = len(mtz_df1)
        print(f"No. unique reflections: {n_refl1} in file {mtz_fi1}")

        # mtz_df2 = mtz_df2[['H', 'K', 'L'] + columns]
        mtz_df2 = mtz_df2.dropna(subset=[column_label_dropna])
        mtz_df2 = mtz_df2.rename(columns=column_labels_dict2)
        n_refl2 = len(mtz_df2)
        print(f"No. unique reflections: {n_refl2} in file {mtz_fi2}")

        # Extract common Miller indices (H, K, L)
        df = pandas.merge(mtz_df1, mtz_df2, on=["H", "K", "L"])
        n_refl = len(df)
        print(
            f"No. unique reflections: {n_refl} in common;"
            f" ratios to the originals: {n_refl / n_refl1}   {n_refl / n_refl2}"
        )
        # print("")
        # print(df.head(10))
        # print(df.describe())
        hkl_common_array = numpy.array(df[["H", "K", "L"]].values, numpy.int32)
        hkl_common_array = numpy.ascontiguousarray(hkl_common_array, dtype=numpy.int32)
        # print(len(hkl_common_array))
        # print(hkl_common_array.flags.c_contiguous)
        # print(hkl_common_array[:10])
        n_refl_list = [n_refl1, n_refl2, n_refl]

        """# Scaling per resolution bins - at least 100 reflections per bin
        n_bins = int(n_refl / 200)  # only starting point
        binner = gemmi.Binner()
        binner.setup(n_bins, gemmi.Binner.Method.Dstar2, hkl_common_array, mtz1.cell)
        bins_tmp = binner.get_bins(hkl_common_array)
        min_n_bins = min(Counter(bins_tmp).values())
        while min_n_bins <= 100:
            n_bins = n_bins - 1
            binner.setup(
                n_bins, gemmi.Binner.Method.Dstar2, hkl_common_array, mtz1.cell
            )
            bins_tmp = binner.get_bins(hkl_common_array)
            min_n_bins = min(Counter(bins_tmp).values())"""
        bins_tmp = binner.get_bins(hkl_common_array)
        min_n_bins = min(Counter(bins_tmp).values())
        if min_n_bins < 100:
            warnings.warn(
                "Less than 100 reflections per bin"
                " - it is recommended to set up a lower number of bins."
            )
        df["BIN"] = bins_tmp
        # print("Binner min_n_bins:", min_n_bins)
        n_bins = len(set(bins_tmp))  # TODO how to use args.n_bins?
        bins_stats = []
        for b in range(n_bins):
            df_bin = df[df["BIN"] == b]
            # scale_delfofo = sum_hkl F1 * F2 / sum_hkl F2**2
            """
            scale_delfofo_numer = (df_bin[f_col + "1"] * df_bin[f_col + "2"]).sum()
            scale_delfofo_denomin = (df_bin[f_col + "2"] ** 2).sum()
            scale_delfofo = scale_delfofo_numer / scale_delfofo_denomin
            ccF_iso = numpy.corrcoef(
                df_bin[f_col + "1"], scale_delfofo * df_bin[f_col + "2"]
            )[0, 1]
            rF_iso_numer = (
                abs(df_bin[f_col + "1"] - scale_delfofo * df_bin[f_col + "2"])
            ).sum()
            rF_iso_denom = (
                abs(df_bin[f_col + "1"] + scale_delFfofo * df_bin[f_col + "2"])
            ).sum()
            rF_iso = 2 * rF_iso_numer / rF_iso_denom"""
            # DELFOFO
            """df.loc[df_bin.index, 'DELFOFO'] = \
                numpy.abs(df_bin[f_col + '1'] - scale_delfofo * df_bin[f_col + '2'])
            # df.loc[df_bin.index, 'PHDELFOFO'] = df_bin['PHFC1']
            df.loc[df_bin.index, 'DELFOFOSIG'] = \
                numpy.sign(df_bin[f_col + '1'] - scale_delfofo * df_bin[f_col + '2'])"""
            """# If FP1 - scale * FP2 < 0, then add/subtract 180deg to phase
            df_bin_noflip = df[(df['BIN'] == b) & (df['DELFOFOSIG'] != -1)]
            df_bin_flip_plus = \
                df[(df['BIN'] == b) & (df['DELFOFOSIG'] == -1) & (df['PHFC1'] <= 0)]
            df_bin_flip_minus = \
                df[(df['BIN'] == b) & (df['DELFOFOSIG'] == -1) & (df['PHFC1'] > 0)]
            # print(len(df_bin_noflip), len(df_bin_flip_plus), len(df_bin_flip_minus))
            df.loc[df_bin_noflip.index, 'PHDELFOFO'] = df_bin_noflip['PHFC1']
            df.loc[df_bin_flip_plus.index, 'PHDELFOFO'] = \
                df_bin_flip_plus['PHFC1'] + 180
            df.loc[df_bin_flip_minus.index, 'PHDELFOFO'] = \
                df_bin_flip_minus['PHFC1'] - 180"""
            scale_delioio = calc_scale_real(
                df_bin, i_col, b, binner.dmax_of_bin(b), binner.dmin_of_bin(b)
            )
            ccI_iso = numpy.corrcoef(
                df_bin[i_col + "1"], scale_delioio * df_bin[i_col + "2"]
            )[0, 1]
            """
            rI_iso_numer = \
                (abs(df_bin[i_col + "1"] - scale_delioio * df_bin[i_col + "2"])).sum()
            rI_iso_denom = \
                (abs(df_bin[i_col + "1"] + scale_delioio * df_bin[i_col + "2"])).sum()
            rI_iso = 2 * rI_iso_numer / rI_iso_denom"""

            bins_stats.append(
                {
                    "bin": b + 1,
                    "dmax": binner.dmax_of_bin(b),
                    "dmin": binner.dmin_of_bin(b),
                    "dmin_star2": 1 / (binner.dmin_of_bin(b) ** 2),
                    "count": len(df_bin),
                    # "scale_delfofo": scale_delfofo,
                    # "ccF_iso": ccF_iso,
                    # "rF_iso": rF_iso,
                    "scale_delioio": scale_delioio,
                    "ccI_iso": ccI_iso,
                    # "rI_iso": rI_iso,
                }
            )

        if (
            bin_stats_list1
            and bin_stats_list2
            and len(bin_stats_list1) == n_bins + 1
            and len(bin_stats_list2) == n_bins + 1
        ):
            # Add CC* from bin_stats_list1 and bin_stats_list2 if available
            for b in range(n_bins):
                bins_stats[b]["CC*1"] = bin_stats_list1[b]["CC*"]
                bins_stats[b]["CC*2"] = bin_stats_list2[b]["CC*"]
                bins_stats[b]["CC12true"] = bins_stats[b]["ccI_iso"] / (
                    bins_stats[b]["CC*1"] * bins_stats[b]["CC*2"]
                )

        bins_stats_df = pandas.DataFrame(bins_stats)
        # Calculate weighted average of cc over bins
        """ccF_iso_avg = (
            bins_stats_df["ccF_iso"] * bins_stats_df["count"]
        ).sum() / bins_stats_df["count"].sum()"""
        ccI_iso_avg = (
            bins_stats_df["ccI_iso"] * bins_stats_df["count"]
        ).sum() / bins_stats_df["count"].sum()
        # cc_iso_avg_list = [ccF_iso_avg, ccI_iso_avg]
        mtz_fi1_base = os.path.splitext(os.path.basename(mtz_fi1))[0]
        mtz_fi2_base = os.path.splitext(os.path.basename(mtz_fi2))[0]

        # Make a plot
        """def star2(x):
            # Vectorized 1/x^2, treating x==0 manually
            x = numpy.array(x, float)
            near_zero = numpy.isclose(x, 0)
            x[near_zero] = 9999
            x[~near_zero] = 1 / (x[~near_zero] ** 2)
            return x
        def star_sqrt(x):
            # Vectorized 1/sqrt(x), treating x<0 manually
            x = numpy.array(x, float)
            negative = numpy.less(x, 0)
            x[negative] = 9999
            x[~negative] = 1 / numpy.sqrt(x[~negative])
            return x"""
        plt.figure(figsize=(8, 6))
        plt.plot(
            bins_stats_df["dmin_star2"],
            bins_stats_df["ccI_iso"],
            marker="o",
            label="ccI_iso",
        )
        if "CC*1" in bins_stats_df.columns:
            plt.plot(
                bins_stats_df["dmin_star2"],
                bins_stats_df["CC*1"],
                marker="s",
                label="CC*1",
            )
        if "CC*2" in bins_stats_df.columns:
            plt.plot(
                bins_stats_df["dmin_star2"],
                bins_stats_df["CC*2"],
                marker="^",
                label="CC*2",
            )
        if "CC12true" in bins_stats_df.columns:
            plt.plot(
                bins_stats_df["dmin_star2"],
                bins_stats_df["CC12true"],
                marker="x",
                label="CC12true",
            )
        plt.xlabel("Resolution (Å⁻²)")
        # Show dmin labels on a secondary x-axis at the bottom
        ax = plt.gca()
        """secax = ax.secondary_xaxis(
            'bottom',
            functions=(lambda x: 1 / numpy.sqrt(x), lambda d: 1 / (d ** 2))
            )
        secax = ax.secondary_xaxis('top', functions=(star_sqrt, star2))"""
        secax = ax.secondary_xaxis("top")
        secax.set_xlabel("Resolution (Å)")
        N = max(1, len(bins_stats_df) // 5)  # Show only some labels to avoid overlap
        ticks = bins_stats_df["dmin_star2"]
        dmins = bins_stats_df["dmin"]
        labels = []
        for i, d in enumerate(dmins):
            if i == 0 or i == len(dmins) - 1:  # Always show the first and last labels
                labels.append(f"{d:.2f}")
            elif i % N == 0:
                labels.append(f"{d:.2f}")
            else:
                labels.append("")
        secax.set_xticks(ticks)
        secax.set_xticklabels(labels)
        plt.ylabel("Correlation")
        plt.title("Correlation statistics per resolution bin")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{mtz_fi1_base}_vs_{mtz_fi2_base}_cc_plot.png")
        plt.close()
        # Round float columns to 4 decimal places and save as fixed-width file
        stats_filename = f"{mtz_fi1_base}_vs_{mtz_fi2_base}_bin_stats.txt"
        write_bin_stats(bins_stats, stats_filename)
        return bins_stats, n_refl_list, ccI_iso_avg

    n_refl_matrix = numpy.zeros((len(mtzs_fi), len(mtzs_fi)), dtype=int)
    ratio_refl_matrix = numpy.identity(len(mtzs_fi), dtype=float)
    # ccF_iso_matrix = numpy.identity(len(mtzs_fi), dtype=float)
    ccI_iso_matrix = numpy.identity(len(mtzs_fi), dtype=float)
    for i in range(len(mtzs_fi)):
        for j in range(i + 1, len(mtzs_fi)):
            # print(i, j)
            bins_stats, n_refl_list, ccI_iso_avg = compare_mtz_fi_pair(
                mtzs_fi[i],
                mtzs_fi[j],
                binner,
                bin_stats_matrix[i][i],
                bin_stats_matrix[j][j],
            )
            bin_stats_matrix[i][j] = bins_stats
            bin_stats_matrix[j][i] = bin_stats_matrix[i][j]
            if i == 0:
                n_refl_matrix[j, j] = n_refl_list[1]
                if j == 1:
                    n_refl_matrix[i, i] = n_refl_list[0]
            n_refl_matrix[i, j] = n_refl_list[2]
            n_refl_matrix[j, i] = n_refl_list[2]
            # No. reflections in common / No. reflections in the first file
            ratio_refl_matrix[i, j] = n_refl_list[2] / n_refl_list[0]
            # No. reflections in common / No. reflections in the second file
            ratio_refl_matrix[j, i] = n_refl_list[2] / n_refl_list[1]
            # ccF_iso_matrix[i, j] = cc_iso_avg_list[0]
            # ccF_iso_matrix[j, i] = cc_iso_avg_list[0]
            ccI_iso_matrix[i, j] = ccI_iso_avg
            ccI_iso_matrix[j, i] = ccI_iso_avg
    print("No. unique reflections:")
    print(n_refl_matrix)
    if n_expected and len(n_expected) == len(mtzs_fi):
        completeness_matrix = n_refl_matrix / max(n_expected)
        print("Completeness:")
        print(completeness_matrix)
    print(
        "Ratio of No. unique reflections in common and No. reflections in a data set:"
    )
    print(ratio_refl_matrix)
    # print("Average CCFiso:")
    # print(ccF_iso_matrix)
    print("Average CCIiso:")
    print(ccI_iso_matrix)
    # TODO: multiplicity
    return bin_stats_matrix, n_refl_matrix, ratio_refl_matrix


def write_mtz_from_df(df, mtz_ref, columns, filename):
    """
    Create a gemmi.Mtz object from a pandas dataframe and save to file.
    The numpy.float32 format is used for the data.

    Args:
        df (pandas.DataFrame): DataFrame containing columns for H, K, L and other data.
        mtz_ref (gemmi.Mtz): Reference MTZ object for cell and spacegroup.
        columns (dict): Dictionary of column names and their MTZ data types
            to include after H, K, L.
        filename (str): Output filename for the MTZ file.
    Returns:
        None
    """
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = mtz_ref.spacegroup
    mtz.set_cell_for_all(mtz_ref.cell)
    mtz.add_dataset(mtz_ref.datasets[0].dataset_name)
    for col_name, col_type in columns.items():
        mtz.add_column(col_name, col_type)
    data = numpy.array(
        df[["H", "K", "L"] + list(columns.keys())].values,
        numpy.float32,
    )
    mtz.set_data(data)
    mtz.write_to_file(filename)
    print(f"Saved {len(df)} reflections to {filename}.")
    return


def adp_analysis_histograms(modelPaths, prefix=""):

    def adp_analysis(modelPath):
        """
        Perform ADP analysis on a given structure model (mmCIF)
        and return statistics.

        Args:
            modelPath (str): Path to the mmCIF model file.
        Returns:
            tuple: Contains the following elements:
                - adp_dict["All"] (list): List of all ADP values.
                - hist (numpy.ndarray): Histogram of ADP values.
                - bin_edges (numpy.ndarray): Edges of the histogram bins.
                - median (float): Median of ADP values.
                - mad (float): Median Absolute Deviation of ADP values.
                - q1 (float): First quartile of ADP values.
                - q3 (float): Third quartile of ADP values.
                - iqr (float): Interquartile range of ADP values.
        """
        print(f"Running ADP analysis for {modelPath}")
        adp_dict = {}
        adp_per_resi = {}
        adp_dict["All"] = []
        cif_block = gemmi.cif.read(modelPath)[0]
        st = gemmi.make_structure_from_block(cif_block)
        # st = gemmi.read_structure(modelPath)
        for model in st:
            for chain in model:
                polymer = chain.get_polymer()
                ptype = polymer.check_polymer_type()
                adp_dict[chain.name] = []
                adp_per_resi[chain.name] = {"resi": [], "adp": [], "adp_sidechain": []}
                for residue in chain:
                    adp_this_resi = []
                    adp_this_resi_sidechain = []
                    for atom in residue:
                        if not atom.is_hydrogen() and atom.occ > 0:
                            if atom.aniso.nonzero():
                                adp_atom = gemmi.calculate_b_est(atom)
                            else:
                                adp_atom = atom.b_iso
                            adp_dict["All"].append(adp_atom)
                            adp_dict[chain.name].append(adp_atom)
                            if (
                                residue.entity_type == gemmi.EntityType.Polymer
                                and ptype
                                in [
                                    gemmi.PolymerType.PeptideL,
                                    gemmi.PolymerType.PeptideD,
                                ]
                                and atom.name not in ["CA", "C", "O", "N", "OXT"]
                            ):
                                adp_this_resi_sidechain.append(adp_atom)
                            else:
                                adp_this_resi.append(adp_atom)
                    try:
                        if residue.seqid.num in adp_per_resi[chain.name]["resi"]:
                            continue  # ignoring insertion codes, sorry
                        adp_per_resi[chain.name]["resi"].append(residue.seqid.num)
                        if adp_this_resi:
                            adp_per_resi[chain.name]["adp"].append(
                                numpy.mean(adp_this_resi)
                            )
                        else:
                            adp_per_resi[chain.name]["adp"].append(None)
                        if adp_this_resi_sidechain:
                            adp_per_resi[chain.name]["adp_sidechain"].append(
                                numpy.mean(adp_this_resi_sidechain)
                            )
                        else:
                            adp_per_resi[chain.name]["adp_sidechain"].append(None)
                    except (KeyError, AttributeError) as e:
                        warnings.warn(
                            f"Error processing residue {residue.seqid}"
                            f" in chain {chain.name}: {e}"
                        )

        median = numpy.median(adp_dict["All"])
        mad = numpy.median(numpy.absolute(adp_dict["All"] - median))
        q1 = numpy.quantile(adp_dict["All"], 0.25)
        q3 = numpy.quantile(adp_dict["All"], 0.75)
        iqr = q3 - q1

        return adp_dict["All"], median, mad, q1, q3, iqr

    plt.figure(figsize=(8, 6))
    max_value = 0
    for modelPath in modelPaths:
        values, median, mad, q1, q3, iqr = adp_analysis(modelPath)
        max_value = max(max_value, max(values))
        plt.hist(
            values, alpha=0.7, histtype="step", label=f"{modelPath}", density=True
        )  # bins=30, edgecolor='black'
        plt.axvline(
            median,
            color="b",
            linestyle="--",
            label=f"Median = {median:.2f}; MAD = {mad:.2f}",
            alpha=0.7,
        )
        plt.axvline(
            q1,
            color="r",
            linestyle="--",
            label=f"Q1 = {q1:.2f}; IQR = {iqr:.2f}",
            alpha=0.7,
        )
        plt.axvline(q3, color="r", linestyle="--", label=f"Q3 = {q3:.2f}", alpha=0.7)
    plt.xlabel("ADP (Atomic Displacement Parameter)")
    plt.ylabel("Frequency")
    plt.title("ADP Histogram")
    plt.legend(loc="upper right")
    plt.xlim(0, max(values) * 1.1)
    png_filename = f"{prefix}adp_histogram.png"
    plt.savefig(png_filename)
    # TODO: same histogram ranges
    return png_filename


def compute_difference_maps_pair(mtz_file_1, mtz_file_2, binner, bin_stats_list=[]):
    """
    Compute difference maps between two MTZ files from `servalcat refine_xtal_norefmac`
    or `servalcat sigmaa`, save the results in a new MTZ file and save the statistics
    for each resolution bin in a txt file.

    Args:
        mtz_file_1 (str): Path to the first MTZ file.
        mtz_file_2 (str): Path to the second MTZ file.
        binner (gemmi.Binner): Binner object for resolution binning.
        bin_stats_list (list of dict): A list where each dictionary represents
            statistics for a resolution bin.
    Returns:
        bin_stats_list (list of dict): Updated list with statistics
            for each resolution bin.
    """

    mtz1 = gemmi.read_mtz_file(mtz_file_1)
    mtz2 = gemmi.read_mtz_file(mtz_file_2)
    columns_fwt = ["FWT", "PHWT"]
    if all(col in mtz1.column_labels() for col in ["F_est", "DFC", "PHDFC"]) and all(
        col in mtz2.column_labels() for col in ["F_est", "DFC", "PHDFC"]
    ):
        columns_fwt += ["Fcombi", "PHDFC"]
        columns_fwt1 = [col + "1" for col in columns_fwt]
        columns_fwt1_dict = {col: col + "1" for col in columns_fwt}
        columns_fwt2 = [col + "2" for col in columns_fwt]
        columns_fwt2_dict = {col: col + "2" for col in columns_fwt}

    mtz_df1 = pandas.DataFrame(data=mtz1.array, columns=mtz1.column_labels())
    mtz_df1 = mtz_df1.astype({name: "int32" for name in ["H", "K", "L"]})
    mtz_fwt_df1 = mtz_df1.copy()
    mtz_fwt_df1["Fcombi"] = mtz_fwt_df1["F_est"].combine_first(mtz_fwt_df1["DFC"])

    mtz_df2 = pandas.DataFrame(data=mtz2.array, columns=mtz2.column_labels())
    mtz_df2 = mtz_df2.astype({name: "int32" for name in ["H", "K", "L"]})
    mtz_fwt_df2 = mtz_df2.copy()
    mtz_fwt_df2["Fcombi"] = mtz_fwt_df2["F_est"].combine_first(mtz_fwt_df2["DFC"])

    if "F_est" in mtz_df1.columns and "F_est" in mtz_df2.columns:
        f_col = "F_est"  # Use also FP?
        columns = ["F_est"]  # Do we need SIGFP?
    else:
        raise ValueError("No column with amplitudes found.")
        raise ValueError("No column with amplitudes found.")
    columns += ["FWT", "PHWT", "FC", "PHFC"]
    # afterwards, rename to FP1, SIGFP1, ..., FP2, SIGFP2, ...
    # columns1 = [col + "1" for col in columns]
    columns1_dict = {col: col + "1" for col in columns}
    # columns2 = [col + "2" for col in columns]
    columns2_dict = {col: col + "2" for col in columns}

    mtz_df1 = mtz_df1[["H", "K", "L"] + columns]  # Select only relevant columns
    mtz_df1 = mtz_df1.dropna(subset=[f_col])  # Select only reflections with F
    mtz_df1 = mtz_df1.rename(columns=columns1_dict)  # Rename
    n_refl1 = len(mtz_df1)
    print("")
    print(f"No. unique reflections: {n_refl1} in file {mtz_file_1}")

    mtz_df2 = mtz_df2[["H", "K", "L"] + columns]
    mtz_df2 = mtz_df2.dropna(subset=[f_col])
    mtz_df2 = mtz_df2.rename(columns=columns2_dict)
    n_refl2 = len(mtz_df2)
    print(f"No. unique reflections: {n_refl2} in file {mtz_file_2}")

    # Extract common Miller indices (H, K, L)
    df = pandas.merge(mtz_df1, mtz_df2, on=["H", "K", "L"])
    n_refl = len(df)
    print(
        f"No. unique reflections: {n_refl} in common;"
        f" ratios to the originals: {n_refl / n_refl1}   {n_refl / n_refl2}"
    )
    hkl_common_array = numpy.array(df[["H", "K", "L"]].values, numpy.int32)
    hkl_common_array = numpy.ascontiguousarray(hkl_common_array, dtype=numpy.int32)
    # print(len(hkl_common_array))  # should be equal to n_refl

    # Scaling per resolution bins
    df["BIN"] = binner.get_bins(hkl_common_array)

    df["FP1RE"] = df[f_col + "1"] * numpy.cos(numpy.deg2rad(df["PHFC1"]))
    df["FP1IM"] = df[f_col + "1"] * numpy.sin(numpy.deg2rad(df["PHFC1"]))
    df["FP2RE"] = df[f_col + "2"] * numpy.cos(numpy.deg2rad(df["PHFC2"]))
    df["FP2IM"] = df[f_col + "2"] * numpy.sin(numpy.deg2rad(df["PHFC2"]))
    df["FWT1RE"] = df["FWT1"] * numpy.cos(numpy.deg2rad(df["PHWT1"]))
    df["FWT1IM"] = df["FWT1"] * numpy.sin(numpy.deg2rad(df["PHWT1"]))
    df["FWT2RE"] = df["FWT2"] * numpy.cos(numpy.deg2rad(df["PHWT2"]))
    df["FWT2IM"] = df["FWT2"] * numpy.sin(numpy.deg2rad(df["PHWT2"]))

    # Difference map types:
    # + DELFOFO  or DELFESFES   SR (scaling real)
    # + DELFOFO2 or DELFESFES2  SC (scaling complex)
    # + DELFWTFWT2              SC (scaling complex)
    # + DELFWTFWT2all           SC (scaling complex)

    if not bin_stats_list:
        bin_stats_list = [
            {
                "bin": b + 1,
                "dmax": binner.dmax_of_bin(b),
                "dmin": binner.dmin_of_bin(b),
            }
            for b in range(binner.size)
        ]
    if len(bin_stats_list) != binner.size:
        warnings.warn(
            f"bin_stats_list has {len(bin_stats_list)} bins,"
            f" but binner has {binner.size} bins.",
        )
    for b in range(len(bin_stats_list)):
        df_bin = df[df["BIN"] == b]
        # scale_delfofo = sum_hkl FP1 * FP2 / sum_hkl FP2**2
        scale_delfofo = calc_scale_real(
            df_bin, f_col, b, bin_stats_list[b]["dmax"], bin_stats_list[b]["dmin"]
        )
        scale_delfofo2sc = calc_scale_complex(
            df_bin,
            "FP",
            f_col + "2",
            b,
            bin_stats_list[b]["dmax"],
            bin_stats_list[b]["dmin"],
        )
        scale_delfwtfwt2sc = calc_scale_complex(
            df_bin,
            "FWT",
            "FWT2",
            b,
            bin_stats_list[b]["dmax"],
            bin_stats_list[b]["dmin"],
        )

        if len(df_bin) < 100:
            warnings.warn(
                f"Less than 100 reflections in bin {b + 1}"
                f" ({bin_stats_list[b]['dmax']:.4f} -"
                f" {bin_stats_list[b]['dmin']:.4f} A)."
            )
        bin_stats_list[b]["scale_delfofo"] = scale_delfofo
        bin_stats_list[b]["delfofo_count"] = len(df_bin)
        bin_stats_list[b]["scale_delfofo2sc"] = scale_delfofo2sc
        bin_stats_list[b]["scale_delfwtfwt2sc"] = scale_delfwtfwt2sc

        # DELFOFO
        df.loc[df_bin.index, "DELFOFO"] = numpy.abs(
            df_bin[f_col + "1"] - scale_delfofo * df_bin[f_col + "2"]
        )
        df.loc[df_bin.index, "DELFOFOSIG"] = numpy.sign(
            df_bin[f_col + "1"] - scale_delfofo * df_bin[f_col + "2"]
        )
        # If FP1 - scale * FP2 < 0, then add/subtract 180deg to phase
        df_bin_noflip = df[(df["BIN"] == b) & (df["DELFOFOSIG"] != -1)]
        df_bin_flip_plus = df[
            (df["BIN"] == b) & (df["DELFOFOSIG"] == -1) & (df["PHFC1"] <= 0)
        ]
        df_bin_flip_minus = df[
            (df["BIN"] == b) & (df["DELFOFOSIG"] == -1) & (df["PHFC1"] > 0)
        ]
        df.loc[df_bin_noflip.index, "PHDELFOFO"] = df_bin_noflip["PHFC1"]
        df.loc[df_bin_flip_plus.index, "PHDELFOFO"] = df_bin_flip_plus["PHFC1"] + 180
        df.loc[df_bin_flip_minus.index, "PHDELFOFO"] = df_bin_flip_minus["PHFC1"] - 180

        # DELFOFO2SC
        df.loc[df_bin.index, "DELFOFO2SCRE"] = (
            df_bin["FP1RE"] - scale_delfofo2sc * df_bin["FP2RE"]
        )
        df.loc[df_bin.index, "DELFOFO2SCIM"] = (
            df_bin["FP1IM"] - scale_delfofo2sc * df_bin["FP2IM"]
        )
        # df.loc[df_bin.index, 'DELFWTFWT2SC'] = numpy.sqrt(
        #    (df_bin['FWT1RE'] - scale_delfwtfwt2sc*df_bin['FWT2RE'])**2 + \
        #    (df_bin['FWT1IM'] - scale_delfwtfwt2sc*df_bin['FWT2IM'])**2)
        df.loc[df_bin.index, "DELFOFO2SC"] = numpy.sqrt(
            df["DELFOFO2SCRE"] ** 2 + df["DELFOFO2SCIM"] ** 2
        )
        df.loc[df_bin.index, "PHDELFOFO2SC"] = numpy.rad2deg(
            numpy.arctan2(df["DELFOFO2SCIM"], df["DELFOFO2SCRE"])
        )

        # DELFWTFWT2SC
        df.loc[df_bin.index, "DELFWTFWT2SCRE"] = (
            df_bin["FWT1RE"] - scale_delfwtfwt2sc * df_bin["FWT2RE"]
        )
        df.loc[df_bin.index, "DELFWTFWT2SCIM"] = (
            df_bin["FWT1IM"] - scale_delfwtfwt2sc * df_bin["FWT2IM"]
        )
        # df.loc[df_bin.index, 'DELFWTFWT2SC'] = numpy.sqrt(
        #    (df_bin['FWT1RE'] - scale_delfwtfwt2sc*df_bin['FWT2RE'])**2 + \
        #    (df_bin['FWT1IM'] - scale_delfwtfwt2sc*df_bin['FWT2IM'])**2)
        df.loc[df_bin.index, "DELFWTFWT2SC"] = numpy.sqrt(
            df["DELFWTFWT2SCRE"] ** 2 + df["DELFWTFWT2SCIM"] ** 2
        )
        df.loc[df_bin.index, "PHDELFWTFWT2SC"] = numpy.rad2deg(
            numpy.arctan2(df["DELFWTFWT2SCIM"], df["DELFWTFWT2SCRE"])
        )

    mtz_fi1_base = os.path.splitext(os.path.basename(mtz_file_1))[0]
    mtz_fi2_base = os.path.splitext(os.path.basename(mtz_file_2))[0]
    output_prefix = f"{mtz_fi1_base}_vs_{mtz_fi2_base}_diffmaps"
    output_mtz = f"{output_prefix}.mtz"
    columns_to_write_list = [
        "DELFOFO",
        "PHDELFOFO",
        "DELFOFO2SC",
        "PHDELFOFO2SC",
        "DELFWTFWT2SC",
        "PHDELFWTFWT2SC",
    ]
    columns_to_write_dict = {
        col: ("F" if not col.startswith("PH") else "P") for col in columns_to_write_list
    }
    write_mtz_from_df(df, mtz1, columns_to_write_dict, output_mtz)

    # For DELFWTFWT2SCall map, use all the reflections
    mtz_fwt_df1 = mtz_fwt_df1.dropna(subset=["FWT"])
    mtz_fwt_df1 = mtz_fwt_df1.rename(columns=columns_fwt1_dict)
    mtz_fwt_df1 = mtz_fwt_df1[["H", "K", "L"] + columns_fwt1]
    mtz_fwt_df2 = mtz_fwt_df2.dropna(subset=["FWT"])
    mtz_fwt_df2 = mtz_fwt_df2.rename(columns=columns_fwt2_dict)
    mtz_fwt_df2 = mtz_fwt_df2[["H", "K", "L"] + columns_fwt2]
    df_fwt = pandas.merge(mtz_fwt_df1, mtz_fwt_df2, on=["H", "K", "L"])
    hkl_common_array_fwt = numpy.array(df_fwt[["H", "K", "L"]].values, numpy.int32)
    hkl_common_array_fwt = numpy.ascontiguousarray(
        hkl_common_array_fwt, dtype=numpy.int32
    )
    output_mtz_fwt = f"{output_prefix}_fwt.mtz"
    print(f"No. reflections in {output_mtz_fwt}: {len(df_fwt)}")
    binner_fwt = binner
    df_fwt["BIN"] = binner_fwt.get_bins(hkl_common_array_fwt)

    df_fwt["FWT1RE"] = df_fwt["FWT1"] * numpy.cos(numpy.deg2rad(df_fwt["PHWT1"]))
    df_fwt["FWT1IM"] = df_fwt["FWT1"] * numpy.sin(numpy.deg2rad(df_fwt["PHWT1"]))
    df_fwt["FWT2RE"] = df_fwt["FWT2"] * numpy.cos(numpy.deg2rad(df_fwt["PHWT2"]))
    df_fwt["FWT2IM"] = df_fwt["FWT2"] * numpy.sin(numpy.deg2rad(df_fwt["PHWT2"]))
    df_fwt["Fcombi1RE"] = df_fwt["Fcombi1"] * numpy.cos(numpy.deg2rad(df_fwt["PHDFC1"]))
    df_fwt["Fcombi1IM"] = df_fwt["Fcombi1"] * numpy.sin(numpy.deg2rad(df_fwt["PHDFC1"]))
    df_fwt["Fcombi2RE"] = df_fwt["Fcombi2"] * numpy.cos(numpy.deg2rad(df_fwt["PHDFC2"]))
    df_fwt["Fcombi2IM"] = df_fwt["Fcombi2"] * numpy.sin(numpy.deg2rad(df_fwt["PHDFC2"]))
    for b in range(len(bin_stats_list)):
        df_fwt_bin = df_fwt[df_fwt["BIN"] == b]
        scale_delfwtfwt2scall = calc_scale_complex(df_fwt_bin, "FWT")
        scale_delfestfest2scall = calc_scale_complex(df_fwt_bin, "Fcombi")
        bin_stats_list[b]["scale_delfwtfwt2scall"] = scale_delfwtfwt2scall
        bin_stats_list[b]["delfwtfwt2scall_count"] = len(df_fwt_bin)
        bin_stats_list[b]["scale_delfestfest2scall"] = scale_delfestfest2scall
        bin_stats_list[b]["scale_delfwtfwt2sc"] = scale_delfwtfwt2sc
        df_fwt.loc[df_fwt_bin.index, "DELFWTFWT2SCallRE"] = (
            df_fwt_bin["FWT1RE"] - scale_delfwtfwt2scall * df_fwt_bin["FWT2RE"]
        )
        df_fwt.loc[df_fwt_bin.index, "DELFWTFWT2SCallIM"] = (
            df_fwt_bin["FWT1IM"] - scale_delfwtfwt2scall * df_fwt_bin["FWT2IM"]
        )
        df_fwt.loc[df_fwt_bin.index, "DELFWTFWT2SCall"] = numpy.sqrt(
            df_fwt["DELFWTFWT2SCallRE"] ** 2 + df_fwt["DELFWTFWT2SCallIM"] ** 2
        )
        df_fwt.loc[df_fwt_bin.index, "PHDELFWTFWT2SCall"] = numpy.rad2deg(
            numpy.arctan2(df_fwt["DELFWTFWT2SCallIM"], df_fwt["DELFWTFWT2SCallRE"])
        )
        df_fwt.loc[df_fwt_bin.index, "DELFestFest2SCallRE"] = (
            df_fwt_bin["Fcombi1RE"] - scale_delfestfest2scall * df_fwt_bin["Fcombi2RE"]
        )
        df_fwt.loc[df_fwt_bin.index, "DELFestFest2SCallIM"] = (
            df_fwt_bin["Fcombi1IM"] - scale_delfestfest2scall * df_fwt_bin["Fcombi2IM"]
        )
        df_fwt.loc[df_fwt_bin.index, "DELFestFest2SCall"] = numpy.sqrt(
            df_fwt["DELFestFest2SCallRE"] ** 2 + df_fwt["DELFestFest2SCallIM"] ** 2
        )
        df_fwt.loc[df_fwt_bin.index, "PHDELFestFest2SCall"] = numpy.rad2deg(
            numpy.arctan2(df_fwt["DELFestFest2SCallIM"], df_fwt["DELFestFest2SCallRE"])
        )
    columns_to_write_list = [
        "DELFWTFWT2SCall",
        "PHDELFWTFWT2SCall",
        "DELFestFest2SCall",
        "PHDELFestFest2SCall",
    ]
    columns_to_write_dict = {
        col: ("F" if not col.startswith("PH") else "P") for col in columns_to_write_list
    }
    write_mtz_from_df(df_fwt, mtz1, columns_to_write_dict, output_mtz_fwt)
    stats_filename = f"{mtz_fi1_base}_vs_{mtz_fi2_base}_bin_stats.txt"
    write_bin_stats(bin_stats_list, stats_filename)
    return bin_stats_list


def compute_difference_maps(refined_mtzs, binner, bin_stats_matrix=[]):
    for i in range(len(refined_mtzs)):
        for j in range(i + 1, len(refined_mtzs)):
            # print(i, j)
            bin_stats_diff = compute_difference_maps_pair(
                refined_mtzs[i],
                refined_mtzs[j],
                binner,
                bin_stats_matrix[i][j],
            )
            if bin_stats_matrix:
                """print(
                    len(bin_stats_matrix[i][j]),
                    len(bin_stats_matrix[j][i]),
                    len(bin_stats_diff),
                )"""
                for b in range(len(bin_stats_matrix[i][j])):
                    try:
                        bin_stats_matrix[i][j][b].update(bin_stats_diff[b])
                        bin_stats_matrix[j][i][b].update(bin_stats_diff[b])
                    except IndexError:
                        warnings.warn(
                            f"IndexError: bin_stats_matrix[{i}][{j}] or"
                            f" bin_stats_matrix[{j}][{i}] does not have"
                            f" enough bins."
                        )
                        break

    return bin_stats_matrix


def compute_structure_differences(refined_mmcifs):
    for i in range(len(refined_mmcifs)):
        for j in range(i + 1, len(refined_mmcifs)):
            compute_structure_differences_pair(
                refined_mmcifs[i],
                refined_mmcifs[j],
                output=f"{refined_mmcifs[i]}_vs_{refined_mmcifs[j]}_differences.csv",
            )


def makeAddressStr(cra):
    address = cra.chain.name + "/" + cra.residue.name + " "
    address += str(cra.residue.seqid.num)
    if cra.residue.seqid.icode.strip():
        address += str(cra.residue.seqid.icode)
    address += "/"
    address += cra.atom.name
    if cra.atom.has_altloc():
        address += "."
        address += cra.atom.altloc
    return address


def compute_structure_differences_pair(
    structure1, structure2, output=None, minCoordDev=0, minAdpDev=0
):

    def search(st1Cras, st2Cras, output, minCoordDev, minAdpDev):
        records = []
        for cra1 in st1Cras:
            for j, cra2 in enumerate(st2Cras):
                if (
                    cra1.atom.name == cra2.atom.name
                    and cra1.atom.altloc == cra2.atom.altloc
                    and cra1.residue.name == cra2.residue.name
                    and cra1.residue.seqid == cra2.residue.seqid
                    and cra1.chain.name == cra2.chain.name
                ):
                    coordDev = cra1.atom.pos.dist(cra2.atom.pos)
                    adpDev = cra2.atom.b_iso - cra1.atom.b_iso
                    if coordDev >= minCoordDev or abs(adpDev) >= minAdpDev:
                        record = {
                            "AtomAddress": makeAddressStr(cra1),
                            "CoordDev": round(coordDev, 2),
                            "ADPDev": round(adpDev, 2),
                        }
                        records.append(record)
                    del st2Cras[j]
                    break
        df = pandas.DataFrame.from_records(records)
        if output:
            df.to_csv(output, index=False)
        return df

    st1 = gemmi.read_structure(structure1)
    st2 = gemmi.read_structure(structure2)
    st1Cras = list(st1[0].all())
    st2Cras = list(st2[0].all())
    if len(st1Cras) != len(st2Cras):
        warnings.warn(
            f"Number of atoms in {structure1} does not match the number"
            f" of atoms in {structure2}."
        )
    return search(st1Cras, st2Cras, output, minCoordDev, minAdpDev)


def bootstrap_dataset(mtz_file, binner, seeds=[1001, 1002, 1003]):
    """
    Bootstrap the dataset from an MTZ file and save the results in new MTZ files.

    Args:
        mtz_file (str): Path to the input MTZ file.
        seeds (list of int): List of random seeds for bootstrapping.
        n_bins (int): Number of resolution bins to use for bootstrapping.
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

    print("\nBootstrapping dataset", mtz_file)
    mtzs_out = []
    mtz = gemmi.read_mtz_file(mtz_file)
    df = pandas.DataFrame(data=mtz.array, columns=mtz.column_labels())
    df = df.astype({name: "int32" for name in ["H", "K", "L"]})

    i_col = "IMEAN"  # can be just "I" after servalcat fw or sigmaa, or IMEAN?
    column_label_dropna = i_col  # or F?
    if column_label_dropna in df.columns:
        df = df.dropna(subset=[column_label_dropna])

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
            warnings.warn(
                f"Sum of weight coefficients {weight_sum} does not match the"
                f" number of reflections {len(df)}."
            )

        # TODO: FreeR_flag
        # Save the llweights in the MTZ file
        mtz_out_name = (
            f"{os.path.splitext(os.path.basename(mtz_file))[0]}_llweight_{i}.mtz"
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
    completeness_std = numpy.std(completeness_list)
    print(
        f"Completeness of bootstrap datasets:"
        f" {completeness_mean:.2%} ± {completeness_std:.2%}"
    )

    return mtzs_out


def bootstrap_analyse_structures(refined_mmcifs, idx=0, prefix="", skip_hydrogen=True):
    """
    Analyse structure models (mmCIF files) to compute mean coordinates and B-factors.
    The structure models are expected to be after refinement against a bootstrapped
    data set. They must have the same number of atoms and the same atom identifiers.

    Args:
        refined_mmcifs (list of str): List of mmCIF filenames.
        idx (int): Index for naming the output files (applies if not set to 0).
        prefix (str): Prefix for the output filenames.
        skip_hydrogen (bool): If True, skip hydrogen atoms in the analysis.

    Returns:
        None: Writes the statistics in 'bootstrap_stats.csv' and
              the mean structure to 'bootstrap_mean_structure.mmcif' with
              where 1000 * sigma_coordinate is saved as B-values.
    """

    # numpy.set_printoptions(threshold=numpy.inf)
    st_master = gemmi.read_structure(refined_mmcifs[0])
    st_master_cras = list(st_master[0].all())
    if skip_hydrogen:
        st_master_cras = [cra for cra in st_master_cras if not cra.atom.is_hydrogen()]
    print(len(st_master_cras), "atoms in the master structure")

    atom_addresses = [makeAddressStr(cra) for cra in st_master_cras]
    coords = numpy.zeros(
        (len(st_master_cras), 3, len(refined_mmcifs)), dtype=numpy.float32
    )
    b_values = numpy.zeros(
        (len(st_master_cras), len(refined_mmcifs)), dtype=numpy.float32
    )

    print(f"Loading {len(refined_mmcifs)} structure models...")
    # Collect coordinates and B-values
    for s, mmcif in enumerate(refined_mmcifs):
        st = gemmi.read_structure(mmcif)
        st_cras = list(st[0].all())
        if skip_hydrogen:
            st_cras = [cra for cra in st_cras if not cra.atom.is_hydrogen()]
        assert len(st_master_cras) == len(st_cras), "Different number of atoms in"
        f" structure models after bootstrapping: {mmcif}."
        for a, (cra_master, cra) in enumerate(zip(st_master_cras, st_cras)):
            assert (
                cra_master.atom.name == cra.atom.name
                and cra_master.atom.altloc == cra.atom.altloc
                and cra_master.residue.name == cra.residue.name
                and cra_master.residue.seqid == cra.residue.seqid
                and cra_master.chain.name == cra.chain.name
            ), f"Inconsistent structure models after bootstrapping: {mmcif}."
            coords[a, :, s] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
            b_values[a, s] = cra.atom.b_iso

    # Compute mean and standard deviation per atom
    mean_coords = numpy.mean(coords, axis=2)  # shape: (n_atoms, 3)
    std_coords = numpy.std(coords, axis=2)  # shape: (n_atoms, 3)
    # std_coords_norm = sqrt(σ_x² + σ_y² + σ_z²)
    #  (when assuming no correlation which is not the case)
    # std_coords_norm = numpy.linalg.norm(std_coords, axis=1)  # shape: (n_atoms,)
    #
    # std_coords_norm = sqrt(σ_x² + σ_y² + σ_z² + 2 * (σ_xy + σ_xz + σ_yz))
    # Calculate joint sigma of coordinates, assuming correlation between x, y, z
    std_coords_norm = numpy.zeros(len(st_master_cras))
    for i in range(len(st_master_cras)):
        cov = numpy.cov(coords[i, :, :])
        std_coords_norm[i] = numpy.sqrt(
            numpy.trace(cov) + 2 * (cov[0, 1] + cov[0, 2] + cov[1, 2])
        )
    mean_b_values = numpy.mean(b_values, axis=1)  # shape: (n_atoms,)
    std_b_values = numpy.std(b_values, axis=1)  # shape: (n_atoms,)

    # Write calculated data as a CSV file
    csv_data = []
    for i, atom_address in enumerate(atom_addresses):
        csv_data.append(
            {
                "atom_id": atom_address,
                "mean_x": mean_coords[i][0],
                "mean_y": mean_coords[i][1],
                "mean_z": mean_coords[i][2],
                "sigma_x": std_coords[i][0],
                "sigma_y": std_coords[i][1],
                "sigma_z": std_coords[i][2],
                "sigma_coord": std_coords_norm[i],
                "mean_b": mean_b_values[i],
                "sigma_b": std_b_values[i],
            }
        )
    df_csv = pandas.DataFrame(csv_data)
    csv_filename = f"{prefix}group{idx}_mean_stats.csv" if idx else "mean_stats.csv"
    df_csv.to_csv(csv_filename, index=False)
    print(f"Mean structure statistics written to {csv_filename}.")

    # Write mean structure as mmCIF
    for i, cra in enumerate(st_master_cras):
        # Replace position with mean coordinates
        cra.atom.pos = gemmi.Position(*mean_coords[i])
        # Replace B-factor with norm of std deviation (or square it if desired)
        cra.atom.b_iso = 1000 * std_coords_norm[i]  # or (8π²/3)*σ² ???
    mmcif_filename = (
        f"{prefix}group{idx}_mean_structure.mmcif" if idx else "mean_structure.mmcif"
    )
    st_master.make_mmcif_document().write_file(mmcif_filename)
    print(f"Mean structure written to {mmcif_filename}.")
    return


def bootstrap_mean_map(refined_mtzs, idx=0, prefix="", binner=None):
    """
    Calculate the mean 2Fo-Fc and Fo-Fc maps from refined MTZ files after bootstrapping.
    The maps are expected to be after refinement against a bootstrapped
    data set.

    Args:
        refined_mtzs (list of str): List of MTZ filenames.
        idx (int): Index for naming the output file (applies if not set to 0).
        prefix (str): Prefix for the output filename.

    Returns:
        None: Writes the mean maps in '{prefix}bootstrap_mean_map.mtz' or
              '{prefix}group{idx}_bootstrap_mean_map.mtz' if idx is set.
    """

    def merge_reflections_bootstrap(
        df_master, mtz_ref=None, prefix="", suffix="", idx=0, binner=None
    ):
        """
        Merge reflections from the master DataFrame and calculate mean maps.

        Args:
            df_master (pandas.DataFrame): DataFrame containing reflections.
                It must contain columns "H", "K", "L", "F_complex", "DEL_F_complex",
            mtz_ref (gemmi.Mtz): Reference MTZ object for cell and spacegroup.
            prefix (str): Prefix for the output filename.
            suffix (str): Suffix for the output filename.
            idx (int): Index for naming the output file.

        Returns:
            pandas.DataFrame: DataFrame with mean maps.
        """

        # noqa: E741
        def is_centric_vectorized(h, k, l):  # noqa: E741
            return mtz_ref.spacegroup.operations().is_reflection_centric(
                (int(h), int(k), int(l))  # noqa: E741
            )

        def calculate_mean_var(df, is_centric=False):
            """Calculate variance of structure factors;
            separately for a/centric reflections."""

            if is_centric:
                var_func = lambda x: (  # noqa: E731
                    numpy.sqrt(numpy.var(numpy.abs(x))) if len(x) > 1 else 0
                )
            else:
                var_func = lambda x: (  # noqa: E731
                    numpy.sqrt(
                        (numpy.var(numpy.real(x)) + numpy.var(numpy.imag(x))) / 2
                    )
                    if len(x) > 1
                    else 0
                )

            df_mean_f = (
                df.groupby(["H", "K", "L"])["F_complex"]
                .agg(
                    [
                        ("F_complex_mean", lambda x: numpy.mean(x)),
                        ("SIGFWT", var_func),
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
                        ("SIGDELFWT", var_func),
                        ("DELFWTcount", "count"),
                    ]
                )
                .reset_index()
            )

            return df_mean_f.merge(df_mean_delf, on=["H", "K", "L"], how="outer")

        df_master = df_master.astype({col: "int32" for col in ["H", "K", "L"]})
        is_centric_vec = numpy.vectorize(is_centric_vectorized)
        centric_mask = is_centric_vec(df_master["H"], df_master["K"], df_master["L"])

        # Calculate mean and variance for a/centric reflections separately
        acentric = df_master[~centric_mask]
        centric = df_master[centric_mask]
        stats_acentric = calculate_mean_var(acentric, is_centric=False)
        stats_centric = calculate_mean_var(centric, is_centric=True)
        df_mean = pandas.concat([stats_acentric, stats_centric], ignore_index=True)

        # Convert to amplitude and phase
        df_mean["FWT"] = numpy.abs(df_mean["F_complex_mean"])
        df_mean["PHWT"] = numpy.rad2deg(numpy.angle(df_mean["F_complex_mean"]))
        df_mean["DELFWT"] = numpy.abs(df_mean["DEL_F_complex_mean"])
        df_mean["PHDELWT"] = numpy.rad2deg(numpy.angle(df_mean["DEL_F_complex_mean"]))

        if mtz_ref and prefix and suffix and idx:
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
                mtz_ref,
                columns,
                filename=mtz_filename,
            )

        # Calculate statistics per bin
        if binner and mtz_ref:
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
                        mtz_ref.cell,
                        mtz_ref.spacegroup,
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

        return df_mean

    print(f"Loading {len(refined_mtzs)} density maps...")
    columns_selected = ["H", "K", "L", "FWT", "PHWT", "DELFWT", "PHDELWT", "llweight"]
    for i, mtz_file in enumerate(refined_mtzs):
        mtz = gemmi.read_mtz_file(mtz_file)
        col_labels = mtz.column_labels()
        df = pandas.DataFrame(data=mtz.array, columns=col_labels)
        df = df[columns_selected]
        if not df.empty:
            if i == 0:
                df_master = df.copy()
                df_master = df_master.astype(
                    {name: "int32" for name in ["H", "K", "L"]}
                )
            else:
                df_master = pandas.concat([df_master, df], ignore_index=True)
        else:
            warnings.warn(f"No reflections in {mtz_file} for FWT/PHWT/DELFWT/PHDELWT.")

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

    mtz_ref = gemmi.read_mtz_file(refined_mtzs[0])
    # save 3 mean maps: all reflections, llweight == 0 and llweight > 0
    merge_reflections_bootstrap(df_master, mtz_ref, prefix, "_all", idx, binner)
    merge_reflections_bootstrap(
        df_master_llweight_0, mtz_ref, prefix, "_llweight0", idx, binner
    )
    merge_reflections_bootstrap(
        df_master_llweight_pos, mtz_ref, prefix, "_llweightpos", idx, binner
    )

    return


def main():
    print("Command line:", " ".join(sys.argv))
    print("Running multixem version:", __version__)

    parser = create_parser()
    args = parser.parse_args()
    print("Arguments parsed:", args)

    pprint.pprint(vars(args))
    if args.prefix:
        prefix = args.prefix
        if args.prefix[-1] != "_":
            prefix += "_"
    else:
        prefix = "multixem_"
    print("Prefix for the output files:", prefix)

    # subcommand mean
    if args.command == "mean":
        # args.func(args)
        import glob

        refined_mmcifs = glob.glob(f"{args.file_name_template}*_refine.mmcif")
        refined_mmcifs2 = glob.glob(f"{args.file_name_template}*_refine.cif")
        refined_mmcifs = refined_mmcifs + refined_mmcifs2
        if refined_mmcifs:
            bootstrap_analyse_structures(refined_mmcifs, 1, prefix)
        else:
            print(
                "No refined mmCIF files found with a filename template"
                f"{args.file_name_template}*_refine.mmcif"
            )
        refined_mtzs = glob.glob(f"{args.file_name_template}*_refine.mtz")
        if refined_mtzs:
            bootstrap_mean_map(refined_mtzs, 1, prefix)
        else:
            print(
                "No refined MTZ files found with a filename template"
                f"{args.file_name_template}*_refine.mtz"
            )
        return

    elif args.command != "pipeline" and args.command is not None:
        parser.print_help()
        return

    # elif args.command == 'pipeline' or args.command is None:
    # Run main pipeline

    working_dir_name = f"multixem_{prefix[:-1]}"
    os.mkdir(working_dir_name)
    os.chdir(working_dir_name)
    print("Current working directory:", os.getcwd())
    n_proc = min(os.cpu_count(), args.n_proc)
    servalcat_args = args.servalcat_args.split() if args.servalcat_args else []
    mtzs_i = []
    bin_stats_lists = []
    n_expected_list = []
    binner_master = None

    if args.hklin_unmerged:
        print("Unmerged diffraction data:", args.hklin_unmerged)
        n_groups = 0
        mtz_groups_i = []
        bin_stats_lists = []
        mtzs_fi = []
        for i, hklin_unmerged in enumerate(args.hklin_unmerged):
            print("")
            print("Unmerged diffraction data file:", hklin_unmerged)
            # TODO: select automatically the number of batches in group (now default 60)
            if len(args.n_batches) == 1:
                n_batches_per_group = args.n_batches[0]
                print("Number of batches in merging group:", n_batches_per_group)
                _mtz_groups_i, _bin_stats_lists, _n_expected_list, _binner_master = (
                    merge_in_groups(
                        hklin_unmerged,
                        args.n_bins,
                        prefix,
                        n_batches_per_group=n_batches_per_group,
                        i_group_prefix=n_groups,
                    )
                )
            else:
                _mtz_groups_i, _bin_stats_lists, _n_expected_list, _binner_master = (
                    merge_in_groups(
                        hklin_unmerged,
                        args.n_bins,
                        prefix,
                        batches_edges=args.n_batches,
                        i_group_prefix=n_groups,
                    )
                )
            mtz_groups_i.extend(_mtz_groups_i)
            n_groups = len(mtz_groups_i)
            bin_stats_lists.extend(_bin_stats_lists)
            n_expected_list.extend(_n_expected_list)
            if i == 0:
                binner_master = _binner_master
            if args.amplitude:
                _mtzs_fi = run_servalcat_fwt(_mtz_groups_i, prefix, n_proc)
                mtzs_fi.extend(_mtzs_fi)
            # TODO: free reflections if not given
            # TODO: check that input files have FI(R?)
            # TODO: mmCIF
        mtzs_i = mtz_groups_i

    if args.hklin:
        print("Merged diffraction data:", args.hklin)
        for i, mtz_i in enumerate(args.hklin):
            print("")
            print("Merged diffraction data file:", mtz_i)
            if mtz_i.lower().endswith(".cif") or mtz_i.lower().endswith(".ent"):
                doc = gemmi.cif.read(mtz_i)
                rblocks = gemmi.as_refln_blocks(doc)
                for rblock in rblocks:
                    if rblock.is_merged():
                        Convert = gemmi.CifToMtz()
                        mtz = Convert.convert_block_to_mtz(rblock)
                        break
            else:
                mtz = gemmi.read_mtz_file(mtz_i)
            print(
                f"Resolution limits: {mtz.resolution_low():.3f}"
                f" - {mtz.resolution_high():.3f} A"
            )
            print(f"Space group: {mtz.spacegroup.hm} (No. {mtz.spacegroup.number})")
            print(
                f"Unit cell: {mtz.cell.a:.3f} {mtz.cell.b:.3f} {mtz.cell.c:.3f}"
                f" {mtz.cell.alpha:.3f} {mtz.cell.beta:.3f} {mtz.cell.gamma:.3f}"
            )
            i_present, f_present, anom_present = check_reflection_file_columns(
                mtz, unmerged=False
            )
            if not i_present and not f_present:
                raise RuntimeError(
                    f"Neither intensities nor amplitudes present in {mtz_i}."
                )
            elif f_present and not i_present:
                warnings.warn(
                    "The file contain only amplitudes but not intensities, however,"
                    " providing intensities is recommended."
                )
            # elif i_present and not f_present: TODO FW
            mtzs_i.append(mtz_i)
            bin_stats_lists.append([])
            # TODO: check and fix n_expected
            n_expected = gemmi.count_reflections(
                mtz.cell, mtz.spacegroup, mtz.resolution_high(), mtz.resolution_low()
            )
            n_expected_list.append(n_expected)
            if not binner_master or mtz.resolution_high() < 1 / numpy.sqrt(
                binner_master.max_1_d2
            ):
                print(
                    "Setting up resolution bins according to the file",
                    mtz_i,
                    f"with resolution limits {mtz.resolution_low():.3f}"
                    f" - {mtz.resolution_high():.3f} A",
                )
                binner_master = gemmi.Binner()
                binner_master.setup_from_1_d2(
                    args.n_bins,
                    gemmi.Binner.Method.Dstar2,
                    mtz.make_1_d2_array(),
                    mtz.get_cell(),
                )

    bin_stats_matrix = len(mtzs_i) * [len(mtzs_i) * [None]]
    for i in range(len(mtzs_i)):
        bin_stats_matrix[i][i] = bin_stats_lists[i]
    if len(mtzs_i) >= 2:
        bin_stats_matrix, n_refl_matrix, ratio_refl_matrix = compare_mtzs_fi(
            mtzs_i, binner_master, bin_stats_matrix, n_expected_list
        )

    if args.model:
        if args.molrep:
            models = []
            for mtz_i in mtzs_i:
                print("Running MolRep to generate a model from the input structure.")
                model_molrep = run_molrep(args.model, mtz_i)
                models.append(model_molrep)
        else:
            models = [args.model]
        refined_mmcifs, refined_mtzs = run_servalcat_refine(
            mtzs_i,
            models,
            mtzs_free=[args.hklin_free],
            arguments=servalcat_args,
            quick=args.quick,
            n_proc=n_proc,
        )
        adp_analysis_histograms(refined_mmcifs, prefix)
        if len(refined_mmcifs) >= 2:
            compute_structure_differences(refined_mmcifs)
            bin_stats_matrix = compute_difference_maps(
                refined_mtzs, binner_master, bin_stats_matrix
            )
        if args.bootstrap:
            if args.amplitude:
                mtzs_in = mtzs_fi
            else:
                mtzs_in = mtzs_i
            for i_mtz, mtz_in in enumerate(mtzs_in):
                mtzs_bootstrap = bootstrap_dataset(
                    mtz_in, binner_master, seeds=range(1001, 1001 + args.bootstrap)
                )
                refined_mmcifs_bootstrap, refined_mtzs_bootstrap = run_servalcat_refine(
                    [mtz_in],
                    models,
                    mtzs_free=mtzs_bootstrap,
                    arguments=servalcat_args + ["--labin_llweight", "llweight"],
                    sigmaa=False,
                    quick=args.quick,
                    n_proc=n_proc,
                )
                bootstrap_analyse_structures(
                    refined_mmcifs_bootstrap, i_mtz + 1, prefix
                )
                bootstrap_mean_map(
                    refined_mtzs_bootstrap, i_mtz + 1, prefix, binner_master
                )
