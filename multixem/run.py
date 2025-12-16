# coding: utf-8
import os
import sys
import argparse
import subprocess
import numpy
import pandas
import gemmi
import matplotlib.pyplot as plt
import matplotlib
import logging
import json
import shlex
import glob
from collections import Counter
import concurrent.futures
from . import __version__
from .tools import write_bin_stats, calc_scale_real, write_mtz_from_df
from .analyse_refinement import (
    adp_analysis_histograms,
    compute_difference_maps,
    compute_structure_differences,
)
from .bootstrap import (
    bootstrap_dataset,
    bootstrap_analyse_stats,
    bootstrap_analyse_structures,
    bootstrap_mean_map,
    select_cids_for_geometry_analysis,
    geometry_analysis_load,
    unrestrain,
)

matplotlib.use("Agg")


def setup_logging():
    """
    Set up logging to output to both console and file.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters - simple format for INFO, detailed for others
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno in [logging.INFO, logging.DEBUG]:
                return record.getMessage()
            else:
                formatted_time = self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S")
                return f"{formatted_time} - {record.levelname} - {record.getMessage()}"

    formatter = CustomFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - capture everything including DEBUG
    file_handler = logging.FileHandler("multixem.log", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Capture warnings and route them through logging
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)

    # Also capture uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow KeyboardInterrupt to be handled normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        # Log uncaught exceptions with full traceback
        import traceback

        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.error(f"Uncaught exception:\n{tb_str}")

    sys.excepthook = handle_exception

    return logger


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

    def existing_directory(path):
        abs_norm_path = os.path.abspath(os.path.normpath(path))
        if not os.path.isdir(abs_norm_path):
            raise argparse.ArgumentTypeError(
                f"Directory does not exist: {abs_norm_path}"
            )
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

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Parent parser for common options
    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument(
        "--n_bins",
        type=positive_int,
        default=20,
        help="Number of resolution bins. Must be a positive integer.",
    )
    common_parent.add_argument(
        "-p", "--prefix", type=str, help="Prefix for the output files."
    )
    common_parent.add_argument(
        "--n_proc",
        type=positive_int,
        default=4,
        help="Number of processes to use for parallelisation."
        + " Must be a positive integer.",
    )
    common_parent.add_argument(
        "--geometry_cids",
        type=existing_file,
        help=(
            "Input file with atomic CIDs defining bonds, angles or torsions to be"
            " analysed while performing bootstrap."
        ),
    )

    common_refinement_parent = argparse.ArgumentParser(add_help=False)
    common_refinement_parent.add_argument(
        "--servalcat_args",
        type=str,
        default=[],
        help="Command line arguments for Servalcat (quoted)."
        + " Do not use options -s, --source and --hout here.",
    )
    common_refinement_parent.add_argument(
        "--model",
        type=existing_file,
        nargs="+",
        help="Input atomic structure model file(s).",
        required=False,
    )
    # TODO more files
    common_refinement_parent.add_argument(
        "--hklin",
        type=existing_file,
        nargs="+",
        help="Input merged diffraction data file(s).",
    )
    common_refinement_parent.add_argument(
        "--hklin_free", type=existing_file, help="Input MTZ file for test flags."
    )
    common_refinement_parent.add_argument(
        "--amplitude",
        action="store_true",
        help="Use amplitude rather than intensities (not recommended).",
    )
    common_refinement_parent.add_argument(
        "--model_dir",
        type=existing_directory,
        help=(
            "Directory containing multiple input atomic structure model files"
            " for bootstrap."
        ),
    )
    common_refinement_parent.add_argument(
        "--unre",
        type=positive_int,
        default=0,
        help="Number of cycles for prior unrestrained refinement during bootstrap.",
    )
    common_refinement_parent.add_argument(
        "--quick",
        action="store_true",
        help="Quick run (only for development).",
    )
    # TODO: if input has Friedel pairs but a user wants to merge them
    common_refinement_parent.add_argument(
        "-s",
        "--source",
        type=str,
        default="xray",
        choices=["xray", "electron", "neutron"],
        help="Radiation source, xray or electron or neutron.",
    )

    # Main pipeline subcommand (default behavior)
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        parents=[common_parent, common_refinement_parent],
        help="Run the main refinement pipeline",
        formatter_class=ArgumentDefaultsHelpFormatterCustom,
    )
    pipeline_parser.add_argument(
        "-u",
        "--hklin_unmerged",
        type=existing_file,
        nargs="+",
        help="Input unmerged diffraction data file(s).",
    )
    pipeline_parser.add_argument(
        "--n_batches",
        type=positive_int,
        nargs="+",
        default=0,  # default values set up in validate_args()
        help="Number of batches per merging group, or list of batch edges"
        + " where to split the data."
        + " Must be a positive integer or space-separated list of positive integers.",
    )
    pipeline_parser.add_argument(
        "--merge_whole_file",
        action="store_true",
        help="Merge all the batches in the input unmerged file(s).",
    )
    pipeline_parser.add_argument(
        "--molrep",
        action="store_true",
        help="Run MolRep for molecular replacement before structure refinement.",
    )
    pipeline_parser.add_argument(
        "--unify_cell",
        action="store_true",
        help=(
            "Set the same unit cell parameters for all datasets, "
            "based on the first dataset."
        ),
    )
    pipeline_parser.add_argument(
        "--bootstrap",
        type=positive_int,
        default=0,
        help=(
            "No. of bootstrap resampled sub data sets to be created and"
            " used for refinement. Must be a positive integer higher than 1."
        ),
    )

    # mean subcommand - mean and std for statistics, model and maps
    mean_parser = subparsers.add_parser(
        "mean",
        parents=[common_parent],
        help="Calculate mean maps from bootstrapped refinement results",
        formatter_class=ArgumentDefaultsHelpFormatterCustom,
    )
    mean_parser.add_argument(
        "file_name_template",
        type=str,
        help=(
            "Template name for input files, e.g. put `dataset`"
            " for `dataset_llweight*_refine.mtz` and `dataset_llweight*_refine.mmcif`."
        ),
    )
    mean_parser.add_argument(
        "--cif",
        type=existing_file,
        help="Path to a small molecule CIF file.",
    )

    # Bootstrap subcommand (optional new command)
    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        parents=[common_parent, common_refinement_parent],
        help="Run multiple refinement against bootstrap sub data sets",
        formatter_class=ArgumentDefaultsHelpFormatterCustom,
    )
    bootstrap_parser.add_argument(
        "n_samples",
        type=positive_int,
        default=5000,
        help=(
            "No. of bootstrap resampled sub data sets to be created and"
            " used for refinement. Must be a positive integer higher than 1."
        ),
    )

    def validate_common(args):
        if args.n_proc <= 0:
            parser.error("--n_proc must be positive")

    def validate_model_dir(args, n_required):
        if not args.model_dir:
            return
        model_files = []
        for pattern in ["*.pdb", "*.cif", "*.mmcif"]:
            model_files.extend(glob.glob(os.path.join(args.model_dir, pattern)))

        if len(model_files) < n_required:
            parser.error(
                f"--model_dir contains {len(model_files)} structure model files, "
                f"but at least {n_required} are required."
            )
        args.models = sorted(model_files)[:n_required]

    def validate_pipeline(args):
        validate_common(args)
        if args.n_batches and not args.hklin_unmerged:
            parser.error("--n_batches requires --hklin_unmerged to be provided.")
        if args.merge_whole_file and not args.hklin_unmerged:
            parser.error("--merge_whole_file requires --hklin_unmerged to be provided.")
        if args.hklin_unmerged and not args.n_batches and not args.merge_whole_file:
            args.n_batches = [60]  # Default
        if args.model and len(args.model) > 1:
            if len(args.model or []) < (
                len(args.hklin or []) + len(args.hklin_unmerged or [])
            ):
                parser.error(
                    "Just a single model can be provided for multiple data sets, "
                    "or the number of models must match the number of data sets."
                )
        if args.bootstrap and args.bootstrap < 2:
            parser.error("--bootstrap must be at least 2.")
        if args.geometry_cids and not args.bootstrap:
            parser.error("--geometry_cids requires --bootstrap to be provided.")
        if args.unre and not args.bootstrap:
            parser.error("--unre requires --bootstrap to be provided.")
        if args.bootstrap and args.model_dir:
            validate_model_dir(args, args.bootstrap)

    def validate_mean(args):
        validate_common(args)

    def validate_bootstrap(args):
        validate_common(args)
        # TODO
        if not args.hklin:
            parser.error("--hklin is required for bootstrap.")
        if args.n_samples < 2:
            parser.error("Number of samples must be at least 2.")
        if args.model_dir:
            model_files = glob.glob(os.path.join(args.model_dir, "*.pdb"))
            model_files += glob.glob(os.path.join(args.model_dir, "*.cif"))
            model_files += glob.glob(os.path.join(args.model_dir, "*.mmcif"))
            if len(model_files) < args.n_samples:
                parser.error(
                    f"--model_dir contains {len(model_files)} model files, "
                    f"but at least {args.n_samples} are required."
                )
            args.models = sorted(model_files)[: args.n_samples]
        if args.model_dir:
            validate_model_dir(args, args.n_samples)

    pipeline_parser.set_defaults(func=validate_pipeline)
    mean_parser.set_defaults(func=validate_mean)
    bootstrap_parser.set_defaults(func=validate_bootstrap)

    # Parse and validate immediately
    original_parse_args = parser.parse_args

    def parse_and_validate(cmd_args=None):
        args = original_parse_args(cmd_args)
        if hasattr(args, "func"):
            args.func(args)
        return args

    parser.parse_args = parse_and_validate

    return parser


def check_reflection_file_columns(hklin, unmerged=False, prefer_amplitude=False):
    """
    Check the input reflection file for the presence of intensities,
    amplitudes and Friedel pairs. Find the column labels and decide which to use.

    Args:
        hklin (str): gemmi.Mtz object or path to the input reflection file.
        unmerged (bool): Whether the input file is unmerged data.
        prefer_amplitude (bool): If both intensities and amplitudes are found,
            prefer amplitudes over intensities.

    Returns:
        tuple: A tuple containing three boolean values:
            - labin (str): Column labels for --labin option in Servalcat.
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

    col_iplusminus = []
    col_sigiplusminus = []
    col_imean = []
    col_fplusminus = []
    col_sigfplusminus = []
    col_fmean = []
    col_sigxmean = []

    for column in m.columns:
        if column.type == "K":
            logging.info(
                f"Column with intensity (type K, Friedel pairs) found: {column.label}"
            )
            logging.info("Friedel pairs will be kept separately.")
            anom = True
            col_iplusminus.append(column.label)
        elif column.type == "M":
            logging.info(
                f"Column with standard deviation associated to intensity column"
                f" (type M, Friedel pairs) found: {column.label}"
            )
            col_sigiplusminus.append(column.label)
        elif column.type == "J":
            logging.info(
                f"Column with intensity (type J, no Friedel pairs)"
                f" found: {column.label}"
            )
            col_imean.append(column.label)
        elif column.type == "Q":
            logging.info(
                f"Column with standard deviation associated to intensity/amplitude"
                f" column (type Q, no Friedel pairs) found: {column.label}"
            )
            col_sigxmean.append(column.label)
        elif column.type == "G":
            logging.info(
                f"Column with amplitude (type G, Friedel pairs) found: {column.label}"
            )
            anom = True
            col_fplusminus.append(column.label)
            if unmerged:
                logging.warning(unexpected_column_warning)
        elif column.type == "L":
            logging.info(
                f"Column with standard deviation associated to amplitude"
                f" (type L, Friedel pairs) found: {column.label}"
            )
            col_sigfplusminus.append(column.label)
        elif column.type == "F":
            logging.info(
                f"Column with amplitude (type F, no Friedel pairs)"
                f" found: {column.label}"
            )
            col_fmean.append(column.label)
            if unmerged:
                logging.warning(unexpected_column_warning)

    labin = None
    if not prefer_amplitude:
        if len(col_iplusminus) >= 2 and len(col_sigiplusminus) >= 2:
            labin = (
                f"{col_iplusminus[0]},{col_sigiplusminus[0]},"
                f"{col_iplusminus[1]},{col_sigiplusminus[1]}"
            )
        elif col_imean and col_sigxmean:
            labin = f"{col_imean[0]},{col_sigxmean[0]}"
    if not labin:
        if len(col_fplusminus) >= 2 and len(col_sigfplusminus) >= 2:
            labin = (
                f"{col_fplusminus[0]},{col_sigfplusminus[0]},"
                f"{col_fplusminus[1]},{col_sigfplusminus[1]}"
            )
        elif col_fmean and col_sigxmean:
            labin = f"{col_fmean[0]},{col_sigxmean[0]}"

    if not labin:
        raise RuntimeError(f"Neither intensities nor amplitudes found in {hklin}.")
    elif (
        (len(col_fplusminus) >= 2 and len(col_sigfplusminus) >= 2)
        or (col_fmean and col_sigxmean)
    ) and not (
        (len(col_iplusminus) >= 2 and len(col_sigiplusminus) >= 2)
        or (col_imean and col_sigxmean)
    ):
        logging.warning(
            "The file contain only amplitudes but not intensities, however,"
            " providing intensities is recommended."
        )
    logging.info(f"Using these columns for refinement: {labin}")

    return labin, anom


def copy_cell_mtz(mtz_input, mtz_reference):
    """
    Copy unit cell parameters from a reference MTZ file to
    an MTZ file using gemmi.

    Args:
        mtz_input (str): Path to input MTZ file
        mtz_reference (str): Path to reference MTZ file
    Returns:
        str: Output MTZ file name
    """
    mtz = gemmi.read_mtz_file(mtz_input)
    mtz_ref = gemmi.read_mtz_file(mtz_reference)
    mtz.set_cell_for_all(mtz_ref.cell)
    mtz_out_filepath = os.path.basename(mtz_input).replace(".mtz", "_cell.mtz")
    mtz.write_to_file(mtz_out_filepath)
    logging.info(
        f"Copied unit cell parameters"
        f" {mtz_ref.cell.a} {mtz_ref.cell.b} {mtz_ref.cell.c}"
        f" {mtz_ref.cell.alpha} {mtz_ref.cell.beta} {mtz_ref.cell.gamma}"
        f" from {mtz_reference} to {mtz_out_filepath}"
    )
    return mtz_out_filepath


def merge_in_groups(
    unmerged,
    n_bins,
    prefix,
    n_batches_per_group=60,
    batches_edges=[],
    merge_whole_file=False,
    i_group_prefix=0,
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
        logging.info(
            f"Merged group {i_group_prefix + i_group + 1} of batches: #reflections:"
            f" {len(intensities.miller_array)}"
            f" => completeness: {completeness:.3f}",
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

    logging.info(
        f"Resolution limits: {dmax:.3f} - {dmin:.3f} A\n"
        f"Space group: {m.spacegroup.hm} (No. {m.spacegroup.number})\n"
        f"Unit cell: {m.cell.a:.3f} {m.cell.b:.3f} {m.cell.c:.3f}"
        f" {m.cell.alpha:.3f} {m.cell.beta:.3f} {m.cell.gamma:.3f}"
    )
    # Scan the columns of the input unmerged MTZ file
    # and check if Friedel pairs are present or not
    anom = False
    labin, anom = check_reflection_file_columns(m, unmerged=True)
    # print(m.dataset(0).wavelength) == 0.0
    # print(m.dataset(1).wavelength) OK
    # print(m.datasets[0].wavelength) == 0.0
    if m.datasets[-1].wavelength:
        wavelength = m.datasets[-1].wavelength
        logging.info(f"Wavelength from input file: {m.datasets[-1].wavelength}")
    else:
        wavelength = 0.0
        logging.info("No wavelength found in input file.")
    logging.info(
        "Setting up resolution bins according to the file"
        f" {unmerged} with resolution limits {dmax:.3f} - {dmin:.3f} A"
    )
    binner_master = gemmi.Binner()
    binner_master.setup_from_1_d2(
        n_bins, gemmi.Binner.Method.Dstar2, m.make_1_d2_array(), m.get_cell()
    )
    # n_expected = len(gemmi.make_miller_array(m.cell, m.spacegroup, 2.2, float('inf')))
    n_expected = gemmi.count_reflections(m.cell, m.spacegroup, dmin, dmax)
    logging.info(
        f"Expected number of reflections for resolution range ({dmax:.3f} - {dmin:.3f}"
        f" A), cell and symmetry from the input file {unmerged}: {str(n_expected)}"
    )

    logging.info(f"No. batches: {len(m.batches)}")
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
            logging.info(
                "Start/end phi of the first batch found:"
                f" {batch.floats[36]}, {batch.floats[37]}"
            )
        else:
            logging.info("Batch start/end phi of the first batch not found.")
    except (IndexError, AttributeError) as e:
        logging.info(f"Batch start and end of phi not found. Error: {e}")

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
    if merge_whole_file:
        batches_split = [0, len(m.batches)]
    elif batches_edges:
        batches_split = [0]
        for i in range(len(batches_edges)):
            batches_split.append(batches_split[-1] + batches_edges[i])
    elif n_batches_per_group:
        batches_split = list(range(0, len(m.batches), n_batches_per_group))
        batches_split.append(len(m.batches))
    logging.info(f"Batch edges for merging groups: {batches_split}")

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
    logging.info(f"Merged MTZ files: {mtz_groups}")
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
    logging.info("Running command: %s", shlex.join(cmd))
    try:
        with open(log_filename, "w") as log_file:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running command: {e}")
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
    logging.info(
        "Running servalcat fw to convert intensities to structure factor amplitudes..."
    )
    mtz_groups_fi = []

    def run_fw_one(args):
        i_group, mtz_group_i = args
        group_fi_prefix = os.path.splitext(os.path.basename(mtz_group_i))[0] + "F"
        log_group_fi = f"{group_fi_prefix}.log"
        mtz_group_fi = f"{group_fi_prefix}.mtz"
        cmd = ["servalcat", "fw", "--hklin", mtz_group_i, "-o", group_fi_prefix]
        logging.info("Running command: %s", shlex.join(cmd))
        try:
            with open(log_group_fi, "w") as log_file:
                subprocess.run(
                    cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
            return mtz_group_fi
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while running command: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as executor:
        results = list(executor.map(run_fw_one, enumerate(mtz_groups_i)))
    mtz_groups_fi.extend([r for r in results if r])
    return mtz_groups_fi


def run_servalcat_refine(
    mtzs_fi,
    labins,
    models,
    mtzs_free=[],
    prefix="multixem_",
    source="xray",
    keyword_file="",
    arguments=[],
    sigmaa=True,
    quick=False,
    n_proc=1,
):
    # TODO: --config
    refined_mmcifs = []
    refined_mtzs = []
    refined_jsons = []

    def refine_one(params):
        i_mtz, (mtz_fi, labin, mtz_free, model, prefix) = params
        local_refined_mmcifs = []
        local_refined_mtzs = []
        local_refined_jsons = []

        if mtzs_free and "--labin_llweight" in arguments:
            prefix += f"llweight{i_mtz}_"
            if "--weight" in arguments:
                prefix += "unre_"
        prefix += "refine"
        log_filename = prefix + ".log"

        cmd = [
            "servalcat",
            "refine_xtal_norefmac",
            "--hklin",
            mtz_fi,
            "--model",
            model,
            "-s",
            source,
            "--labin",
            labin,
            "--hout",
            "-o",
            prefix,
        ]
        if mtz_free:
            cmd.extend(["--hklin_free", mtz_free])
        if keyword_file:
            cmd.extend(["--keyword_file", keyword_file])
        if arguments:
            cmd.extend(arguments)
        if quick:
            cmd.extend(["--ncycle", "1"])
        logging.info("Running command: %s", shlex.join(cmd))
        try:
            with open(log_filename, "w") as log_file:
                process = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.PIPE, text=True, bufsize=1
                )
                for line in process.stderr:
                    logging.error(line.rstrip())
                    log_file.write(line)
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while running command: {e}")
        json_filename = prefix + "_stats.json"
        if not os.path.exists(json_filename):
            raise FileNotFoundError(f"Expected stats file not found: {json_filename}")
        with open(json_filename, "r") as stats_file:
            stats = json.load(stats_file)
            stats_line_list = [
                f"{stat} = {stats[-1]['data']['summary'][stat]:.4f}"
                for stat in stats[-1]["data"]["summary"]
                if stat != "-LL"
            ]
            logging.info(f"Finished: {prefix}.mmcif {', '.join(stats_line_list)}")
        if sigmaa:
            log_filename_sigmaa = prefix + "_sigmaa.log"
            cmd_sigmaa = [
                "servalcat",
                "sigmaa",
                "--hklin",
                mtz_fi,
                "--model",
                prefix + ".mmcif",
                "-s",
                source,
                "-o",
                prefix + "_sigmaa",
            ]
            if mtz_free:
                cmd_sigmaa.extend(["--hklin_free", mtz_free])
            # if arguments:
            #     cmd_sigmaa.extend(arguments)
            logging.info("Running command: %s", shlex.join(cmd_sigmaa))
            try:
                with open(log_filename_sigmaa, "w") as log_file_sigmaa:
                    subprocess.run(
                        cmd_sigmaa,
                        check=True,
                        stdout=log_file_sigmaa,
                        stderr=subprocess.STDOUT,
                    )
            except subprocess.CalledProcessError as e:
                logging.error(f"Error occurred while running command: {e}")
            local_refined_mtzs.append(prefix + "_sigmaa.mtz")
        else:
            local_refined_mtzs.append(prefix + ".mtz")
        local_refined_mmcifs.append(prefix + ".mmcif")
        local_refined_jsons.append(prefix + "_stats.json")
        return local_refined_mmcifs[0], local_refined_mtzs[0], local_refined_jsons[0]

    if len(mtzs_fi) == len(models) >= 2:
        models_list = models
    elif len(models) == len(mtzs_free) >= 2:
        models_list = models
    else:
        models_list = [models[0]] * max(len(mtzs_fi), len(mtzs_free))

    if mtzs_free and len(mtzs_free) >= 2 and len(mtzs_fi) == 1:
        # refinement after bootstrapping
        params = zip(
            mtzs_fi * len(mtzs_free),
            labins * len(mtzs_free),
            mtzs_free,
            models_list,
            [prefix] * len(mtzs_free),
        )
    elif not mtzs_free:
        # refinement after merging, no free set provided
        params = zip(
            mtzs_fi, labins, [None] * len(mtzs_fi), models_list, [prefix] * len(mtzs_fi)
        )
    elif len(mtzs_free) == 1:
        # refinement after merging, single free set provided
        params = zip(
            mtzs_fi,
            labins,
            mtzs_free * len(mtzs_fi),
            models_list,
            [prefix] * len(mtzs_fi),
        )
    else:
        # unexpected case, should not happen
        raise ValueError(
            "Unexpected case: both mtzs_fi and mtzs_free have more than one element."
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as executor:
        results = list(executor.map(refine_one, enumerate(params)))
    for mmcif, mtz, stats in results:
        refined_mmcifs.append(mmcif)
        refined_mtzs.append(mtz)
        refined_jsons.append(stats)
    return refined_mmcifs, refined_mtzs, refined_jsons


def compare_mtzs_fi(mtzs_fi, binner, bin_stats_matrix=[], n_expected=[]):

    # noqa: E501
    def compare_mtz_fi_pair(
        mtz_fi1, mtz_fi2, binner, bin_stats_list1=[], bin_stats_list2=[]
    ):
        # f_col = "F"
        i_col = "IMEAN"  # can be just "I" after servalcat fw
        column_labels_dropna = [i_col, f"SIG{i_col}"]  # or F?
        mtz1 = gemmi.read_mtz_file(mtz_fi1)
        mtz2 = gemmi.read_mtz_file(mtz_fi2)
        logging.info("")
        logging.info(
            f"Unit cell {str(mtz1.cell.a)} {str(mtz1.cell.b)} {str(mtz1.cell.c)}"
            f" {str(mtz1.cell.alpha)} {str(mtz1.cell.beta)} {str(mtz1.cell.gamma)}"
            f" in file {mtz_fi1}"
        )
        logging.info(
            f"Unit cell {str(mtz2.cell.a)} {str(mtz2.cell.b)} {str(mtz2.cell.c)}"
            f" {str(mtz2.cell.alpha)} {str(mtz2.cell.beta)} {str(mtz2.cell.gamma)}"
            f" in file {mtz_fi2}"
        )
        if mtz1.cell != mtz2.cell:
            logging.warning("Unit cell parameters are different.")
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
            subset=column_labels_dropna
        )  # Select only reflections with F
        mtz_df1 = mtz_df1.rename(columns=column_labels_dict1)  # Rename
        # print("")
        # print(mtz_df1.head(10))
        n_refl1 = len(mtz_df1)
        logging.info(f"No. unique reflections: {n_refl1} in file {mtz_fi1}")

        # mtz_df2 = mtz_df2[['H', 'K', 'L'] + columns]
        mtz_df2 = mtz_df2.dropna(subset=column_labels_dropna)
        mtz_df2 = mtz_df2.rename(columns=column_labels_dict2)
        n_refl2 = len(mtz_df2)
        logging.info(f"No. unique reflections: {n_refl2} in file {mtz_fi2}")

        # Extract common Miller indices (H, K, L)
        df = pandas.merge(mtz_df1, mtz_df2, on=["H", "K", "L"])
        n_refl = len(df)
        logging.info(
            f"No. unique reflections: {n_refl} in common;"
            f" ratios to the originals: {n_refl / n_refl1:.4f}   {n_refl / n_refl2:.4f}"
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
            logging.warning(
                "Less than 100 reflections per bin"
                " - it is recommended to set up a lower number of bins."
            )
        df["BIN"] = bins_tmp
        # print("Binner min_n_bins:", min_n_bins)
        n_bins = len(set(bins_tmp))  # TODO how to use args.n_bins?
        bins_stats = []
        df2_scaled = pandas.DataFrame()
        # mtz_df2_scaled = mtz_df2[["H", "K", "L"]].copy()
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
            df_bin_scaled = df_bin.copy()
            df_bin_scaled[f"{i_col}2_scaled"] = (
                scale_delioio * df_bin_scaled[f"{i_col}2"]
            )
            df_bin_scaled[f"SIG{i_col}2_scaled"] = (
                scale_delioio * df_bin_scaled[f"SIG{i_col}2"]
            )
            df2_scaled = pandas.concat(
                [df2_scaled, df_bin_scaled]
            )  # , ignore_index=True)
            ccI_iso = numpy.corrcoef(
                df_bin[i_col + "1"], df_bin_scaled[i_col + "2_scaled"]
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

        mtz_df2 = mtz_df2.drop(columns=[f"{i_col}2", f"SIG{i_col}2"])
        df2_scaled = df2_scaled.reset_index()
        mtz_df2 = mtz_df2.merge(
            df2_scaled[["H", "K", "L", f"{i_col}2_scaled", f"SIG{i_col}2_scaled"]],
            on=["H", "K", "L"],
            how="left",
        )
        mtz_df2 = mtz_df2.rename(
            columns={
                f"{i_col}2_scaled": f"{i_col}",
                f"SIG{i_col}2_scaled": f"SIG{i_col}",
            }
        )
        write_mtz_from_df(
            mtz_df2,
            mtz2,
            columns={f"{i_col}": "J", f"SIG{i_col}": "Q"},
            filename=f"{mtz_fi2_base}_scaled_to_{mtz_fi1_base}.mtz",
        )

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
    logging.info("\nNo. unique reflections:")
    logging.info(n_refl_matrix)
    if n_expected and len(n_expected) == len(mtzs_fi):
        completeness_matrix = n_refl_matrix / max(n_expected)
        logging.info("\nCompleteness:")
        logging.info(completeness_matrix)
    logging.info(
        "\nRatio of No. unique reflections in common and No. reflections in a data set:"
    )
    logging.info(ratio_refl_matrix)
    # print("Average CCFiso:")
    # print(ccF_iso_matrix)
    logging.info("\nAverage CCIiso:")
    logging.info(ccI_iso_matrix)
    logging.info("")
    # TODO: multiplicity
    return bin_stats_matrix, n_refl_matrix, ratio_refl_matrix


def main():
    parser = create_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)

    # subcommand mean
    if args.command == "mean":
        setup_logging()

        n_proc = min(os.cpu_count(), args.n_proc)
        if args.prefix:
            prefix = args.prefix
        else:
            prefix = args.file_name_template
        if prefix[-1] != "_":
            prefix += "_"

        refined_jsons = glob.glob(
            f"{args.file_name_template}_llweight*_refine_stats.json"
        )
        refined_json_ref_candidate = f"{args.file_name_template}_refine_stats.json"
        refined_json_ref = (
            refined_json_ref_candidate
            if os.path.isfile(refined_json_ref_candidate)
            else None
        )
        if refined_jsons:
            bootstrap_analyse_stats(refined_jsons, refined_json_ref, 1, prefix)

        refined_mmcif_ref_candidate = f"{args.file_name_template}_refine.mmcif"
        refined_mmcif_ref = (
            refined_mmcif_ref_candidate
            if os.path.isfile(refined_mmcif_ref_candidate)
            else None
        )

        refined_mtz_ref_candidate = f"{args.file_name_template}_refine.mtz"
        refined_mtz_ref = (
            refined_mtz_ref_candidate
            if os.path.isfile(refined_mtz_ref_candidate)
            else None
        )

        for f in [refined_json_ref, refined_mmcif_ref, refined_mtz_ref]:
            if f:
                logging.info(f"Reference file found: {f}")

        refined_mmcifs = glob.glob(f"{args.file_name_template}_llweight*_refine.mmcif")
        refined_mmcifs2 = glob.glob(f"{args.file_name_template}_llweight*_refine.cif")
        refined_mmcifs = refined_mmcifs + refined_mmcifs2

        if refined_mmcifs:
            geometry_objects_ref = []
            if args.geometry_cids and refined_mmcif_ref:
                st_ref = gemmi.read_structure(refined_mmcif_ref)
                geometry_objects_ref = select_cids_for_geometry_analysis(
                    args.geometry_cids
                )
                geometry_objects_ref = geometry_analysis_load(
                    st_ref, geometry_objects_ref
                )
            bootstrap_analyse_structures(
                refined_mmcifs,
                refined_mmcif_ref,
                idx=1,
                prefix=prefix,
                skip_hydrogen=True,
                smcif=args.cif,
                geometry_cids_file=args.geometry_cids,
                geometry_objects_ref=geometry_objects_ref,
            )
        else:
            logging.warning(
                f"No refined mmCIF files found with a filename template"
                f" {args.file_name_template}_llweight*_refine.mmcif"
            )
        refined_mtzs = glob.glob(f"{args.file_name_template}_llweight*_refine.mtz")
        if refined_mtzs:
            if refined_mtz_ref:
                m = gemmi.read_mtz_file(refined_mtz_ref)
                binner = gemmi.Binner()
                binner.setup_from_1_d2(
                    args.n_bins,
                    gemmi.Binner.Method.Dstar2,
                    m.make_1_d2_array(),
                    m.get_cell(),
                )
                bootstrap_mean_map(
                    refined_mtzs,
                    idx=1,
                    prefix=prefix,
                    binner=binner,
                    mtz_ref=refined_mtz_ref,
                    n_proc=n_proc,
                )
        else:
            logging.warning(
                f"No refined MTZ files found with a filename template"
                f"{args.file_name_template}_llweight*_refine.mtz"
            )
        return

    elif args.command not in ["pipeline", "bootstrap"] and args.command is not None:
        parser.print_help()
        return

    # subcommand pipeline or bootstrap
    # elif args.command in ["pipeline", "bootstrap"] or args.command is None:

    if args.prefix:
        prefix = args.prefix
        if prefix[-1] != "_":
            prefix += "_"
        working_dir_name = f"multixem_{prefix[:-1]}"
    else:
        prefix = "multixem_"
        working_dir_name = "multixem_project"

    os.mkdir(working_dir_name)
    os.chdir(working_dir_name)

    setup_logging()

    logging.info(f"Command line: {shlex.join(sys.argv)}")
    logging.info(f"Running multixem version: {__version__}")
    args_dict = vars(args)
    logging.info("Parsed arguments:")
    for key, value in args_dict.items():
        logging.info(f"  {key}: {value}")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Prefix for the output files: {prefix}")

    n_proc = min(os.cpu_count(), args.n_proc)
    # parse servalcat args preserving quoted groups
    servalcat_args = shlex.split(args.servalcat_args) if args.servalcat_args else []

    mtzs_i = []
    labins = []
    bin_stats_lists = []
    n_expected_list = []
    binner_master = None

    if args.command == "pipeline" and args.hklin_unmerged:
        logging.info(f"Unmerged diffraction data files: {args.hklin_unmerged}")
        n_groups = 0
        mtz_groups_i = []
        bin_stats_lists = []
        mtzs_fi = []
        for i, hklin_unmerged in enumerate(args.hklin_unmerged):
            logging.info("")
            logging.info(f"Unmerged diffraction data file: {hklin_unmerged}")
            if args.merge_whole_file:
                logging.info(
                    "All the reflections in all the batches"
                    " in this file will be merged."
                )
                _mtz_groups_i, _bin_stats_lists, _n_expected_list, _binner_master = (
                    merge_in_groups(
                        hklin_unmerged,
                        args.n_bins,
                        prefix,
                        merge_whole_file=args.merge_whole_file,
                        i_group_prefix=n_groups,
                    )
                )
            # TODO: select automatically the number of batches in group (now default 60)
            elif len(args.n_batches) == 1:
                n_batches_per_group = args.n_batches[0]
                logging.info(
                    f"Number of batches in merging group: {n_batches_per_group}"
                )
                _mtz_groups_i, _bin_stats_lists, _n_expected_list, _binner_master = (
                    merge_in_groups(
                        hklin_unmerged,
                        args.n_bins,
                        prefix,
                        n_batches_per_group=n_batches_per_group,
                        merge_whole_file=False,
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
                        merge_whole_file=False,
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
                labins.extend(["FMEAN,SIGFMEAN"] * n_groups)
            else:
                labins.extend(["IMEAN,SIGIMEAN"] * n_groups)
            # TODO: free reflections if not given
            # TODO: check that input files have FI(R?)
            # TODO: mmCIF
        mtzs_i = mtz_groups_i

    if args.hklin:
        logging.info(f"Merged diffraction data files: {args.hklin}")
        for i, hklin_i in enumerate(args.hklin):
            logging.info("")
            logging.info(f"Merged diffraction data file: {hklin_i}")
            if hklin_i.lower().endswith(".cif") or hklin_i.lower().endswith(".ent"):
                mtz = None
                doc = gemmi.cif.read(hklin_i)
                rblocks = gemmi.as_refln_blocks(doc)
                for rblock in rblocks:
                    if rblock.is_merged():
                        Convert = gemmi.CifToMtz()
                        mtz = Convert.convert_block_to_mtz(rblock)
                        break
                if mtz:
                    mtz_filename = (
                        f"{os.path.splitext(os.path.basename(hklin_i))[0]}.mtz"
                    )
                    mtz.write_to_file(mtz_filename)
                    mtzs_i.append(mtz_filename)
                else:
                    try:
                        from servalcat import utils as servalcat_utils

                        mtz, _, _ = servalcat_utils.fileio.read_smcif_shelx(hklin_i)
                    except RuntimeError:
                        pass
                if mtz:
                    mtz_filename = (
                        f"{os.path.splitext(os.path.basename(hklin_i))[0]}.mtz"
                    )
                    mtz.write_to_file(mtz_filename)
                    mtzs_i.append(mtz_filename)
                else:
                    raise RuntimeError(
                        f"Could not recognise format of diffraction data file {hklin_i}"
                    )
            elif hklin_i.lower().endswith(".hkl") or hklin_i.lower().endswith(".ent"):
                from servalcat import utils as servalcat_utils

                mtz = servalcat_utils.fileio.read_smcif_hkl(hklin_i)
                mtz_filename = f"{os.path.splitext(os.path.basename(hklin_i))[0]}.mtz"
                mtz.write_to_file(mtz_filename)
                mtzs_i.append(mtz_filename)
            else:
                mtz = gemmi.read_mtz_file(hklin_i)
                mtzs_i.append(hklin_i)
            dmax = mtz.resolution_high()
            dmin = mtz.resolution_low()
            # Check for None or nan values and recalculate if necessary
            if dmax is None or dmin is None or numpy.isnan(dmax) or numpy.isnan(dmin):
                d_array = mtz.cell.calculate_d_array(mtz.make_miller_array())
                dmax = max(d_array)
                dmin = min(d_array)
            logging.info(f"Resolution limits: {dmax:.3f}" f" - {dmin:.3f} A")
            logging.info(
                f"Space group: {mtz.spacegroup.hm} (No. {mtz.spacegroup.number})"
            )
            logging.info(
                f"Unit cell: {mtz.cell.a:.3f} {mtz.cell.b:.3f} {mtz.cell.c:.3f}"
                f" {mtz.cell.alpha:.3f} {mtz.cell.beta:.3f} {mtz.cell.gamma:.3f}"
            )
            labin, anom_present = check_reflection_file_columns(
                mtz, unmerged=False, prefer_amplitude=args.amplitude
            )
            # elif i_present and not f_present: TODO FW
            labins.append(labin)
            bin_stats_lists.append([])
            # TODO: check and fix n_expected
            n_expected = gemmi.count_reflections(mtz.cell, mtz.spacegroup, dmin, dmax)
            n_expected_list.append(n_expected)
            if not binner_master or dmin < 1 / numpy.sqrt(binner_master.max_1_d2):
                logging.info(
                    "Setting up resolution bins according to the file"
                    f" {hklin_i} with resolution limits {dmax:.3f}"
                    f" - {dmin:.3f} A"
                )
                binner_master = gemmi.Binner()
                binner_master.setup_from_1_d2(
                    args.n_bins,
                    gemmi.Binner.Method.Dstar2,
                    mtz.make_1_d2_array(),
                    mtz.get_cell(),
                )
    assert len(mtzs_i) == len(labins)

    bin_stats_matrix = len(mtzs_i) * [len(mtzs_i) * [None]]
    for i in range(len(mtzs_i)):
        bin_stats_matrix[i][i] = bin_stats_lists[i]
    if len(mtzs_i) >= 2:
        bin_stats_matrix, n_refl_matrix, ratio_refl_matrix = compare_mtzs_fi(
            mtzs_i, binner_master, bin_stats_matrix, n_expected_list
        )

    if args.command == "pipeline" and args.unify_cell:
        for i, mtz_i in enumerate(mtzs_i):
            if i == 0:
                continue
            mtzs_i[i] = copy_cell_mtz(mtzs_i[i], mtzs_i[0])

    models = []
    if args.model:
        if args.command == "pipeline" and args.molrep:
            models_molrep = []
            for model, mtz_i in zip(args.model, mtzs_i):
                logging.info(
                    "Running MolRep to generate a model from the input structure."
                )
                model_molrep = run_molrep(model, mtz_i)
                models_molrep.append(model_molrep)
            models = models_molrep
        else:
            models = args.model
        refined_mmcifs, refined_mtzs, refined_jsons = run_servalcat_refine(
            mtzs_i,
            labins,
            models,
            mtzs_free=[args.hklin_free],
            prefix=prefix,
            source=args.source,
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
        if args.command == "bootstrap" or (
            args.command == "pipeline" and args.bootstrap
        ):
            n_samples = (
                args.bootstrap
                if (args.command == "pipeline" and args.bootstrap)
                else args.n_samples
            )
            # TODO: some features assume only single model...
            mtzs_in = mtzs_i
            for i_mtz, (mtz_in, labin, model) in enumerate(
                zip(mtzs_in, labins, models)
            ):
                geometry_objects_ref = []
                restraints_file = ""
                if args.geometry_cids:
                    st_ref = gemmi.read_structure(refined_mmcifs[i_mtz])
                    geometry_objects_ref = select_cids_for_geometry_analysis(
                        args.geometry_cids
                    )
                    geometry_objects_ref = geometry_analysis_load(
                        st_ref, geometry_objects_ref
                    )
                    restraints_file = unrestrain(geometry_objects_ref, model)
                mtzs_bootstrap = bootstrap_dataset(
                    mtz_in,
                    binner_master,
                    seeds=range(1001, 1001 + n_samples),
                    labin=labin,
                )
                if args.model_dir and args.models:
                    input_model_s = args.models
                else:
                    input_model_s = [model]
                if args.unre:
                    (
                        refined_mmcifs_bootstrap_unre,
                        _,
                        _,
                    ) = run_servalcat_refine(
                        [mtz_in],
                        [labin],
                        input_model_s,
                        mtzs_free=mtzs_bootstrap,
                        prefix=f"{prefix}group{i_mtz + 1}_",
                        source=args.source,
                        keyword_file="",
                        arguments=servalcat_args
                        + ["--labin_llweight", "llweight"]
                        + [
                            "--weight",
                            "2.0",
                            "--hydrogen",
                            "yes",
                            "--ncycle",
                            str(args.unre),
                        ],
                        sigmaa=False,
                        quick=args.quick,
                        n_proc=n_proc,
                    )
                    input_model_s = refined_mmcifs_bootstrap_unre
                (
                    refined_mmcifs_bootstrap,
                    refined_mtzs_bootstrap,
                    refined_jsons_bootstrap,
                ) = run_servalcat_refine(
                    [mtz_in],
                    [labin],
                    input_model_s,
                    mtzs_free=mtzs_bootstrap,
                    prefix=f"{prefix}group{i_mtz + 1}_",
                    source=args.source,
                    keyword_file=restraints_file,
                    arguments=servalcat_args + ["--labin_llweight", "llweight"],
                    sigmaa=False,
                    quick=args.quick,
                    n_proc=n_proc,
                )
                bootstrap_analyse_stats(
                    refined_jsons_bootstrap, refined_jsons[i_mtz], 1, prefix
                )
                if os.path.splitext(model)[1] == ".cif":
                    bootstrap_analyse_structures(
                        refined_mmcifs_bootstrap,
                        refined_mmcifs[i_mtz],
                        idx=i_mtz + 1,
                        prefix=prefix,
                        skip_hydrogen=True,
                        smcif=model,
                        geometry_cids_file=args.geometry_cids,
                        geometry_objects_ref=geometry_objects_ref,
                    )
                else:
                    bootstrap_analyse_structures(
                        refined_mmcifs_bootstrap,
                        refined_mmcifs[i_mtz],
                        idx=i_mtz + 1,
                        prefix=prefix,
                        skip_hydrogen=True,
                        smcif="",
                        geometry_cids_file=args.geometry_cids,
                        geometry_objects_ref=geometry_objects_ref,
                    )
                bootstrap_mean_map(
                    refined_mtzs_bootstrap,
                    idx=i_mtz + 1,
                    prefix=prefix,
                    binner=binner_master,
                    mtz_ref=refined_mtzs[i_mtz],
                    n_proc=n_proc,
                )
