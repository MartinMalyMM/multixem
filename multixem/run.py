# coding: utf-8
import os
import argparse
import subprocess
import pprint
import numpy
import pandas
import gemmi
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from . import __version__


def create_parser():
    """
    Create the argument parser for the command-line interface.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """

    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
        return ivalue

    parser = argparse.ArgumentParser(
        prog="multixem",
        description="Refinement pipeline for multiple data sets in structure biology.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
        help="show version and exit",
    )
    parser.add_argument("-p", "--prefix", type=str, help="Prefix for the output files.")
    parser.add_argument(
        "-u",
        "--hklin_unmerged",
        type=str,
        nargs="+",
        help="Input unmerged diffraction data file(s).",
    )  # TODO - file exists?
    # TODO more files
    parser.add_argument(
        "--hklin_free", type=str, help="Input MTZ file for test flags."
    )  # TODO - file exists?
    parser.add_argument(
        "--model", type=str, help="Input atomic structure model file."
    )  # TODO - file exists?
    parser.add_argument(
        "--n_batches",
        type=positive_int,
        default=60,
        help="Number of batches per merging group. Must be a positive integer.",
    )
    parser.add_argument(
        "--n_bins",
        type=positive_int,
        default=20,
        help="Number of resolution bins. Must be a positive integer.",
    )
    parser.add_argument(
        "--amplitude",
        action="store_true",
        help="Use amplitude rather than intensities (not recommended).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run (only for development).",
    )
    # TODO: if input has Friedel pairs but a user wants to merge them

    def validate_args(args):
        if args.n_batches and not args.hklin_unmerged:
            parser.error("--n_batches requires --hklin_unmerged to be provided.")

    parser.set_defaults(func=validate_args)
    # TO DO: at least two --hklin or one --hklin_unmerged
    return parser


def write_bin_stats(bin_stats_list, filename):
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


def merge_in_groups(unmerged, n_batches_in_group, n_bins, prefix, i_group_prefix=0):

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
        # binner10 = gemmi.Binner()
        # binner10.setup(10, gemmi.Binner.Method.Dstar2, intensities)
        if anom:
            intensities.prepare_for_merging(gemmi.DataType.Anomalous)
        else:
            intensities.prepare_for_merging(gemmi.DataType.Mean)
        bin_stats = intensities.calculate_merging_stats(binner)
        # binner_bincount_obs = numpy.bincount(
        #     binner.get_bins(intensities.miller_array))
        # TODO fix n_obs and n-unique (and completeness) per bin
        # TODO: add I/sigma
        # TODO add multiplicity
        # Collect bin statistics into a list of dictionaries
        bin_stats_list = []
        for n, stats in enumerate(bin_stats):
            bin_stats_list.append(
                {
                    "bin": n + 1,
                    "dmax": binner.dmax_of_bin(n),
                    "dmin": binner.dmin_of_bin(n),
                    # "n_obs": binner_bincount_obs[n],  # NOT GOOD
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
                "n_obs": len(intensities.miller_array),
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
        # After meringing, add n_unique completeness multiplicity to statistics
        # binner_bincount_unique = numpy.bincount(
        #     binner.get_bins(intensities.miller_array)
        # )
        for b in range(binner.size):
            bin_n_obs_expected = gemmi.count_reflections(
                cell,
                spacegroup,
                binner.dmin_of_bin(b),
                binner.dmax_of_bin(b),
                unique=False,
            )
            bin_n_unique_expected = gemmi.count_reflections(
                cell,
                spacegroup,
                binner.dmin_of_bin(b),
                binner.dmax_of_bin(b),
                unique=True,
            )
            # bin_stats_list[b]["n_unique"] = int(binner_bincount_unique[b])  # NOT GOOD
            bin_stats_list[b]["n_obs_expected"] = bin_n_obs_expected
            bin_stats_list[b]["n_unique_expected"] = bin_n_unique_expected
            # bin_stats_list[b]["completeness"] = (
            #     bin_stats_list[b]["n_unique"] / bin_n_unique_expected  # NOT GOOD
            # )
            # bin_stats_list[b]["multiplicity"] = (
            #     bin_stats_list[b]["n_obs"] / bin_stats_list[b]["n_unique"]  # NOT GOOD
            # )
        bin_stats_list[-1]["n_unique"] = len(intensities.miller_array)
        bin_stats_list[-1]["n_obs_expected"] = gemmi.count_reflections(
            cell,
            spacegroup,
            intensities.resolution_range()[1],
            intensities.resolution_range()[0],
            unique=False,
        )
        bin_stats_list[-1]["n_unique_expected"] = gemmi.count_reflections(
            cell,
            spacegroup,
            intensities.resolution_range()[1],
            intensities.resolution_range()[0],
            unique=True,
        )
        bin_stats_list[-1]["completeness"] = (
            bin_stats_list[-1]["n_unique"] / bin_stats_list[-1]["n_unique_expected"]
        )
        bin_stats_list[-1]["multiplicity"] = (
            bin_stats_list[-1]["n_obs"] / bin_stats_list[-1]["n_unique"]
        )
        print("average multiplicity", intensities.nobs_array.mean())
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
        mtz_group_merged = intensities.prepare_merged_mtz(with_nobs=True)
        if wavelength:
            mtz_group_merged.dataset(0).wavelength = wavelength
        if n_groups:
            g_with_leading_zeros = str(i_group_prefix + i_group + 1).zfill(
                len(str(n_groups))
            )
        else:
            g_with_leading_zeros = i_group_prefix + i_group + 1
        mtz_group_merged_filename = f"{prefix}group{g_with_leading_zeros}_I.mtz"
        mtz_group_merged.write_to_file(mtz_group_merged_filename)

        return mtz_group_merged_filename, bin_stats_list

    if unmerged.lower().endswith(".hkl"):
        xds_ascii = gemmi.read_xds_ascii(unmerged)
        m = xds_ascii.to_mtz()
        # Read resolution range from the unmerged file if present
        with open(unmerged, "r") as f:
            for line in f:
                if line.strip().startswith("!INCLUDE_RESOLUTION_RANGE="):
                    dmax_dmin = line.strip().split("=")[-1].split()
                    if len(dmax_dmin) >= 2:
                        dmax, dmin = float(dmax_dmin[0]), float(dmax_dmin[1])
                    break
    else:
        m = gemmi.read_mtz_file(unmerged)
        dmax = m.resolution_low()
        dmin = m.resolution_high()
    print(m)
    print(list(m.columns))

    # Scan the columns of the input unmerged MTZ file
    # and check if Friedel pairs are present or not
    anom = False
    for column in m.columns:
        if column.type == "J":
            print(
                "Column with intensity (type J, no Friedel pairs) found:", column.label
            )
        elif column.type == "Q":
            print(
                "Column with standard deviation associated to intensity/amplitude"
                " column (type Q, no Friedel pairs) found:",
                column.label,
            )
        if column.type == "K":
            print(
                "Column with intensity (type K, Friedel pairs)" " found:", column.label
            )
            print("Friedel pairs will be kept separately.")
            anom = True
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
            print(
                "This is quite unusual for unmerged data file,"
                " are you sure about the file?"
            )
        elif column.type == "L":
            print(
                "Column with standard deviation associated to amplitude"
                " (type L, Friedel pairs) found:",
                column.label,
            )
            print(
                "This is quite unusual for unmerged data file,"
                " are you sure about the file?"
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
    batches_split = list(range(0, len(m.batches), n_batches_in_group))
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


def run_servalcat_fwt(mtz_groups_i, prefix=""):
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
    for i_group, mtz_group_i in enumerate(mtz_groups_i):
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
            mtz_groups_fi.append(mtz_group_fi)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {e}")
    return mtz_groups_fi


def run_servalcat_refine(
    mtzs_fi,
    model,
    mtz_free="",
    source="xray",
    sigmaa=True,
    quick=False,
):  # , prefix=""):
    # TO DO: source -s
    # TO DO: command line parameters for servalcat, --keyword_file, --config
    refined_mmcifs = []
    refined_mtzs = []
    for i_mtz, mtz_fi in enumerate(mtzs_fi):
        prefix = os.path.splitext(os.path.basename(mtz_fi))[0] + "_refine"
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
            "--hout",
            "-o",
            prefix,
        ]
        if mtz_free:
            cmd.extend(["--hklin_free", mtz_free])
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
            log_filename = prefix + "_sigmaa.log"
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
            print("Running command:", " ".join(cmd_sigmaa))
            try:
                with open(log_filename, "w") as log_file:
                    subprocess.run(
                        cmd_sigmaa,
                        check=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running command: {e}")
            refined_mtzs.append(prefix + "_sigmaa.mtz")
        else:
            refined_mtzs.append(prefix + ".mtz")
        refined_mmcifs.append(prefix + ".mmcif")
    return refined_mmcifs, refined_mtzs


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
        hkl_common_array = numpy.array(df[["H", "K", "L"]].values, numpy.int8)
        hkl_common_array = numpy.ascontiguousarray(hkl_common_array, dtype=numpy.int8)
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
            scale_delioio_nomin = (df_bin[i_col + "1"] * df_bin[i_col + "2"]).sum()
            scale_delioio_denumer = (df_bin[i_col + "2"] ** 2).sum()
            scale_delioio = scale_delioio_nomin / scale_delioio_denumer
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


def compute_difference_maps_pair(mtz_file_1, mtz_file_2, binner, bin_stats_list=[]):

    def calc_scale_complex(df, column="F_est"):
        # scale_complex = 2 * sum_hkl (F1RE * F2RE + F1IM * F2IM) / sum_hkl F2**2
        scale_complex_nomin = (
            (df[column + "1RE"] * df[column + "2RE"])
            + (df[column + "1IM"] * df[column + "2IM"])
        ).sum()
        scale_complex_denomin = (df[column + "2"] ** 2).sum()
        # equivalent to the previous line:
        # scale_complex = (df[column + '2RE']**2 + df[column + '2IM']**2).sum()
        if not numpy.isclose(scale_complex_denomin, 0):
            scale_complex = scale_complex_nomin / scale_complex_denomin
        else:
            warnings.warn(
                f"scale denominator for bin {b + 1} is zero,"
                " setting scale for this bin to 1."
            )
            scale_complex = 1

        return scale_complex

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
        raise ("No column with amplitudes found.")
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
    hkl_common_array = numpy.array(df[["H", "K", "L"]].values, numpy.int8)
    hkl_common_array = numpy.ascontiguousarray(hkl_common_array, dtype=numpy.int8)
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
        scale_delfofo_nomin = (df_bin[f_col + "1"] * df_bin[f_col + "2"]).sum()
        scale_delfofo_denomin = (df_bin[f_col + "2"] ** 2).sum()
        if not numpy.isclose(scale_delfofo_denomin, 0):
            scale_delfofo = scale_delfofo_nomin / scale_delfofo_denomin
        else:
            warnings.warn(
                f"scale_delfofo denominator for bin {b + 1} is zero"
                f" ({bin_stats_list[b]['dmax']:.4f} -"
                f" {bin_stats_list[b]['dmin']:.4f} A),"
                " setting scale for this bin to 1."
            )
            scale_delfofo = 1

        # scale_delfofo2sc= 2* sum_hkl (FP1RE * FP2RE + FP1IM * FP2IM) / sum_hkl FP2^**2
        scale_delfofo2sc_nomin = (
            (df_bin["FP1RE"] * df_bin["FP2RE"]) + (df_bin["FP1IM"] * df_bin["FP2IM"])
        ).sum()
        scale_delfofo2sc_denomin = (df_bin[f_col + "2"] ** 2).sum()
        # equivalent to the previous line:
        # scale_delfofo2sc_denomin = (df_bin['FP2RE']**2 + df_bin['FP2IM']**2).sum()
        if not numpy.isclose(scale_delfofo2sc_denomin, 0):
            scale_delfofo2sc = scale_delfofo2sc_nomin / scale_delfofo2sc_denomin
        else:
            warnings.warn(
                f"scale_delfofo2sc denominator for bin {b + 1} is zero,"
                " setting scale for this bin to 1."
            )
            scale_delfofo2sc = 1

        # scale_delfwtfwt2sc= 2*sum_hkl(FWT1RE*FWT2RE+FWT1IM*FWT2IM)/sum_hkl FWT2^**2
        scale_delfwtfwt2sc_nomin = (
            (df_bin["FWT1RE"] * df_bin["FWT2RE"])
            + (df_bin["FWT1IM"] * df_bin["FWT2IM"])
        ).sum()
        scale_delfwtfwt2sc_denomin = (df_bin["FWT2"] ** 2).sum()
        # equivalent to the previous line:
        # scale_delfwtfwt2sc_denomin = (df_bin['FWT2RE']**2 + df_bin['FWT2IM']**2).sum()
        if not numpy.isclose(scale_delfwtfwt2sc_denomin, 0):
            scale_delfwtfwt2sc = scale_delfwtfwt2sc_nomin / scale_delfwtfwt2sc_denomin
        else:
            warnings.warn(
                f"scale_delfwtfwt2sc denominator for bin {b + 1} is zero,"
                f" ({bin_stats_list[b]['dmax']:.4f} -"
                f" {bin_stats_list[b]['dmin']:.4f} A),"
                " setting scale for this bin to 1."
            )
            scale_delfwtfwt2sc = 1

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

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = mtz1.spacegroup
    mtz.set_cell_for_all(mtz1.cell)
    mtz.add_dataset(mtz1.datasets[0].dataset_name)
    mtz.add_column("DELFOFO", "F")
    mtz.add_column("PHDELFOFO", "P")
    mtz.add_column("DELFOFO2SC", "F")
    mtz.add_column("PHDELFOFO2SC", "P")
    mtz.add_column("DELFWTFWT2SC", "F")
    mtz.add_column("PHDELFWTFWT2SC", "P")

    data = numpy.array(
        df[
            [
                "H",
                "K",
                "L",
                "DELFOFO",
                "PHDELFOFO",
                "DELFOFO2SC",
                "PHDELFOFO2SC",
                "DELFWTFWT2SC",
                "PHDELFWTFWT2SC",
            ]
        ].values,
        numpy.float32,
    )
    mtz.set_data(data)
    mtz_fi1_base = os.path.splitext(os.path.basename(mtz_file_1))[0]
    mtz_fi2_base = os.path.splitext(os.path.basename(mtz_file_2))[0]
    output_prefix = f"{mtz_fi1_base}_vs_{mtz_fi2_base}_diffmaps"
    output_mtz = f"{output_prefix}.mtz"
    mtz.write_to_file(output_mtz)
    print(f"Saved: {output_mtz}")

    # For DELFWTFWT2SCall map, use all the reflections
    mtz_fwt_df1 = mtz_fwt_df1.dropna(subset=["FWT"])
    mtz_fwt_df1 = mtz_fwt_df1.rename(columns=columns_fwt1_dict)
    mtz_fwt_df1 = mtz_fwt_df1[["H", "K", "L"] + columns_fwt1]
    mtz_fwt_df2 = mtz_fwt_df2.dropna(subset=["FWT"])
    mtz_fwt_df2 = mtz_fwt_df2.rename(columns=columns_fwt2_dict)
    mtz_fwt_df2 = mtz_fwt_df2[["H", "K", "L"] + columns_fwt2]
    df_fwt = pandas.merge(mtz_fwt_df1, mtz_fwt_df2, on=["H", "K", "L"])
    hkl_common_array_fwt = numpy.array(df_fwt[["H", "K", "L"]].values, numpy.int8)
    hkl_common_array_fwt = numpy.ascontiguousarray(
        hkl_common_array_fwt, dtype=numpy.int8
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
    mtz_fwt = gemmi.Mtz(with_base=True)
    mtz_fwt.spacegroup = mtz1.spacegroup
    mtz_fwt.set_cell_for_all(mtz1.cell)
    mtz_fwt.add_dataset(mtz1.datasets[0].dataset_name)
    mtz_fwt.add_column("DELFWTFWT2SCall", "F")
    mtz_fwt.add_column("PHDELFWTFWT2SCall", "P")
    mtz_fwt.add_column("DELFestFest2SCall", "F")
    mtz_fwt.add_column("PHDELFestFest2SCall", "P")
    data = numpy.array(
        df_fwt[
            [
                "H",
                "K",
                "L",
                "DELFWTFWT2SCall",
                "PHDELFWTFWT2SCall",
                "DELFestFest2SCall",
                "PHDELFestFest2SCall",
            ]
        ].values,
        numpy.float32,
    )
    mtz_fwt.set_data(data)
    mtz_fwt.write_to_file(output_mtz_fwt)
    print(f"Saved: {output_mtz_fwt}")
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
                for b in range(len(bin_stats_diff)):
                    bin_stats_matrix[i][j][b].update(bin_stats_diff[b])
                    bin_stats_matrix[j][i][b].update(bin_stats_diff[b])
                filename = f"{refined_mtzs[i]}_vs_{refined_mtzs[j]}_bin_stats.txt"
                write_bin_stats(bin_stats_matrix[i][j], filename)

    return bin_stats_matrix


def main():
    print("Running multixem version:", __version__)
    parser = create_parser()
    args = parser.parse_args()
    print("Arguments parsed:", args)
    pprint.pprint(vars(args))

    if args.prefix:
        prefix = args.prefix
        if args.prefix[-1] != "_":
            prefix += "_"
        print("Prefix for the output files:", prefix)
    else:
        prefix = ""

    os.mkdir("multixem_proc")
    os.chdir("multixem_proc")
    print("Current working directory:", os.getcwd())

    if args.hklin_unmerged:
        print("Unmerged diffraction data:", args.hklin_unmerged)
        n_groups = 0
        mtz_groups_i = []
        bin_stats_lists = []
        n_expected_list = []
        mtzs_fi = []
        for i, hklin_unmerged in enumerate(args.hklin_unmerged):
            # TODO: select automatically the number of batches in group (now default 60)
            n_batches_per_group = args.n_batches
            print("Number of batches in merging group:", n_batches_per_group)
            _mtz_groups_i, _bin_stats_lists, _n_expected_list, _binner_master = (
                merge_in_groups(
                    hklin_unmerged, n_batches_per_group, args.n_bins, prefix, n_groups
                )
            )
            mtz_groups_i.extend(_mtz_groups_i)
            n_groups = len(mtz_groups_i)
            bin_stats_lists.extend(_bin_stats_lists)
            n_expected_list.extend(_n_expected_list)
            if i == 0:
                binner_master = _binner_master
            if args.amplitude:
                _mtzs_fi = run_servalcat_fwt(_mtz_groups_i, prefix)
                mtzs_fi.extend(_mtzs_fi)
            # TODO: free reflections if not given

    # TODO: check that input files have FI(R?)
    # TODO: mmCIF
    bin_stats_matrix = len(bin_stats_lists) * [len(bin_stats_lists) * [None]]
    for i in range(len(bin_stats_lists)):
        bin_stats_matrix[i][i] = bin_stats_lists[i]

    mtzs_i = mtz_groups_i
    bin_stats_matrix, n_refl_matrix, ratio_refl_matrix = compare_mtzs_fi(
        mtzs_i, binner_master, bin_stats_matrix, n_expected_list
    )
    if args.model:
        refined_mmcifs, refined_mtzs = run_servalcat_refine(
            mtzs_i,
            args.model,
            mtz_free=args.hklin_free,
            quick=args.quick,
        )
        bin_stats_matrix = compute_difference_maps(
            refined_mtzs, binner_master, bin_stats_matrix
        )

    # compute_difference_maps(mtz_groups[0], mtz_groups[-1], "output_prefix")


if __name__ == "__main__":
    main()
