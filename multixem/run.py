# coding: utf-8
import os
import argparse
import subprocess
import numpy
import pandas
import gemmi
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
        "-u", "--hklin_unmerged", type=str, help="Input unmerged diffraction data."
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
    # TODO: if input has Friedel pairs but a user wants to merge them

    def validate_args(args):
        if args.n_batches and not args.hklin_unmerged:
            parser.error("--n_batches requires --hklin_unmerged to be provided.")

    parser.set_defaults(func=validate_args)
    # TO DO: at least two --hklin or one --hklin_unmerged
    return parser


def merge_in_groups(unmerged, n_batches_in_group, prefix):  # noqa: C901

    def merge_group(
        df_groups,
        i_group,
        cell,
        spacegroup,
        n_expected,
        wavelength=0,
        n_groups=0,
        anom=True,
        prefix="",
    ):
        intensities = gemmi.Intensities()
        intensities.set_data(
            cell,
            spacegroup,
            df_groups[i_group][["H", "K", "L"]].values,
            df_groups[i_group]["I"].values,
            df_groups[i_group]["SIGI"].values,
        )
        binner10 = gemmi.Binner()
        binner10.setup(10, gemmi.Binner.Method.Dstar2, intensities)
        if anom:
            intensities.prepare_for_merging(gemmi.DataType.Anomalous)
        else:
            intensities.prepare_for_merging(gemmi.DataType.Mean)
        bin_stats = intensities.calculate_merging_stats(binner10)
        print("")
        print(" dmax - dmin  CC1/2    CC*  Rmeas   Rpim")
        for n, stats in enumerate(bin_stats):
            dmax, dmin = binner10.dmax_of_bin(n), binner10.dmin_of_bin(n)
            # TODO I/sigma ? (see gemmi/include/intensit.hpp)
            # TODO #refl #unique multiplicity
            print(
                f"{dmax:5.2f} - {dmin:4.2f}"
                f" {stats.cc_half():6.3f} {stats.cc_star():6.3f}"
                f" {stats.r_meas():6.3f} {stats.r_pim():6.3f}"
            )
        overall_stats = intensities.calculate_merging_stats(None)
        dmax, dmin = intensities.resolution_range()
        print(
            f"{dmax:5.2f} - {dmin:4.2f}"
            f" {overall_stats[0].cc_half():6.3f} {overall_stats[0].cc_star():6.3f}"
            f" {overall_stats[0].r_meas():6.3f} {overall_stats[0].r_pim():6.3f}"
        )

        if anom:
            intensities.merge_in_place(gemmi.DataType.Anomalous)
        else:
            intensities.merge_in_place(gemmi.DataType.Mean)
        # SIGI from merging:  1/sqrt(∑w), where w=1/sigma^2
        completeness = len(intensities.miller_array) / n_expected
        print(
            f"Merged group {i_group + 1} of batches: #reflections:",
            len(intensities.miller_array),
            " => completeness:",
            f"{completeness:.3f}",
        )
        mtz_group_merged = intensities.prepare_merged_mtz(with_nobs=True)
        if wavelength:
            mtz_group_merged.dataset(0).wavelength = wavelength
        if n_groups:
            g_with_leading_zeros = str(i_group + 1).zfill(len(str(n_groups)))
        else:
            g_with_leading_zeros = i_group + 1
        mtz_group_merged_filename = f"{prefix}group{g_with_leading_zeros}_I.mtz"
        mtz_group_merged.write_to_file(mtz_group_merged_filename)
        return mtz_group_merged_filename

    m = gemmi.read_mtz_file(unmerged)
    # TODO gemmi.read_xds_ascii()
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
    dmax = m.resolution_low()
    dmin = m.resolution_high()
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
    mtz_groups = []
    for i_group in range(len(batches_split) - 1):
        # print(batches_split[i_group], batches_split[i_group+1])
        df_group = df.loc[
            (df["BATCH"] >= batches_split[i_group])
            & (df["BATCH"] < batches_split[i_group + 1])
        ]
        df_groups.append(df_group)
        mtz_group = merge_group(
            df_groups,
            i_group,
            m.cell,
            m.spacegroup,
            n_expected,
            wavelength,
            n_groups=len(batches_split),
            anom=anom,
            prefix=prefix,
        )
        mtz_groups.append(mtz_group)
    print("Merged MTZ files:", mtz_groups)
    return mtz_groups, n_expected


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
        group_fi_prefix = f"{prefix}group{i_group + 1}_FI"
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


def run_servalcat_refine(mtzs_fi, model, mtz_free="", source="xray"):  # , prefix=""):
    # TO DO: source -s
    # TO DO: command line parameters for servalcat, --keyword_file, --config
    refined_mmcifs = []
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
            "-o",
            prefix,
        ]
        if mtz_free:
            cmd.extend(["--hklin_unmerged", mtz_free])
        print("Running command:", " ".join(cmd))
        try:
            with open(log_filename, "w") as log_file:
                subprocess.run(
                    cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {e}")
        refined_mmcifs.append(prefix + ".mmcif")
    return refined_mmcifs


def compare_mtzs_fi(mtzs_fi, n_expected=0):

    # noqa: E501
    def compare_mtz_fi_pair(mtz_fi1, mtz_fi2):
        f_col = "F"
        column_label_dropna = "F"
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

        # Scaling per resolution bins - at least 100 reflections per bin
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
            min_n_bins = min(Counter(bins_tmp).values())
        df["BIN"] = bins_tmp
        # print("Binner min_n_bins:", min_n_bins)
        bins_stats = []
        for b in range(n_bins):
            df_bin = df[df["BIN"] == b]
            # scale_delfofo = sum_hkl F1 * F2 / sum_hkl F2**2
            scale_delfofo_numer = (df_bin[f_col + "1"] * df_bin[f_col + "2"]).sum()
            scale_delfofo_denomin = (df_bin[f_col + "2"] ** 2).sum()
            scale_delfofo = scale_delfofo_numer / scale_delfofo_denomin
            ccF_iso = numpy.corrcoef(
                df_bin[f_col + "1"], scale_delfofo * df_bin[f_col + "2"]
            )[0, 1]
            """
            rF_iso_numer = (
                abs(df_bin[f_col + "1"] - scale_delfofo * df_bin[f_col + "2"])
            ).sum()
            rF_iso_denom = (
                abs(df_bin[f_col + "1"] + scale_delfofo * df_bin[f_col + "2"])
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
            scale_delioio_nomin = (df_bin["I1"] * df_bin["I2"]).sum()
            scale_delioio_denumer = (df_bin["I2"] ** 2).sum()
            scale_delioio = scale_delioio_nomin / scale_delioio_denumer
            ccI_iso = numpy.corrcoef(df_bin["I1"], scale_delioio * df_bin["I2"])[0, 1]
            """
            rI_iso_numer = (abs(df_bin["I1"] - scale_delioio * df_bin["I2"])).sum()
            rI_iso_denom = (abs(df_bin["I1"] + scale_delioio * df_bin["I2"])).sum()
            rI_iso = 2 * rI_iso_numer / rI_iso_denom"""

            bins_stats.append(
                {
                    "i": b,
                    "count": len(df_bin),
                    "scale_delfofo": scale_delfofo,
                    "ccF_iso": ccF_iso,
                    # "rF_iso": rF_iso,
                    "scale_delioio": scale_delfofo,
                    "ccI_iso": ccI_iso,
                    # "rI_iso": rI_iso,
                }
            )

        bins_stats_df = pandas.DataFrame(bins_stats)
        # Calculate weighted average of cc over bins
        ccF_iso_avg = (
            bins_stats_df["ccF_iso"] * bins_stats_df["count"]
        ).sum() / bins_stats_df["count"].sum()
        ccI_iso_avg = (
            bins_stats_df["ccI_iso"] * bins_stats_df["count"]
        ).sum() / bins_stats_df["count"].sum()
        cc_iso_avg_list = [ccF_iso_avg, ccI_iso_avg]
        mtz_fi1_base = os.path.basename(mtz_fi1)
        mtz_fi2_base = os.path.basename(mtz_fi2)
        bins_stats_df.to_csv(
            f"{mtz_fi1_base}_bins_stats_{mtz_fi2_base}.csv",
            index=False,
            sep="\t",
            float_format="%.4f",
        )
        # pprint.pprint(bins_stats)
        return n_refl_list, cc_iso_avg_list

    n_refl_matrix = numpy.zeros((len(mtzs_fi), len(mtzs_fi)), dtype=int)
    ratio_refl_matrix = numpy.identity(len(mtzs_fi), dtype=float)
    ccF_iso_matrix = numpy.identity(len(mtzs_fi), dtype=float)
    ccI_iso_matrix = numpy.identity(len(mtzs_fi), dtype=float)
    for i in range(len(mtzs_fi)):
        for j in range(i + 1, len(mtzs_fi)):
            # print(i, j)
            n_refl_list, cc_iso_avg_list = compare_mtz_fi_pair(mtzs_fi[i], mtzs_fi[j])
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
            ccF_iso_matrix[i, j] = cc_iso_avg_list[0]
            ccF_iso_matrix[j, i] = cc_iso_avg_list[0]
            ccI_iso_matrix[i, j] = cc_iso_avg_list[1]
            ccI_iso_matrix[j, i] = cc_iso_avg_list[1]
    print("No. unique reflections:")
    print(n_refl_matrix)
    if n_expected:
        completeness_matrix = n_refl_matrix / n_expected
        print("Completeness:")
        print(completeness_matrix)
    print(
        "Ratio of No. unique reflections in common and No. reflections in a data set:"
    )
    print(ratio_refl_matrix)
    print("Average CCFiso:")
    print(ccF_iso_matrix)
    print("Average CCIiso:")
    print(ccI_iso_matrix)
    # TODO: multiplicity
    return n_refl_matrix, ratio_refl_matrix


def main():
    print("Running multixem version:", __version__)
    parser = create_parser()
    args = parser.parse_args()
    print("Arguments parsed:", args)

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

    n_expected = 0
    if args.hklin_unmerged:
        print("Unmerged diffraction data:", args.hklin_unmerged)
        # TODO: select automatically the number of batches in group (now default 60)
        n_batches_per_group = args.n_batches
        print("Number of batches in merging group:", n_batches_per_group)
        mtz_groups_i, n_expected = merge_in_groups(
            args.hklin_unmerged, n_batches_per_group, prefix
        )
        mtzs_fi = run_servalcat_fwt(mtz_groups_i, prefix)
        # TODO: free reflections if not given

    # TODO: check that input files have FI(R?)
    # TODO: mmCIF
    compare_mtzs_fi(mtzs_fi, n_expected)
    if args.model:
        run_servalcat_refine(mtzs_fi, args.model, mtz_free=args.hklin_free)

    # compute_difference_maps(mtz_groups[0], mtz_groups[-1], "output_prefix")


if __name__ == "__main__":
    main()
