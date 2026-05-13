# coding: utf-8
import os
import numpy
import pandas
import gemmi
import matplotlib.pyplot as plt
import matplotlib
import logging
from .tools import (
    write_bin_stats,
    write_mtz_from_df,
    makeAddressStr,
    calc_scale_real,
    calc_scale_complex,
)

matplotlib.use("Agg")


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
        logging.info(f"Running ADP analysis for {modelPath}")
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
                        logging.warning(
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


def compute_difference_maps_pair(
    mtz_file_1,
    mtz_file_2,
    binner,
    bin_stats_list=[],
    amplitude=False,
):
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
        amplitude (bool): Whether to use
                            intensities (False, e.g. MTZ files from servalcat sigmaa
                                         so F_est column is present),
                            or amplitudes (True)
    Returns:
        bin_stats_list (list of dict): Updated list with statistics
            for each resolution bin.
    """

    mtz1 = gemmi.read_mtz_file(mtz_file_1)
    mtz2 = gemmi.read_mtz_file(mtz_file_2)
    columns_fwt = ["FWT", "PHWT"]
    F_est_avail = False
    if (
        not amplitude
        and all(col in mtz1.column_labels() for col in ["F_est", "DFC", "PHDFC"])
        and all(col in mtz2.column_labels() for col in ["F_est", "DFC", "PHDFC"])
    ):
        F_est_avail = True
        f_col = "F_est"
        columns_fwt += ["Fcombi", "PHDFC"]
    else:
        f_col = "FP"
    columns = [f_col]
    columns_fwt1 = [col + "1" for col in columns_fwt]
    columns_fwt1_dict = {col: col + "1" for col in columns_fwt}
    columns_fwt2 = [col + "2" for col in columns_fwt]
    columns_fwt2_dict = {col: col + "2" for col in columns_fwt}

    mtz_df1 = pandas.DataFrame(data=mtz1.array, columns=mtz1.column_labels())
    mtz_df1 = mtz_df1.astype({name: "int32" for name in ["H", "K", "L"]})
    mtz_fwt_df1 = mtz_df1.copy()
    if F_est_avail:
        mtz_fwt_df1["Fcombi"] = mtz_fwt_df1["F_est"].combine_first(mtz_fwt_df1["DFC"])

    mtz_df2 = pandas.DataFrame(data=mtz2.array, columns=mtz2.column_labels())
    mtz_df2 = mtz_df2.astype({name: "int32" for name in ["H", "K", "L"]})
    mtz_fwt_df2 = mtz_df2.copy()
    if F_est_avail:
        mtz_fwt_df2["Fcombi"] = mtz_fwt_df2["F_est"].combine_first(mtz_fwt_df2["DFC"])

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
    logging.info(f"No. unique reflections: {n_refl1} in file {mtz_file_1}")

    mtz_df2 = mtz_df2[["H", "K", "L"] + columns]
    mtz_df2 = mtz_df2.dropna(subset=[f_col])
    mtz_df2 = mtz_df2.rename(columns=columns2_dict)
    n_refl2 = len(mtz_df2)
    logging.info(f"No. unique reflections: {n_refl2} in file {mtz_file_2}")

    # Extract common Miller indices (H, K, L)
    df = pandas.merge(mtz_df1, mtz_df2, on=["H", "K", "L"])
    n_refl = len(df)
    logging.info(
        f"No. unique reflections: {n_refl} in common;"
        f" ratios to the originals: {n_refl / n_refl1:.4f}   {n_refl / n_refl2:.4f}"
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
    # + DELFWTFWT2all           SC (scaling complex) (if F_est is available)

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
        logging.warning(
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
            logging.warning(
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
        df.loc[df_bin.index, "DELFOFO2SC"] = numpy.hypot(
            df["DELFOFO2SCRE"].astype(numpy.float64),
            df["DELFOFO2SCIM"].astype(numpy.float64),
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
        df.loc[df_bin.index, "DELFWTFWT2SC"] = numpy.hypot(
            df["DELFWTFWT2SCRE"].astype(numpy.float64),
            df["DELFWTFWT2SCIM"].astype(numpy.float64),
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
    logging.info(f"No. reflections in {output_mtz_fwt}: {len(df_fwt)}")
    binner_fwt = binner
    df_fwt["BIN"] = binner_fwt.get_bins(hkl_common_array_fwt)

    df_fwt["FWT1RE"] = df_fwt["FWT1"] * numpy.cos(numpy.deg2rad(df_fwt["PHWT1"]))
    df_fwt["FWT1IM"] = df_fwt["FWT1"] * numpy.sin(numpy.deg2rad(df_fwt["PHWT1"]))
    df_fwt["FWT2RE"] = df_fwt["FWT2"] * numpy.cos(numpy.deg2rad(df_fwt["PHWT2"]))
    df_fwt["FWT2IM"] = df_fwt["FWT2"] * numpy.sin(numpy.deg2rad(df_fwt["PHWT2"]))
    if F_est_avail:
        df_fwt["Fcombi1RE"] = df_fwt["Fcombi1"] * numpy.cos(
            numpy.deg2rad(df_fwt["PHDFC1"])
        )
        df_fwt["Fcombi1IM"] = df_fwt["Fcombi1"] * numpy.sin(
            numpy.deg2rad(df_fwt["PHDFC1"])
        )
        df_fwt["Fcombi2RE"] = df_fwt["Fcombi2"] * numpy.cos(
            numpy.deg2rad(df_fwt["PHDFC2"])
        )
        df_fwt["Fcombi2IM"] = df_fwt["Fcombi2"] * numpy.sin(
            numpy.deg2rad(df_fwt["PHDFC2"])
        )
    for b in range(len(bin_stats_list)):
        df_fwt_bin = df_fwt[df_fwt["BIN"] == b]
        scale_delfwtfwt2scall = calc_scale_complex(df_fwt_bin, "FWT")
        if F_est_avail:
            scale_delfestfest2scall = calc_scale_complex(df_fwt_bin, "Fcombi")
            bin_stats_list[b]["scale_delfestfest2scall"] = scale_delfestfest2scall
        bin_stats_list[b]["scale_delfwtfwt2scall"] = scale_delfwtfwt2scall
        bin_stats_list[b]["delfwtfwt2scall_count"] = len(df_fwt_bin)
        df_fwt.loc[df_fwt_bin.index, "DELFWTFWT2SCallRE"] = (
            df_fwt_bin["FWT1RE"] - scale_delfwtfwt2scall * df_fwt_bin["FWT2RE"]
        )
        df_fwt.loc[df_fwt_bin.index, "DELFWTFWT2SCallIM"] = (
            df_fwt_bin["FWT1IM"] - scale_delfwtfwt2scall * df_fwt_bin["FWT2IM"]
        )
        df_fwt.loc[df_fwt_bin.index, "DELFWTFWT2SCall"] = numpy.hypot(
            df_fwt["DELFWTFWT2SCallRE"].astype(numpy.float64),
            df_fwt["DELFWTFWT2SCallIM"].astype(numpy.float64),
        )
        df_fwt.loc[df_fwt_bin.index, "PHDELFWTFWT2SCall"] = numpy.rad2deg(
            numpy.arctan2(df_fwt["DELFWTFWT2SCallIM"], df_fwt["DELFWTFWT2SCallRE"])
        )
        if F_est_avail:
            df_fwt.loc[df_fwt_bin.index, "DELFestFest2SCallRE"] = (
                df_fwt_bin["Fcombi1RE"]
                - scale_delfestfest2scall * df_fwt_bin["Fcombi2RE"]
            )
            df_fwt.loc[df_fwt_bin.index, "DELFestFest2SCallIM"] = (
                df_fwt_bin["Fcombi1IM"]
                - scale_delfestfest2scall * df_fwt_bin["Fcombi2IM"]
            )
            df_fwt.loc[df_fwt_bin.index, "DELFestFest2SCall"] = numpy.hypot(
                df_fwt["DELFestFest2SCallRE"].astype(numpy.float64),
                df_fwt["DELFestFest2SCallIM"].astype(numpy.float64),
            )
            df_fwt.loc[df_fwt_bin.index, "PHDELFestFest2SCall"] = numpy.rad2deg(
                numpy.arctan2(
                    df_fwt["DELFestFest2SCallIM"], df_fwt["DELFestFest2SCallRE"]
                )
            )
    columns_to_write_list = [
        "DELFWTFWT2SCall",
        "PHDELFWTFWT2SCall",
    ]
    if F_est_avail:
        columns_to_write_list += ["DELFestFest2SCall", "PHDELFestFest2SCall"]
    columns_to_write_dict = {
        col: ("F" if not col.startswith("PH") else "P") for col in columns_to_write_list
    }
    write_mtz_from_df(df_fwt, mtz1, columns_to_write_dict, output_mtz_fwt)
    stats_filename = f"{mtz_fi1_base}_vs_{mtz_fi2_base}_bin_stats.txt"
    write_bin_stats(bin_stats_list, stats_filename)
    return bin_stats_list


def compute_difference_maps(refined_mtzs, binner, bin_stats_matrix=[], amplitude=False):
    """Compute difference maps for all pairs of MTZ files in `refined_mtzs`
    and update the `bin_stats_matrix` with the statistics
    for each pair and each resolution bin."""

    for i in range(len(refined_mtzs)):
        for j in range(i + 1, len(refined_mtzs)):
            # print(i, j)
            bin_stats_diff = compute_difference_maps_pair(
                refined_mtzs[i],
                refined_mtzs[j],
                binner,
                bin_stats_matrix[i][j],
                amplitude=amplitude,
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
                        logging.warning(
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
        logging.warning(
            f"Number of atoms in {structure1} does not match the number"
            f" of atoms in {structure2}."
        )
    return search(st1Cras, st2Cras, output, minCoordDev, minAdpDev)
