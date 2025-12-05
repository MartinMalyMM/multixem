# coding: utf-8
import os
import numpy
import gemmi
import logging
import pandas
import re


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
    logging.info(f"Saved statistics to {filename}")


def write_mtz_from_df(df, mtz_ref, columns, filename):
    """
    Create a gemmi.Mtz object from a pandas dataframe and save to file.
    The numpy.float32 format is used for the data.

    Args:
        df (pandas.DataFrame): DataFrame containing columns for H, K, L and other data.
        mtz_ref (gemmi.Mtz): Reference MTZ file or object for cell and spacegroup.
        columns (dict): Dictionary of column names and their MTZ data types
            to include after H, K, L.
        filename (str): Output filename for the MTZ file.
    Returns:
        None
    """
    mtz = gemmi.Mtz(with_base=True)
    if isinstance(mtz_ref, str):
        mtz_ref = gemmi.read_mtz_file(mtz_ref)
    mtz.spacegroup = mtz_ref.spacegroup
    mtz.set_cell_for_all(mtz_ref.cell)
    mtz.add_dataset(mtz_ref.datasets[0].dataset_name)
    for col_name, col_type in columns.items():
        mtz.add_column(col_name, col_type)
    data = numpy.array(
        df[["H", "K", "L"] + list(columns.keys())].values.astype(numpy.float64)
    )
    f32max = numpy.finfo(numpy.float32).max
    data = numpy.clip(data, -f32max, f32max).astype(numpy.float32)

    mtz.set_data(data)
    mtz.write_to_file(filename)
    logging.info(f"Saved {len(df)} reflections to {filename}")
    return


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
            logging.warning(
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
        logging.warning(
            f"Scale denominator for bin {b + 1} is zero"
            f" ({dmax} - {dmin} A),"
            " setting scale for this bin to 1."
        )
        return 1.0


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


def CID2RefmacRestraint(geometry_object):
    """geometry_object is a dict with keys:
    atom1, atom2, atom3, atom4 - CIDs of the atoms involved
    values - list of reference values, should contain one value only"""

    ref_value = geometry_object.get("values", [None])[0]
    refmacAddresses = []
    for cid in [
        geometry_object["atom1"],
        geometry_object["atom2"],
        geometry_object["atom3"],
        geometry_object["atom4"],
    ]:
        if not cid:
            continue
        cid_symm = len(cid.split("@")) > 1
        cid_split = cid.split("@")[0].split("/")
        assert len(cid_split) >= 5, f"Invalid CID: {cid}"
        seqid_split = cid_split[3].split(".")
        atom_split = cid_split[4].split(":")
        refmacAddress = f"chain {cid_split[2]} "
        refmacAddress += f"resi {seqid_split[0]} "
        if len(seqid_split) > 1:
            refmacAddress += f"inse {seqid_split[1]} "
        refmacAddress += f"atom {atom_split[0]}"
        if len(atom_split) > 1:
            refmacAddress += f" alte {atom_split[1]}"
        if cid_symm:
            refmacAddress += " symm y"
        refmacAddresses.append(refmacAddress)

    if not geometry_object["atom2"]:
        # single atom in refmac syntax - may not be useful...
        return refmacAddresses[0]
    elif not geometry_object["atom3"]:
        # external distance restraint
        assert "@" not in geometry_object["atom1"], (
            "For interatomic distances, the first atom must not be from"
            f" a symmetry mate: {geometry_object['atom1']} {geometry_object['atom2']}"
        )
        if ref_value is None:
            ref_value = 2.2
        restraint = f"exte dist first {refmacAddresses[0]} second {refmacAddresses[1]}"
        restraint += f" value {ref_value:<.2f} sigma 99 type 0"
    elif not geometry_object["atom4"]:
        # external angle restraint
        assert "@" not in geometry_object["atom2"], (
            "For angles, the second atom must not be from a symmetry mate:"
            f" {geometry_object['atom1']} {geometry_object['atom2']}"
            f" {geometry_object['atom3']}"
        )
        if ref_value is None:
            ref_value = 120.0
        restraint = f"exte angle first {refmacAddresses[0]} next {refmacAddresses[1]}"
        restraint += f" next {refmacAddresses[2]}"
        restraint += f" value {ref_value:<.2f} sigma 999 type 0"
    else:
        # external torsion restraint
        if ref_value is None:
            ref_value = 120.0
        restraint = f"exte torsion first {refmacAddresses[0]} next {refmacAddresses[1]}"
        restraint += f" next {refmacAddresses[2]} next {refmacAddresses[3]}"
        restraint += f" value {ref_value:<.2f} sigma 999 type 0"
    return restraint


def filename_replace_char(filename):
    filename = filename.replace("=", "_equals_")
    filename = filename.replace(">", "_gt_")
    filename = filename.replace("<", "_lt_")
    filename = re.sub(r"[^A-Za-z0-9_\-.]", "_", filename)
    return filename


def json_numpy_converter(o):
    if isinstance(o, numpy.generic):
        return o.item()
    if isinstance(o, numpy.ndarray):
        return o.tolist()
    # Fallback: convert unknown objects to string
    return str(o)


def scale_reflections(refl1, refl2, binner, bin_stats_list=[], output_mtz2_prefix=""):
    """
    Scale reflections from refl2 to refl1 in resolution bins defined by binner.

    Args:
        refl1 (str or gemmi.Mtz or pandas.DataFrame): First reflection dataset.
        refl2 (str or gemmi.Mtz or pandas.DataFrame): Second reflection dataset.
        binner (gemmi.Binner): Binner object defining resolution bins.
        bin_stats_list (list of dict): List to store statistics for each resolution bin.
        output_mtz2_prefix (str): If provided, save scaled refl2 to MTZ file
            with this prefix.

    Returns:
        df2_scaled (pandas.DataFrame): Scaled reflections from refl2.
        bin_stats_list (list of dict): Updated list with statistics
            for each resolution bin.
    """

    if isinstance(refl1, str):
        if not os.path.isfile(refl1):
            raise FileNotFoundError(f"Reflection file not found: {refl1}")
        mtz1 = gemmi.read_mtz_file(refl1)
        df1 = pandas.DataFrame(data=mtz1.array, columns=mtz1.column_labels())
        df1 = df1.astype({name: "int32" for name in ["H", "K", "L"]})
    elif isinstance(refl1, gemmi.Mtz):
        df1 = pandas.DataFrame(data=refl1.array, columns=refl1.column_labels())
        df1 = df1.astype({name: "int32" for name in ["H", "K", "L"]})
    else:
        df1 = refl1

    if isinstance(refl2, str):
        if not os.path.isfile(refl2):
            raise FileNotFoundError(f"Reflection file not found: {refl2}")
        mtz2 = gemmi.read_mtz_file(refl2)
        df2 = pandas.DataFrame(data=mtz2.array, columns=mtz2.column_labels())
        df2 = df2.astype({name: "int32" for name in ["H", "K", "L"]})
    elif isinstance(refl2, gemmi.Mtz):
        df2 = pandas.DataFrame(data=refl2.array, columns=refl2.column_labels())
        df2 = df2.astype({name: "int32" for name in ["H", "K", "L"]})
    else:
        df2 = refl2

    f_col = "DELFWT"  # Select only observed reflections
    columns = ["FWT", "PHWT", "DELFWT", "PHDELWT"]
    # afterwards, rename to FWT1, PHWT1, ..., FWT2, PHWT2, ...
    # columns1 = [col + "1" for col in columns]
    columns1_dict = {col: col + "1" for col in columns}
    # columns2 = [col + "2" for col in columns]
    columns2_dict = {col: col + "2" for col in columns}

    df1 = df1[["H", "K", "L"] + columns]  # Select only relevant columns
    df1 = df1.dropna(subset=[f_col])  # Select only observed reflections
    df1 = df1.rename(columns=columns1_dict)  # Rename
    # n_refl1 = len(df1)
    # logging.info(f"No. unique reflections: {n_refl1} in file {mtz_file_1}")

    llweight_col = ["llweight"] if "llweight" in df2.columns else []
    df2 = df2[["H", "K", "L"] + columns + llweight_col]
    df2 = df2.dropna(subset=[f_col])  # Select only observed reflections
    if llweight_col:
        df2 = df2.dropna(subset=llweight_col)
    df2 = df2.rename(columns=columns2_dict)
    # n_refl2 = len(df2)
    # logging.info(f"No. unique reflections: {n_refl2} in file {mtz_file_2}")

    # Extract common Miller indices (H, K, L)
    df = pandas.merge(df1, df2, on=["H", "K", "L"])
    # n_refl = len(df)
    # logging.info(
    #     f"No. unique reflections: {n_refl} in common;"
    #     f" ratios to the originals: {n_refl / n_refl1:.4f}   {n_refl / n_refl2:.4f}"
    # )
    hkl_common_array = numpy.array(df[["H", "K", "L"]].values, numpy.int32)
    hkl_common_array = numpy.ascontiguousarray(hkl_common_array, dtype=numpy.int32)
    # print(len(hkl_common_array))  # should be equal to n_refl

    # Scaling per resolution bins
    df["BIN"] = binner.get_bins(hkl_common_array)

    df["FWT1RE"] = df["FWT1"] * numpy.cos(numpy.deg2rad(df["PHWT1"]))
    df["FWT1IM"] = df["FWT1"] * numpy.sin(numpy.deg2rad(df["PHWT1"]))
    df["FWT2RE"] = df["FWT2"] * numpy.cos(numpy.deg2rad(df["PHWT2"]))
    df["FWT2IM"] = df["FWT2"] * numpy.sin(numpy.deg2rad(df["PHWT2"]))
    df["DELFWT1RE"] = df["DELFWT1"] * numpy.cos(numpy.deg2rad(df["PHDELWT1"]))
    df["DELFWT1IM"] = df["DELFWT1"] * numpy.sin(numpy.deg2rad(df["PHDELWT1"]))
    df["DELFWT2RE"] = df["DELFWT2"] * numpy.cos(numpy.deg2rad(df["PHDELWT2"]))
    df["DELFWT2IM"] = df["DELFWT2"] * numpy.sin(numpy.deg2rad(df["PHDELWT2"]))

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

        scale_fwt = calc_scale_complex(
            df_bin,
            "FWT",
            "FWT2",
            b,
            bin_stats_list[b]["dmax"],
            bin_stats_list[b]["dmin"],
        )
        """scale_delfwt = calc_scale_complex(
            df_bin,
            "DELFWT",
            "DELFWT2",
            b,
            bin_stats_list[b]["dmax"],
            bin_stats_list[b]["dmin"],
        )"""

        if len(df_bin) < 100:
            logging.warning(
                f"Less than 100 reflections in bin {b + 1}"
                f" ({bin_stats_list[b]['dmax']:.4f} -"
                f" {bin_stats_list[b]['dmin']:.4f} A)."
            )
        bin_stats_list[b]["scale_fwt"] = scale_fwt
        bin_stats_list[b]["bin_count"] = len(df_bin)
        fwt2_count = int(
            pandas.to_numeric(df_bin["FWT2"], errors="coerce").notna().sum()
        )
        bin_stats_list[b]["fwt_count"] = fwt2_count
        # bin_stats_list[b]["scale_delfwt"] = scale_delfwt
        delfwt2_count = int(
            pandas.to_numeric(df_bin["DELFWT2"], errors="coerce").notna().sum()
        )
        bin_stats_list[b]["delfwt_count"] = min(delfwt2_count, delfwt2_count)

        # FWT
        df.loc[df_bin.index, "FWT2SCRE"] = scale_fwt * df_bin["FWT2RE"]
        df.loc[df_bin.index, "FWT2SCIM"] = scale_fwt * df_bin["FWT2IM"]
        # DELFWT
        df.loc[df_bin.index, "DELFWT2SCRE"] = scale_fwt * df_bin["DELFWT2RE"]
        df.loc[df_bin.index, "DELFWT2SCIM"] = scale_fwt * df_bin["DELFWT2IM"]

    df["FWT"] = numpy.hypot(
        df["FWT2SCRE"].astype(numpy.float64),
        df["FWT2SCIM"].astype(numpy.float64),
    )
    df["PHWT"] = numpy.rad2deg(numpy.arctan2(df["FWT2SCIM"], df["FWT2SCRE"]))
    df["DELFWT"] = numpy.hypot(
        df["DELFWT2SCRE"].astype(numpy.float64),
        df["DELFWT2SCIM"].astype(numpy.float64),
    )
    df["PHDELWT"] = numpy.rad2deg(numpy.arctan2(df["DELFWT2SCIM"], df["DELFWT2SCRE"]))
    df2_scaled = df[
        ["H", "K", "L", "FWT", "PHWT", "DELFWT", "PHDELWT"] + llweight_col
    ].copy()

    if output_mtz2_prefix:
        output_mtz2 = f"{output_mtz2_prefix}_scaled.mtz"
        columns_to_write_list = [
            "FWT",
            "PHWT",
            "DELFWT",
            "PHDELWT",
        ]
        columns_to_write_dict = {
            col: ("F" if not col.startswith("PH") else "P")
            for col in columns_to_write_list
        }
        if llweight_col:
            columns_to_write_dict["llweight"] = "I"
        write_mtz_from_df(df, mtz1, columns_to_write_dict, output_mtz2)
        stats_filename = f"{output_mtz2_prefix}_scaled_bin_stats.txt"
        write_bin_stats(bin_stats_list, stats_filename)

    return df2_scaled, bin_stats_list
