# coding: utf-8
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
    logging.info(f"Saved {len(df)} reflections to {filename}.")
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
