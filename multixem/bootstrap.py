# coding: utf-8
import os
import numpy
import pandas
import gemmi
import logging
import warnings
import re
from .tools import write_bin_stats, write_mtz_from_df, makeAddressStr


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

    logging.info(f"\nBootstrapping dataset {mtz_file}")
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
            logging.warning(
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
    completeness_std = numpy.std(completeness_list, ddof=1, mean=completeness_mean)
    logging.info(
        f"Completeness of bootstrap datasets:"
        f" {completeness_mean:.2%} ± {completeness_std:.2%}"
    )

    return mtzs_out


def bootstrap_analyse_structures(
    refined_mmcifs, idx=0, prefix="", skip_hydrogen=False, smcif=""
):
    """
    Analyse structure models (mmCIF files) to compute mean coordinates and B-factors.
    The structure models are expected to be after refinement against a bootstrapped
    data set. They must have the same number of atoms and the same atom identifiers.

    Args:
        refined_mmcifs (list of str): List of mmCIF filenames.
        idx (int): Index for naming the output files (applies if not set to 0).
        prefix (str): Prefix for the output filenames.
        skip_hydrogen (bool): If True, skip hydrogen atoms in the analysis.
        smcif (str): Path to a corresponding small molecule CIF file.

    Returns:
        None: Writes the statistics in '{prefix}group{idx}_mean_stats.csv' and
              the mean structure to '{prefix}group{idx}_mean_structure.mmcif'
              where 1000 * sigma_coordinate is saved as B-value.
    """

    def get_smcif_tables(smcif_block):
        """Extract relevant tables from a small molecule CIF block and their columns."""

        def get_table_and_columns(col_names):
            try:
                table = smcif_block.find(col_names)
                columns = [table.find_column(col) for col in col_names]
                return table, columns
            except RuntimeError as e:
                logging.warning(f"Table does not found in mmcif: {e}")
                return None, []

        coords_cols = [
            "_atom_site_label",
            # "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            # "_atom_site_occupancy",
            "_atom_site_U_iso_or_equiv",
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
            "_geom_bond_site_symmetry_2",
        ]
        angle_cols = [
            "_geom_angle_atom_site_label_1",
            "_geom_angle_atom_site_label_2",
            "_geom_angle_atom_site_label_3",
            "_geom_angle",
            "_geom_angle_site_symmetry_1",
            "_geom_angle_site_symmetry_3",
        ]
        torsion_cols = [
            "_geom_torsion_atom_site_label_1",
            "_geom_torsion_atom_site_label_2",
            "_geom_torsion_atom_site_label_3",
            "_geom_torsion_atom_site_label_4",
            "_geom_torsion",
            "_geom_torsion_site_symmetry_1",
            "_geom_torsion_site_symmetry_2",
            "_geom_torsion_site_symmetry_3",
            "_geom_torsion_site_symmetry_4",
        ]

        coords_table, coords_columns = get_table_and_columns(coords_cols)
        u_aniso_table, u_aniso_columns = get_table_and_columns(u_aniso_cols)
        bond_table, bond_columns = get_table_and_columns(bond_cols)
        angle_table, angle_columns = get_table_and_columns(angle_cols)
        torsion_table, torsion_columns = get_table_and_columns(torsion_cols)

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
        e.g. '0.1234(5)' -> (0.1234, 0.0005)
        """
        match = re.match(r"([0-9.]+)\((\d+)\)", value)
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
    ):
        """Collect atom lists (for bonds, angles, torsions).
        Do not include atoms from symmetry-related molecules."""
        geom_list = [
            {f"atom{i + 1}": atom_cols[i][j] for i in range(len(atom_cols))}
            for j in range(len(table))
            if not symmetry_cols or [col[j] == "." for col in symmetry_cols]
        ]

        if value_sigma_cols:
            for j in range(len(table)):
                for i in range(len(value_sigma_cols)):
                    value, sigma = extract_value_and_stdev(value_sigma_cols[i][j])
                    geom_list[j][f"{value_sigma_cols_names[i]}_deposit"] = value
                    geom_list[j][f"sigma_{value_sigma_cols_names[i]}_deposit"] = sigma

        return geom_list

    def collect_values_smcif(smcif):
        """
        Collect values about geometry from a small molecule CIF file from SHELX.
        """
        smcif_block = gemmi.cif.read(smcif).sole_block()
        value_shelx_res_file = smcif_block.find_value("_shelx_res_file")
        value_computing_structure_refinement = smcif_block.find_value(
            "_computing_structure_refinement"
        )
        bonds_list = angles_list = torsions_list = []
        if (
            value_shelx_res_file
            or "shelx" in value_computing_structure_refinement.lower()
        ):
            st = gemmi.read_small_structure(smcif)
            (
                (table_coords, coords_cols),
                (table_u_aniso, u_aniso_cols),
                (table_bond, bond_columns),
                (table_angle, angle_columns),
                (table_torsion, torsion_columns),
            ) = get_smcif_tables(smcif_block)

            atom_col, x_fract_col, y_fract_col, z_fract_col, u_iso_col = coords_cols
            atoms_list = collect_geometry_lists(
                table_coords,
                [atom_col],
                [],
                [x_fract_col, y_fract_col, z_fract_col, u_iso_col],
                ["x_frac", "y_frac", "z_frac", "u_iso"],
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
        else:
            atoms_list = []

        return atoms_list, u_aniso_list, bonds_list, angles_list, torsions_list

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

    def calculate_torsion_angle(
        atom1_pos, atom2_pos, atom3_pos, atom4_pos, degrees=True
    ):
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
            numpy.mean(numpy.sin(angles_rad)) ** 2
            + numpy.mean(numpy.cos(angles_rad)) ** 2
        )
        # Circular standard deviation in degrees
        return numpy.rad2deg(numpy.sqrt(-2 * numpy.log(R)))

    # numpy.set_printoptions(threshold=numpy.inf)
    st_master = gemmi.read_structure(refined_mmcifs[0])
    st_master_cras = [
        cra
        for cra in st_master[0].all()
        if not skip_hydrogen or not cra.atom.is_hydrogen()
    ]
    logging.info(
        f"{len(st_master_cras)} atoms in the master structure will be analysed."
    )
    if skip_hydrogen:
        logging.info("(Not taking into account hydrogen atoms)")

    atom_addresses = [makeAddressStr(cra) for cra in st_master_cras]
    coords = numpy.zeros(
        (len(st_master_cras), 3, len(refined_mmcifs)), dtype=numpy.float32
    )
    b_values = numpy.zeros(
        (len(st_master_cras), len(refined_mmcifs)), dtype=numpy.float32
    )
    u_aniso = numpy.zeros(
        (len(st_master_cras), 6, len(refined_mmcifs)), dtype=numpy.float32
    )

    if smcif:
        atoms_list, u_aniso_list, bonds_list, angles_list, torsions_list = (
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
        st_cras = [
            cra
            for cra in st[0].all()
            if not skip_hydrogen or not cra.atom.is_hydrogen()
        ]
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
            u_aniso[a, :, s] = [
                cra.atom.aniso.u11,
                cra.atom.aniso.u22,
                cra.atom.aniso.u33,
                cra.atom.aniso.u12,
                cra.atom.aniso.u13,
                cra.atom.aniso.u23,
            ]

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
                else "mean_bonds_stats.csv"
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
                else "mean_angles_stats.csv"
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
                else "mean_torsions_stats.csv"
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
    std_coords_norm = numpy.zeros(len(st_master_cras))
    for i in range(len(st_master_cras)):
        cov = numpy.cov(coords[i, :, :])
        std_coords_norm[i] = numpy.sqrt(
            numpy.trace(cov) + 2 * (cov[0, 1] + cov[0, 2] + cov[1, 2])
        )
    mean_b_values = numpy.mean(b_values, axis=1)  # shape: (n_atoms,)
    std_b_values = numpy.std(b_values, ddof=1, axis=1)  # shape: (n_atoms,)
    mean_u_aniso = numpy.mean(u_aniso, axis=2)  # shape: (n_atoms, 6)
    std_u_aniso = numpy.std(u_aniso, ddof=1, axis=2)  # shape: (n_atoms, 6)

    # Write calculated data as a CSV file
    csv_data = []
    i_aniso = 0
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
            }
        )
        if smcif and atoms_list:
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
            if (
                u_aniso_list
                and i_aniso < len(u_aniso_list)
                and u_aniso_list[i_aniso]["u_aniso_atom"] == st_master_cras[i].atom.name
            ):
                for key in [
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
                ]:
                    csv_data[i][f"{key}_deposit"] = u_aniso_list[i_aniso][
                        f"{key}_deposit"
                    ]
                i_aniso += 1
            else:
                csv_data[i][f"{key}_deposit"] = None
    df_csv = pandas.DataFrame(csv_data)
    df_csv = df_csv.round(6)
    csv_filename = f"{prefix}group{idx}_mean_stats.csv" if idx else "mean_stats.csv"
    df_csv.to_csv(csv_filename, index=False)
    logging.info(f"Mean structure statistics written to {csv_filename}.")

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
    logging.info(f"Mean structure written to {mmcif_filename}.")
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

        """# noqa: E741
        def is_centric_vectorized(h, k, l):  # noqa: E741
            return mtz_ref.spacegroup.operations().is_reflection_centric(
                (int(h), int(k), int(l))  # noqa: E741
            )"""

        def calculate_mean_std_count(df):
            """Calculate mean and standard deviation and number of structure factors."""

            def stats_func(x):
                if len(x) <= 1:
                    return pandas.Series([numpy.mean(x), 0.0, len(x)])

                mean_val = numpy.mean(x)
                real_mean = numpy.real(mean_val)
                imag_mean = numpy.imag(mean_val)
                real_part = numpy.real(x)
                imag_part = numpy.imag(x)
                real_var = numpy.var(real_part, ddof=1, mean=real_mean)
                imag_var = numpy.var(imag_part, ddof=1, mean=imag_mean)
                std_val = numpy.sqrt(real_var + imag_var)

                return pandas.Series([mean_val, std_val, len(x)])

            # F_complex
            df_mean_f = df.groupby(["H", "K", "L"])["F_complex"].apply(stats_func)
            df_mean_f = df_mean_f.unstack(level=-1)  # This converts Series to DataFrame
            df_mean_f.columns = ["F_complex_mean", "SIGFWT", "FWTcount"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
                df_mean_f["SIGFWT"] = df_mean_f["SIGFWT"].astype(numpy.float32)
                df_mean_f["FWTcount"] = df_mean_f["FWTcount"].astype(numpy.int32)
            df_mean_f = df_mean_f.reset_index()

            df_mean_delf = df.groupby(["H", "K", "L"])["DEL_F_complex"].apply(
                stats_func
            )
            df_mean_delf = df_mean_delf.unstack(
                level=-1
            )  # This converts Series to DataFrame
            df_mean_delf.columns = ["DEL_F_complex_mean", "SIGDELFWT", "DELFWTcount"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
                df_mean_delf["SIGDELFWT"] = df_mean_delf["SIGDELFWT"].astype(
                    numpy.float32
                )
                df_mean_delf["DELFWTcount"] = df_mean_delf["DELFWTcount"].astype(
                    numpy.int32
                )
            df_mean_delf = df_mean_delf.reset_index()

            df_mean_d_delf = df_mean_f.merge(
                df_mean_delf, on=["H", "K", "L"], how="outer"
            )
            return df_mean_d_delf

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
        df_mean = calculate_mean_std_count(df_master)

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

    logging.info(f"Loading {len(refined_mtzs)} density maps...")
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
            logging.warning(
                f"No reflections in {mtz_file} for FWT/PHWT/DELFWT/PHDELWT."
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
