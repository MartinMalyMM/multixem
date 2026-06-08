# coding: utf-8
import json
import logging
import os
import re
from importlib import resources
import gemmi
import numpy
import pandas
from .bootstrap_statistics import df_scatter_plot, plot_histogram
from .tools import (
    # CID2RefmacRestraint,
    # CRA2CID,
    makeAddressStr,
    json_numpy_converter,
    select_CIDs_of_residues,
)


'''
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
'''


def unrestrain_yaml(geometry_objects_ref):
    """geometry_objects_ref is a list of dicts with keys:
    atom1, atom2, atom3, atom4 - CIDs of the atoms involved"""
    cids = set()
    for geometry_object in geometry_objects_ref:
        cids.update(select_CIDs_of_residues(geometry_object))
    return {
        "refine": {
            "vdw_exclusion": {
                "selections": list(cids),
                "pair_selections": [],
            }
        }
    }


def floating_origin_detect(st: gemmi.Structure):
    """
    Detect if there is a floating origin problem in the point group
    and in which directions.
    """

    try:
        Rrot_sum_eigen_path = resources.files("multixem.data").joinpath(
            "Rrot_sum_eigen.json"
        )
        with Rrot_sum_eigen_path.open("r", encoding="utf-8") as f:
            point_group_results = json.load(f)
    except FileNotFoundError:
        from multixem.scripts.Rrot_sum_eigen import get_Rrot_sum_eigen

        point_group_results = get_Rrot_sum_eigen()
    if not point_group_results:
        logging.warning("Could not load point group results for origin shift analysis.")
        return [], []

    sg = st.find_spacegroup()
    pg = sg.point_group_hm()
    logging.info(
        f"Point group: {pg}, space group: {sg.hm}, unit cell parameters: {st.cell}"
    )
    if point_group_results[pg]["subspace_dimension"] == 0:
        logging.info(
            f"In point group {pg}, the origin is well defined."
            " No shift analysis needed."
        )
        return [], []

    axis_dict = {0: "x", 1: "y", 2: "z"}
    basis_vectors = [
        numpy.array(vec) for vec in point_group_results[pg]["eigenvectors"]
    ]
    eigenindices_nonzero = point_group_results[pg]["eigenindices_nonzero"]
    logging.info(
        "There is a float origin problem. Directions in which origin can be shifted:"
    )
    for i, axis in axis_dict.items():
        if i in eigenindices_nonzero:
            logging.info(f"  {axis}: {basis_vectors[i]}")
    logging.info(f"Orthogonalization matrix: {st.cell.orth.mat}")
    logging.info(f"Fractionalization matrix: {st.cell.frac.mat}")
    return basis_vectors, eigenindices_nonzero


def load_fractional_coords(st: gemmi.Structure) -> numpy.ndarray:
    """
    Load a structure with Gemmi and return fractional coordinates and B-factors.

    Args:
        st (gemmi.Structure): The structure to load.

    Returns:
        coords_frac (numpy.ndarray): Array of shape (N, 3) with fractional coordinates.
    """
    cell = st.cell
    cras = list(st[0].all())
    coords_frac = numpy.array(
        [cell.fractionalize(cra.atom.pos).tolist() for cra in cras], dtype=numpy.float64
    )
    return coords_frac


def calculate_alpha_displacement(
    coords_diff_frac: numpy.ndarray,
    basis_vectors: list[numpy.ndarray],
    eigenindices_nonzero: list[int],
    atom_weights: numpy.ndarray,
    occs: numpy.ndarray,
) -> numpy.ndarray:
    """
    Compute coefficients alpha for the translation expressed in basis_vectors.

        alpha_i = 1 / N * sum_j (delta_coords_j . basis_vector_i)
    for j in all atoms and i = {x, y, z} for eigenvectors with non-zero eigenvalues.
    N is the number of atoms, delta_coords_j is the difference in fract. coord.
    (orthornomal system)

    Args:
        coords_diff_frac: (N, 3) array with differences in fractional coordinates
        basis_vectors: list of k vectors, each shape (3,), with k in [0, 3]
        eigenindices_nonzero: list of indices for which eigen vectors
                              with non-zero eigenvalues were found
        atom_weights: (N,) array with weights for each atom
        occs: (N,) array with occupancies for each atom

    Returns:
        alpha: (3,) array
    """

    alpha = numpy.array([0.0, 0.0, 0.0])
    weights = atom_weights * occs
    mean_disp = numpy.average(coords_diff_frac, axis=0, weights=weights)
    for i, basis_vector in enumerate(basis_vectors):
        if i in eigenindices_nonzero:
            alpha[i] = numpy.dot(mean_disp, basis_vector)

    return alpha


def apply_alpha_to_structure_cartesian(
    st: gemmi.Structure,
    alpha: numpy.ndarray,
):

    for cra in st[0].all():
        cra.atom.pos = cra.atom.pos + alpha

    return st


def floating_origin_shift(
    st1: gemmi.Structure, st2: gemmi.Structure, basis_vectors, eigenindices_nonzero
):
    """
    Apply a shift to st2 to correct for the floating origin problem based
    on the difference in coordinates between st1 and st2.
    """

    coords1_frac = load_fractional_coords(st1)
    coords2_frac = load_fractional_coords(st2)
    atom_weights = numpy.array([float(cra.atom.element.weight) for cra in st1[0].all()])
    occs = numpy.array([float(cra.atom.occ) for cra in st1[0].all()])
    coords_diff_frac = coords1_frac - coords2_frac
    coords_diff_frac = numpy.array(
        [
            gemmi.Fractional(d[0], d[1], d[2]).wrap_to_zero().tolist()
            for d in coords_diff_frac
        ],
        dtype=numpy.float64,
    )

    # Scalar: mean over atoms of (dx² + dy² + dz²)
    sum_coords_diff_frac_sq = numpy.sum(
        atom_weights
        * (
            coords_diff_frac[:, 0] ** 2
            + coords_diff_frac[:, 1] ** 2
            + coords_diff_frac[:, 2] ** 2
        )
    )
    # individual components also
    sum_coords_diff_frac_sq_xyz = numpy.sum(coords_diff_frac**2, axis=0)
    # logging.info(f"Initial average squared difference in fractional coordinates:"
    # " {avg_coords_diff_frac_sq:.7f} {avg_coords_diff_frac_sq_z:.7f} in z")

    alpha_array = calculate_alpha_displacement(
        coords_diff_frac, basis_vectors, eigenindices_nonzero, atom_weights, occs
    )
    alpha_frac = gemmi.Fractional(alpha_array[0], alpha_array[1], alpha_array[2])
    alpha_cart = st1.cell.orthogonalize(alpha_frac)
    st2_shifted = apply_alpha_to_structure_cartesian(st2, alpha_cart)
    # Save the shifted structure ?

    coords2_frac_shifted = load_fractional_coords(st2_shifted)
    coords_diff_frac_shifted = coords1_frac - coords2_frac_shifted
    coords_diff_frac_shifted = numpy.array(
        [
            gemmi.Fractional(d[0], d[1], d[2]).wrap_to_zero().tolist()
            for d in coords_diff_frac_shifted
        ]
    )

    sum_coords_diff_frac_sq_shifted = numpy.sum(
        atom_weights
        * (
            coords_diff_frac_shifted[:, 0] ** 2
            + coords_diff_frac_shifted[:, 1] ** 2
            + coords_diff_frac_shifted[:, 2] ** 2
        )
    )
    sum_coords_diff_frac_sq_shifted_xyz = numpy.sum(coords_diff_frac_shifted**2, axis=0)
    sum_coords_diff_frac_sqs = (
        sum_coords_diff_frac_sq,
        sum_coords_diff_frac_sq_shifted,
        sum_coords_diff_frac_sq_xyz,
        sum_coords_diff_frac_sq_shifted_xyz,
    )

    return st2_shifted, alpha_frac, alpha_cart, sum_coords_diff_frac_sqs


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
                if "symmetry" in col or "disorder" in col:  # add None if not found
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
        "?_atom_site_disorder_assembly",
        "?_atom_site_disorder_group",
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
    Do not include atoms from symmetry-related molecules specified in symmetry_cols.
    If elem_col is provided, hydrogen atoms will be excluded.

    Args:
        table: CIF table to process.
        atom_cols: List of columns with atom labels.
        symmetry_cols: List of columns with symmetry information (optional).
                       Atoms with non-empty values in these columns will be excluded.
        value_sigma_cols: List of columns with values and standard deviations (optional)
        value_sigma_cols_names: List of base names for value/sigma columns (optional).
                                If "disorder" is in the name,
                                    only the value will be extracted without sigma.
        elem_col: Column with chemical element (optional).
                  Hydrogen atoms (with elem_col value "H") will be excluded.
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
                if "disorder" in value_sigma_cols_names[i].lower():
                    entry[f"{value_sigma_cols_names[i]}"] = (
                        value_sigma_cols[i][j_idx]
                        if value_sigma_cols[i][j_idx] not in [".", "?", "", None]
                        else None
                    )
                else:
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
            disorder_assembly_col,
            disorder_group_col,
        ) = coords_cols
        if not skip_hydrogen:
            elem_col = ""
        atoms_list = collect_geometry_lists(
            table_coords,
            [atom_col],
            [],
            [
                x_fract_col,
                y_fract_col,
                z_fract_col,
                u_iso_col,
                disorder_assembly_col,
                disorder_group_col,
            ],
            [
                "x_frac",
                "y_frac",
                "z_frac",
                "u_iso",
                "disorder_assembly",
                "disorder_group",
            ],
            elem_col=elem_col,  # exclude hydrogens
        )

        st = gemmi.read_small_structure(smcif)
        if skip_hydrogen:
            st.remove_hydrogens()
        assert len(atoms_list) == len(st.sites), (
            f"Number of atoms in coordinates table ({len(atoms_list)}) does not match"
            f" number of atoms in structure {smcif} ({len(st.sites)})"
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
            st.change_occupancies_to_crystallographic()
            occ_list = collect_geometry_lists(
                table_coords,
                [atom_col],
                [],
                [occ_col],
                ["occupancy"],
                elem_col=elem_col,
            )
            assert len(occ_list) == len(atoms_list), (
                f"Number of atoms in occupancy table ({len(occ_list)}) does not match"
                f" number of atoms in coordinates table ({len(atoms_list)})"
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
    no_origin_shift=False,
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
        geometry_cids_file (str)
        geometry_objects_ref (list)
        no_origin_shift (bool): Do not perform correction for floating origin

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

        if not no_origin_shift:
            basis_vectors, eigenindices_nonzero = floating_origin_detect(st_ref)
        else:
            basis_vectors = eigenindices_nonzero = []
        if eigenindices_nonzero:
            alphas_frac = numpy.zeros((3, len(refined_mmcifs)), dtype=numpy.float32)
            alphas_cart = numpy.zeros((3, len(refined_mmcifs)), dtype=numpy.float32)
            sum_coords_diff_frac_sq = numpy.zeros(
                (len(refined_mmcifs)), dtype=numpy.float32
            )
            sum_coords_diff_frac_sq_xyz = numpy.zeros(
                (3, len(refined_mmcifs)), dtype=numpy.float32
            )
            sum_coords_diff_frac_sq_shifted = numpy.zeros(
                (len(refined_mmcifs)), dtype=numpy.float32
            )
            sum_coords_diff_frac_sq_shifted_xyz = numpy.zeros(
                (3, len(refined_mmcifs)), dtype=numpy.float32
            )

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

        # If there is a float origin problem in a point group, shift the coordinates
        # to match the first structure
        if mmcif_ref and os.path.isfile(mmcif_ref) and eigenindices_nonzero:
            st_shifted, alpha_frac, alpha_cart, alpha_sums = floating_origin_shift(
                st_first, st, basis_vectors, eigenindices_nonzero
            )
            alphas_frac[:, s] = numpy.array(
                [alpha_frac.x, alpha_frac.y, alpha_frac.z], dtype=numpy.float32
            )
            alphas_cart[:, s] = numpy.array(
                [alpha_cart.x, alpha_cart.y, alpha_cart.z], dtype=numpy.float32
            )
            (
                sum_coords_diff_frac_sq[s],
                sum_coords_diff_frac_sq_shifted[s],
                sum_coords_diff_frac_sq_xyz[:, s],
                sum_coords_diff_frac_sq_shifted_xyz[:, s],
            ) = alpha_sums
        else:
            st_shifted = st

        st_cras = [
            cra for cra in st[0].all() if not (skip_hydrogen and cra.atom.is_hydrogen())
        ]
        assert len(st_first_cras) == len(st_cras), (
            f"Different number of atoms in structure model after bootstrapping: {mmcif}"
            f". Expected {len(st_first_cras)} atoms, got {len(st_cras)}."
        )
        st_shifted_cras = [
            cra
            for cra in st_shifted[0].all()
            if not (skip_hydrogen and cra.atom.is_hydrogen())
        ]
        for a, (cra_first, cra) in enumerate(zip(st_first_cras, st_shifted_cras)):
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
            csv_data[i]["disorder_assembly"] = atoms_list[i]["disorder_assembly"]
            csv_data[i]["disorder_group"] = atoms_list[i]["disorder_group"]
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
    logging.info(f"Mean structure statistics written to {csv_filename}")

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

    if mmcif_ref and os.path.isfile(mmcif_ref) and eigenindices_nonzero:
        # Write floating origin analysis results
        floating_origin_dict = dict()
        for d in range(3):
            if d in eigenindices_nonzero:
                floating_origin_dict[f"alpha_frac_{'xyz'[d]}"], _, _ = plot_histogram(
                    alphas_frac[d],
                    f"mean shift, fractional coordinates ({'xyz'[d]} direction)",
                    ref={},
                    idx=idx,
                    prefix=prefix,
                    outlier_factor=99,
                )
                floating_origin_dict[f"alpha_cart_{'xyz'[d]}"], _, _ = plot_histogram(
                    alphas_cart[d],
                    f"mean shift, Cartesian coordinates ({'xyz'[d]} direction)",
                    ref={},
                    idx=idx,
                    prefix=prefix,
                    outlier_factor=99,
                )
                floating_origin_dict[f"sum_coords_diff_frac_sq_{'xyz'[d]}"], _, _ = (
                    plot_histogram(
                        sum_coords_diff_frac_sq_xyz[d],
                        "sum of coordinate differences squared,"
                        f" calculated in fractional coordinates ({'xyz'[d]} direction)",
                        ref={},
                        idx=idx,
                        prefix=prefix,
                        outlier_factor=99,
                    )
                )
                (
                    floating_origin_dict[f"sum_coords_diff_frac_sq_shifted_{'xyz'[d]}"],
                    _,
                    _,
                ) = plot_histogram(
                    sum_coords_diff_frac_sq_shifted_xyz[d],
                    "sum of coordinate differences squared after shift,"
                    f" calculated in fractional coordinates ({'xyz'[d]} direction)",
                    ref={},
                    idx=idx,
                    prefix=prefix,
                    outlier_factor=99,
                )
        floating_origin_dict["sum_coords_diff_frac_sq"], _, _ = plot_histogram(
            sum_coords_diff_frac_sq,
            "sum of coordinate differences squared,"
            " calculated in fractional coordinates",
            ref={},
            idx=idx,
            prefix=prefix,
            outlier_factor=99,
        )
        floating_origin_dict["sum_coords_diff_frac_sq_shifted"], _, _ = plot_histogram(
            sum_coords_diff_frac_sq_shifted,
            "sum of coordinate differences squared after shift,"
            " calculated in fractional coordinates",
            ref={},
            idx=idx,
            prefix=prefix,
            outlier_factor=99,
        )
        # mean_alpha_frac = numpy.mean(alphas_frac, axis=1)
        # mean_alpha_cart = numpy.mean(alphas_cart, axis=1)
        floating_origin_json_filename = (
            f"{prefix}group{idx}_bootstrap_floating_origin_analysis.json"
            if idx
            else f"{prefix}bootstrap_floating_origin_analysis.json"
        )
        with open(floating_origin_json_filename, "w") as f:
            json.dump(floating_origin_dict, f, indent=2, default=json_numpy_converter)
        logging.info(
            f"Floating origin analysis results written to"
            f" {floating_origin_json_filename}"
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
