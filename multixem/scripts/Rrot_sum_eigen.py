import gemmi
import numpy
from pprint import pprint
import json
from collections import OrderedDict


def get_Rrot_sum_eigen():
    """
    Compute the sum of rotation symmetry operation matrices and its
    eigen decomposition for the highest-numbered space group in each point group.
    Save results to Rrot_sum_eigen.json.
    """
    # Find the highest space group per point group
    point_group_to_sg = {}
    # Iterate over all 230 space groups
    for sg in gemmi.spacegroup_table():
        pg = sg.point_group_hm()
        # Keep the space group with the highest number per point group
        if pg not in point_group_to_sg or sg.number > point_group_to_sg[pg].number:
            point_group_to_sg[pg] = sg

    print("Point groups:", list(point_group_to_sg.keys()))

    point_group_results = OrderedDict({})
    # Iterate over the selected space groups, one each for a point group
    for pg, sg in point_group_to_sg.items():
        ops = sg.operations()
        # Keep rotation matrices as integers to avoid numerical artifacts
        rots = [numpy.array(op.rot, dtype=int) for op in ops.sym_ops]
        Rsum = numpy.array(numpy.sum(rots, axis=0) / 24, dtype=int)
        # Eigen decomposition
        eigvals, eigvecs = numpy.linalg.eig(Rsum)
        eigvals_all = numpy.round(eigvals).astype(int)
        eigvecs_all = numpy.round(eigvecs).astype(int)
        # Get eigenvalues and related eigenvectors that are truly non-zero
        nonzero_idx = numpy.where(numpy.abs(eigvals) > 1e-12)[0]
        subspace_dim = len(nonzero_idx)
        eigvals_nonzero = numpy.round(eigvals[nonzero_idx]).astype(int)
        eigvecs_nonzero = numpy.round(eigvecs[:, nonzero_idx]).astype(int).T
        point_group_results[pg] = OrderedDict(
            {
                "space_group_highest": sg.hm,
                "space_group_highest_number": sg.number,
                "Rrot_sum": Rsum.tolist(),
                "eigenindices_nonzero": nonzero_idx.tolist(),
                "eigenvalues": eigvals_all.tolist(),
                "eigenvectors": eigvecs_all.tolist(),
                "subspace_dimension": subspace_dim,
                "eigenvalues_nonzero": eigvals_nonzero.tolist(),
                "eigenvectors_nonzero": eigvecs_nonzero.tolist(),
            }
        )

    pprint(point_group_results)
    with open("Rrot_sum_eigen.json", "w") as f:
        json.dump(point_group_results, f, indent=2)
    print("Rrot_sum_eigen.json written with results for each point group.")


if __name__ == "__main__":
    get_Rrot_sum_eigen()
