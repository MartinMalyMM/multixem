# coding: utf-8
import os
import argparse
import gemmi
import pandas
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
    parser.add_argument(
        "--n_batches",
        type=positive_int,
        default=60,
        help="Number of batches per merging group. Must be a positive integer.",
    )

    def validate_args(args):
        if args.n_batches and not args.hklin_unmerged:
            parser.error("--n_batches requires --hklin_unmerged to be provided.")

    parser.set_defaults(func=validate_args)
    return parser


def merge_in_groups(unmerged, n_batches_in_group, prefix):

    def merge_group(
        df_groups,
        i_group,
        cell,
        spacegroup,
        n_expected,
        wavelength=0,
        n_groups=0,
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
        intensities.merge_in_place(gemmi.DataType.Anomalous)
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
        mtz_group_merged.write_to_file(f"{prefix}group{g_with_leading_zeros}.mtz")

    m = gemmi.read_mtz_file(unmerged)
    print(m)
    print(list(m.columns))
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
        " A), cell and symmetry from the input file {unmerged}:",
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

    for i_group in range(len(batches_split) - 1):
        # print(batches_split[i_group], batches_split[i_group+1])
        df_group = df.loc[
            (df["BATCH"] >= batches_split[i_group])
            & (df["BATCH"] < batches_split[i_group + 1])
        ]
        df_groups.append(df_group)
        merge_group(
            df_groups,
            i_group,
            m.cell,
            m.spacegroup,
            n_expected,
            wavelength,
            n_groups=len(batches_split),
            prefix=prefix,
        )


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

    if args.hklin_unmerged:
        print("Unmerged diffraction data:", args.hklin_unmerged)
        # TODO: select automatically the number of batches in group (now default 60)
        n_batches_per_group = args.n_batches
        print("Number of batches in merging group:", n_batches_per_group)
        merge_in_groups(args.hklin_unmerged, n_batches_per_group, prefix)


if __name__ == "__main__":
    main()
