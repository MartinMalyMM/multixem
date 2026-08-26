# coding: utf-8
import os
import logging
import warnings
import gemmi
import numpy
import pandas
from .tools import write_mtz_from_df


def bootstrap_dataset(
    mtz_file: str,
    binner: gemmi.Binner,
    seeds=[1001, 1002, 1003],
    labin: str = "",
    draw_factor: float = 1.0,
    random_weights: bool = False,
    col_free: str = "",
    fraction_zero: float = 0.05,
):
    """
    Bootstrap the dataset from an MTZ file and save the results in new MTZ files.

    Args:
        mtz_file (str): Path to the input MTZ file.
        binner (gemmi.Binner): gemmi.Binner object for resolution binning.
        seeds (list of int): List of random seeds for bootstrapping.
        labin (str): Column label (e.g. `IMEAN,SIGIMEAN`)
            to apply `df.dropna(subset=[labin.split(",")[0]])`.
        draw_factor (float): Factor for a number of draws in resampling.
                              By default, the number of draws is equal to the number of
                              reflections in each bin (draw_factor==1.0).
        random_weights (bool): Assign random weights instead of
                               resampling with replacement.
        col_free (str): Label for free R flag in `mtz_file` which would be used to set
                        zero weight for reflections of the free set
                        if `fraction_zero` is zero
        fraction zero (float): Sets fraction of reflections with zero weight [0.0, 1.0)
    Returns:
        list of str: List of output MTZ filenames created during bootstrapping.
    """

    def resample(
        n: int,
        seed: int = 1001,
        draw_factor: float = 1.0,
        column_name: str = "llweight",
    ):
        """
        Create a DataFrame`llweight` column using resampling with replacement.

        Args:
            n (int): Number of items to resample.
            seed (int): Random seed for reproducibility.
            draw_factor (float): Factor for a number of draws in resampling.
                                  By default, the number of draws is equal to the
                                  number of reflections in each bin (draw_factor==1.0).
            column_name (str): Name of the column to create in the DataFrame.
        Returns:
            pandas.Series: Series with the bootstrap weights for each reflection.
        """
        n_draws = int(n * draw_factor)
        rng = numpy.random.default_rng(seed)
        df_random = pandas.DataFrame(
            rng.integers(1, n + 1, size=n_draws), columns=["index_resample"]
        )
        df_weight = (
            df_random.groupby(["index_resample"])
            .size()
            .reindex(range(1, n + 1), fill_value=0)
        )
        if draw_factor != 1.0:
            df_weight = df_weight / draw_factor

        return df_weight.rename(column_name)

    def resample_random(
        n: int,
        zero_mask: numpy.ndarray,
        seed: int = 1001,
        column_name: str = "llweight",
    ):
        """
        Create a DataFrame`llweight` column using random resampling and keeping
        a fraction of zero weights.

        Args:
            n (int): Number of items to resample.
            seed (int): Random seed for reproducibility.
            zero_mask (numpy.ndarray): Optional boolean mask;
                                       rows set to True are forced to zero.
                                       Applied if `fraction_zero` is zero.
            column_name (str): Name of the column to create in the DataFrame.
        Returns:
            pandas.Series: Series with the weights for each reflection.
        """
        rng = numpy.random.default_rng(seed)
        df_random = pandas.DataFrame(rng.random(size=n), columns=["index_resample"])
        df_weight = df_random["index_resample"].copy()
        # Set a fraction of weights to zero based on the provided mask
        zero_mask_array = numpy.asarray(zero_mask, dtype=bool)
        if zero_mask_array.shape[0] != n:
            raise ValueError(
                "Length mismatch between zero_mask and bin size in "
                f"random resampling: {zero_mask_array.shape[0]} != {n}"
            )
        df_weight.loc[zero_mask_array] = 0.0
        # Renormalize so that the total weight equals n
        weight_sum = df_weight.sum()
        assert weight_sum > 0
        df_weight = df_weight * (n / weight_sum)

        return df_weight.rename(column_name)

    if random_weights:
        if abs(fraction_zero) > 1e-6:
            logging.info(
                f"\nBootstrapping dataset {mtz_file} (using random resampling"
                f" and keeping fraction of zero weights {fraction_zero})",
            )
        elif col_free:
            logging.info(
                f"\nBootstrapping dataset {mtz_file} (using random resampling,"
                " free reflections will be excluded)."
            )
        else:
            raise RuntimeError(
                f"Free reflections were not found in {mtz_file} and --fraction_zero"
                " is not set. Cannot assign reflections with zero weight. Aborting."
            )
    else:
        logging.info(
            "\n"
            f"Bootstrapping dataset {mtz_file} (using a draw factor of {draw_factor})",
        )
    mtzs_out = []
    mtz = gemmi.read_mtz_file(mtz_file)
    df = pandas.DataFrame(data=mtz.array, columns=mtz.column_labels())
    df = df.astype({name: "int32" for name in ["H", "K", "L"]})
    columns_dict = {
        col.label: col.type for col in mtz.columns if col.label not in ["H", "K", "L"]
    }

    n_unique_orig = 0
    # i_col = "IMEAN"  # can be just "I" after servalcat fw or sigmaa, or IMEAN?
    # dropping reflections can cause problems, let's save the filtered dataset as MTZ
    if labin and labin.split(",")[0] in df.columns:
        df = df.dropna(subset=[labin.split(",")[0]])
        # Save the filtered dataset as MTZ, preserving all original columns
        mtz_filtered_name = (
            f"{os.path.splitext(os.path.basename(mtz_file))[0]}_filtered.mtz"
        )
        write_mtz_from_df(df, mtz, columns=columns_dict, filename=mtz_filtered_name)
        n_unique_orig = df.shape[0]
    else:
        warnings.warn(
            f"Column {labin} not found in MTZ file {mtz_file}. "
            f"Using all reflections for bootstrapping."
        )
    n_unique_expected = gemmi.count_reflections(
        mtz.cell,
        mtz.spacegroup,
        mtz.resolution_high(),
        mtz.resolution_low(),
        unique=True,
    )

    hkl_array = numpy.array(df[["H", "K", "L"]].values, numpy.int32)
    hkl_array = numpy.ascontiguousarray(hkl_array, dtype=numpy.int32)
    df["bin"] = binner.get_bins(hkl_array)
    # print("No. unique reflections:", len(df))
    # print(df.head(10))
    # print(df.describe())

    bins = [bin for _, bin in df.groupby("bin")]

    # Create per-bin masks once and keep them for all bootstrap samples
    zero_mask_bins = []
    if random_weights:
        if abs(fraction_zero) > 1e-6:
            # Create a random mask for reflections with zero weight
            rng = numpy.random.default_rng(seeds[0])
            zero_mask_bins = [rng.random(len(bin)) < fraction_zero for bin in bins]
        elif col_free:
            # Use free reflections to create a mask for refls with zero weight
            zero_mask_bins = [bin[col_free].eq(0).to_numpy() for bin in bins]
        else:
            raise RuntimeError(
                f"Free reflections were not found in {mtz_file} and --fraction_zero"
                " is not set. Cannot assign reflections with zero weight. Aborting."
            )

    completeness_list = []
    for i, seed in enumerate(seeds):
        if random_weights:
            assert len(zero_mask_bins) == len(bins)
            parts = []
            for b, bin in enumerate(bins):
                w = resample_random(
                    len(bin),
                    zero_mask_bins[b],
                    seed=seed,
                )
                parts.append(pandas.Series(w.values, index=bin.index, name="llweight"))
            df_bootstrap1_weight = pandas.concat(parts).sort_index()
        else:
            parts = []
            for bin in bins:
                w = resample(len(bin), seed, draw_factor)
                parts.append(pandas.Series(w.values, index=bin.index, name="llweight"))
            df_bootstrap1_weight = pandas.concat(parts).sort_index()

        # Keep original reflection order to preserve HKL-to-weight mapping.
        df_bootstrap1_weight_hkl = df[["H", "K", "L"]].copy()
        df_bootstrap1_weight_hkl["llweight"] = df_bootstrap1_weight.reindex(
            df.index
        ).values
        weight_sum = df_bootstrap1_weight.sum()
        if abs(weight_sum - len(df)) > 1e-6:
            logging.warning(
                f"Sum of weight coefficients ({weight_sum}) from bootstrap resampling"
                f" {i} does not match the number of reflections ({len(df)})."
            )

        # Save the llweights in the MTZ file
        mtz_out_name = (
            f"{os.path.splitext(os.path.basename(mtz_file))[0]}_llweight{i}.mtz"
        )
        write_mtz_from_df(
            df_bootstrap1_weight_hkl,
            mtz,
            columns={"llweight": "R"},
            filename=mtz_out_name,
        )
        mtzs_out.append(mtz_out_name)

        # Compute completeness
        n_unique = len(
            df_bootstrap1_weight_hkl[df_bootstrap1_weight_hkl["llweight"] > 0]
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
    if random_weights:
        if n_unique_orig:
            logging.info(
                "Completeness of bootstrap datasets"
                " after exclusion of zero-weighted reflections:"
                f" {completeness_mean:.2%}"
            )
        else:
            logging.info(
                "Completeness of bootstrap datasets"
                " after exclusion of zero-weighted reflections:"
                f" {completeness_mean:.2%}\n"
            )
    else:
        completeness_std = numpy.std(completeness_list, ddof=1, mean=completeness_mean)
        logging.info(
            "Completeness of bootstrap datasets"
            " after exclusion of zero-weighted reflections:"
            f" {completeness_mean:.2%} ± {completeness_std:.2%}"
            f" (using a draw factor of {draw_factor})\n"
        )
    if n_unique_orig:
        completeness_orig = n_unique_orig / n_unique_expected
        logging.info(f"Completeness of the original dataset: {completeness_orig:.2%}\n")

    return mtzs_out
