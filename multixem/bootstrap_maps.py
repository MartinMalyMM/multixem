# coding: utf-8
import concurrent.futures
import json
import logging
import os
import warnings
import gemmi
import numpy
import pandas
from .tools import (
    json_numpy_converter,
    scale_reflections,
    write_bin_stats,
    write_mtz_from_df,
)


def bscale_reflections_fc(
    df: pandas.DataFrame,
    delta_b: float,
) -> pandas.DataFrame:
    """
    Scale the FC amplitudes in the DataFrame `df` according to the B-value.
    s^2 has to be pre-calculated and present in the DataFrame as a column named "s2".

    Args:
        df (pandas.DataFrame): DataFrame containing reflections.
        delta_b (float): delta B-value for scaling.

    Returns:
        pandas.DataFrame: DataFrame with scaled FC values.
    """
    df["bscale_factor"] = numpy.exp(-delta_b * df["s2"] / 4)
    df["FC"] *= df["bscale_factor"]
    df.drop(columns=["bscale_factor"], inplace=True)

    return df


def bootstrap_mean_map(
    refined_mtzs: list[str],
    idx: int = 0,
    prefix: str = "",
    binner: gemmi.Binner = None,
    mtz_ref: str = "",
    n_proc: int = 4,
    mean_b_values: numpy.ndarray = numpy.array([]),
    mean_mean_b_value: float = 0.0,
):
    """
    Calculate the mean 2Fo-Fc and Fo-Fc maps from refined MTZ files after bootstrapping.
    The maps are expected to be after refinement against a bootstrapped
    data set.

    Args:
        refined_mtzs (list of str): List of MTZ filenames.
        idx (int): Index for naming the output file (applies if not set to 0).
        prefix (str): Prefix for the output filename.
        binner (gemmi.Binner): Binner object for resolution bins (optional).
        mtz_ref (str): Reference MTZ file for scaling (optional).
        n_proc (int): Number of parallel processes to use.
        mean_b_values (list of float): Mean B-values for each data set for
                                       mean map scaling (optional).
        mean_mean_b_value (float): Mean of the mean B-values for
                                   mean map scaling (optional).

    Returns:
        None: Writes the mean maps in
            '{prefix}[group{idx}]_bootstrap_mean_map_all.mtz'
            '{prefix}[group{idx}]_bootstrap_mean_map_llweight0.mtz'
            '{prefix}[group{idx}]_bootstrap_mean_map_llweightpos.mtz'
            '{prefix}[group{idx}]_bootstrap_mean_map_llweightposw.mtz'
    """

    def merge_reflections_bootstrap(
        df_master: pandas.DataFrame,
        mtz_first: gemmi.Mtz = None,
        prefix: str = "",
        suffix: str = "",
        idx: int = 0,
        binner: gemmi.Binner = None,
        mtz_ref: str = "",
        do_llweighting: bool = False,
    ) -> pandas.DataFrame:
        """
        Merge reflections from the master DataFrame and calculate mean maps.

        Args:
            df_master (pandas.DataFrame): DataFrame containing reflections.
                It must contain columns "H", "K", "L", "F_complex", "DEL_F_complex",
            mtz_first (gemmi.Mtz): Reference MTZ object for cell and spacegroup.
            prefix (str): Prefix for the output filename.
            suffix (str): Suffix for the output filename.
            idx (int): Index for naming the output file.
            binner (gemmi.Binner): Binner object for resolution bins.
            mtz_ref (str): Reference MTZ file for scaling.
            do_llweighting (bool): Whether to apply weighting based on llweight column

        Returns:
            pandas.DataFrame: DataFrame with mean maps.
        """

        """# noqa: E741
        def is_centric_vectorized(h, k, l):  # noqa: E741
            return mtz_first.spacegroup.operations().is_reflection_centric(
                (int(h), int(k), int(l))  # noqa: E741
            )"""

        def calculate_mean_std_count(
            df: pandas.DataFrame,
            do_llweighting: bool = False,
        ) -> pandas.DataFrame:
            """Calculate mean and standard deviation and number of structure factors."""

            def stats_func(
                miller_index_df: pandas.DataFrame,
                column_name: str,
                do_llweighting: bool = do_llweighting,
            ) -> pandas.Series:
                """
                Compute weighted or unweighted mean, std, and count
                for one Miller index.

                Args:
                    miller_index_df (pandas.DataFrame): DataFrame for
                                                        a particular Miller index.
                    column_name (str): Column name to compute stats on.
                    do_llweighting (bool): Whether to apply llweighting.

                Returns:
                    pandas.Series: Series containing mean, std, and count.
                """
                x = numpy.asarray(
                    miller_index_df[column_name].to_numpy(),
                    dtype=numpy.complex128,
                )
                if len(x) <= 1:
                    return pandas.Series([numpy.mean(x), 0.0, len(x)])

                mean_val: complex
                real_mean: float
                imag_mean: float
                real_var: float
                imag_var: float
                if do_llweighting and "llweight" in miller_index_df.columns:
                    # Weighted mean and variance
                    w = numpy.asarray(
                        miller_index_df["llweight"].to_numpy(),
                        dtype=numpy.float64,
                    )
                    w = w / numpy.sum(w)  # normalize weights
                    mean_val = complex(numpy.sum(w * x))
                    real_mean = float(numpy.real(mean_val))
                    imag_mean = float(numpy.imag(mean_val))
                    real_part = numpy.real(x)
                    imag_part = numpy.imag(x)
                    real_var = float(numpy.sum(w * (real_part - real_mean) ** 2))
                    imag_var = float(numpy.sum(w * (imag_part - imag_mean) ** 2))
                else:
                    # Unweighted mean and variance
                    mean_val = complex(numpy.mean(x))
                    real_mean = float(numpy.real(mean_val))
                    imag_mean = float(numpy.imag(mean_val))
                    real_part = numpy.real(x)
                    imag_part = numpy.imag(x)
                    real_var = float(numpy.var(real_part, ddof=1, mean=real_mean))
                    imag_var = float(numpy.var(imag_part, ddof=1, mean=imag_mean))

                std_val = numpy.sqrt(real_var + imag_var)

                return pandas.Series(
                    [mean_val, std_val, len(x)],
                    dtype=object,
                )

            # F_complex: apply stats_func to each Miller index
            df_mean_fwt = df.groupby(["H", "K", "L"], as_index=False).apply(
                lambda d: stats_func(d, "F_complex", do_llweighting=do_llweighting),
                include_groups=False,
            )
            # This converts Series to DataFrame
            # df_mean_fwt = df_mean_fwt.unstack(level=-1)
            df_mean_fwt = df_mean_fwt.set_axis(
                [
                    "H",
                    "K",
                    "L",
                    "F_complex_mean",
                    "SIGFWT",
                    "FWTcount",
                ],
                axis="columns",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
                df_mean_fwt["SIGFWT"] = df_mean_fwt["SIGFWT"].astype(numpy.float32)
                df_mean_fwt["FWTcount"] = df_mean_fwt["FWTcount"].astype(numpy.int32)
            df_mean_fwt = df_mean_fwt.reset_index()

            # DEL_F_complex: apply stats_func to each Miller index
            df_mean_delfwt = df.groupby(["H", "K", "L"], as_index=False).apply(
                lambda d: stats_func(d, "DEL_F_complex", do_llweighting=do_llweighting),
                include_groups=False,
            )
            # This converts Series to DataFrame
            # df_mean_delfwt = df_mean_delfwt.unstack(level=-1)
            df_mean_delfwt = df_mean_delfwt.set_axis(
                [
                    "H",
                    "K",
                    "L",
                    "DEL_F_complex_mean",
                    "SIGDELFWT",
                    "DELFWTcount",
                ],
                axis="columns",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
                df_mean_delfwt["SIGDELFWT"] = df_mean_delfwt["SIGDELFWT"].astype(
                    numpy.float32
                )
                df_mean_delfwt["DELFWTcount"] = df_mean_delfwt["DELFWTcount"].astype(
                    numpy.int32
                )
            df_mean_delfwt = df_mean_delfwt.reset_index()

            # mean FC for 2 Fo-<Fc> and Fo-<Fc> maps
            # FC_complex: apply stats_func to each Miller index
            df_mean_fc = df.groupby(["H", "K", "L"], as_index=False).apply(
                lambda d: stats_func(d, "FC_complex", do_llweighting=do_llweighting),
                include_groups=False,
            )
            df_mean_fc = df_mean_fc.set_axis(
                [
                    "H",
                    "K",
                    "L",
                    "FC_complex",
                    "SIGFC_complex",
                    "FC_complexcount",
                ],
                axis="columns",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
                df_mean_fc["SIGFC_complex"] = df_mean_fc["SIGFC_complex"].astype(
                    numpy.float32
                )
                df_mean_fc["FC_complexcount"] = df_mean_fc["FC_complexcount"].astype(
                    numpy.int32
                )
            df_mean_fc = df_mean_fc.reset_index()

            df_mean_fwt_delfwt = df_mean_fwt.merge(
                df_mean_delfwt, on=["H", "K", "L"], how="outer"
            )
            df_mean_all = df_mean_fwt_delfwt.merge(
                df_mean_fc, on=["H", "K", "L"], how="outer"
            )
            # df_mean_all = df_mean_all.reindex(df_ref.index)

            return df_mean_all

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
        df_mean = calculate_mean_std_count(df_master, do_llweighting=do_llweighting)

        """
        if mtz_ref:  # TODO use Fo data
            mtz_ref_read = gemmi.read_mtz_file(mtz_ref)
            col_labels_ref = mtz_ref_read.column_labels()
            df_ref = pandas.DataFrame(data=mtz_ref_read.array, columns=col_labels_ref)
            df_ref = df_ref.astype({col: "int32" for col in ["H", "K", "L"]})
            df_ref = df_ref[["H", "K", "L", "FP"]]
        else:
            df_ref = pandas.DataFrame()
        """

        # Convert to amplitude and phase
        df_mean["FWT"] = numpy.abs(df_mean["F_complex_mean"])
        df_mean["PHWT"] = numpy.rad2deg(numpy.angle(df_mean["F_complex_mean"]))
        df_mean["DELFWT"] = numpy.abs(df_mean["DEL_F_complex_mean"])
        df_mean["PHDELWT"] = numpy.rad2deg(numpy.angle(df_mean["DEL_F_complex_mean"]))
        df_mean["FC"] = numpy.abs(df_mean["FC_complex"])
        df_mean["PHFC"] = numpy.rad2deg(numpy.angle(df_mean["FC_complex"]))

        if mtz_first and prefix and suffix:
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
                "FC": "F",
                "PHFC": "P",
            }
            """
            if mtz_ref and not df_ref.empty:
                # should drop reflections without FP...
                df_mean["PHTWOFPFC"] = df_mean["PHFC"]
                df_mean["PHFPFC"] = df_mean["PHTWOFPFC"]
                df_mean["TWOFPFC"] = 2 * df_ref["FP"] - df_mean["FC"]
                df_mean["FPFC"] = df_ref["FP"] - df_mean["FC"]
                columns.update(
                    {
                        "TWOFPFC": "F",
                        "PHTWOFPFC": "P",
                        # "SIGTWOFPFC": "Q",
                        # "TWOFPFCcount": "I",
                        "FPFC": "F",
                        "PHFPFC": "P",
                        # "SIGFPFC": "Q",
                    }
                )
            """
            mtz_filename = (
                f"{prefix}group{idx}_bootstrap_mean_map{suffix}.mtz"
                if idx
                else f"{prefix}bootstrap_mean_map{suffix}.mtz"
            )
            write_mtz_from_df(
                df_mean[["H", "K", "L"] + list(columns.keys())],
                mtz_first,
                columns,
                filename=mtz_filename,
            )

        # Calculate statistics per bin
        if binner and mtz_first:
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
                        mtz_first.cell,
                        mtz_first.spacegroup,
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

        if mtz_ref and binner and prefix and suffix:
            mtz_scaled_prefix = (
                f"{prefix}group{idx}_bootstrap_mean_map{suffix}"
                if idx
                else f"{prefix}bootstrap_mean_map{suffix}"
            )
            df_scaled, bin_stats_scaled = scale_reflections(
                mtz_ref,
                df_mean.copy(),
                binner,
                output_mtz2_prefix=mtz_scaled_prefix,
            )
            """if mtz_ref and prefix and suffix and idx:
                # Save the scaled mean maps as an MTZ file
                columns_scaled = {
                    "FWT": "F",
                    "PHWT": "P",
                    "DELFWT": "F",
                    "PHDELWT": "P",
                }
                write_mtz_from_df(
                    df_scaled[["H", "K", "L"] + list(columns_scaled.keys())],
                    mtz_ref,
                    columns_scaled,
                    mtz_scaled_filename,
                )"""

            # Calculate statistics per bin
            # if binner and mtz_ref:
            hkl_array = numpy.array(df_scaled[["H", "K", "L"]].values, numpy.int32)
            hkl_array = numpy.ascontiguousarray(hkl_array, dtype=numpy.int32)
            df_scaled["bin"] = binner.get_bins(hkl_array)
            for b in range(binner.size):
                df_bin = df_scaled[df_scaled["bin"] == b]
                if not df_bin.empty:
                    mean_fwt = df_bin["FWT"].mean()
                    # mean_sigfwt = df_bin["SIGFWT"].mean()
                    # mean_fwt_sigfwt = mean_fwt / mean_sigfwt if mean_sigfwt else 0.0
                    # fwt_count = df_bin["FWTcount"].sum()
                    mean_delfwt = df_bin["DELFWT"].mean()
                    # mean_sigdelfwt = df_bin["SIGDELFWT"].mean()
                    # mean_delfwt_sigdelfwt = (
                    # #     mean_delfwt / mean_sigdelfwt if mean_sigdelfwt else 0.0
                    # )
                    # delfwt_count = df_bin["DELFWTcount"].sum()
                    bin_n_unique = len(df_bin)
                    """bin_n_unique_expected = gemmi.count_reflections(
                        mtz_first.cell,
                        mtz_first.spacegroup,
                        binner.dmin_of_bin(b),
                        binner.dmax_of_bin(b),
                        unique=True,
                    )
                    completeness = bin_n_unique / bin_n_unique_expected"""
                    bin_stats_scaled[b].update(
                        {
                            "bin": b + 1,
                            "dmax": binner.dmax_of_bin(b),
                            "dmin": binner.dmin_of_bin(b),
                            "mean_FWT": mean_fwt,
                            # "mean_SIGFWT": mean_sigfwt,
                            # "mean_FWT_SIGFWT": mean_fwt_sigfwt,
                            # "FWTcount": fwt_count,
                            "mean_DELFWT": mean_delfwt,
                            # "mean_SIGDELFWT": mean_sigdelfwt,
                            # "mean_DELFWT_SIGDELFWT": mean_delfwt_sigdelfwt,
                            # "DELFWTcount": delfwt_count,
                            "count": bin_n_unique,
                            # "completeness": completeness,
                        }
                    )
                else:
                    bin_stats_scaled[b].update(
                        {
                            "bin": b + 1,
                            "dmax": binner.dmax_of_bin(b),
                            "dmin": binner.dmin_of_bin(b),
                            "mean_FWT": 0.0,
                            # "mean_SIGFWT": 0.0,
                            # "mean_FWT_SIGFWT": 0.0,
                            # "FWTcount": 0,
                            "mean_DELFWT": 0.0,
                            # "mean_SIGDELFWT": 0.0,
                            # "mean_DELFWT_SIGDELFWT": 0.0,
                            # "DELFWTcount": 0,
                            "count": 0,
                            # "completeness": 0.0,
                        }
                    )
            stats_scaled_filename = mtz_scaled_prefix + "_scaled_stats.txt"
            write_bin_stats(bin_stats_scaled, stats_scaled_filename)

        return df_scaled

    logging.info(f"\nLoading {len(refined_mtzs)} density maps...")
    columns_selected = ["H", "K", "L", "FWT", "PHWT", "DELFWT", "PHDELWT", "llweight"]
    columns_selected += ["FC", "PHFC"]  # include also FP?
    if binner:
        if mtz_ref:
            logging.info(f"Scaling reflections to {mtz_ref}")
        else:
            mtz_first = gemmi.read_mtz_file(refined_mtzs[0])
            col_labels_first = mtz_first.column_labels()
            df_first = pandas.DataFrame(data=mtz_first.array, columns=col_labels_first)
            df_first = df_first[columns_selected]
            logging.info(f"Scaling reflections to {refined_mtzs[0]}")

    # Process MTZ files in parallel
    def _add_reflections(worker_args):
        """
        Worker function to scale one MTZ file for bootstrap mean map calculation.
        Returns (df, bin_stats) or (None, None) on failure.
        """
        mtz_file, columns_selected, binner, mtz_ref, delta_b = worker_args
        try:
            mtz = gemmi.read_mtz_file(mtz_file)
            col_labels = mtz.column_labels()
            df = pandas.DataFrame(data=mtz.array, columns=col_labels)
            df = df[columns_selected]
            if df.empty:
                columns_selected_str = ", ".join(columns_selected)
                logging.warning(
                    f"No reflections in {mtz_file} for columns {columns_selected_str}"
                )
                return None, None

            bin_stats = None
            if binner and mtz_ref:
                # scale per resolution bin
                mtz_file_base = os.path.splitext(os.path.basename(mtz_file))[0]
                df, bin_stats = scale_reflections(
                    mtz_ref, df, binner, output_mtz2_prefix=mtz_file_base
                )
                if delta_b:
                    # Scale FC values according to delta_b for this MTZ file
                    hkl_array = numpy.array(df[["H", "K", "L"]].values, numpy.int32)
                    # TODO: now suboptimal: calculate s^2 for each MTZ...
                    df["s2"] = binner.cell.calculate_1_d2_array(hkl_array)
                    df = bscale_reflections_fc(df, delta_b)
                    df.drop(columns=["s2"], inplace=True)

            df = df.astype({name: "int32" for name in ["H", "K", "L"]})
            return df, bin_stats

        except Exception as e:
            logging.error(f"Error processing {mtz_file}: {e}")
            import traceback

            traceback.print_exc()
            return None, None

    if mean_mean_b_value:
        if len(mean_b_values) != len(refined_mtzs):
            logging.warning(
                "Length of mean_b_values does not match number of refined MTZs. "
                "Skipping B-value scaling."
            )
            delta_bs = [0.0] * len(refined_mtzs)
        else:
            # Compensate for differences in mean B-values between structures
            delta_bs = [mean_mean_b_value - b for b in mean_b_values]
    else:
        delta_bs = [0.0] * len(refined_mtzs)
    worker_args_list = [
        (mtz_file, columns_selected, binner, mtz_ref, delta_b)
        for mtz_file, delta_b in zip(refined_mtzs, delta_bs)
    ]
    df_list = []
    bin_stats_bootstrap_scale = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc) as executor:
        futures = [
            executor.submit(_add_reflections, worker_args)
            for worker_args in worker_args_list
        ]
        for future in concurrent.futures.as_completed(futures):
            df, bin_stats = future.result()
            if df is not None:
                df_list.append(df)
                if bin_stats is not None:
                    bin_stats_bootstrap_scale.append(bin_stats)
    if df_list:
        df_master = pandas.concat(df_list, ignore_index=True)
    else:
        logging.error("No valid MTZ files processed.")
        return

    if bin_stats_bootstrap_scale:
        try:
            json_filename = (
                f"{prefix}group{idx}_bootstrap_map_scaling_stats.json"
                if idx
                else f"{prefix}bootstrap_map_scaling_stats.json"
            )
            with open(json_filename, "w") as f_json:
                json.dump(
                    bin_stats_bootstrap_scale,
                    f_json,
                    indent=4,
                    default=json_numpy_converter,
                )
            logging.info(f"Saved bootstrap map scaling stats to {json_filename}")
        except Exception as e:
            logging.warning(
                f"Could not write bootstrap map scaling stats JSON file: {e}"
            )

    logging.info("")
    # Convert FWT&PHWT, DELFWT&PHDELWT and FC&PHFC to complex numbers
    df_master["F_complex"] = df_master["FWT"] * numpy.exp(
        1j * numpy.deg2rad(df_master["PHWT"])
    )
    df_master["DEL_F_complex"] = df_master["DELFWT"] * numpy.exp(
        1j * numpy.deg2rad(df_master["PHDELWT"])
    )
    df_master["FC_complex"] = df_master["FC"] * numpy.exp(
        1j * numpy.deg2rad(df_master["PHFC"])
    )

    # print(df_master.head(10))
    # print(df_master[["H", "K", "L", "FWT", "PHWT"]].describe())
    df_master_llweight_0 = df_master[df_master["llweight"] == 0].copy()
    df_master_llweight_pos = df_master[df_master["llweight"] > 0].copy()

    mtz_first = gemmi.read_mtz_file(refined_mtzs[0])
    # save 4 mean maps: all reflections, llweight == 0,
    # llweight > 0 and llweight > 0 weighted average
    merge_reflections_bootstrap(
        df_master, mtz_first, prefix, "_all", idx, binner, mtz_ref
    )
    merge_reflections_bootstrap(
        df_master_llweight_0, mtz_first, prefix, "_llweight0", idx, binner, mtz_ref
    )
    merge_reflections_bootstrap(
        df_master_llweight_pos, mtz_first, prefix, "_llweightpos", idx, binner, mtz_ref
    )
    merge_reflections_bootstrap(
        df_master_llweight_pos,
        mtz_first,
        prefix,
        "_llweightposw",
        idx,
        binner,
        mtz_ref,
        do_llweighting=True,
    )

    return
