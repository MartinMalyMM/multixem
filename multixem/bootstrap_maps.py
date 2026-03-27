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


def bootstrap_mean_map(
    refined_mtzs, idx=0, prefix="", binner=None, mtz_ref="", n_proc=4
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

    Returns:
        None: Writes the mean maps in
            '{prefix}group{idx}_bootstrap_mean_map_all.mtz'
            '{prefix}group{idx}_bootstrap_mean_map_llweight0.mtz'
            '{prefix}group{idx}_bootstrap_mean_map_llweightpos.mtz'
            '{prefix}group{idx}_bootstrap_mean_map_llweightposw.mtz'
    """

    def merge_reflections_bootstrap(
        df_master,
        mtz_first=None,
        prefix="",
        suffix="",
        idx=0,
        binner=None,
        mtz_ref="",
        do_llweighting=False,
    ):
        """
        Merge reflections from the master DataFrame and calculate mean maps.

        Args:
            df_master (pandas.DataFrame): DataFrame containing reflections.
                It must contain columns "H", "K", "L", "F_complex", "DEL_F_complex",
            mtz_first (gemmi.Mtz): Reference MTZ object for cell and spacegroup.
            prefix (str): Prefix for the output filename.
            suffix (str): Suffix for the output filename.
            idx (int): Index for naming the output file.

        Returns:
            pandas.DataFrame: DataFrame with mean maps.
        """

        """# noqa: E741
        def is_centric_vectorized(h, k, l):  # noqa: E741
            return mtz_first.spacegroup.operations().is_reflection_centric(
                (int(h), int(k), int(l))  # noqa: E741
            )"""

        def calculate_mean_std_count(df, do_llweighting=False):
            """Calculate mean and standard deviation and number of structure factors."""

            def stats_func(miller_index_df, column_name, do_llweighting=do_llweighting):
                """
                Compute weighted or unweighted mean, std, and count
                for one Miller index.

                Args:
                    miller_index_df (pandas.DataFrame): DataFrame for
                        a specific Miller index.
                    column_name (str): Column name to compute stats on.
                    do_llweighting (bool): Whether to apply llweighting.

                Returns:
                    pandas.Series: Series containing mean, std, and count.
                """
                x = miller_index_df[column_name].values

                if len(x) <= 1:
                    return pandas.Series([numpy.mean(x), 0.0, len(x)])

                if do_llweighting and "llweight" in miller_index_df.columns:
                    # Weighted mean and variance
                    w = miller_index_df["llweight"].values
                    w = w / numpy.sum(w)  # normalize weights
                    mean_val = numpy.sum(w * x)

                    real_mean = numpy.real(mean_val)
                    imag_mean = numpy.imag(mean_val)
                    real_part = numpy.real(x)
                    imag_part = numpy.imag(x)
                    real_var = numpy.sum(w * (real_part - real_mean) ** 2)
                    imag_var = numpy.sum(w * (imag_part - imag_mean) ** 2)
                else:
                    # Unweighted mean and variance
                    mean_val = numpy.mean(x)
                    real_mean = numpy.real(mean_val)
                    imag_mean = numpy.imag(mean_val)
                    real_part = numpy.real(x)
                    imag_part = numpy.imag(x)
                    real_var = numpy.var(real_part, ddof=1, mean=real_mean)
                    imag_var = numpy.var(imag_part, ddof=1, mean=imag_mean)

                std_val = numpy.sqrt(real_var + imag_var)
                return pandas.Series([mean_val, std_val, len(x)])

            # F_complex: apply stats_func to each Miller index
            df_mean_fwt = df.groupby(["H", "K", "L"], as_index=False).apply(
                lambda d: stats_func(d, "F_complex", do_llweighting=do_llweighting),
                include_groups=False,
            )
            # This converts Series to DataFrame
            # df_mean_fwt = df_mean_fwt.unstack(level=-1)
            df_mean_fwt.columns = [
                "H",
                "K",
                "L",
                "F_complex_mean",
                "SIGFWT",
                "FWTcount",
            ]
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
            df_mean_delfwt.columns = [
                "H",
                "K",
                "L",
                "DEL_F_complex_mean",
                "SIGDELFWT",
                "DELFWTcount",
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
                df_mean_delfwt["SIGDELFWT"] = df_mean_delfwt["SIGDELFWT"].astype(
                    numpy.float32
                )
                df_mean_delfwt["DELFWTcount"] = df_mean_delfwt["DELFWTcount"].astype(
                    numpy.int32
                )
            df_mean_delfwt = df_mean_delfwt.reset_index()

            df_mean_fwt_delfwt = df_mean_fwt.merge(
                df_mean_delfwt, on=["H", "K", "L"], how="outer"
            )
            return df_mean_fwt_delfwt

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

        # Convert to amplitude and phase
        df_mean["FWT"] = numpy.abs(df_mean["F_complex_mean"])
        df_mean["PHWT"] = numpy.rad2deg(numpy.angle(df_mean["F_complex_mean"]))
        df_mean["DELFWT"] = numpy.abs(df_mean["DEL_F_complex_mean"])
        df_mean["PHDELWT"] = numpy.rad2deg(numpy.angle(df_mean["DEL_F_complex_mean"]))

        if mtz_first and prefix and suffix and idx:
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

    logging.info(f"Loading {len(refined_mtzs)} density maps...")
    columns_selected = ["H", "K", "L", "FWT", "PHWT", "DELFWT", "PHDELWT", "llweight"]
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
        mtz_file, columns_selected, binner, mtz_ref = worker_args
        try:
            mtz = gemmi.read_mtz_file(mtz_file)
            col_labels = mtz.column_labels()
            df = pandas.DataFrame(data=mtz.array, columns=col_labels)
            df = df[columns_selected]
            if df.empty:
                logging.warning(
                    f"No reflections in {mtz_file} for FWT/PHWT/DELFWT/PHDELWT."
                )
                return None, None

            bin_stats = None
            if binner and mtz_ref:
                # scale per resolution bin
                mtz_file_base = os.path.splitext(os.path.basename(mtz_file))[0]
                df, bin_stats = scale_reflections(
                    mtz_ref, df, binner, output_mtz2_prefix=mtz_file_base
                )

            df = df.astype({name: "int32" for name in ["H", "K", "L"]})
            return df, bin_stats

        except Exception as e:
            logging.error(f"Error processing {mtz_file}: {e}")
            import traceback

            traceback.print_exc()
            return None, None

    worker_args_list = [
        (mtz_file, columns_selected, binner, mtz_ref) for mtz_file in refined_mtzs
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
