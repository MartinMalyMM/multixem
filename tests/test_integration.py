import os
import subprocess
import shutil
import sys
import pytest
from pathlib import Path
from multixem.tools import filename_replace_char
from .conftest import adjust_bootstrap_cmd, TEST_DATA_DIR


def check_output_files(out_dir: Path, expected_files: list[str]) -> None:
    """Fail if any of expected_files is missing or empty in out_dir."""
    assert out_dir.is_dir(), f"Output directory {out_dir} not found"
    missing = [f for f in expected_files if not (out_dir / f).is_file()]
    assert not missing, f"Missing output files in {out_dir}: {missing}"
    empty = [f for f in expected_files if (out_dir / f).stat().st_size == 0]
    assert not empty, f"Empty output files in {out_dir}: {empty}"


def run_multixem_and_cleanup(
    cmd: list[str],
    work: Path,
    expected_files: list[str] = [],
    out_dir_name: str = "",
) -> None:
    """Run multixem with cmd list, fail with stderr/stdout,
    optionally check that expected_files exist in work/out_dir_name,
    and remove work dir on success."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # use multixem and servalcat from the environment running the tests,
    # not whichever comes first in PATH
    bin_dir = str(Path(sys.executable).parent)
    env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ.get("PATH", ""))
    result = subprocess.run(
        ["multixem", *cmd],
        cwd=work,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert (
        result.returncode == 0
    ), f"Command failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    if expected_files:
        out_dir = work / out_dir_name if out_dir_name else work
        check_output_files(out_dir, expected_files)
    # keep the work dir for debugging if anything above failed
    shutil.rmtree(work)


def geometry_cids_labels(geometry_cids_file: str) -> list[str]:
    """Labels used in histogram file names for --geometry_cids:
    2 atoms = distance, 3 = angle, 4 = torsion angle."""
    kinds = {2: "distance", 3: "angle", 4: "torsion angle"}
    labels = []
    with open(geometry_cids_file) as f:
        for line in f:
            atoms = line.split()
            if len(atoms) in kinds:
                labels.append(
                    filename_replace_char(f"{kinds[len(atoms)]} {' '.join(atoms)}")
                )
    return labels


def expected_files_bootstrap(
    prefix,
    hklin_name,
    n_bootstrap,
    smcif=False,
    intensities=False,
    restrained=True,
    geometry_cids_file: str = "",
) -> list[str]:
    """Files expected in multixem_{prefix}/ after a bootstrap run.

    prefix: -p/--prefix; hklin_name: base name of the --hklin file
    smcif: --hklin and --model is a small molecule CIF, i.e. converted
        to MTZ and reference values for the mean structure are available
    intensities: the data are intensities (CCI) rather than amplitudes (CCF)
    restrained: refinement without --unre (Rmsd/RmsZ statistics)
    geometry_cids_file: --geometry_cids
    """
    boot = f"{prefix}_bootstrap"
    files = [
        "multixem.log",
        "coot_FWT_DELFWT.py",
        f"{prefix}_adp_histogram.png",
        f"{prefix}_refine.log",
        f"{prefix}_refine.mmcif",
        f"{prefix}_refine.mtz",
        f"{prefix}_refine.pdb",
        f"{prefix}_refine_stats.json",
        f"{prefix}_refine_sigmaa.log",
        f"{prefix}_refine_sigmaa.mtz",
        f"{boot}_all_models.mmcif.gz",
        f"{boot}_statistics.json",
        f"{boot}_mean_stats.csv",
        f"{boot}_mean_structure.mmcif",
        f"{boot}_mean_structure.pdb",
        f"{boot}_map_scaling_stats.json",
        f"{boot}_mean_map_sigmaa.log",
        f"{boot}_mean_map_sigmaa.mtz",
    ]
    # bootstrap data sets and their refinements
    for i in range(n_bootstrap):
        files.append(f"{hklin_name}_llweight{i}.mtz")
        files.extend(
            f"{prefix}_llweight{i}_refine{suffix}"
            for suffix in (
                ".log",
                ".mmcif",
                ".mtz",
                ".pdb",
                "_stats.json",
                "_scaled.mtz",
                "_scaled_bin_stats.txt",
            )
        )
    # mean maps
    for mean_map in ("all", "llweight0", "llweightpos", "llweightposw"):
        files.extend(
            f"{boot}_mean_map_{mean_map}{suffix}"
            for suffix in (".mtz", ".txt", "_scaled.mtz", "_scaled_bin_stats.txt")
        )
    # histograms of refinement statistics: csv + values csv + scatter plot
    cc = "CCI" if intensities else "CCF"
    stats = ["R", "R_llw_equals_0", "R_llw_gt_0", "-LL", "FOM"]
    stats += [f"{cc}avg", f"{cc}_llw_equals_0_avg", f"{cc}_llw_gt_0_avg"]
    if restrained:
        stats += [
            "Rmsd_Bond_angles__non_H",
            "Rmsd_Bond_distances__non_H",
            "RmsZ_Bond_angles__non_H",
            "RmsZ_Bond_distances__non_H",
        ]
    for stat in stats:
        files.extend(
            [
                f"{prefix}_histogram_{stat}.csv",
                f"{prefix}_histogram_{stat}_values.csv",
                f"{prefix}_scatter_histogram_{stat}.png",
            ]
        )
    # histograms of B-values and of the --geometry_cids objects: csv + png
    labels = ["Average_B-value"]
    if geometry_cids_file:
        files.extend(["servalcat_config.yaml", f"{prefix}_mean_geometry_stats.txt"])
        labels += geometry_cids_labels(geometry_cids_file)
    for label in labels:
        files.extend(
            [
                f"{prefix}_histogram_{label}.csv",
                f"{prefix}_histogram_{label}_values.csv",
                f"{prefix}_histogram_{label}.png",
            ]
        )
    if smcif:
        # CIF converted to MTZ
        files.extend([f"{hklin_name}.mtz", f"{hklin_name}_filtered.mtz"])
        # comparison with the reference values from the CIF
        for plot in ("angles", "bonds", "torsions", "xyzb", "xyzb_per_element"):
            files.append(f"{boot}_mean_stats_plot_{plot}.png")
        for stat in ("angles", "bonds", "torsions"):
            files.append(f"{prefix}_mean_{stat}_stats.csv")
    return files


@pytest.mark.slow
def test_bootstrap_1pgj_integration(
    bootstrap_cli_1pgj_args: list[str],
    real_1pgj_paths: dict[str, Path],
):
    """Integration test for bootstrap command."""

    # Apply overrides for integration test
    cmd = adjust_bootstrap_cmd(
        bootstrap_cli_1pgj_args,
        overrides={"--prefix": "1PGJ_test_slow_run", "--servalcat_args": "--ncycle 1"},
    )

    work = Path.cwd() / "tmp_multixem_bootstrap_run"

    prefix = "1PGJ_test_slow_run"
    expected_files = expected_files_bootstrap(
        prefix,
        hklin_name=real_1pgj_paths["hklin"].stem,
        n_bootstrap=int(cmd[1]),
        geometry_cids_file=str(real_1pgj_paths["geom"]),
    )
    run_multixem_and_cleanup(cmd, work, expected_files, f"multixem_{prefix}")


@pytest.mark.slow
def test_pipeline_insulin_integration(real_insulin_paths):
    """Integration test for pipeline command."""

    cmd = [
        "pipeline",
        "--hklin_unmerged",
        str(real_insulin_paths["cow_unmerged"]),
        str(real_insulin_paths["pig_unmerged"]),
        str(real_insulin_paths["people_unmerged"]),
        "--hklin_free",
        str(real_insulin_paths["free"]),
        "-p",
        "insulin_unmerged2",
        "--model",
        str(real_insulin_paths["model_cow"]),
        str(real_insulin_paths["model_pig"]),
        str(real_insulin_paths["model_people"]),
        "--n_bins",
        "30",
        "--n_proc",
        "8",
        # "--servalcat_args", "",
        "--unify_cell",
        "--merge_whole_file",
        "--quick",
    ]

    work = Path.cwd() / "tmp_multixem_pipeline_run"

    run_multixem_and_cleanup(cmd, work)


SMCIF_COD = TEST_DATA_DIR / "smcif" / "2022519.cif"
N_BOOTSTRAP_SMCIF = 3


@pytest.mark.slow
def test_bootstrap_smcif_integration(tmp_path):
    """Bootstrap with a small molecule CIF (COD 2022519) as both
    the reflection data and the model; check all expected output files."""
    if not SMCIF_COD.exists():
        pytest.skip(f"{SMCIF_COD} not found. Run: git lfs pull")

    cmd = [
        "bootstrap",
        str(N_BOOTSTRAP_SMCIF),
        "-p",
        "COD",
        "--hklin",
        str(SMCIF_COD),
        "--model",
        str(SMCIF_COD),
        "--n_bins",
        "5",
        "--servalcat_args",
        "--unre --hydrogen yes --refine_h --adp aniso --no_solvent "
        "--adpr_weight 0 --ncycle 3",
        "--n_proc",
        "1",
        "--random_weights",
        "--fraction_zero",
        "0.04",
        "--dirichlet_alpha",
        "0.95",
    ]
    expected_files = expected_files_bootstrap(
        "COD",
        hklin_name=SMCIF_COD.stem,
        n_bootstrap=N_BOOTSTRAP_SMCIF,
        smcif=True,
        intensities=True,
        restrained=False,  # --unre
    )
    run_multixem_and_cleanup(cmd, tmp_path / "work", expected_files, "multixem_COD")
