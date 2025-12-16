import subprocess
import shutil
import pytest
from pathlib import Path
from .conftest import have_files, adjust_bootstrap_cmd


def run_multixem_and_cleanup(cmd, work: Path):
    """Run multixem with cmd list, fail with stderr/stdout
    and remove work dir on success."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["multixem", *cmd],
        cwd=work,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode == 0:
        shutil.rmtree(work)
    assert (
        result.returncode == 0
    ), f"Command failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.mark.slow
def test_bootstrap_1pgj_integration(bootstrap_cli_1pgj_args):
    """Integration test for bootstrap command."""

    # Apply overrides for integration test
    cmd = adjust_bootstrap_cmd(
        bootstrap_cli_1pgj_args,
        overrides={"--prefix": "1PGJ_test_slow_run", "--servalcat_args": "--ncycle 1"},
    )

    work = Path.cwd() / "tmp_bootstrap_run"

    run_multixem_and_cleanup(cmd, work)


@pytest.mark.slow
def test_pipeline_insulin_integration(real_insulin_paths):
    """Integration test for pipeline command."""
    files = [
        real_insulin_paths["cow_unmerged"],
        real_insulin_paths["pig_unmerged"],
        real_insulin_paths["people_unmerged"],
        real_insulin_paths["free"],
        real_insulin_paths["model_cow"],
        real_insulin_paths["model_pig"],
        real_insulin_paths["model_people"],
    ]
    if not have_files(files):
        pytest.skip("Real input files not available")

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

    work = Path.cwd() / "tmp_pipeline_run"

    run_multixem_and_cleanup(cmd, work)
