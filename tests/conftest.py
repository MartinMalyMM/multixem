import os
import pytest
from pathlib import Path

# Base locations for real-data integration tests
TEST_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_1PGJ_DIR = TEST_DATA_DIR / "1PGJ"
DEFAULT_INSULIN_DIR = TEST_DATA_DIR / "insulin"

# Allow environment overrides for CI/other setups (guard against None)
_env_1pgj = os.environ.get("MULTIXEM_1PGJ_DIR")
if _env_1pgj:
    DEFAULT_1PGJ_DIR = Path(_env_1pgj)

_env_insulin = os.environ.get("MULTIXEM_INSULIN_DIR")
if _env_insulin:
    DEFAULT_INSULIN_DIR = Path(_env_insulin)


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory."""
    if TEST_DATA_DIR.exists():
        return TEST_DATA_DIR
    pytest.skip("tests/data not found. Run: git lfs pull")
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def real_1pgj_paths():
    """Paths to 1PGJ test data files."""
    base = DEFAULT_1PGJ_DIR
    return {
        "base": base,
        "hklin": base / "1PGJ-sfR.mtz",
        "model": base / "1PGJ.pdb",
        "free": base / "1PGJ-sfR.mtz",
        "geom": base / "1PGJ_geometry_obj_test.txt",
    }


@pytest.fixture(scope="session")
def real_insulin_paths():
    """Paths to insulin test data files."""
    base = DEFAULT_INSULIN_DIR
    return {
        "base": base,
        "cow_unmerged": base / "insulin_cow_unmerged_reindex.mtz",
        "pig_unmerged": base / "insulin_pig_unmerged_reindex.mtz",
        "people_unmerged": base / "insulin_people_unmerged_reindex.mtz",
        "free": base / "insulin_cow_reindex.mtz",
        "model_cow": base / "insulin_cow_mm04_serval.pdb",
        "model_pig": base / "insulin_pig_mm03_serval.pdb",
        "model_people": base / "insulin_people_mm02_serval.pdb",
    }


@pytest.fixture
def bootstrap_cli_1pgj_args(real_1pgj_paths):
    """Reusable default CLI args for the bootstrap command using 1PGJ data set."""
    return [
        "bootstrap",
        "8",
        "--hklin",
        str(real_1pgj_paths["hklin"]),
        "--model",
        str(real_1pgj_paths["model"]),
        "--n_bins",
        "15",
        "--n_proc",
        "8",
        "--prefix",
        "1PGJ_bootstrap",
        "--servalcat_args",
        "--ncycle 2 --hydrogen yes",
        "--geometry_cids",
        str(real_1pgj_paths["geom"]),
        "--source",
        "xray",
        "--hklin_free",
        str(real_1pgj_paths["free"]),
        # "--unre", "2",
        # "--quick",
    ]


def adjust_bootstrap_cmd(cmd, overrides=None):
    """Apply overrides to command arguments."""
    cmd = cmd[:]
    if overrides:
        for key, val in overrides.items():
            for i, t in enumerate(cmd):
                if t == key:
                    cmd[i + 1] = str(val)
    return cmd
