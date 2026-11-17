import logging
import subprocess
import sys
import numpy
import pytest
from pathlib import Path

from multixem import run
from multixem.run import read_small_molecule_cif

pytestmark = pytest.mark.skipif(
    not run.servalcat_imported, reason="servalcat is not installed"
)
import gemmi  # noqa: E402 - servalcat must be imported before gemmi

SMCIF = Path(__file__).parent / "data" / "smcif" / "2022519.cif"
WAVELENGTH = 0.71073  # _diffrn_radiation_wavelength in 2022519.cif


@pytest.fixture(scope="module")
def smcif_path():
    if not SMCIF.exists():
        pytest.skip(f"{SMCIF} not found. Run: git lfs pull")
    return SMCIF


def test_read_small_molecule_cif(smcif_path):
    mtz = read_small_molecule_cif(str(smcif_path))
    assert mtz is not None
    assert mtz.nreflections > 0
    assert "IMEAN" in mtz.column_labels() or "I(+)" in mtz.column_labels()
    # datasets[0] is HKL_base, the data are in datasets[1]
    assert len(mtz.datasets) > 1
    assert mtz.datasets[1].wavelength == pytest.approx(WAVELENGTH)


def test_wavelength_written_to_mtz_file(smcif_path, tmp_path):
    mtz = read_small_molecule_cif(str(smcif_path))
    mtz_out = tmp_path / "2022519.mtz"
    mtz.write_to_file(str(mtz_out))
    mtz_read = gemmi.read_mtz_file(str(mtz_out))
    assert mtz_read.datasets[1].wavelength == pytest.approx(WAVELENGTH)


def test_same_as_servalcat_sm2mm(smcif_path, tmp_path):
    # the same interpreter, not whichever servalcat is first in PATH
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "servalcat",
            "util",
            "sm2mm",
            str(smcif_path),
            "-o",
            "sm2mm",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ref = gemmi.read_mtz_file(str(tmp_path / "sm2mm.mtz"))

    mtz = read_small_molecule_cif(str(smcif_path))
    mtz_out = tmp_path / "multixem.mtz"
    mtz.write_to_file(str(mtz_out))
    mtz_read = gemmi.read_mtz_file(str(mtz_out))

    assert mtz_read.column_labels() == ref.column_labels()
    assert [d.wavelength for d in mtz_read.datasets] == pytest.approx(
        [d.wavelength for d in ref.datasets]
    )
    assert numpy.array_equal(
        numpy.array(mtz_read, copy=False),
        numpy.array(ref, copy=False),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "content",
    [
        "data_x\n_cell_length_a 10\n",  # no coordinates, no reflections
        "data_x\n_cell_length_a 10\nloop_\n_foo\n",  # broken CIF
        "garbage\n",  # not a CIF
    ],
)
def test_invalid_file_returns_none(tmp_path, caplog, content):
    cif = tmp_path / "invalid.cif"
    cif.write_text(content)
    with caplog.at_level(logging.ERROR):
        assert read_small_molecule_cif(str(cif)) is None
    assert str(cif) in caplog.text


def test_missing_file_returns_none(tmp_path, caplog):
    cif = tmp_path / "missing.cif"
    with caplog.at_level(logging.ERROR):
        assert read_small_molecule_cif(str(cif)) is None
    assert str(cif) in caplog.text


def test_servalcat_exception_is_caught(monkeypatch, caplog):
    def raise_error(files):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(
        run.servalcat_utils.fileio, "read_small_molecule_files", raise_error
    )
    with caplog.at_level(logging.ERROR):
        assert read_small_molecule_cif("any.cif") is None
    assert "simulated failure" in caplog.text
