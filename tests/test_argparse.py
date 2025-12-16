import os
import sys
import pytest
from pathlib import Path
import argparse
from multixem.run import create_parser


# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestParserCreation:
    """Test parser creation and basic functionality."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "multixem"

    def test_version_flag(self):
        """Test that --version flag works."""
        parser = create_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])

        assert exc_info.value.code == 0

    def test_subcommands_exist(self):
        """Test that all expected subcommands are registered."""
        parser = create_parser()

        # Get subparsers action from the parser
        subparsers_actions = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]

        assert len(subparsers_actions) == 1

        # Check that expected subcommands are registered
        subparser_action = subparsers_actions[0]
        subcommands = list(subparser_action.choices.keys())

        assert "pipeline" in subcommands
        assert "bootstrap" in subcommands
        assert "mean" in subcommands
        assert len(subcommands) == 3


class TestBootstrapCommand:
    """Test bootstrap subcommand."""

    def test_bootstrap_parser_args(self, real_1pgj_paths):
        """Test bootstrap subcommand argument parsing with real files."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "bootstrap",
                "8",
                "--hklin",
                str(real_1pgj_paths["hklin"]),
                "--model",
                str(real_1pgj_paths["model"]),
            ]
        )

        assert args.command == "bootstrap"
        assert args.n_samples == 8
        assert args.hklin == [str(real_1pgj_paths["hklin"])]
        assert args.model == [str(real_1pgj_paths["model"])]

    def test_bootstrap_required_args(self):
        """Test that bootstrap requires essential arguments."""
        parser = create_parser()

        # Missing --hklin
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "bootstrap",
                    "20",
                    "--model",
                    "test.pdb",
                ]
            )

        # Missing --model
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "bootstrap",
                    "20",
                    "--hklin",
                    "test.mtz",
                ]
            )

    def test_bootstrap_parser_args_nonexisting_files(self):
        """Test bootstrap subcommand rejects non-existing files."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "bootstrap",
                    "8",
                    "--hklin",
                    "nonexistent.mtz",
                    "--model",
                    "nonexistent.pdb",
                    "--geometry_cids",
                    "nonexistent.txt",
                ]
            )

    def test_bootstrap_with_all_options(self, bootstrap_cli_1pgj_args):
        """Test bootstrap with all options using fixture."""
        parser = create_parser()
        args = parser.parse_args(bootstrap_cli_1pgj_args)

        # Verify structure and values (paths come from fixture automatically)
        assert args.command == "bootstrap"
        assert args.n_samples == 8
        assert len(args.hklin) == 1
        assert args.n_bins == 15
        assert args.n_proc == 8
        assert args.prefix == "1PGJ_bootstrap"
        assert args.servalcat_args == "--ncycle 2 --hydrogen yes"
        assert args.hklin_free is not None
        assert args.geometry_cids is not None
        assert args.source == "xray"

    def test_bootstrap_with_model_dir(self, tmp_path):
        """Test bootstrap with --model_dir."""
        parser = create_parser()

        # Create a temporary model directory with real files
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "model1.pdb").write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           C\n"
        )
        (model_dir / "model2.pdb").write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           C\n"
        )
        (model_dir / "model3.pdb").write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           C\n"
        )

        # Create dummy MTZ file
        hklin = tmp_path / "test.mtz"
        hklin.touch()

        args = parser.parse_args(
            [
                "bootstrap",
                "3",
                "--hklin",
                str(hklin),
                "--model_dir",
                str(model_dir),
                "--n_bins",
                "10",
            ]
        )

        assert args.model_dir == str(model_dir)

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "bootstrap",
                    "8",
                    "--hklin",
                    str(hklin),
                    "--model_dir",
                    str(model_dir),
                    "--n_bins",
                    "10",
                ]
            )


class TestPipelineCommand:
    """Test pipeline subcommand."""

    @pytest.fixture
    def pipeline_files(self, tmp_path):
        """Provide paths to pipeline test files."""
        # Create dummy files for testing
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        hklin_unmerged = [
            str(data_dir / "cow_unmerged.mtz"),
            str(data_dir / "pig_unmerged.mtz"),
            str(data_dir / "people_unmerged.mtz"),
        ]
        models = [
            str(data_dir / "cow_model.pdb"),
            str(data_dir / "pig_model.pdb"),
            str(data_dir / "people_model.pdb"),
        ]
        hklin_free = str(data_dir / "free.mtz")

        # Create dummy files
        for f in hklin_unmerged + models + [hklin_free]:
            Path(f).touch()

        return {
            "hklin_unmerged": hklin_unmerged,
            "models": models,
            "hklin_free": hklin_free,
        }

    def test_pipeline_parser_args(self, pipeline_files):
        """Test pipeline subcommand argument parsing."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "pipeline",
                "--hklin_unmerged",
                pipeline_files["hklin_unmerged"][0],
                pipeline_files["hklin_unmerged"][1],
                "--model",
                pipeline_files["models"][0],
                pipeline_files["models"][1],
            ]
        )

        assert args.command == "pipeline"
        assert len(args.hklin_unmerged) == 2
        assert len(args.model) == 2

    def test_pipeline_with_unmerged_data(self, pipeline_files):
        """Test pipeline with unmerged diffraction data."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "pipeline",
                "--hklin_unmerged",
                *pipeline_files["hklin_unmerged"],
                "--model",
                *pipeline_files["models"],
                "--merge_whole_file",
                "--hklin_free",
                pipeline_files["hklin_free"],
            ]
        )

        assert len(args.hklin_unmerged) == 3
        assert len(args.model) == 3
        assert args.merge_whole_file is True
        assert args.hklin_free == pipeline_files["hklin_free"]

    def test_pipeline_with_n_batches(self, pipeline_files):
        """Test pipeline with batch splitting."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "pipeline",
                "--hklin_unmerged",
                pipeline_files["hklin_unmerged"][0],
                "--model",
                pipeline_files["models"][0],
                "--n_batches",
                "60",
                "120",
                "180",
            ]
        )

        assert args.n_batches == [60, 120, 180]

    def test_pipeline_with_all_options(self, pipeline_files):
        """Test pipeline with all optional arguments."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "pipeline",
                "--hklin_unmerged",
                pipeline_files["hklin_unmerged"][0],
                pipeline_files["hklin_unmerged"][1],
                "--model",
                pipeline_files["models"][0],
                pipeline_files["models"][1],
                "--hklin_free",
                pipeline_files["hklin_free"],
                "--n_bins",
                "30",
                "--n_proc",
                "8",
                "--prefix",
                "insulin_",
                "--servalcat_args",
                "--labin IMEAN,SIGIMEAN",
                "--source",
                "xray",
                "--unify_cell",
                "--merge_whole_file",
                "--molrep",
                "--bootstrap",
                "20",
                "--amplitude",
                "--quick",
            ]
        )

        assert len(args.hklin_unmerged) == 2
        assert len(args.model) == 2
        assert args.hklin_free == pipeline_files["hklin_free"]
        assert args.n_bins == 30
        assert args.n_proc == 8
        assert args.prefix == "insulin_"
        assert args.unify_cell is True
        assert args.merge_whole_file is True
        assert args.molrep is True
        assert args.bootstrap == 20
        assert args.amplitude is True
        assert args.quick is True

    def test_pipeline_bootstrap_option(self, pipeline_files):
        """Test pipeline with bootstrap resampling."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "pipeline",
                "--hklin_unmerged",
                pipeline_files["hklin_unmerged"][0],
                "--model",
                pipeline_files["models"][0],
                "--bootstrap",
                "50",
                "--merge_whole_file",
            ]
        )

        assert args.bootstrap == 50

    def test_pipeline_multiple_datasets(self, pipeline_files):
        """Test pipeline with multiple datasets."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "pipeline",
                "--hklin_unmerged",
                *pipeline_files["hklin_unmerged"],
                "--model",
                *pipeline_files["models"],
                "--hklin_free",
                pipeline_files["hklin_free"],
                "--n_batches",
                "60",
            ]
        )

        assert len(args.hklin_unmerged) == 3
        assert len(args.model) == 3


class TestCommonOptions:
    """Test common options across subcommands."""

    @pytest.fixture
    def dummy_files(self, tmp_path):
        """Create dummy files for validation tests."""
        test_mtz = tmp_path / "test.mtz"
        test_pdb = tmp_path / "test.pdb"
        test_mtz.touch()
        test_pdb.touch()
        return {"mtz": str(test_mtz), "pdb": str(test_pdb)}

    @pytest.mark.parametrize("command", ["bootstrap", "pipeline"])
    def test_n_proc_validation(self, command, dummy_files):
        """Test that --n_proc must be positive."""
        parser = create_parser()

        # Valid n_proc
        if command == "bootstrap":
            args = parser.parse_args(
                [
                    command,
                    "5",
                    "--hklin",
                    dummy_files["mtz"],
                    "--model",
                    dummy_files["pdb"],
                    "--n_proc",
                    "4",
                ]
            )
        else:
            args = parser.parse_args(
                [
                    command,
                    "--hklin_unmerged",
                    dummy_files["mtz"],
                    "--model",
                    dummy_files["pdb"],
                    "--n_proc",
                    "4",
                ]
            )
        assert args.n_proc == 4

        # Invalid n_proc
        with pytest.raises(SystemExit):
            if command == "bootstrap":
                parser.parse_args(
                    [
                        command,
                        "5",
                        "--hklin",
                        dummy_files["mtz"],
                        "--model",
                        dummy_files["pdb"],
                        "--n_proc",
                        "0",
                    ]
                )
            else:
                parser.parse_args(
                    [
                        command,
                        "--hklin_unmerged",
                        dummy_files["mtz"],
                        "--model",
                        dummy_files["pdb"],
                        "--n_proc",
                        "-1",
                    ]
                )

    @pytest.mark.parametrize("command", ["bootstrap", "pipeline"])
    def test_n_bins_validation(self, command, dummy_files):
        """Test that --n_bins must be positive."""
        parser = create_parser()

        # Valid n_bins
        if command == "bootstrap":
            args = parser.parse_args(
                [
                    command,
                    "5",
                    "--hklin",
                    dummy_files["mtz"],
                    "--model",
                    dummy_files["pdb"],
                    "--n_bins",
                    "20",
                ]
            )
        else:
            args = parser.parse_args(
                [
                    command,
                    "--hklin_unmerged",
                    dummy_files["mtz"],
                    "--model",
                    dummy_files["pdb"],
                    "--n_bins",
                    "20",
                ]
            )
        assert args.n_bins == 20

        # Invalid n_bins
        with pytest.raises(SystemExit):
            if command == "bootstrap":
                parser.parse_args(
                    [
                        command,
                        "5",
                        "--hklin",
                        dummy_files["mtz"],
                        "--model",
                        dummy_files["pdb"],
                        "--n_bins",
                        "0",
                    ]
                )
            else:
                parser.parse_args(
                    [
                        command,
                        "--hklin_unmerged",
                        dummy_files["mtz"],
                        "--model",
                        dummy_files["pdb"],
                        "--n_bins",
                        "-5",
                    ]
                )

    def test_prefix_handling(self, dummy_files):
        """Test that prefix is handled correctly."""
        parser = create_parser()

        # With trailing underscore
        args = parser.parse_args(
            [
                "bootstrap",
                "5",
                "--hklin",
                dummy_files["mtz"],
                "--model",
                dummy_files["pdb"],
                "--prefix",
                "my_prefix_",
            ]
        )
        assert args.prefix == "my_prefix_"

        # Without trailing underscore
        args2 = parser.parse_args(
            [
                "bootstrap",
                "5",
                "--hklin",
                dummy_files["mtz"],
                "--model",
                dummy_files["pdb"],
                "--prefix",
                "my_prefix",
            ]
        )
        assert args2.prefix == "my_prefix"

    def test_servalcat_args_parsing(self, dummy_files):
        """Test that Servalcat args are parsed correctly."""
        parser = create_parser()

        servalcat_str = "--labin IMEAN,SIGIMEAN --weight 1.5"
        args = parser.parse_args(
            [
                "bootstrap",
                "5",
                "--hklin",
                dummy_files["mtz"],
                "--model",
                dummy_files["pdb"],
                "--servalcat_args",
                servalcat_str,
            ]
        )

        assert args.servalcat_args == servalcat_str


class TestMeanCommand:
    """Test mean subcommand."""

    def test_mean_parser_args(self):
        """Test mean subcommand argument parsing."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "mean",
                "dataset_name",
            ]
        )

        assert args.command == "mean"
        assert args.file_name_template == "dataset_name"

    def test_mean_with_cif(self, tmp_path):
        """Test mean with optional CIF file."""
        parser = create_parser()

        cif_file = tmp_path / "small_molecule.cif"
        cif_file.touch()

        args = parser.parse_args(
            [
                "mean",
                "dataset_name",
                "--cif",
                str(cif_file),
            ]
        )

        assert args.cif == str(cif_file)

    def test_mean_with_options(self):
        """Test mean with common options."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "mean",
                "my_dataset",
                "--prefix",
                "mean_",
                "--n_bins",
                "15",
                "--n_proc",
                "4",
            ]
        )

        assert args.file_name_template == "my_dataset"
        assert args.prefix == "mean_"
        assert args.n_bins == 15
        assert args.n_proc == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
