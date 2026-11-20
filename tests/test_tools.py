import re

import pytest

from multixem.tools import filename_replace_char


class TestFilenameReplaceChar:
    """filename_replace_char() makes file names from statistics labels.

    The expected names are those of the output files of multixem
    (see expected_files_bootstrap() in test_integration.py).
    """

    @pytest.mark.parametrize(
        "label, expected",
        [
            # refinement statistics
            ("R", "R"),
            ("R_llw=0", "R_llw_equals_0"),
            ("R_llw>0", "R_llw_gt_0"),
            ("CCF_llw=0_avg", "CCF_llw_equals_0_avg"),
            ("CCI_llw>0_avg", "CCI_llw_gt_0_avg"),
            ("-LL", "-LL"),
            ("Rmsd Bond distances, non H", "Rmsd_Bond_distances__non_H"),
            ("RmsZ Bond angles, non H", "RmsZ_Bond_angles__non_H"),
            ("Average B-value", "Average_B-value"),
            # --geometry_cids objects
            (
                "distance //A/505/O1 //B/453/NH1",
                "distance___A_505_O1___B_453_NH1",
            ),
            (
                "angle //A/192/OE2 //A/542/O //A/289/NH2",
                "angle___A_192_OE2___A_542_O___A_289_NH2",
            ),
            (
                "torsion angle //A/505/S //A/505/O4 //A/542/O //A/289/NH2",
                "torsion_angle___A_505_S___A_505_O4___A_542_O___A_289_NH2",
            ),
        ],
    )
    def test_labels_of_output_files(self, label, expected):
        assert filename_replace_char(label) == expected

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("histogram_R_llw=0.csv", "histogram_R_llw_equals_0.csv"),
            ("histogram_-LL_values.csv", "histogram_-LL_values.csv"),
            ("scatter_histogram_R_llw>0.png", "scatter_histogram_R_llw_gt_0.png"),
        ],
    )
    def test_extension_kept(self, filename, expected):
        """The dot of the file extension is kept."""
        assert filename_replace_char(filename) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("a=b", "a_equals_b"),
            ("a>b", "a_gt_b"),
            ("a<b", "a_lt_b"),
            ("a<=b", "a_lt__equals_b"),
        ],
    )
    def test_comparison_operators(self, text, expected):
        assert filename_replace_char(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("a b", "a_b"),
            ("a/b\\c", "a_b_c"),
            ("a:b@c", "a_b_c"),
            ("(a),[b]", "_a___b_"),
            ("B in Å²", "B_in___"),  # non-ASCII characters
            ("", ""),
        ],
    )
    def test_other_characters_replaced(self, text, expected):
        assert filename_replace_char(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "R_llw=0",
            "torsion angle //A/505/S //A/505/O4 //A/542/O //A/289/NH2",
            "a<=b > c (d) ~!#$%^&*'\"|?;,{}`Å",
        ],
    )
    def test_result_is_safe_and_stable(self, text):
        """Only letters, digits, "_", "-" and "." remain;
        applying it again changes nothing."""
        result = filename_replace_char(text)
        assert re.fullmatch(r"[A-Za-z0-9_\-.]*", result)
        assert filename_replace_char(result) == result
