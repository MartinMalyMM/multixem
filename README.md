**Multixem** Refinement pipeline for multiple data sets in structure biology
_________________

[![PyPI version](https://badge.fury.io/py/multixem.svg)](http://badge.fury.io/py/multixem)
[![Test Status](https://github.com/MartinMalyMM/multixem/workflows/Test/badge.svg?branch=develop)](https://github.com/MartinMalyMM/multixem/actions?query=workflow%3ATest)
[![Downloads](https://pepy.tech/badge/multixem)](https://pepy.tech/project/multixem)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
_________________

Tests
=====

Automatic tests are implemented using `pytest`. They can be run using the following command:

```bash
python -m pytest -s -v
```

The integration tests use data from 6-phosphogluconate dehydrogenase (PDB [1PGJ](https://www.rcsb.org/structure/1PGJ), [manuscript](https://doi.org/10.1006/jmbi.1998.2059)) and human, porcine, and bovine insulin ([manuscript](https://doi.org/10.1107/S2059798325004589), [raw data (DOI 10.5281/zenodo.13890874)](https://doi.org/10.5281/zenodo.13890874)).
