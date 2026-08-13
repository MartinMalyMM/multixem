**Multixem** Refinement pipeline for multiple data sets in structure biology
_________________

[![PyPI version](https://badge.fury.io/py/multixem.svg)](http://badge.fury.io/py/multixem)
[![Test Status](https://github.com/MartinMalyMM/multixem/workflows/Test/badge.svg?branch=develop)](https://github.com/MartinMalyMM/multixem/actions?query=workflow%3ATest)
[![Downloads](https://pepy.tech/badge/multixem)](https://pepy.tech/project/multixem)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
_________________

*This is still work in progress.*


Installation
============

Install the released package from PyPI with `pip`:

```bash
pip install multixem
```

It is recommended to use a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install multixem
```

You will also need [Servalcat](https://github.com/kyamashita/servalcat) installed, typically from a recent `CCP4` installation which also includes the [Monomer Library](https://github.com/MonomerLibrary/monomers).


Bootstrap
=========

Subcommand `bootstrap`: Perform the bootstrap protocol - run multiple refinements in parallel against resampled sub data sets.

Example:

```bash
multixem bootstrap 10000 \
	--hklin 1PGJ_data.mtz \
	--model 1PGJ_model.pdb \
    --hklin_free 1PGJ_data.mtz \
	--prefix 1PGJ_bootstrap10000 \
	--servalcat_args "--ncycle 10" \
    --servalcat_confing "config.yaml" \
	--n_bins 30 \
	--n_proc 16 \
    --geometry_cids 1PGJ_geometry_obj.txt
```

The file given in `1PGJ_geometry_obj.txt` defines parameters/features of the structure model under investigation. Each lines specifies an object using atomic CIDs divided by spaces. One CID in row denotes an occupancy, two an interactomic distance, three an angle between the atoms and four a torsion angle. For instance:

```
//A/505/O2
//A/505/O2 //A/262/CG
//A/505/O2 //A/262/CG //A/262/CD
```

Regarding the distances and angles, the procedure is suitable for investigation non-covalently linked atoms. The environment is locally unrestrained (Van der Waals anti-bumping restraints switched off) to provide unbiased results.

All currently available options are listed using:

```bash
multixem bootstrap --help
```

Comparison of isomorphous data sets
===================================

Subcommand `pipeline`: Compare the given diffraction data sets, refine given structure model(s) against them and compare them including calculation of isomorphous difference density maps (|Fobs,n|e^iɸn – k|Fobs,1|e^iɸ1). The input data set should be cut at the same resolution.

Example: comparison of bovine, pork and human insulin:

```bash
multixem pipeline \
	--hklin insuling_cow.mtz insuling_pig.mtz insuling_people.mtz \
	--hklin_free insuling_people.mtz \
	-p insulin \
	--model insuling_cow.pdb insuling_pig.pdb insuling_people.pdb \
	--n_bins 30 \
	--n_proc 4 \
	--unify_cell
```

Example: Merging and comparison of batches of unmerged diffraction data:

```bash
multixem pipeline \
	--hklin_unmerged insuling_people_unmerged.mtz \
	--hklin_free insuling_people.mtz \
	-p insulin_people_600 \
	--model insuling_people.pdb \
	--n_bins 30 \
	--n_proc 4 \
    --n_batches 600
```

All currently available options are listed using:


```bash
multixem pipeline --help
```

Tests
=====

Automatic tests are implemented using `pytest`. They can be run using the following command:

```bash
python -m pytest -s -v
```

The integration tests use data from 6-phosphogluconate dehydrogenase (PDB [1PGJ](https://www.rcsb.org/structure/1PGJ), [manuscript](https://doi.org/10.1006/jmbi.1998.2059)) and human, porcine, and bovine insulin ([manuscript](https://doi.org/10.1107/S2059798325004589), [raw data (DOI 10.5281/zenodo.13890874)](https://doi.org/10.5281/zenodo.13890874)).