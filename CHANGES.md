# Changelog
All notable changes to patteRNA are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [2.0.0] - 2021-02-23
### Changed
- Major version release will almost a full rewrite of the method. [PR]
- Addition of `--HDSL` flag to compute local structure levels. [PR]
- Addition of a new Discretized Observation Model (DOM) emission model scheme, which is more precise for scoring and faster than a GMM. [PR]
- New c-score distribution sampling procedure is much more efficient than before. [PR]
- Now using human-readable `.json` format for saving a loading trained models. [PR]
- Matplotlib backend to `svg`. [PR]
- Dependencies: `humanfriendly`. [PR]

### Removed
- Temporarily removed `--reference` and `--config` options. [PR]

## [1.2.2] - 2020-01-08
### Changed
- Updated sample data with corrected structures. [PR]
- Updated README to reflect current developers. [PR]
- Fixed sample data. [PR]

## [1.2.1] - 2019-04-10
### Changed
- Installation procedure. [ML]
- Matplotlib backend to `Agg`. [ML]

## [1.2.0] - 2018-06-11
- Supervised initialization of Model's parameters based on reference RNA secondary structures in dot-bracket notation supplied via the new `--reference` flag. Note that `--reference` supports RNAstructure's `ct2dot` output format. [ML]
- Simulation framework for testing (devs only). [ML]
- Scoring motifs now returns, by default, a c-score based on a fitted Null distribution in addition to the original score. [PR/ML]
- Flag `--no-cscores` to turn off the computation of c-scores. [ML]
- The training set is now built automatically using KL divergence metrics (via option `--KL-div`). Data-dense transcripts are prioritized. [ML]
- Infinite values in structure profiles are now supported. [ML]
- Added a checkpoint to ensure paired/unpaired states are never flipped in the model. [ML]
- Dependencies: `cairosvg` (needed by `pygal`). [ML]
- Dependencies: `matplotlib`. [ML]

### Changed
- Motifs were all observations are missing now return NaN scores. [ML]
- Progress bar during scoring tracks individual transcripts instead of batches. [PR]
- The behavior of the `--min-density/-d` CLI flag was changed to affect both training and scoring. [ML]
- The default value for the `--min-density/-d` CLI flag was changed to 0 (i.e. all transcripts are used by default). [ML]
- Now renders all plots as PNG instead of SVG. [ML]
- Re-vamped user messages printed during the task. [ML]

### Removed
- Removed `-n` CLI flag (obsolete). [ML]
- Removed `--filter-test` CLI flag (obsolete). [ML]


## [1.1.4] - 2018-02-13
### Fixed
- Bugfix. Sequence constraints contained a bug affecting non fully nested target motifs. [PR]

## [1.1.3] - 2018-02-13
### Fixed
- Bugfix. Output Viterbi and posterior files were not deleted if already existing. [ML]


## [1.1.2] - 2018-02-06
### Changed
- Unsupervised initialization now uses by default an initial transition probability matrix derived from the Weeks set and GMM means based on data percentiles for increased robustness. [ML]

### Fixed
- Bugfix over v1.1.1 which was removed [ML]

## [1.1.0] - 2018-01-18
### Added
- The number of Gaussian components (`-k`) can be determined automatically using Aikaike Information Criteria (AICs). [ML]
- Automatic detection of the experimental assay based on the input observation filename extension. [ML]
- Motifs are sorted by scores in `scores.txt`. [ML]
- Dependencies: `scikit-learn` and `scipy` (needed by `scikit-learn`). [ML]
- Motif dot-brackets are declared using the option `--motif` and by default sequence constraints are applied. [ML]
- Pairing state sequences are declared using the option `--path` and used either alone or as a mask to `--motif`. [ML]

### Changed
- Structure profiling observations were removed from the output score file `scores.txt` to minimize file size. [ML]
- Versioned the latest tested patteRNA distributions to `latest`. [ML]
- Refactored `file_handler.py` to handle all FASTA-like files within a single function. [ML]
- Option `--gammas` is now named `--posteriors`. [ML]
- Option `--nogmm` is now devs only. [ML]
- Options `--pattern` and `-s/--seq` are now legacy and will be deprecated in the future. [ML]

### Removed
- Removed parameter for final state probabilities `rho` (obsolete). [ML]
- Removed `-wmin` as it is now obsolete. [ML]
- Removed `--PARS` CLI flag as file extensions are used to determine the assay (obsolete). [ML]
- Removed `--debug` CLI flag (obsolete). [ML]

### Fixed
- Fixed a bug where the last entry of input files would not be read. [ML]
- Minor bugfixes. [ML]
- Minor runtime optimizations. [ML]

## [1.0.0] - 2017-12-12
### Added
- Initial release. [ML]
