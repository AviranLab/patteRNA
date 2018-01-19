# Changelog
All notable changes to patteRNA are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased


## [1.1.0] - 2018-01-18
### Added
- The number of Gaussian components (`-k`) can be determined automatically using Aikaike Information Criteria (AICs).
- Automatic detection of the experimental assay based on the input observation filename extension.
- Motifs are sorted by scores in `scores.txt`.
- Dependencies: `scikit-learn` and `scipy` (needed by `scikit-learn`).
- Motif dot-brackets are declared using the option `--motif` and by default sequence constraints are applied.
- Pairing state sequences are declared using the option `--path` and used either alone or as a mask to `--motif`.

### Changed
- Structure profiling observations were removed from the output score file `scores.txt` to minimize file size.
- Versioned the latest tested patteRNA distributions to `latest`.
- Refactored `file_handler.py` to handle all FASTA-like files within a single function.
- Option `--gammas` is now named `--posteriors`.
- Option `--nogmm` is now devs only.
- Options `--pattern` and `-s/--seq` are now legacy and will be deprecated in the future.

### Removed
- Removed parameter for final state probabilities `rho` (obsolete).
- Removed `-wmin` as it is now obsolete.
- Removed `--PARS` CLI flag as file extensions are used to determine the assay (obsolete).
- Removed `--debug` CLI flag (obsolete).

### Fixed
- Fixed a bug where the last entry of input files would not be read.
- Minor bugfixes.
- Minor runtime optimizations.

## [1.0.0] - 2017-12-12
### Added
- Initial release.
