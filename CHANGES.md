# Changelog
All notable changes to patteRNA are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased


## [1.1.0] - 2018-01-17
### Added
- The number of Gaussian components can be determined automatically using Aikaike Information Criteria (AICs).
- Automatic detection of the experimental assay based on the input observation filename extension.
- Dependencies: `scikit-learn`
- Dependencies: `scipy` (needed by `scikit-learn`).
- Return motifs sorted by scores in `score.txt`.
- Motif dot-brackets are now input using the option `--motif` and by default sequence constraints are applied.
- Pairing state sequences can now be input using the option `--path` and used either alone or as a mask to `--motif`.

### Changed
- Structure profiling observations were removed from the output score file `score.txt` to minimize file size.
- Versioned the latest tested patteRNA distributions to `latest`.
- Refactored `file_handler.py` to handle all fasta-like files within a single function.
- Refactored global variables to be configured and stored within a single `GLOBALS` dictionary.
- Options `--pattern` and `-s/--seq` are now legacy and will be deprecated in the future.
- Option `--nogmm` is now meant to devs only.
- Minor refactoring.

### Removed
- Removed parameter for final state probabilities `rho` (obsolete).
- Removed `--PARS` CLI flag as file extensions are used to determine the assay (obsolete).
- Removed `--debug` CLI flag (obsolete).
- Backup copies of the original observation data in `RNA.obj` (obsolete as observations are not anymore written to the output).

### Fixed
- Minor runtime optimizations.
- Fixed a bug where the last entry of input files would not be read.

## [1.0.0] - 2017-12-12
### Added
- Initial release.
