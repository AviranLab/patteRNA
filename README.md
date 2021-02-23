# We are excited to share an initial implementation of the next major patteRNA release, patteRNA 2.0!

This version includes new capabilities in addition to several improvements, including:

- A discretized observation model (DOM) of reactivity, which is more robust to different data distributions and almost always provides a strong improvement to the precision and speed of motif mining when compared to a GMM
- A `--hairpins` flag to search for a representative set of hairpins automatically, without having to manually specify an extended dot-bracket notation
- An `--HDSL` flag that computes a "hairpin-driven structure level" profile for RNA transcripts, which provides a quantification of local structure based on local pairing probabilities and the presence of highly-scored hairpin elements
- An updated implementation for the computation of c-scores which scales much better for searches with a large number of motifs

This new release involved a fundamental re-write of most of the codebase. As a consequence, some legacy features may not work as expected. The following features are still being brought over to the new release:

- The use of reference structures to initialize training via the `--reference` flag
- The use of a configuration file to define detailed run parameters via the `--config` flag

These features will be added in a future release in the coming weeks.

# patteRNA

Rapid mining of RNA secondary structure motifs from structure probing data.


## What Is It?

patteRNA is an unsupervised pattern recognition algorithm that rapidly mines RNA structure motifs from structure profiling data. It features a GMM-HMM algorithm that learns automatically from the data, hence bypassing the need for known reference structures. patteRNA is compatible with most current probing technique (e.g. SHAPE, DMS, PARS) and can be used on dataset of any sizes, from small scale experiments to transcriptome-wide assays.



## Getting Started

These instructions will get you a copy of patteRNA up and running on your local machine. Note that patteRNA has only a command-line interface so a terminal is required.

**For Windows users**, use a Linux emulator such as [cygwin](https://www.cygwin.com/). As we will build patteRNA from source, this should work (not tested however).

### Prerequisites

**Python3**. To get the current installed version on your system, run `python3 -V`. If the command failed or if your version is anterior to v3.5, install the latest version of Python3 for your OS from the [official Python website](https://www.python.org/downloads/).

You also need the latest versions of `pip` and `setuptools`, which can be installed by typing:

```
sudo python3 -m pip install -U pip setuptools
```

### Installation

Installation is done directly from source. For that, clone this repository using the commands:

```
git clone https://github.com/AviranLab/patteRNA.git
cd patteRNA
```

If you do have `sudo` permissions, such as on a personal laptop, then use the command:

```
sudo python3 setup.py install
```

Otherwise, run a local installation using the commands:

```
python3 setup.py install --user
echo 'export PATH="$PATH:~/.local/bin"' >> ~/.bashrc; source ~/.bashrc
```

*Note for macOS Big Sur users:* Due to an issue, you must use `pip` to run the installation. Use the commands:

```
python3 -m pip install .
```
or
```
python3 -m pip install . --user
```

### Running a test

To make sure patteRNA is properly installed, run the following command:

```
patteRNA --version
```

This should output the current version of patteRNA. You can now do a test by entering the following command:

```
patteRNA sample_data/weeks_set.shape sample_output -f sample_data/weeks_set.fa --motif "((((...))))" -v
```

This will run patteRNA in verbose mode (`-v`) and create an output directory `sample_output` in the current folder.



## Usage

### General usage and available options

```
patteRNA <probing> <output> <OPTIONS>
```

All available options are accessible via `patteRNA -h` as listed below. Recommendations (when applicable) are given in the option caption. Note that switches, i.e. boolean options that do not need arguments, have defaults set to `False`.

```
usage: patteRNA [-h] [--version] [-f fasta] [--reference] [-v] [-l] [-k]
                [--KL-div] [-e] [-i] [-nt] [--model] [--config] [--motif]
                [--hairpins] [--posteriors] [--viterbi] [--HDSL] [--nan]
                [--no-prompt] [--GMM] [--no-cscores] [--min-cscores]
                [--batch-size]
                probing output

Rapid mining of RNA secondary structure motifs from profiling data.

positional arguments:
  probing               FASTA-like file of probing data.
  output                Output directory

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -f fasta, --fasta fasta
                        FASTA file of RNA sequences (default: None)
  --reference           FASTA-like file of reference RNA secondary structures
                        in dot-bracket notation. (default: None)
  -v, --verbose         Print progress (default: False)
  -l, --log             Log transform input data (default: False)
  -k                    Number of Gaussian components per pairing state in the
                        GMM model. By default, K is determined automatically
                        using Bayesian Information Criteria. If K <= 0,
                        automatic detection is enabled. Increasing K manually
                        will make the model fit the data tighter but could
                        result in overfitting. Fitted data should always be
                        visually inspected after training to gauge if the
                        model is adequate (default: -1)
  --KL-div              Minimum Kullback–Leibler divergence criterion for
                        building the training set. The KL divergence measures
                        the difference in information content between the full
                        dataset and the training set. The smaller the value,
                        the more representative the training set will be with
                        respect to the full dataset. However, this will
                        produce a larger training set and increase both
                        runtime and RAM consumption during training. (default:
                        0.001)
  -e , --epsilon        Convergence criterion (default: 0.01)
  -i , --maxiter        Maximum number of training iterations (default: 250)
  -nt , --n-tasks       Number of parallel processes. By default all available
                        CPUs are used (default: -1)
  --model               Trained .json model (version 2.0+ models only)
                        (default: None)
  --config              Currently unsupported (default: None)
  --motif               Score target motif declared using an extended dot-
                        bracket notation. Paired and unpaired bases are
                        denoted using parentheses '()' and dots '.',
                        respectively. A stretch of consecutive characters is
                        declared using the format <char>{<from>, <to>}. Can be
                        used in conjunction with --mask to modify the expected
                        underlying sequence of pairing states. (default: None)
  --hairpins            Score a representative set of hairpins (stem lengths 4
                        to 15; loop lengths 3 to 10). Automatically enabled
                        when the --HDSL flag is used. This flag overrides any
                        motif syntaxes provided via --motif. (default: False)
  --posteriors          Output the posterior probabilities of pairing states
                        (i.e. the probability Trellis) (default: False)
  --viterbi             Output the most likely sequence of pairing states for
                        entire transcripts (i.e. Viterbi paths) (default:
                        False)
  --HDSL                Use scores a representative set of hairpins (stem
                        lengths 4 to 15; loop lengths 3 to 10) to quantify
                        structuredness across the input data. This flag
                        overrides any motif syntaxes provided via --motif and
                        also activates --posteriors (default: False)
  --nan                 If NaN are considered informative in term of pairing
                        state, use this flag. However, note that this can lead
                        to unstable results and is therefore not recommended
                        if data quality is low or long runs of NaN exist in
                        the data (default: False)
  --no-prompt           Do not prompt a question if existing output files
                        could be overwritten. Useful for automation using
                        scripts or for running patteRNA on computing servers
                        (default: False)
  --GMM                 Train a Gaussian Mixture Model (GMM) during training
                        instead of a Discretized ObservationModel (DOM)
                        (default: False)
  --no-cscores          Suppress the computation of c-scores during the
                        scoring phase (default: False)
  --min-cscores         Minimum number of scores to sample during construction
                        of null distributions to usefor c-score normalization
                        (default: 1000)
  --batch-size          Number of transcripts to process at once using a pool
                        of parallel workers (default: 100)
```

### Inputs
patteRNA uses a FASTA-like convention for probing data (see this [example file](sample_data/weeks_set.shape)). As patteRNA learns from data, non-normalized data can be used directly. Also, patteRNA fully supports negatives and zero values, even when applying a log-transformation to the data (via the `-l` flag). We recommend to **not** artificially set negative values to 0. Missing data values must be set to `nan`, `NA` or `-999`.

### Training a model on a new dataset

By default, patteRNA will learn its model from the data. Run an example training phase using the command:

```
patteRNA sample_data/weeks_set.shape sample_output -vl
```

> If you ran the test during installation, you will be prompted about overwriting files in the existing directory `test`. Answer `y`/`yes`. Note that in this example we run patteRNA in verbose-mode (`-v`) and we log transform (`-l`) the input data.

This command will generate an output folder `sample_output` in the current directory which contains:

- A log file: `<date>.log`
- Trained model: `trained_model.json`
- A plot of the fitted data: `fit.png`/`fit.svg`
- A plot of the model's log-likelihood convergence: `logL.png`/`logL.svg`

### Scoring motifs
patteRNA supports structural motifs (via [`--motif`](#motifs) or [`--path`](#paths)) that contain no gaps. These options can be used in conjunction with training to perform both training and scoring using a single command. However, we recommend to train patteRNA first and use the trained model in subsequent searches for motifs. The trained model is saved in `trained_model.json` and can be loaded using the flag `--model`.

#### Motifs <a name="motifs"></a>
Standard motifs (flag `--motif`) can be declared using an extended dot-bracket notation where stretches of consecutive repeats are denoted by curly brackets. For instance, an hairpin of stem size 4 and loop size 5 can be declared by `((((.....))))` (full form) or alternatively `({4}.{5}){4}` (short form). Curly brackets can also be used to indicate stretches of varying length using the convention `{<from>,<to>}`. For example, all loops of size 2 to 7 can be declared as `.{2,7}`. By default, RNA sequences are used to ensure a scored region sequence is compatible with the folding of the motif. RNA sequences must be provided in a FASTA file inputted using the option `-f <FASTA.fa>`. See [example commands](#examples).

#### Paths <a name="paths"></a>
As an alternative to `--motif`, an expected sequence of numerical pairing state can be used via the flag `--path`, with 0 and 1 representing unpaired and paired nucleotides, respectively. Similar to [`--motif`](#motifs), short form notation can be used for stretches of identical pairing states, e.g. `1111000001111` can be denoted `1{4}0{5}1{4}`. Curly brackets can also be used to indicate stretches of varying length using the convention `{<from>,<to>}`. If `--path` is used in conjunction to `--motif`, then it is used as a mask on top of the declared motif. This can be useful when it is known that specific nucleotides do not behave under the expected model, for example a single nucleotide being paired based on the dot-bracket notation but that generates SP data more consistent with an unpaired nucleotide.

#### Output
Scored motifs are available in the file `scores.txt` in the output directory. This file contains the following columns:

- Transcript name
- Motif start position (uses a 1-based encoding)
- Motif score
- Motif c-score
- Motif in dot-bracket notation
- RNA sequence at the motif's location
- Path in binary notation (if `--path` used in conjunction with `--motif`)

### Additional outputs
#### Viterbi path
patteRNA can return the most likely sequence of pairing states across an entire transcript, called the Viterbi path, using the `--viterbi` flag. This will create a FASTA-like file called `viterbi.txt` in the output directory, with numerical pairing states encoded as 0/1 for unpaired/paired bases, respectively.

#### Posterior probabilities
The posterior probabilities of pairing states at each nucleotides can be requested using the flag `--posteriors`. This will output a FASTA-like file called `posteriors.txt` where the first and second lines (after the header) correspond to unpaired and paired probabilities, respectively.

#### Hairpin-derived structure level (HDSL)
HDSL is a measure of local structure that assists in converting patteRNA's predicted hairpins into a quantitative assenment of structuredness. This will output a FASTA-like file called `hdsl.txt` with HDSL profiles for all transcripts in the input data.
### Examples <a name="examples"></a>

* Train the model and search for any loop of length 5:

    ```
    patteRNA sample_data/weeks_set.shape test -vl --motif ".{5}" -f sample_data/weeks_set.fa
    ```

* Search for all loops of length 5 using a trained model:

    ```
    patteRNA sample_data/weeks_set.shape test -vl --model test/trained_model.json --motif ".{5}" -f sample_data/weeks_set.fa
    ```

* Search for hairpins of variable stem size 4 to 6 and loop size 5:

    ```
    patteRNA sample_data/weeks_set.shape test -vl --model test/trained_model.json -f sample_data/weeks_set.fa --motif "({4,6}.{5}){4,6}"
    ```

* Request HDSL profiles and the posterior state probabilities using a trained model:

    ```
    patteRNA sample_data/weeks_set.shape test -vl --model test/trained_model.json --viterbi --posteriors
    ```

> Note that in the examples provided above we use the same probing data file for both training and scoring. However, one can train the model and score motifs using two different files (e.g. to use a defined set of transcripts for training).

## Citation

Version 2.0: \
Citation TBA

Version 1.0–1.2: \
Ledda M. and Aviran S. (2018) “PATTERNA: Transcriptome-Wide Search for Functional RNA Elements via Structural Data Signatures.” *Genome Biology* 19(28). https://doi.org/10.1186/s13059-018-1399-z.



## Reporting bugs and requesting features

patteRNA is actively supported and all changes are listed in the [CHANGELOG](CHANGES.md). To report a bug open a ticket in the [issues tracker](https://github.com/AviranLab/patteRNA/issues). Features can be requested by opening a ticket in the [pull request](https://github.com/AviranLab/patteRNA/pulls).



<!-- ## Versioning

We use the [SemVer](http://semver.org/) convention for versioning. For the versions available, see the [tags on this repository](TBA/tags). -->



## Contributors

* [**Pierce Radecki**](https://aviranlab.bme.ucdavis.edu/2018/02/13/about-pierce/) - *Version 2 developer and current maintainer*
* [**Mirko Ledda**](https://mirkoledda.github.io/) - *Initial implementation and developer*
* **Rahul Uppuluri** - *Undergraduate researcher*
* **Kaustubh Deshpande** - *Undergraduate researcher*  
* [**Sharon Aviran**](https://bme.ucdavis.edu/aviranlab/) - *Supervisor*


## License

patteRNA is licensed under the BSD-2 License - see the [LICENSE](LICENSE.txt) file for details.
