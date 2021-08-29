![patteRNA](patterna.png)

# *patteRNA*

Rapid mining of RNA secondary structure motifs from structure profiling data.


## What Is It?

*patteRNA* is an unsupervised pattern recognition algorithm that rapidly mines RNA structure motifs from structure profiling (SP) data. 

It features a discretized observation model, hidden Markov model (DOM-HMM) of reactivity that enables automated and calibrated processing of SP data without a dependence on reference structures. It is compatible with most current probing technique (e.g. SHAPE, DMS, PARS) and can help analyze datasets of any sizes, from small scale experiments to transcriptome-wide assays. It scales well to millions or billions of nucleotides.

The training and scoring implementations are parallelized, so the algorithm can benefit greatly when deployed in a high CPU count environment.



## Getting Started

These instructions will set up patteRNA as a command line tool on your machine.

### Prerequisites

**Python 3.7 or newer**. To check the current installed version on your system, run `python3 -V`. If your version is anterior to 3.7, install version 3.7 or later. We recommend the use of virtual environments with [venv](https://docs.python.org/3/library/venv.html).

You also need the latest versions of `pip` and `setuptools`, which can be installed by typing:

```
python -m pip install -U pip setuptools
```

### Installation

Installation is done directly from source. For that, clone this repository using the commands:

```
git clone https://github.com/AviranLab/patteRNA.git
cd patteRNA
```

To install the Python module such that it can be executed from the command line, run the setup script.

```
python setup.py install
```

You can also specify a [local installation](https://packaging.python.org/tutorials/installing-packages/#installing-to-the-user-site) using the command:

```
python setup.py install --user
```

***Note for macOS Big Sur and Apple M1 users***: Due to an issue, you must use `pip` to run the installation. Be sure to update pip and setuptools before attempting the installation (`python -m pip install -U pip setuptools`). Use the commands:

```
python -m pip install .
```
or
```
python -m pip install . --user
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

All available options are accessible via `patteRNA -h` as listed below. Recommendations (when applicable) are given in the option caption. Note that switches (i.e. boolean options that do not need arguments), have defaults set to `False` and are set to `True` if provided.

```
usage: patteRNA [-h] [--version] [-f fasta] [--reference] [-v] [-l] [--GMM]
                [-k] [--KL-div] [-e] [-i] [-nt] [--model] [--config] [--motif]
                [--hairpins] [--posteriors] [--viterbi] [--HDSL] [--SPP]
                [--nan] [--no-prompt] [--min-cscores] [--no-cscores]
                [--batch-size]
                probing output

Rapid mining of RNA secondary structure motifs from profiling data.

positional arguments:
  probing               FASTA-like file of probing data
  output                Output directory

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -f fasta, --fasta fasta
                        FASTA file of RNA sequences (default: None)
  --reference           FASTA-like file of reference RNA secondary structures
                        in dot-bracket notation (default: None)
  -v, --verbose         Print progress (default: False)
  -l, --log             Log transform input data (default: False)
  --GMM                 Train a Gaussian Mixture Model (GMM) during training
                        instead of a Discretized ObservationModel (DOM)
                        (default: False)
  -k                    Number of Gaussian components per pairing state in the
                        GMM model. By default, K is determined automatically
                        using Bayesian Information Criteria. If K <= 0,
                        automatic detection is enabled. Increasing K manually
                        will make the model fit the data tighter but could
                        result in overfitting. Fitted data should always be
                        visually inspected after training to gauge if the
                        model is adequate. (default: -1)
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
                        CPUs are used. (default: -1)
  --model               Trained .json model (version 2.0+ models only)
                        (default: None)
  --config 
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
                        also activates --posteriors. (default: False)
  --SPP                 Smoothed P(paired). Quantifies structuredness across
                        the input data via local pairing probabilities. This
                        flag activates --posteriors. (default: False)
  --nan                 If NaN are considered informative in term of pairing
                        state, use this flag. However, note that this can lead
                        to unstable results and is therefore not recommended
                        if data quality is low or long runs of NaN exist in
                        the data. (default: False)
  --no-prompt           Do not prompt a question if existing output files
                        could be overwritten. Useful for automation using
                        scripts or for running patteRNA on computing servers.
                        (default: False)
  --min-cscores         Minimum number of scores to sample during construction
                        of null distributions to usefor c-score normalization
                        (default: 1000)
  --no-cscores          Suppress the computation of c-scores during the
                        scoring phase (default: False)
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

### Scoring
patteRNA supports structural motifs (via the `--motif` flag) that contain no gaps. These options can be used in conjunction with training to perform both training and scoring using a single command. However, we recommend to train patteRNA first and use the trained model in subsequent searches for motifs. Trained models are saved as `trained_model.json` and can be loaded using the flag `--model`.

#### Motif Specification
Standard motifs can be declared using an extended dot-bracket notation where stretches of consecutive repeats are denoted by curly brackets. For instance, an hairpin of stem size 4 and loop size 5 can be declared by `((((.....))))` (full form) or alternatively `({4}.{5}){4}` (short form). Curly brackets can also be used to indicate stretches of varying length using the convention `{<from>,<to>}`. For example, all loops of size 2 to 7 can be declared as `.{2,7}`. By default, RNA sequences are used to ensure a scored region sequence is compatible with the folding of the motif. RNA sequences must be provided in a FASTA file inputted using the option `-f <fasta-file>`. See [example commands](#examples).


#### Score Files
The results of a motif search are saved in the file `scores.txt` in the output directory. This file contains putative sites  the following columns:

- Transcript name
- Site start position (uses a 1-based encoding)
- Score
- c-score
- Motif in dot-bracket notation
- Nucleotide sequence

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
    patteRNA sample_data/weeks_set.shape example_outputs/loop -vl --motif ".{5}" -f sample_data/weeks_set.fa
    ```

* Search for all loops of length 5 using a trained model:

    ```
    patteRNA sample_data/weeks_set.shape example_outputs/loop_pretrained -vl --model test/trained_model.json --motif ".{5}" -f sample_data/weeks_set.fa
    ```

* Search for hairpins of variable stem size 4 to 6 and loop size 5:

    ```
    patteRNA sample_data/weeks_set.shape example_outputs/hairpin -vl --model test/trained_model.json -f sample_data/weeks_set.fa --motif "({4,6}.{5}){4,6}"
    ```


* Request HDSL profiles and the posterior state probabilities using a trained model:

    ```
    patteRNA sample_data/weeks_set.shape example_outputs/hdsl -vl --model test/trained_model.json --HDSL
    ```
  
* Train the model using a set of reference transcripts:

    ```
    patteRNA sample_data/weeks_set.shape example_outputs/loop -vl -f sample_data/weeks_set.fa --reference sample_data/weeks_set.dot
    ```
  
## Citation

If you used patteRNA in your research, please reference the following citations depending on which version of patteRNA you utilized.

**Version 2.0**: \
Radecki P., Uppuluri R., and Aviran S. (2021) "Rapid Structure-Function Insights via Hairpin-Centric Analysis of Big RNA Structure Probing Datasets." *NAR Genomics and Bioinformatics* 3(3). doi: [10.1093/nargab/lqab073](https://doi.org/10.1093/nargab/lqab073).

**Version 1.0–1.2**: \
Ledda M. and Aviran S. (2018) “PATTERNA: Transcriptome-Wide Search for Functional RNA Elements via Structural Data Signatures.” *Genome Biology* 19(28). doi: [10.1186/s13059-018-1399-z](https://doi.org/10.1186/s13059-018-1399-z).



## Issue Reporting

patteRNA is actively supported and all changes are listed in the [CHANGELOG](CHANGES.md). To report a bug open a ticket in the [issues tracker](https://github.com/AviranLab/patteRNA/issues). Features can be requested by opening a ticket in the [pull request](https://github.com/AviranLab/patteRNA/pulls).



## Contributors

* [**Pierce Radecki**](https://pierceradecki.com/) - *Version 2 developer and current maintainer*
* [**Mirko Ledda**](https://mirkoledda.github.io/) - *Initial implementation and developer*
* **Rahul Uppuluri** - *Undergraduate contributor*
* **Kaustubh Deshpande** - *Undergraduate contributor*  
* [**Sharon Aviran**](https://bme.ucdavis.edu/aviranlab/) - *Principal Investigator*


## License

patteRNA is licensed under the BSD-2 License - see the [LICENSE](LICENSE.txt) file for details.


## Notes

* You can run patteRNA directly from source without formal installation by running `src.patteRNA` as a module. For example, `python -m src.patteRNA --version`
* If you are working with transcriptome-wide profiling data, consider using ribosomial RNAs as reference transcripts to achieve the highest quality model possible.