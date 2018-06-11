# patteRNA

Rapid mining of RNA secondary structure motifs from structure profiling data.


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

You can install patteRNA directly from source. First, download the distribution tarball by entering the command:

```
wget https://raw.github.com/AviranLab/patteRNA/master/dist/patteRNA-latest.tar.gz
```

Once downloaded, extract the tarball and move into the extracted folder:

```
tar -xzf patteRNA-latest.tar.gz
cd patteRNA-latest
```

Run the installation by entering the command:

```
sudo python3 setup.py install
```

You should now be able to call patteRNA as an executable from anywhere. If it failed, make sure `pip` and `setuptools` are up-to-date, see [Prerequisites](#prerequisites)

> For more advanced users, the use of a virtual python environment is recommended (see [venv](https://docs.python.org/3/library/venv.html#module-venv) and [pyenv](https://github.com/pyenv/pyenv)). By default, the binary is created in the `bin` folder of the active python distribution. To change this behavior, use the command `python3 setup.py install --install-script=<DEST>` where `<DEST>` is the destination folder. If you do this, don't forget to add `<DEST>` to your `$PATH` variable so the binary is discoverable by the OS.

### Running a test

To make sure patteRNA is properly installed, run the following command:

```
patteRNA --version
```

This should output the current version of patteRNA. You can now do a test by entering the following command:

```
patteRNA sample_data/weeks_set.shape dummy_test -f sample_data/weeks_set.fa -vl --motif "((..))"
```

This will run patteRNA in verbose mode (`-v`) and create an output directory `dummy_test` in the current folder.



## Usage

### General usage and available options

```
patteRNA <probing> <output> <OPTIONS>
```

All available options are accessible via `patteRNA -h` as listed below. Recommendations (when applicable) are given in the option caption. Note that switches, i.e. boolean options that do not need arguments, have defaults set to `False`.

```
mandatory arguments:
  probing              FASTA-like file of probing data. The type of assay is
                       automatically detected based on the filename extension.
                       Extensions currently supported are [.shape, .pars,
                       .dms].
  output               Output directory

optional arguments:
  -h, --help           show this help message and exit
  --version            show program's version number and exit
  -f , --fasta         FASTA file of RNA sequences (default: None)
  --reference          FASTA-like file of reference RNA secondary structures
                       in dot-bracket notation. (default: None)
  -v, --verbose        Print progress (default: False)
  -l, --log            Log transform input data (default: False)
  --no-cscores         Do not compute c-scores during scoring (default: False)
  --config             Config parameters in YAML format. Has priority over CLI
                       options (default: None)
  -k                   Number of Gaussian components per pairing state in the
                       GMM model. By default, K is determined automatically
                       using Bayesian Information Criteria. If K <= 0,
                       automatic detection is enabled. Increasing K manually
                       will make the model fit the data tighter but could
                       result in overfitting. Fitted data should always be
                       visually inspected after training to gauge if the model
                       is adequate (default: -1)
  -d , --min-density   Transcripts with data density below this threshold will
                       be rejected from analysis. Valid range is 0 to 1. For
                       example, 0.5 means that only transcripts containing
                       less than 50% missing values will be used. (default: 0)
  --KL-div             Minimum Kullback–Leibler divergence criterion for
                       building the training set. The KL divergence measures
                       the difference in information content between the full
                       dataset and the training set. The smaller the value,
                       the more representative the training set will be with
                       respect to the full dataset. However, this will produce
                       a larger training set and increase both runtime and RAM
                       consumption during training. (default: 0.01)
  -e , --epsilon       Convergence criterion (default: 0.0001)
  -i , --maxiter       Maximum number of training iterations (default: 100)
  -nt , --n-tasks      Number of parallel processes. By default all available
                       CPUs are used (default: -1)
  --model              Trained .pickle GMMHMM model (default: None)
  --motif              Score target motif declared using an extended dot-
                       bracket notation. Paired and unpaired bases are denoted
                       using parentheses '()' and dots '.', respectively. A
                       stretch of consecutive characters is declared using the
                       format <char>{<from>, <to>}. Can be used in conjunction
                       with --mask to modify the expected underlying sequence
                       of pairing states. (default: None)
  --path               Expected sequence of numerical pairing states for the
                       motif with 0=unpaired and 1=paired nucleotides. A
                       stretch of consecutive states is declared using the
                       format <state>{<from>, <to>}. Can be used in
                       conjunction with --motif to apply sequence constraints.
                       (default: None)
  --forbid-N-pairs     Pairs involving a N are considered invalid. Must be
                       used in conjunction with --motif to take
                       effect (default: False)
  --posteriors         Output the posterior probabilities of pairing states
                       (i.e. the probability Trellis) (default: False)
  --viterbi            Output the most likely sequence of pairing states for
                       entire transcripts (i.e. Viterbi paths) (default:
                       False)
  --NAN                If NaN are considered informative in term of pairing
                       state, use this flag. However, note that this can lead
                       to unstable results and is therefore not recommended
                       (default: False)
  --no-prompt          Do not prompt a question if existing output files could
                       be overwritten. Useful for automation using scripts or
                       for running patteRNA on computing servers (default:
                       False)
```

### Inputs
patteRNA uses a FASTA-like convention for probing data (see this [example file](sample_data/weeks_set.shape)). As patteRNA learns from data, non-normalized data can be used directly. Also, patteRNA fully supports negatives and zero values, even when applying a log-transformation to the data (via the `-l` flag). We recommend to **not** artificially set negative values to 0. Also, non available values must be set to `NA` or `nan` and **not to `-999`**.

The type of experimental assay is automatically detected using the `<probing>` filename extension. Currently supported extensions are listed [here](docs/supported_extensions.md).

### Training a model on a new dataset

By default, patteRNA will learn its model from the data. Run an example training phase using the command:

```
patteRNA sample_data/weeks_set.shape dummy_test -vl
```

> If you ran the test during installation, you will be prompted about overwriting files in the existing directory `dummy_test`. Answer `yes`. Note that in this example we run patteRNA in verbose-mode (`-v`) and we log transform (`-l`) the input data.

This command will generate an output folder `dummy_test` in the current directory which contains:

- A log file: `<date>.log`
- Parameters of the trained model (not meant to be read by humans): `trained_model.pickle`
- A plot of the fitted data: `fit.png`
- A plot of the model's log-likelihood convergence: `logL.png`

### Scoring motifs
patteRNA currently supports structural motifs (via [`--motif`](#motifs) or [`--path`](#paths)) that contain no gaps. These options can be used in conjunction with training to perform both training and scoring using a single command. However, we recommend to train patteRNA first and use the trained model in subsequent searches for motifs. The trained model is saved in `trained_model.pickle` and can be loaded using the flag `--model`.

#### Motifs <a name="motifs"></a>
Standard motifs (flag `--motif`) can be declared using an extended dot-bracket notation where stretches of consecutive repeats are denoted by curly brackets. For instance, an hairpin of stem size 4 and loop size 5 can be declared by `((((.....))))` (full form) or alternatively `({4}.{5}){4}` (short form). Curly brackets can also be used to indicate stretches of varying length using the convention `{<from>,<to>}`. For example, all loops of size 2 to 7 can be declared as `.{2,7}`. By default, RNA sequences are used to ensure a scored region sequence is compatible with the folding of the motif. RNA sequences must be provided in a FASTA file inputted using the option `-f <FASTA.fa>`. See [example commands](#examples).

#### Paths <a name="paths"></a>
As an alternative to `--motif`, or in conjunction with it, an expected sequence of numerical pairing state can be used via the flag `--path`, with 0 and 1 representing unpaired and paired nucleotides, respectively. Similar to [`--motif`](#motifs), short form notation can be used for stretches of identical pairing states, e.g. `1111000001111` can be denoted `1{4}0{5}1{4}`. Curly brackets can also be used to indicate stretches of varying length using the convention `{<from>,<to>}`. If `--path` is used in conjunction to `--motif`, then it is used as a mask on top of the declared motif. This can be useful when it is known that specific nucleotides do not behave under the expected model, for example a single nucleotide being paired based on the dot-bracket notation but that generates SP data more consistent with an unpaired nucleotide.

#### Output
Scored motifs are available in the file `scores.txt` in the output directory. This file contains the following columns:

- Transcript name
- Motif start position (uses a 0-based encoding)
- Motif end position (ends not included)
- Motif score
- Motif c-score
- Motif in dot-bracket notation
- Motif in numerical state-sequence (i.e. path) encoded as 0/1 for unpaired/paired bases, respectively.
- RNA sequence at the motif's location

### Additional outputs
#### Viterbi path
patteRNA can return the most likely sequence of pairing states across an entire transcript, called the Viterbi path, using the `--viterbi` flag. This will create a FASTA-like file called `viterbi.txt` in the output directory, with numerical pairing states encoded as 0/1 for unpaired/paired bases, respectively.

#### Posterior probabilities
The posterior probabilities of pairing states at each nucleotides can be requested using the flag `--posteriors`. This will output a FASTA-like file called `posteriors.txt` where the first and second lines (after the header) correspond to unpaired and paired probabilities, respectively.

### Examples <a name="examples"></a>

* Train the model and search for any loop of length 5:

    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vl --motif ".{5}" -f sample_data/weeks_set.fa
    ```

* Search for all loops of length 5 using a trained model:

    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vl --model dummy_test/trained_model.pickle --motif ".{5}" -f sample_data/weeks_set.fa
    ```

* Search for all enclosed loops (i.e. neighbored by paired nucleotides, but constrained to be paired together) of length 5 using a trained model:

    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vl --model dummy_test/trained_model.pickle --path "10{5}1"
    ```

    > Note that the considered path in this case is `1000001` when written in its full format.

* Search for hairpins of variable stem size 4 to 6 and loop size 5:

    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vl --model dummy_test/trained_model.pickle -f sample_data/weeks_set.fa --motif "({4,6}.{5}){4,6}"
    ```

* Request the Viterbi path and the Posterior state probabilities using a trained model:

    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vl --model dummy_test/trained_model.pickle --viterbi --posteriors
    ```

> Note that in the examples provided above we use the same probing data file for both training and scoring. However, one can train the model and score motifs using two different files (e.g. to use a defined set of transcripts for training).

### Using a config file

Because we strongly believe in automation and replicable processes, all options can be passed to patteRNA via the flag `--config` and a configuration file written in YAML. The config file contains more options compared to CLI options. Most notably, all initial values of models' parameters can be controlled via the config file. Note that options passed to patteRNA via the config file have priority over CLI options. An example config file with all options currently available is provided [here](sample_data/config.yaml).



## Citation

Ledda M. and Aviran S. (2018) “PATTERNA: Transcriptome-Wide Search for Functional RNA Elements via Structural Data Signatures.” *Genome Biology* 19(28). https://doi.org/10.1186/s13059-018-1399-z.



## Reporting bugs and requesting features

patteRNA is actively supported and all changes are listed in the [CHANGELOG](CHANGES.md). To report a bug open a ticket in the [issues tracker](https://github.com/AviranLab/patteRNA/issues). Features can be requested by opening a ticket in the [pull request](https://github.com/AviranLab/patteRNA/pulls).



<!-- ## Versioning

We use the [SemVer](http://semver.org/) convention for versioning. For the versions available, see the [tags on this repository](TBA/tags). -->



## Contributors

* [**Mirko Ledda**](https://mirkoledda.github.io/) - *Initial implementation and developer*
* [**Pierce Radecki**](https://github.com/peradecki) - *Developer*
* [**Sharon Aviran**](https://bme.ucdavis.edu/aviranlab/) - *Supervisor*


## License

patteRNA is licensed under the BSD-2 License - see the [LICENSE](LICENSE.txt) file for details.
