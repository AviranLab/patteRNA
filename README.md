# patteRNA

Rapid mining of RNA secondary structure motifs from structure profiling data.

## What Is It?

patteRNA is an unsupervised pattern recognition algorithm that rapidly mines RNA structure motifs from structure profiling data. It features a GMM-HMM algorithm that learns automatically from the data, hence bypassing the need for known reference structures. patteRNA is compatible with most current probing technique (e.g. SHAPE, DMS, PARS) and can be used on dataset of any sizes, from small scale experiments to transcriptome-wide assays.



## Getting Started

These instructions will get you a copy of patteRNA up and running on your local machine. Note that patteRNA has only a command-line interface so a terminal is required.

**For Windows users**, use a Linux emulator such as [cygwin](https://www.cygwin.com/). As we will build patteRNA from source, this should work (not tested however).

### Prerequisites

The only prerequisite for patteRNA is Python v3 which does not always come pre-packaged with operating systems. To check if Python3 is installed on your system run the following command:

```
python3 -V
```

This command should print the current installed version of Python3. If it did, fast-forward to [Installation](#installation). If it failed, install Python3 for you OS from the [official Python website](https://www.python.org/downloads/) and retry the previous command to ensure installation was completed.

### Installation

You can install patteRNA directly from the source distribution. First, download the source distribution tarball by entering the command:

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

You should now be able to call patteRNA as an executable from anywhere.

> For more advanced users, the use of a virtual python environment is recommended (see [venv](https://docs.python.org/3/library/venv.html#module-venv) and [pyenv](https://github.com/pyenv/pyenv)). By default, the binary is created in the `bin` folder of the active python distribution. To change this behavior, use the command `python3 setup.py install --install-script=<DEST>` where `<DEST>` is the destination folder. If you do this, don't forget to add `<DEST>` to `$PATH` so the binary is discoverable by the OS.

### Running a test

To make sure patteRNA is properly installed, run the following command:

```
patteRNA --version
```

This should output the current version of patteRNA. You can now do a test run by entering the following command:

```
patteRNA sample_data/weeks_set.shape dummy_test -f sample_data/weeks_set.fa -vl --pattern "((..))"
```

This will run patteRNA in verbose mode (`-v`) and create an output directory `dummy_test` in the current folder.



## Usage

### General usage and available options

```
patteRNA <INPUT> <OUTPUT> <OPTIONS>
```

All available options are listed and described in the help output accessed by running the command:

```
patteRNA -h
```

Recommendations (when applicable) are given in the option caption. Note that flags, i.e. options that do not need arguments, have defaults set to `False`.

### Inputs
patteRNA uses a FASTA-like convention to parse input probing data. See this [example](sample_data/weeks_set.shape).

### Training a model on a new dataset

When using a new dataset, patteRNA needs to learn its model from the data. Training is performed using an iterative EM algorithm that stops when the convergence criterion is met. Run an example training phase using the command:

```
patteRNA sample_data/weeks_set.shape dummy_test -vl
```

> If you ran the test during installation, you will be prompted about overwriting files in the existing directory `dummy_test`. You can safely answer `yes`. Note that in this example we run patteRNA in verbose-mode (`-v`) and we log transform (`-l`) the input data.

This command will generate an output folder `dummy_test` in the current directory which contains:

- A log file: `<date>.log`
- Parameters of the trained model (not meant to be read by humans): `trained_model.pickle`
- A plot of the fitted data: `fit.svg`
- A plot of the model's log-likelihood convergence: `logL.svg`
- A folder `iterative_learning/` that contains plots of the fitted data at each training iteration.

### Scoring structural motifs
patteRNA currently supports structural motifs that contain no gaps as well as G-quadruplexes. Motifs are declared using the `--pattern` option. This option can be used in conjunction with training to perform both training and scoring within a single command. However, we recommend to train patteRNA first and use the trained model in subsequent searches for motifs. The trained model is saved in `trained_model.pickle` and can be loaded using the flag `--model`.

#### Standard patterns
For standard motifs, we use an extended dot-bracket notation where stretches of consecutive repeats are denoted by curly brackets. For instance, an hairpin of stem size 4 and loop size 5 can be declared by `((((.....))))` (full form) or alternatively `({4}.{5}){4}` (short form). Curly brackets can also be used to indicate stretches of varying length using the covention `{<from>,<to>}`. For example, all loops of size 2 to 7 can be declared as `.{2,7}`. Constraining the motif to only regions where the RNA sequences would allow it to fold is done by using the flag `-s`. To apply sequence constraints, RNA sequences must be provided in a FASTA file inputted using the option `-f <FASTA.fa>`. See [example commands](#examples).

#### G-quadruplexes
G-quadruplexes are declared using the notation `GQ[min, max quartet, min, max loop]`, where loop indicates the space between G-columns. For example `GQ[2,3,3,7]` will request a search for all G-quadruplexes with quartets of size 2 to 3 and where G-columns are separated by loops of size 3 to 7. Because putative G-quadruplexes are identified using RNA sequences, these need to be provided in a FASTA file inputted using the option `-f <FASTA.fa>`. See [example commands](#examples).

#### Output
Scored motifs are available in the file `score.txt` in the output directory. This file contains the following columns:

- Transcript name
- Motif's start position (uses a 0-based encoding)
- Motif's end position (ends not included)
- Motif's score
- Motif's numerical state-sequence (i.e. path) encoded as 0/1 for unpaired/paired bases, respectively.
- RNA sequence at the motif's location

### Additional outputs
#### Viterbi path
patteRNA can return the most likely sequence of pairing states across an entire transcript, called the Viterbi path, using the `--viterbi` flag. This will create a FASTA-like file called `viterbi.txt` in the output directory, with numerical pairing states encoded as 0/1 for unpaired/paired bases, respectively.

#### Posterior probabilities
The posterior probabilities of pairing states at each nucleotides can be requested using the flag `--gammas`. This will output a FASTA-like file called `gammas.txt` where the first and second lines (after the header) correspond to unpaired and paired probabilities, respectively.

### Examples <a name="examples"></a>
Note that in the examples provided below we use the same probing data file for both training and scoring. However, one can train the model and score motifs using two different files (e.g. to use a defined set of transcripts for training).

* Train the model and search for all loops of length 5:
    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vl --pattern ".{5}"
    ```

* Search for all loops of length 5 using a trained model:
    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vl --model dummy_test/trained_model.pickle --pattern ".{5}"
    ```

* Search for hairpins of variable stem size 4 to 6 and loop size 5:

    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vls --model dummy_test/trained_model.pickle -f sample_data/weeks_set.fa --pattern "({4,6}.{5}){4,6}"
    ```

    > Note that sequence constraints "-s" was used in this example to ensure we scored hairpins specifically.

* Search for G-quadruplex with quartet size 2 to 3 and separating loops of length 3 to 7:

    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vls --model dummy_test/trained_model.pickle -f sample_data/weeks_set.fa --pattern "GQ[2,3,3,7]"
    ```

    > Note that sequence constraints are automatically applied regardless of the `-s` flag when searching for G-quadruplex.

* Request the Viterbi path and the Posterior state probabilities using a trained model:

    ```
    patteRNA sample_data/weeks_set.shape dummy_test -vl --model dummy_test/trained_model.pickle --viterbi --gammas
    ```

### Using a config file

Because we strongly believe in automation and replicable processes, all options can be passed to patteRNA via the flag `--config` and a configuration file written in YAML. The config file contains more options compared to CLI options. Most notably, all initial values of models' parameters can be controlled via the config file. Note that options passed to patteRNA via the config file have priority over CLI options. An example config file with all options currently available is provided [here](sample_data/config.yaml).



## Citation

TBA



## Reporting bugs and requesting features

To report a bug open a ticket in the [issues tracker](https://github.com/AviranLab/patteRNA/issues). Features can be requested by opening a ticket in the [pull request](https://github.com/AviranLab/patteRNA/pulls).



<!-- ## Versioning

We use the [SemVer](http://semver.org/) convention for versioning. For the versions available, see the [tags on this repository](TBA/tags). -->



## Contributors

* [**Mirko Ledda**](https://mirkoledda.github.io/) - *Initial implementation*
* [**Sharon Aviran**](https://bme.ucdavis.edu/aviranlab/) - *Supervisor*


## License

patteRNA is licensed under the BSD-2 License - see the [LICENSE](LICENSE.txt) file for details.
