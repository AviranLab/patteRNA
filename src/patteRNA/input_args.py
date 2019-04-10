"""Parse and handle input arguments"""

import argparse
import os
import yaml
import numpy as np

from . import globalbaz
from . import _version


def parse_cl_args(inputargs):
    # Define the args parser
    parser = argparse.ArgumentParser(prog="patteRNA",
                                     description="Rapid mining of RNA secondary structure motifs from profiling data.",
                                     epilog="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version",
                        action="version",
                        version="%(prog)s {}".format(_version.__version__))
    parser.add_argument("input",
                        metavar="probing",
                        type=str,
                        help="FASTA-like file of probing data. The type of assay is automatically detected based on "
                             "the filename extension. Extensions currently supported are [.shape, .pars, .dms]. "
                             "See Github page at https://github.com/AviranLab/patteRNA/docs/supported_extensions.md"
                             " for more information.")
    parser.add_argument("output",
                        metavar="output",
                        type=str,
                        help="Output directory")
    parser.add_argument("-f", "--fasta",
                        metavar="",
                        default=None,
                        type=str,
                        help="FASTA file of RNA sequences")
    parser.add_argument("--reference",
                        metavar="",
                        default=None,
                        type=str,
                        help="FASTA-like file of reference RNA secondary structures in dot-bracket notation.")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Print progress")
    parser.add_argument("-l", "--log",
                        action="store_true",
                        help="Log transform input data")
    parser.add_argument("--no-cscores",
                        action="store_true",
                        help="Do not compute c-scores during scoring")
    parser.add_argument("--config",
                        metavar="",
                        type=str,
                        default=None,
                        help="Config parameters in YAML format. Has priority over CLI options")
    parser.add_argument("-k",
                        metavar="",
                        type=int,
                        default=-1,
                        help="Number of Gaussian components per pairing state in the GMM model. By default, K is "
                             "determined automatically using Bayesian Information Criteria. If K <= 0, automatic "
                             "detection is enabled. Increasing K manually will make the model fit the data tighter but "
                             "could result in overfitting. Fitted data should always be visually inspected after "
                             "training to gauge if the model is adequate")
    parser.add_argument("-d", "--min-density",
                        metavar="",
                        type=float,
                        default=0,
                        help="Transcripts with data density below this threshold will be rejected from analysis. "
                             "Valid range is 0 to 1. For example, 0.5 means that only transcripts containing "
                             "less than 50%% missing values will be used.")
    parser.add_argument("--KL-div",
                        metavar="",
                        type=float,
                        default=0.01,
                        help="Minimum Kullback–Leibler divergence criterion for building the training set. The KL "
                             "divergence measures the difference in information content between the full dataset "
                             "and the training set. The smaller the value, the more representative the training "
                             "set will be with respect to the full dataset. However, this will produce a "
                             "larger training set and increase both runtime and RAM consumption during training.")
    parser.add_argument("-e", "--epsilon",
                        metavar="",
                        type=float,
                        default=1e-4,
                        help="Convergence criterion")
    parser.add_argument("-i", "--maxiter",
                        metavar="",
                        type=int,
                        default=100,
                        help="Maximum number of training iterations")
    parser.add_argument("-nt", "--n-tasks",
                        metavar="",
                        type=int,
                        default=-1,
                        help="Number of parallel processes. By default all available CPUs are used")
    parser.add_argument("--model",
                        metavar="",
                        type=str,
                        default=None,
                        help="Trained .pickle GMMHMM model")
    parser.add_argument("--motif",
                        metavar="",
                        type=str,
                        default=None,
                        help="Score target motif declared using an extended dot-bracket notation. Paired and unpaired "
                             "bases are denoted using parentheses '()' and dots '.', respectively. A stretch of "
                             "consecutive characters is declared using the format <char>{<from>, <to>}. Can be used in "
                             "conjunction with --mask to modify the expected underlying sequence of pairing states.")
    parser.add_argument("--GQ",
                        metavar="",
                        type=str,
                        default=None,
                        #help="Score G-quadruplexes. GQ are declared by passing a string of format "
                        #     "\"[<min quartet>, <max quartet>, <min loop>, <max loop>]\". Quartet denotes the"
                        #     "number of quartets in the GQ and loops the spacing between G-columns."
                        help=argparse.SUPPRESS)
    parser.add_argument("--path",
                        metavar="",
                        type=str,
                        default=None,
                        help="Expected sequence of numerical pairing states for the motif with 0=unpaired and 1=paired "
                             "nucleotides. A stretch of consecutive states is declared using the format "
                             "<state>{<from>, <to>}. Can be used in conjunction with --motif to apply sequence "
                             "constraints.")
    parser.add_argument("--forbid-N-pairs",
                        action="store_true",
                        help="Pairs involving a N are considered invalid. Must be used in conjunction with "
                             "--motif to take effect")
    parser.add_argument("--posteriors",
                        action="store_true",
                        help="Output the posterior probabilities of pairing states (i.e. the probability Trellis)")
    parser.add_argument("--viterbi",
                        action="store_true",
                        help="Output the most likely sequence of pairing states for entire transcripts (i.e. Viterbi "
                             "paths)")
    parser.add_argument("--NAN",
                        action="store_true",
                        help="If NaN are considered informative in term of pairing state, use this flag. However, "
                             "note that this can lead to unstable results and is therefore not recommended")
    parser.add_argument("--no-prompt",
                        action="store_true",
                        help="Do not prompt a question if existing output files could be overwritten. Useful for "
                             "automation using scripts or for running patteRNA on computing servers")
    parser.add_argument("--nogmm",
                        action="store_true",
                        # help="Learn only HMM parameters but do not train the GMM. This should only be used if the "
                        #      "full GMM (i.e. means, sigmas and weights) is known and is configured using the YAML "
                        #      "config file (see option --config)",
                        help=argparse.SUPPRESS)  # DEVS

    args = parser.parse_args(inputargs)  # Parse input args

    if args.min_density > 1:
        args.min_density = 1

    return args


def parse_config_yaml(fp):
    to_array = ["pi", "phi", "upsilon", "A", "mu", "sigma", "w"]
    possible_scientific = ["epsilon", "min_density", "wmin"]

    # Read input params
    file_args = {}
    if fp:
        with open(fp, "r") as f:
            file_args = yaml.load(f)

    # Convert arguments to proper types
    for arg, v in file_args.items():

        # Numpy array
        if arg in to_array:
            file_args[arg] = np.array(v, dtype=float)

        # String "None" -> None
        if v == "None":
            file_args[arg] = None

        # String -> Float
        if arg in possible_scientific:
            file_args[arg] = float(v)

    return file_args


def add_more_defaults(cl_args):
    # Add additional defaults available via YAML config only
    defaults = {"pi": None,
                "A": None,
                "phi": None,
                "upsilon": None,
                "mu": None,
                "sigma": None,
                "w": None}

    for arg, v in defaults.items():
        setattr(cl_args, arg, v)


def merge_args(yaml_args, cl_args):
    for arg, v in yaml_args.items():
        setattr(cl_args, arg, v)


def summarize_config(args):
    """Text summary of config arguments."""
    hline = "========================================================="
    text = "\n{}\n" \
           "Running pattern mining with the following parameters:\n" \
           "{}\n".format(hline, hline)

    for key in sorted(args.__dict__):
        text += "{}: {}\n".format(key, args.__dict__[key])

    text += hline

    return text


def check_obs_extensions(files):
    """Check that extensions of observation filenames are supported and determine type of experimental assays.

    Args:
        files (list): List of filenames

    """

    accepted_files = []
    assay_types = []

    for file in files:
        file = file.strip()
        _, file_extension = os.path.splitext(file)
        if file_extension in globalbaz.GLOBALS["extensions"].keys():
            accepted_files.append(file)
            assay_types.append(globalbaz.GLOBALS["extensions"][file_extension])

    return accepted_files, assay_types


if __name__ == '__main__':
    pass
