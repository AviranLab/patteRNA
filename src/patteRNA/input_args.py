"""Parse and handle input arguments"""

import argparse
import yaml
import numpy as np


def parse_cl_args(inputargs):

    # Define the args parser
    parser = argparse.ArgumentParser(prog="patteRNA",
                                     description="Rapid mining of RNA secondary structure motifs from profiling data.",
                                     epilog="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s 1.0.0')
    parser.add_argument("input",
                        metavar="probing",
                        type=str,
                        help="FASTA-like file of probing data")
    parser.add_argument("output",
                        metavar="output",
                        type=str,
                        help="Output directory")
    parser.add_argument("-f", "--fasta",
                        metavar="",
                        default=None,
                        type=str,
                        help="FASTA file of RNA sequences")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Print progress")
    parser.add_argument("-l", "--log",
                        action="store_true",
                        help="Log transform input data")
    parser.add_argument("--PARS",
                        action="store_true",
                        help="Use this flag for PARS experiments")
    parser.add_argument("--config",
                        metavar="",
                        type=str,
                        default=None,
                        help="Config parameters in YAML format. Has priority over CLI options")
    parser.add_argument("-k",
                        metavar="",
                        type=int,
                        default=10,
                        help="Number of Gaussian components per pairing state in the GMM model. Increasing this "
                             "will make a tighter fit on the data but could also result on overfitting. Fitted data "
                             "can and should be visually inspected after training to gauge if the model is adequate")
    parser.add_argument("-n",
                        metavar="",
                        type=int,
                        default=-1,
                        help="Number of transcripts used for training. We recommend using about 500 to 1000 "
                             "transcripts for large datasets. Transcripts are randomly selected if -n is lower than "
                             "the total number of available transcripts")
    parser.add_argument("-d", "--min-density",
                        metavar="",
                        type=float,
                        default=0.5,
                        help="Minimum data density allowed in each transcript used for training.")
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
    parser.add_argument("--pattern",
                        metavar="",
                        type=str,
                        default=None,
                        help="Pattern of the target structural motif in the extended dot-bracket notation. Paired and "
                             "unpaired nucleotides are represented by parentheses '()' and dots '.', respectively. A "
                             "stretch of consecutive characters is declared using the format <char>{<from>, <to>}. "
                             "For example, use .{2,4} to declare 2 to 4 consecutive repeats of unpaired nucleotides "
                             "in the pattern. For G-Quadruplexes, use GQ[min, max quartet, min, max loop]")
    parser.add_argument("-s", "--seq",
                        action="store_true",
                        help="Use sequence constraints when searching for motifs")
    parser.add_argument("--forbid-N-pairs",
                        action="store_true",
                        help="Pairs involving a N are considered invalid (must be used in conjunction with -s/--seq "
                             "to take effect")
    parser.add_argument("--gammas",
                        action="store_true",
                        help="Output the posterior probabilities of pairing states (i.e. the probability Trellis)")
    parser.add_argument("--viterbi",
                        action="store_true",
                        help="Output the most likely sequence of pairing states for entire transcripts (i.e. Viterbi "
                             "paths)")
    parser.add_argument("-wmin",  # Not currently supported
                        metavar="",
                        type=float,
                        default=0,
                        # help="Drop Gaussian mixture components below this weight."
                        #      "Note that if this value is set too high then all components "
                        #      "will be dropped resulting in a error."
                        help=argparse.SUPPRESS)
    parser.add_argument("--filter-test",
                        action="store_true",
                        help="Apply the density filter used for the training to the test set as well. Not recommended "
                             "as regions with sparse profiling data will tend to score generally poorly")
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
                        help="Learn only HMM parameters but do not train the GMM. This should only be used if the "
                             "full GMM (i.e. means, sigmas and weights) is known and is configured using the YAML "
                             "config file (see option --config)")
    parser.add_argument("--debug",
                        action="store_true",
                        help=argparse.SUPPRESS)

    args = parser.parse_args(inputargs)  # Parse input args

    if args.min_density > 1:
        args.min_density = 1

    return args


def parse_config_yaml(fp):
    to_array = ["pi", "phi", "upsilon", "A", "mu", "sigma", "w"]
    possible_scientific = ["epsilon", "maxpnan", "wmin"]

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


if __name__ == '__main__':
    pass
