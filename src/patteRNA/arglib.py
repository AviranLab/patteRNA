"""Parse and handle input arguments"""

import argparse
import logging
import multiprocessing
from patteRNA import version

logger = logging.getLogger(__name__)


def parse_cl_args(inputargs):
    """
    Parse command line arguments.
    Args:
        inputargs (list): List of input arguments.

    Returns:
        input_file (dict): Dictionary of input file names
        run_config (dict): Dictionary of run configuration options

    """
    parser = argparse.ArgumentParser(prog="patteRNA",
                                     description="Rapid mining of RNA secondary structure motifs from profiling data.",
                                     epilog="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version",
                        action="version",
                        version="%(prog)s {}".format(version.__version__))
    parser.add_argument("probing",
                        metavar="probing",
                        type=str,
                        help="FASTA-like file of probing data")
    parser.add_argument("output",
                        metavar="output",
                        type=str,
                        help="Output directory")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Print detailed progress logs")
    parser.add_argument("-f", "--fasta",
                        metavar="fasta",
                        default=None,
                        type=str,
                        help="FASTA file of RNA sequences")
    parser.add_argument("--reference",
                        metavar="reference",
                        default=None,
                        type=str,
                        help="FASTA-like file of reference RNA secondary structures in dot-bracket notation")
    parser.add_argument("-l", "--log",
                        action="store_true",
                        help="Log transform input data")
    parser.add_argument('--no-vienna',
                        action='store_true',
                        help="Do not attempt to use ViennaRNA libraries. Turns off LBC scoring classifier.")
    parser.add_argument("--GMM",
                        action="store_true",
                        help="Train a Gaussian Mixture Model (GMM) during training instead of a Discretized Observation"
                             "Model (DOM)")
    parser.add_argument("-k",
                        metavar="kernels",
                        type=int,
                        default=-1,
                        help="Number of kernels per pairing state to use in the emission model. By default, k is "
                             "determined automatically using Bayesian Information Criteria. Increasing k manually can "
                             "more precisely fit=the data, but could result in overfitting. Fitted data should always "
                             "be visually inspected after training to gauge if the model is adequate.")
    parser.add_argument("--KL-div",
                        metavar="KL-div",
                        type=float,
                        default=0.001,
                        help="Minimum Kullback–Leibler divergence criterion for building the training set. The KL "
                             "divergence measures the difference in information content between the full dataset "
                             "and the training set. The smaller the value, the more representative the training "
                             "set will be with respect to the full dataset. However, this will produce a "
                             "larger training set and increase both runtime and RAM consumption during training.")
    parser.add_argument("-e", "--epsilon",
                        metavar="eps",
                        type=float,
                        default=1e-2,
                        help="Convergence criterion")
    parser.add_argument("-i", "--maxiter",
                        metavar="iter",
                        type=int,
                        default=250,
                        help="Maximum number of training iterations")
    parser.add_argument("-t", "--n-tasks",
                        metavar="tasks",
                        type=int,
                        default=-1,
                        help="Number of parallel processes. By default all available CPUs are used.")
    parser.add_argument("--model",
                        metavar="model",
                        type=str,
                        default=None,
                        help="Trained .json model (version 2.0+ models only)")
    parser.add_argument("--motif",
                        metavar="motif",
                        type=str,
                        default=None,
                        help="Score target motif declared using an extended dot-bracket notation. Paired and unpaired "
                             "bases are denoted using parentheses '()' and dots '.', respectively. A stretch of "
                             "consecutive characters is declared using the format <char>{<from>, <to>}. Can be used in "
                             "conjunction with --mask to modify the expected underlying sequence of pairing states.")
    parser.add_argument("--path",
                        metavar="path",
                        type=str,
                        default=None,
                        help="Target binary state sequence. When used in conjunction with --motif, sequence constraints"
                             "from the motif applied, but the state sequence provided by --path is used to compute"
                             "raw scores.")
    parser.add_argument("--hairpins",
                        action="store_true",
                        help="Score a representative set of hairpins (stem lengths 4 to 15; loop lengths 3 to 10). "
                             "Automatically enabled when the --HDSL flag is used. This flag overrides any motif "
                             "syntaxes provided via --motif.")
    parser.add_argument("--posteriors",
                        action="store_true",
                        help="Output the posterior probabilities of pairing states (i.e. the probability Trellis)")
    parser.add_argument("--viterbi",
                        action="store_true",
                        help="Output the most likely sequence of pairing states for entire transcripts (i.e. Viterbi "
                             "paths)")
    parser.add_argument("--HDSL",
                        action="store_true",
                        help="Use scores a representative set of hairpins (stem lengths 4 to 15; loop lengths 3 to 10) "
                             "to quantify structuredness across the input data. This flag overrides any motif "
                             "syntaxes provided via --motif and also activates --posteriors.")
    parser.add_argument("--SPP",
                        action="store_true",
                        help="Smoothed P(paired). Quantifies structuredness across the input data via local pairing "
                             "probabilities. This flag activates --posteriors.")
    parser.add_argument("--nan",
                        action="store_true",
                        help="To attempt statistical inferences on the pairing state of nucleotides with missing "
                             "data when training, set this flag. Note that this can lead to meaningless results if "
                             "observation quality is low or long runs of missing data exist in the data.")
    parser.add_argument("--print-nan",
                        action="store_true",
                        help="Include NaN scores when writing scores to file. If the data contain large runs of"
                             "missing data, setting this flag may make score files very large.")
    parser.add_argument("--no-prompt",
                        action="store_true",
                        help="Do not prompt a question if existing output files could be overwritten. Files in output "
                             "directory will be overwritten if present. Useful for automation using scripts or for "
                             "running patteRNA on computing servers. ")
    parser.add_argument("--min-cscores",
                        metavar="min",
                        type=int,
                        default=1000,
                        help="Minimum number of scores to sample during construction of null distributions to use"
                             "for c-score normalization")
    parser.add_argument("--no-cscores",
                        action="store_true",
                        help="Suppress the computation of c-scores during the scoring phase")
    parser.add_argument("--batch-size",
                        metavar="size",
                        type=int,
                        default=100,
                        help="Number of transcripts to process at once using a pool of parallel workers")
    parser.add_argument('-c', '--context',
                        metavar="length",
                        type=int,
                        default=40,
                        help="Flanking distance to use when computing motif energy loss")

    args = parser.parse_args(inputargs)  # Parse input args

    input_files = dict.fromkeys(['probing', 'fasta', 'reference', 'model'], None)
    run_config = dict()

    for attr, value in args.__dict__.items():

        if attr in input_files.keys():
            input_files[attr] = value
        else:
            run_config[attr] = value

    # Process HDSL constraints
    if run_config['HDSL']:
        if run_config['no_cscores']:
            logger.error("Configuration error: --HDSL cannot be used in conjunction with --no-cscores")
        if run_config['motif'] is not None:
            logger.warning('Configuration warning: --motif cannot be used in conjunction with --HDSL')

    # Process hairpin constraints
    if run_config['hairpins']:
        if run_config['motif'] is not None:
            logger.warning('Configuration warning: --motif cannot be used in conjunction with --hairpins')

    if run_config['HDSL']:
        run_config['motif'] = "({4,15}.{3,10}){4,15}"
        run_config['hairpins'] = True
        run_config['posteriors'] = True

    if run_config['hairpins']:
        run_config['motif'] = "({4,15}.{3,10}){4,15}"

    if run_config['SPP']:
        run_config['posteriors'] = True

    # Set training configuration flag based on provided model
    if input_files['model'] is not None:
        run_config['training'] = False
    else:
        run_config['training'] = True

    # Set scoring configuration flag
    if run_config['posteriors'] or run_config['motif'] or run_config['viterbi']:
        run_config['scoring'] = True
    else:
        run_config['scoring'] = False

    if run_config['n_tasks'] == -1:
        run_config['n_tasks'] = multiprocessing.cpu_count()

    run_config['reference'] = False  # Even if reference data is provided, we need to inspect it before deciding to
    # actually use it for training. So, for now, run_config['reference'] is False.
    run_config['hdsl_params'] = (1.2, 0.5)

    return input_files, run_config


def summarize_job(input_files, run_config):
    """
    Generate concise text summary of config arguments for command line printing.

    Args:
        input_files (dict): Dictionary of input files
        run_config (dict): Dictionary of configuration parameters

    Returns:
        text (str): String representation of run configuration
    """

    space = '\n'
    tab = ' ' * 7
    text = "Summarizing job ... \n"
    text += space
    text += tab + "Input files:\n"
    text += 2 * tab + "probing: {}\n".format(input_files['probing'])
    if input_files['fasta'] is not None:
        text += 2 * tab + "fasta: {}\n".format(input_files['fasta'])
    if input_files['reference'] is not None:
        text += 2 * tab + "reference: {}\n".format(input_files['reference'])
    if input_files['model'] is not None:
        text += 2 * tab + "model: {}\n".format(input_files['model'])
    text += space

    text += tab + "Configuration parameters:\n"
    text += 2 * tab + "log: {}\n".format(run_config['log'])
    text += 2 * tab + "training: {}\n".format(run_config['training'])
    text += 2 * tab + "scoring: {}\n".format(run_config['scoring'])
    if run_config['k'] == -1:
        text += 2 * tab + "k: auto\n"
    else:
        text += 2 * tab + "k: {}\n".format(run_config['k'])
    if run_config['GMM']:
        text += 2 * tab + "GMM: True\n"
    else:
        text += 2 * tab + "DOM: True\n"
    if run_config['hairpins']:
        text += 2 * tab + "hairpins: {}\n".format(run_config['hairpins'])
    if run_config['motif'] is not None:
        text += 2 * tab + "motif: {}\n".format(run_config['motif'])
    if run_config['path'] is not None:
        text += 2 * tab + "path: {}\n".format(run_config['path'])
    if run_config['posteriors']:
        text += 2 * tab + "posteriors: {}\n".format(run_config['posteriors'])
    if run_config['HDSL']:
        text += 2 * tab + "HDSL: {}\n".format(run_config['HDSL'])
    if run_config['SPP']:
        text += 2 * tab + "SPP: {}\n".format(run_config['SPP'])
    if run_config['viterbi']:
        text += 2 * tab + "viterbi: {}\n".format(run_config['viterbi'])
    text += space

    text += tab + "Using {} parallel processes\n".format(run_config['n_tasks'])

    return text


def summarize_config(input_files, run_config):
    """
    Generate exhaustive text summary of config arguments for logger.

    Args:
        input_files (dict): Dictionary of input files
        run_config (dict): Dictionary of configuration parameters

    Returns:
        text (str): String representation of run configuration
    """

    hline = "========================================================="
    text = "Summarizing configuration:" \
           "\n{}\n" \
           "Running patteRNA with the following parameters:" \
           "\n{}\n".format(hline, hline)

    text += "Input files:\n"
    for key in input_files:
        if input_files[key]:
            text += "\t{}: {}\n".format(key, input_files[key])

    text += "\nConfiguration parameters:\n"
    for key in run_config:
        text += "\t{}: {}\n".format(key, run_config[key])

    text += hline

    return text


if __name__ == '__main__':
    pass
