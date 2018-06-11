import numpy as np
import logging
import sys
import regex
import shutil
import os

from . import globalbaz
from . import patternlib

# Initialize the logger
logger = logging.getLogger(__name__)

# Set globals
GLOBALS = globalbaz.GLOBALS


def read_fastalike(fp):
    """Reads a fasta-like file and returns a dictionary."""

    rnas = {}
    tr_name = None
    content = ""

    try:
        with open(fp, "r") as f:

            while True:

                line = f.readline().strip()

                if not line:
                    break

                if line.startswith(">"):
                    if tr_name is not None:  # Store this transcript
                        rnas[tr_name] = content

                    tr_name = line.split(">")[1].strip()  # Get the new transcript
                    content = ""
                else:
                    content += line

            # Append the last entry of the file
            if tr_name is not None:
                rnas[tr_name] = content

    except FileNotFoundError:
        logger.error("No file found at {}.".format(fp))
        sys.exit()

    return rnas


def read_sequences(fp):
    """Reads a fasta file of sequence and returns a dictionary."""

    rnas = {}

    file_content = read_fastalike(fp)

    for tr_name, field in file_content.items():
        seq = check_sequence(field, tr_name=tr_name)  # Check if the sequence is valid
        rnas[tr_name] = seq

    return rnas


def read_observations(fp):
    """Reads a fasta-like formatted file of observations and returns a dictionary."""

    rnas = {}

    file_content = read_fastalike(fp)

    for tr_name, field in file_content.items():
        obs = field.strip().split()
        obs = [word.replace('NA', 'nan') for word in obs]  # Handle NA
        obs = np.array(obs, dtype=GLOBALS["dtypes"]["obs"])
        obs[np.isinf(obs)] = np.nan  # Handle infinite values
        rnas[tr_name] = obs

    return rnas


def read_reference_structures(fp):
    """Read a FASTA-like file of reference secondary structures in dot-bracket notation."""

    rnas = {}
    with open(fp, "r") as f:

        while True:
            line1 = f.readline().strip()  # Header
            _ = f.readline()  # Sequence
            line3 = f.readline().strip()  # Dot-bracket

            if not line3:
                break

            tr_name = line1.split(">")[1].strip()
            ref_dot = check_dot(line3, tr_name=tr_name)
            ref_dot = patternlib.dot2states(ref_dot)
            rnas[tr_name] = ref_dot

    return rnas


def read_path(fp):
    """Read a .path or viterbi.txt file."""

    rnas = {}

    file_content = read_fastalike(fp)

    for tr_name, field in file_content.items():
        rnas[tr_name] = np.array(list(field), dtype=GLOBALS["dtypes"]["path"])

    return rnas


def read_posteriors(fp):
    """Read a posteriors.txt file."""

    rnas = {}
    with open(fp, "r") as f:

        while True:
            line1 = f.readline()
            _ = f.readline()
            line3 = f.readline()

            if not line3:
                break

            tr_name = line1.split(">")[1].strip()
            rnas[tr_name] = np.array(line3.split(), dtype=float)  # Line 2 is not used as it is 1 - line3

    return rnas


def check_sequence(seq, tr_name="unnamed"):
    """Check if an RNA sequence is valid.

    Args:
        seq (str): RNA sequences
        tr_name (str): Transcript name

    Returns:
        seq (str): Processed RNA sequence

    """

    invalid_nucleotides = regex.compile("[^ACGTN]+")  # matches non allowed bases in a sequence

    seq = seq.upper()
    seq = seq.replace("U", "T")  # RNA -> DNA-like bases
    m = regex.search(invalid_nucleotides, seq)

    # Check for invalid bases.
    if m:
        logger.error("Invalid nucleotide(s) found for transcript {}".format(tr_name))
        sys.exit()

    return seq


def check_dot(dot, tr_name="unnamed"):
    """Check if a dot-bracket structure is valid.

    Args:
        dot (str): RNA secondary structure in dot-bracket notation
        tr_name (str): Transcript name

    Returns:
        dot (str): Processed dot-bracket

    """

    invalid_symbols = regex.compile("[^\.\(\)\<\>\{\}]+")  # matches non allowed symbols in a dot-bracket

    m = regex.search(invalid_symbols, dot)

    # Check for invalid symbols.
    if m:
        logger.error("Invalid symbol(s) found in the dot-bracket string for transcript {}".format(tr_name))
        sys.exit()

    return dot


def check_overwrites(args, switch):
    """Check if existing file will be overwritten.

    Args:
        args: CLI input arguments.
        switch (dict): CLI switches.

    """

    # Check that no file overwriting will occur
    overwrite_files = False
    delete_model = False
    delete_fit_plot = False
    delete_logL_plot = False
    delete_scores = False
    delete_viterbi = False
    delete_posteriors = False

    if os.path.isdir(args.output):  # Output directory already exist
        if switch["do_training"]:
            # if os.path.isdir(os.path.join(args.output, GLOBALS["output_name"]["training"])):
            #     overwrite_files = True
            #     delete_training = True
            if os.path.isfile(os.path.join(args.output, GLOBALS["output_name"]["model"])):
                delete_model = True
                overwrite_files = True
            if os.path.isfile(os.path.join(args.output, GLOBALS["output_name"]["fit_plot"])):
                delete_fit_plot = True
                overwrite_files = True
            if os.path.isfile(os.path.join(args.output, GLOBALS["output_name"]["logL_plot"])):
                delete_logL_plot = True
                overwrite_files = True

        if switch["do_scan"] and os.path.isfile(os.path.join(args.output, GLOBALS["output_name"]["scores"])):
            delete_scores = True
            overwrite_files = True
        if args.posteriors and os.path.isfile(os.path.join(args.output, GLOBALS["output_name"]["posteriors"])):
            delete_posteriors = True
            overwrite_files = True
        if args.viterbi and os.path.isfile(os.path.join(args.output, GLOBALS["output_name"]["viterbi"])):
            delete_viterbi = True
            overwrite_files = True

    else:
        pass

    while True:
        if overwrite_files:
            if args.no_prompt:
                response = "Yes"
            else:
                response = input("Some output files already exist. Overwrite them? [yes/no] ")

            if response in GLOBALS["user_prompt"]["yes"]:
                # if delete_training:
                #     shutil.rmtree(os.path.join(args.output, GLOBALS["output_name"]["training"]), ignore_errors=False)
                if delete_model:
                    os.remove(os.path.join(args.output, GLOBALS["output_name"]["model"]))
                if delete_fit_plot:
                    os.remove(os.path.join(args.output, GLOBALS["output_name"]["fit_plot"]))
                if delete_logL_plot:
                    os.remove(os.path.join(args.output, GLOBALS["output_name"]["logL_plot"]))
                if delete_scores:
                    os.remove(os.path.join(args.output, GLOBALS["output_name"]["scores"]))
                if delete_posteriors:
                    os.remove(os.path.join(args.output, GLOBALS["output_name"]["posteriors"]))
                if delete_viterbi:
                    os.remove(os.path.join(args.output, GLOBALS["output_name"]["viterbi"]))
                break

            elif response in GLOBALS["user_prompt"]["no"]:
                sys.exit()
        else:
            break


def write_shape(fp, shape):
    """Write reactivities to a .shape formatted file."""

    with open(fp, "w") as f:
        for nuc in range(len(shape)):
            if np.isnan(shape[nuc]):
                f.write("{} {}\n".format(nuc + 1, -999))
            else:
                if shape[nuc] < 0:
                    f.write("{} {:.4f}\n".format(nuc + 1, 0))
                else:
                    f.write("{} {:.4f}\n".format(nuc + 1, shape[nuc]))


def h0_pull(fp, fp_partial, h0_set):
    """Write h0 to temporary files.

    Here we will split paths for which enough h0 paths were scored (written to fp) and paths for which additional
    h0 scores are required (written to fp_partial).

    Completed h0 paths will be reported in the partial file as a space-delimited string after a dummy path titled ">".

    """

    h0_list = []

    with open(fp_partial, "w") as f_partial, open(fp, "a") as f:

        for path, scores in h0_set.items():
            if scores is None:  # Means we already have enough h0 info for this path
                h0_list.append(path)
            else:
                txt = " ".join([str(x) for x in scores])

                if len(scores) == GLOBALS["h0_max_size"]:  # We reached enough h0 scores if this is true
                    f.write("{}\n{}\n".format(path, txt))
                    h0_list.append(path)
                else:
                    f_partial.write("{}\n{}\n".format(path, txt))

        # Write the list of paths with completed h0 in the partial file with title ">"
        txt = " ".join(h0_list)
        f_partial.write(">\n{}\n".format(txt))


def h0_fetch(fp):
    """Fetch h0 scores from a temporary file."""

    h0_paths = {}
    try:
        with open(fp, "r") as f:

            while True:

                line1 = f.readline()
                line2 = f.readline()

                if not line2:
                    break

                path = line1.strip()

                if path.startswith(">"):  # Means we are reading the list of completed paths in the partial file
                    cpaths = line2.strip().split()
                    for cpath in cpaths:
                        h0_paths[cpath] = None  # Completed paths gets None scores to ease further processing
                else:
                    scores = list(np.array(line2.strip().split(), dtype=globalbaz.GLOBALS["dtypes"]["p"]))
                    h0_paths[path] = scores

    except FileNotFoundError:
        pass

    return h0_paths


def h0_merge(fp, fp_partial):
    """Merge the partial h0 file into the complete h0 files."""

    h0_partial = h0_fetch(fp_partial)
    h0 = h0_fetch(fp)

    with open(fp, "a") as f:

        for path, scores in h0_partial.items():
            if path not in h0.keys():
                txt = " ".join([str(x) for x in scores])
                f.write("{}\n{}\n".format(path, txt))


def sort_score_file(fp_in, fp_out, column_ix=4, decreasing=False):
    """Sort the scores.txt output file by p-values.

    Args:
        fp_in (str): Input unsorted scores.txt file path
        fp_out (str): Output sorted scores.txt file path
        column_ix (int): Sort by this column index (default is by p-values).
        decreasing (bool); Sort in decreasing order?

    """

    # Read input score and sort
    with open(fp_in, "r") as f:
        header = f.readline()
        vals = []
        lines = []
        scores = []

        for line in f:
            lines.append(line)
            vals.append(line.split()[column_ix])
            scores.append(line.split()[4])

    if decreasing:
        sort_ix = np.argsort(-np.array(vals, dtype=GLOBALS["dtypes"]["p"]))
    else:
        sort_ix = np.argsort(np.array(vals, dtype=GLOBALS["dtypes"]["p"]))

    max_score = None
    scores = np.array(scores, dtype='float64')
    # Output sorted scores
    with open(fp_out, "w") as f:
        f.write(header)

        for i in sort_ix:
            newline = lines[i].split()
            if newline[4] == "inf":
                if max_score is None:
                    mask = np.array(np.isfinite(np.array(scores, dtype=GLOBALS["dtypes"]["p"])))
                    max_score = np.max(scores[mask])
                newline[4] = str(max_score)
            f.write(" ".join(newline)+"\n")

# def read_and_sort_scores(fp, by=None, descending=False, formats=None):
#     """Return a sorted numpy array from a patteRNA score.txt output file.
#
#     Args:
#         fp (str): Filename
#         by (str): Index of the field to sort
#         descending (bool): Sort by descending order?
#         formats (tuple): Tuple of data format for each column in score.txt
#
#     """
#
#     colnames = ["transcript", "start", "end", "score", "path", "seq"]
#
#     if formats is None:
#         formats = ["U256", "int64", "int64", "float32", "U256", "U256"]
#
#     dtype = {"names": colnames,
#              "formats": formats}
#
#     X = np.genfromtxt(fp, skip_header=1, dtype=dtype)
#
#     if by is not None:
#         X.sort(order=by)
#
#     if descending:
#         X = np.flipud(X)  # To get descending ordered data
#
#     return X


# def ct2seq(fp):
#     """Read an RNA sequence from a .ct file
#
#     Args:
#         fp (str): Filename.
#
#     Returns:
#         seq: RNA sequence.
#
#     """
#
#     with open(fp) as f:
#         next(f)  # skip the header
#         rows = (line.split() for line in f)
#         seq = [row[1] for row in rows]
#
#     return seq

if __name__ == '__main__':
    pass
