
import numpy as np
import logging
import sys
import regex

from . import globalbaz

# Initialize globals
DTYPES = globalbaz.DTYPES

# Initialize the logger
logger = logging.getLogger(__name__)


def read_fasta(fp):
    """Reads a fasta file and returns a dictionary."""

    rnas = {}
    tr_name = None
    seq = ""

    try:
        with open(fp, "r") as f:

            while True:

                line = f.readline().strip()

                if not line:
                    break

                if line.startswith(">"):
                    if tr_name is not None:  # We will store this RNA
                        seq = check_sequence(seq, tr_name=tr_name)  # Check if the sequence is valid
                        rnas[tr_name] = seq

                    tr_name = line.split(">")[1].strip()
                    seq = ""
                else:
                    seq += line.strip()

            # Read the last entry
            if tr_name is not None:
                seq = check_sequence(seq, tr_name=tr_name)  # Check if the sequence is valid
                rnas[tr_name] = seq

    except FileNotFoundError:
        logger.error("No input sequence file found at {}.".format(fp))
        sys.exit()

    return rnas


def check_sequence(seq, tr_name="unnamed"):
    """Check if an RNA sequence is valid.

    Args:
        seq (str): RNA sequences
        tr_name (str): Transcript name

    Returns:
        seq (str): Processed RNA sequence

    """

    invalid_nucleotides = regex.compile("[^ACGTN]+$")  # matches non allowed bases in a sequence

    seq = seq.upper()
    seq = seq.replace("U", "T")  # RNA -> DNA-like bases
    m = regex.search(invalid_nucleotides, seq)
    # Check for invalid bases.
    if m:
        logger.error("Invalid nucleotide(s) found for transcript {}".format(tr_name))
        sys.exit()

    return seq


def read_fastaobs(fp):
    """Reads a fasta-like formatted file of observations and returns a dictionary."""

    rnas = {}
    tr_name = None
    obs = ""

    try:
        with open(fp, "r") as f:

            while True:

                line = f.readline().strip()

                if not line:
                    break

                if line.startswith(">"):
                    if tr_name is not None:  # We will store this RNA
                        obs = obs.strip().split()
                        obs = [word.replace('NA', 'nan') for word in obs]  # Handle NA
                        rnas[tr_name] = np.array(obs, dtype=DTYPES["obs"])

                    tr_name = line.split(">")[1].strip()
                    obs = ""
                else:
                    obs += line.strip()

            # Read the last entry
            if tr_name is not None:
                obs = obs.strip().split()
                obs = [word.replace('NA', 'nan') for word in obs]  # Handle NA
                rnas[tr_name] = np.array(obs, dtype=DTYPES["obs"])

    except FileNotFoundError:
        logger.error("No input observation file found at {}.".format(fp))
        sys.exit()

    return rnas


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


if __name__ == '__main__':
    pass
