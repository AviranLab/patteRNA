"""
This module holds all functions and classes related to RNA secondary structures.

It is organized using the following general class hierarchy:
RNAset -> RNA

Arguments are passed through classes using declared named variables first followed by all remaining attributes fed
using the variable `feed_attr`.
Classes are not directly inherited (i.e. parent -> child) but attributes are, via keyword dictionaries.

Attributes are declared in each class definition and inheritance occurs for attributes with the same name between
two classes.

No __main__, just defining classes and functions.

"""

import numpy as np
import logging
import warnings
import random

from . import globalbaz
from . import file_handler

# Initialize globals
DTYPES = globalbaz.DTYPES

# Initialize the logger
logger = logging.getLogger(__name__)


# Classes
class RNAset:
    """Set of RNAs.

    Attributes:
        rnas (list): List of RNA objects.
        name (list): RNA names.
        n_rna (int): Number of RNAs in the set.
        T (int): Cumulative length of all observations in the set.
        T_nan (int): Cumulative length of all missing observations.
        T_0 (int): Cumulative length of all zeros observations.
        max_T (int): Length of the longest RNA.
        min_obs (float): Minimum value of the continuous observations.
        max_obs (float): Maximum value of the continuous observations.
        mean_obs (float): Average of the continuous observations.
        median_obs (float): Median of the continuous observations.
        stdev_obs (float): Standard deviation of the continuous observations.
        histogram (dict): Contains the continuous data histogram bins and densities. Used for plotting.

    """

    def __init__(self):
        """Initialize RNAset attributes."""
        self.rnas = []
        self.name = []
        self.n_rna = 0
        self.T = 0
        self.T_nan = 0
        self.T_0 = 0
        self.max_T = 0
        self.min_obs = None
        self.max_obs = None
        self.mean_obs = None
        self.median_obs = None
        self.stdev_obs = None
        self.histogram = None

    def add_rna(self, rna):
        """Add an RNA to the set.
            
        Upon adding an RNA, will update all RNAset-related attributes.
            
        Args:
            rna (RNA): An RNA.

        """

        self.n_rna += 1  # Increment RNA counter
        # If an RNA with the same name was encountered, add a dummy character to its name
        if rna.name in self.name:
            rna.name += "_"
        rna.name = rna.name.replace(" ", "_")  # Replace spaces in the name with underscores
        self.name.append(rna.name)  # Add the RNA name to the list
        self.rnas.append(rna)  # Append the RNA to the set
        self.T += rna.T  # Increment the number of total observed probing values
        self.T_nan += rna.T_nan  # Increment the number of missing observed probing values
        self.T_0 += rna.T_0  # Increment the number of zeros observed probing values
        self.max_T = rna.T if rna.T > self.max_T else self.max_T  # Update the longest RNA if needed

    def compute_stats(self):
        """Compute some basic statistics for the set."""
        continuous_obs = []

        for rna in self.rnas:
            continuous_obs += list(rna.obs[~rna.mask_nan & ~rna.mask_0])  # Append observed probing values to a list

        # Compute basic stats
        self.min_obs = np.min(continuous_obs)
        self.max_obs = np.max(continuous_obs)
        # self.mean_obs = np.mean(continuous_obs)
        self.median_obs = np.median(continuous_obs)
        self.stdev_obs = np.std(continuous_obs)

        # Build histogram bins for future plotting
        self.histogram = dict()
        self.histogram["dens"], self.histogram["bins"] = np.histogram(continuous_obs,
                                                                      bins="auto", normed=True)
        self.histogram["n"] = len(self.histogram["bins"])

    def add_temporary_rna(self, rna):
        """Add an RNA in "temporary"-mode, i.e. doesn't update the attributes of the set to reduce runtime."""

        self.rnas.append(rna)  # Append the RNA to the set

    def log_transform(self):
        """Log transform all observations."""

        for rna in self.rnas:
            rna.log_transform()

    def qc_and_build(self, min_density, n):
        """QC RNAs and reject all containing too many NaNs or Zeros.
        
        Args:
            min_density (float): Minimum data density allowed in the observation vector.
            n (int): Maximum number of RNAs in the set.
            
        Returns:

        """

        if n <= 0:
            n = np.inf

        if self.rnas:
            random.shuffle(self.rnas)

        new_set = RNAset()
        cnt = 0

        for rna in self.rnas:

            if rna.T == len(rna.seq):  # Ensure sequence and observations match in length
                p_nan = np.sum(rna.T_nan | rna.T_0) / rna.T

                if ((1-p_nan) >= min_density) & (cnt < n):
                    new_set.add_rna(rna)
                    cnt += 1

        return new_set


class RNA:
    """This class holds attributes related to an RNA.

    To ensure proper indexing, the nucleotide motif_position index starts at 0 and all attributes are mapped to it.

    Attributes:
        name (str): RNA name.
        obs (np.array): Chemical probing reactivities.
        original_obs (np.array): Original chemical probing reactivities before any transformations.
        seq (str): RNA sequence.
        mask_nan (np.array): Boolean mask to select missing observations.
        mask_0 (np.array): Boolean mask to select zero observations.
        T (int): Length of the RNA
        T_nan (int): Number of missing observations
        T_0 (int): Number of zeros observations
        T_continuous (np.array): Number of continuous (non-missing, non-zero) observations

    """

    def __init__(self, name, obs, seq):
        """Initialize attributes."""

        # Set attributes
        self.name = name  # RNA name
        self.obs = np.array(obs, dtype=DTYPES["obs"])  # RNA probing profile
        self.original_obs = np.array(obs, dtype=DTYPES["obs"])  # Keep a copy of the original probing profile
        self.seq = seq  # RNA sequence

        self.mask_nan = np.isnan(self.obs)
        self.mask_0 = self.obs == 0
        self.T = len(self.obs)
        self.T_nan = np.sum(self.mask_nan)
        # noinspection PyTypeChecker
        self.T_0 = np.sum(self.mask_0)
        self.T_continuous = self.T - (self.T_nan + self.T_0)

    def log_transform(self):
        # Assign negatives to 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.obs[self.obs <= 0] = 0
        self.mask_0 = self.obs == 0  # Update the mask for zeros

        self.obs[self.mask_0] = np.nan  # Mask zeros and negatives
        self.obs = np.log(self.obs)  # log transform
        self.obs[self.mask_0] = -np.inf  # Replace original negatives and zeros with -inf
        # noinspection PyTypeChecker
        self.T_0 = np.sum(self.mask_0)  # Update the T_0 counter


# noinspection PyUnusedLocal
def build_rnalib_from_files(fp_seq, fp_obs):
    """Reads both the sequence and the observations transcriptome-wide.
    
    Args:
        fp_seq (str): Pointer to the FASTA file containing the sequences.
        fp_obs (str): Pointer to the observation file (in FASTA-like format)

    Returns:
        rna_set (RNAset): Set of RNAs with both sequences and observations.

    """

    putative_rnas = {}
    rna_set = RNAset()

    # Read the observation data
    observations = file_handler.read_fastaobs(fp_obs)

    for rna, obs in observations.items():
        putative_rnas[rna] = {"seq": "N"*len(obs),
                              "obs": np.array(obs, dtype=DTYPES["obs"])}
    observations = None  # rm entry

    # Read sequences from the fasta file if it exist and update the dictionary
    if fp_seq:
        fasta = file_handler.read_fasta(fp_seq)

        for rna, seq in fasta.items():
            if rna in putative_rnas.keys():
                putative_rnas[rna]["seq"] = seq
        fasta = None  # rm entry

    # Build the set of RNAs
    for rna in putative_rnas.keys():
        current_rna = RNA(name=rna,
                          obs=putative_rnas[rna]["obs"],
                          seq=putative_rnas[rna]["seq"])
        rna_set.add_temporary_rna(current_rna)
        putative_rnas[rna] = None  # rm entry

    return rna_set


if __name__ == '__main__':
    pass
