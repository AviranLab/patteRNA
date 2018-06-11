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
from scipy.stats import entropy
# todo remove that when finished
# import matplotlib.pyplot as plt

from . import globalbaz
from . import file_handler

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
        continuous_obs (np.array): Concatenated continuous observations.
        min_obs (float): Minimum value of the continuous observations.
        max_obs (float): Maximum value of the continuous observations.
        mean_obs (float): Average of the continuous observations.
        median_obs (float): Median of the continuous observations.
        sigma_obs (float): Variance of the continuous observations.
        percentile_obs (np.array): Percentiles to anchor the GMM means.
        histogram (dict): Contains the continuous data histogram bins and densities. Used for plotting.
        real_KL_div (float): KL divergence of the training set.

    """

    def __init__(self):
        """Initialize RNAset attributes."""
        self.rnas = []
        self.name = []
        self.n_rna = 0
        self.n_rna_unfiltered = 0
        self.T = 0
        self.T_nan = 0
        self.T_0 = 0
        self.T_continuous = 0
        self.max_T = 0
        self.continuous_obs = None
        self.min_obs = None
        self.max_obs = None
        self.mean_obs = None
        self.median_obs = None
        self.sigma_obs = None
        self.percentile_obs = None
        self.histogram = None

    def add_rna(self, rna, min_density=0):
        """Add an RNA to the set.
            
        Upon adding an RNA, will update all RNAset-related attributes.
            
        Args:
            rna (RNA): An RNA.
            min_density (float): Minimum data density to consider a transcript

        """

        self.n_rna_unfiltered += 1  # Increment RNA before filtering

        # Reject RNA if data density is too poor
        if (((rna.T_continuous + rna.T_0) / rna.T) >= min_density) and (rna.T_nan != rna.T):

            self.n_rna += 1  # Increment RNA counter

            # If an RNA with the same name was encountered, add a dummy character to its name
            if rna.name in self.name:
                rna.name += "_"
            rna.name = rna.name.replace(" ", "_")  # Replace spaces in the name with underscores
            self.name.append(rna.name)  # Add the RNA name to the list
            self.rnas.append(rna)  # Append the RNA to the set
            self.T += rna.T  # Increment the number of total observed probing values
            self.T_continuous += rna.T_continuous  # Increment the number of continuous observed probing values
            self.T_nan += rna.T_nan  # Increment the number of missing observed probing values
            self.T_0 += rna.T_0  # Increment the number of zeros observed probing values
            self.max_T = rna.T if rna.T > self.max_T else self.max_T  # Update the longest RNA if needed

    def build_continuous_obs(self):
        self.continuous_obs = []

        for rna in self.rnas:
            self.continuous_obs += list(rna.obs[~rna.mask_nan & ~rna.mask_0])

        self.continuous_obs = np.array(self.continuous_obs)

    def compute_stats(self, K):
        """Compute some basic statistics for the set."""

        # Compute basic stats
        self.min_obs = np.min(self.continuous_obs)
        self.max_obs = np.max(self.continuous_obs)
        # self.mean_obs = np.mean(self.continuous_obs)
        self.median_obs = np.median(self.continuous_obs)
        self.sigma_obs = np.var(self.continuous_obs)
        percentile_anchors = np.arange(0, 1, (1 / (K * 2 + 1)))[1:] * 100
        self.percentile_obs = np.percentile(self.continuous_obs, percentile_anchors).reshape(2, -1)

    def build_histogram(self):

        # Build histogram bins for future plotting
        self.histogram = dict()
        self.histogram["dens"], self.histogram["bins"] = np.histogram(self.continuous_obs,
                                                                      bins="auto", normed=True)
        self.histogram["n"] = len(self.histogram["bins"])

    def log_transform(self):
        """Log transform all observations."""

        for rna in self.rnas:
            rna.log_transform()

    def qc_and_build(self, is_training=False, KL_threshold=None):
        """QC RNAs and reject all containing too many NaNs or Zeros.

        Args:
            is_training (bool): Build the training set?
            KL_threshold (float): Kullback–Leibler divergence criterion to build the training set

        """

        if is_training:
            # Compute the probability density over the full continuous observation set (RAM savvy)
            self.build_continuous_obs()
            self.build_histogram()

            # Compute the data density for each transcript
            density = []
            for rna in self.rnas:
                density.append(rna.T_continuous / rna.T)

            # Sort transcripts by density
            density = np.array(density)
            density_ix = np.argsort(density)[::-1]
        else:
            density_ix = np.arange(len(self.rnas))

        # Initialize the QCed and reference set
        qc_set = RNAset()  # QCed set
        ref_set = RNAset()  # Set with reference secondary structures

        # Variables required for building the training set
        cnt_subset = 0  # Subset counter to activate KL divergence computation when required
        train_obs = []
        do_KL = True
        qc_set.KL_div = np.inf

        for i in density_ix:
            rna = self.rnas[i]

            # Assess that length of observation and sequence match
            if rna.T == len(rna.seq):

                if is_training:
                    if do_KL:
                        # Add the transcript to the train subset
                        qc_set.add_rna(rna)
                        cnt_subset += rna.T_continuous

                        # Add new observations to the set used to compute the probability density of the data
                        train_obs += list(rna.obs[~rna.mask_nan & ~rna.mask_0])

                        # Compute KL divergence if the subset of data is large enough
                        if cnt_subset >= globalbaz.GLOBALS["KL_interval"]:

                            # Compute the probability density of the data
                            train_dens, _ = np.histogram(train_obs,
                                                         bins=self.histogram["bins"],
                                                         normed=True)
                            train_dens[train_dens == 0] = 1E-6

                            # Compute KL divergence
                            qc_set.KL_div = entropy(self.histogram["dens"], train_dens)

                            # todo remove this part when finished
                            # print(qc_set.KL_div)
                            # plt.plot(self.histogram["bins"][1:], train_dens, color="red")
                            # plt.plot(self.histogram["bins"][1:], self.histogram["dens"], color="blue")
                            # plt.fill_between(self.histogram["bins"][1:], 0, self.histogram["dens"],
                            #                  color="blue", alpha=0.2)
                            # plt.title("# {} {:.3f}".format(len(train_obs), KL_div))
                            # plt.ylim([0, 0.5])
                            # plt.savefig("/Users/mledda/Desktop/KL/n_{:d}.png".format(len(train_obs)))
                            # plt.close()

                            cnt_subset = 0  # Reset the subset size

                        if qc_set.T_continuous >= globalbaz.GLOBALS["min_train_n"]:  # Reached minimum train set size
                            if qc_set.KL_div <= KL_threshold:  # Train set summarizes well the full data
                                do_KL = False

                else:  # Doing scoring so all transcripts will be used
                    qc_set.add_rna(rna)

                if rna.ref_dot is not None:
                    # Assess that length of observation and structure match
                    if rna.T == len(rna.ref_dot):
                        ref_set.add_rna(rna)
                    else:
                        # Raise warning for transcripts that do not match in length
                        logger.warning("Dot-bracket is not matching the length of transcript -> {}".format(rna.name))

            else:
                # Raise warning for transcripts that do not match in length
                logger.warning("Input data are not matching in length for transcript -> {}".format(rna.name))

        if np.isinf(qc_set.KL_div):
            qc_set.KL_div = 0

        return qc_set, ref_set


class RNA:
    """This class holds attributes related to an RNA.

    To ensure proper indexing, the nucleotide motif_position index starts at 0 and all attributes are mapped to it.

    Attributes:
        name (str): RNA name.
        obs (np.array): Chemical probing reactivities.
        seq (str): RNA sequence.
        ref_dot (str): Reference secondary structure in dot-bracket notation.
        mask_nan (np.array): Boolean mask to select missing observations.
        mask_0 (np.array): Boolean mask to select zero observations.
        T (int): Length of the RNA
        T_nan (int): Number of missing observations
        T_0 (int): Number of zeros observations
        T_continuous (np.array): Number of continuous (non-missing, non-zero) observations

    """

    def __init__(self, name, obs, seq, ref_dot):
        """Initialize attributes."""

        # Set attributes
        self.name = name  # RNA name
        self.obs = np.array(obs, dtype=globalbaz.GLOBALS["dtypes"]["obs"])  # RNA probing profile
        self.seq = seq  # RNA sequence
        self.ref_dot = ref_dot  # Reference secondary structure
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
def build_rnalib_from_files(fp_seq, fp_obs, fp_ref, min_density):
    """Reads both the sequence and the observations transcriptome-wide.
    
    Args:
        fp_seq (str): Pointer to the FASTA file containing the sequences
        fp_obs (str): Pointer to the observation file (in FASTA-like format)
        fp_ref (str): FASTA-like file of reference secondary structures in dot-bracket notation
        min_density (float): Minimum data density to consider a transcript

    Returns:
        rna_set (RNAset): Set of RNAs with both sequences and observations.

    """

    putative_rnas = {}
    rna_set = RNAset()

    # Read the observation data
    observations = file_handler.read_observations(fp_obs)

    for rna, obs in observations.items():
        putative_rnas[rna] = {"seq": "N" * len(obs),
                              "obs": obs,
                              "ref_dot": None}
    observations = None  # rm entry

    # Read sequences if provided and update the dictionary
    if fp_seq:
        fasta = file_handler.read_sequences(fp_seq)

        for rna, seq in fasta.items():
            if rna in putative_rnas.keys():
                putative_rnas[rna]["seq"] = seq
        fasta = None  # garbage collection

    # Read reference secondary structures if provided and update the dictionary
    if fp_ref:
        ref_structures = file_handler.read_reference_structures(fp_ref)

        for rna, ref_dot in ref_structures.items():
            if rna in putative_rnas.keys():
                putative_rnas[rna]["ref_dot"] = ref_dot
        ref_structures = None  # garbage collection

    # Build the set of RNAs
    initial_n_rnas = len(putative_rnas.keys())
    for rna in putative_rnas.keys():
        current_rna = RNA(name=rna,
                          obs=putative_rnas[rna]["obs"],
                          seq=putative_rnas[rna]["seq"],
                          ref_dot=putative_rnas[rna]["ref_dot"])
        rna_set.add_rna(current_rna, min_density=min_density)
        putative_rnas[rna] = None  # garbage collection

    return rna_set, initial_n_rnas


if __name__ == '__main__':
    pass
