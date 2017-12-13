"""Set globals."""

import numpy as np

N_TASKS = None  # Number of processes to use
VERBOSE = None  # Verbose?
SEQ_CONSTRAINTS = None  # Sequence constraints?
OUTPUT = None  # Output directory
PATTERN = None  # Store all possible patterns dot-bracket patterns
NO_GMM = None  # Don't train the GMM?
PARS = None  # Are those PARS data?
NAN = None  # Are NaNs considered informative?
MAXTASKPERCHILD = 1000  # Number of tasks performed by a worker before recycling the process
TEST_BATCH_SIZE = 100  # Number of RNAs in each batch during scoring

# Output file names
OUTPUT_NAME = {"pattern": "score.txt",
               "gammas": "gammas.txt",
               "viterbi": "viterbi.txt",
               "training": "iterative_learning",
               "model": "trained_model.pickle",
               "fit_plot": "fit.svg",
               "logL_plot": "logL.svg"}

# Memory configs
DTYPES = {"obs": np.float32,
          "p": np.float64,
          "path": np.int8}

# Allowed pairing pairs (with N considered valid pairings)
PAIRING_TABLE = {"A": ["T", "N"],
                 "T": ["A", "G", "N"],
                 "G": ["C", "T", "N"],
                 "C": ["G", "N"],
                 "N": ["A", "T", "G", "C", "N"]}

# Allowed pairing pairs (with N considered invalid pairings)
PAIRING_TABLE_NO_N = {"A": ["T"],
                      "T": ["A", "G"],
                      "G": ["C", "T"],
                      "C": ["G"],
                      "N": []}

# User prompt answer compatibility list
USER_PROMPT = {"yes": ["y", "Y", "yes", "Yes"],
               "no": ["n", "N", "no", "No"]}

if __name__ == '__main__':
    pass
