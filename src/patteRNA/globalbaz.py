"""Set globals."""

import numpy as np

# noinspection PyDictCreation
GLOBALS = {"n_tasks": None,  # Number of processes to use
           "verbose": None,  # Verbose?
           "seq_constraints": None,  # Sequence constraints?
           "output": None,  # Output directory
           "pattern": None,  # Store all possible patterns dot-bracket patterns
           "no_gmm": None,  # Don't train the GMM?
           "pars": None,  # Are those PARS data?
           "nan": None,  # Are NaNs considered informative?
           "maxtasksperchild": 1000,  # Number of tasks performed by a worker before recycling the process
           "test_batch_size": 100,  # Number of RNAs in each batch during scoring
           "h0_min_size": 100,  # Min # of points over which to estimate the Null score distribution for each motif
           "h0_max_size": 10000,  # Max # of points over which to estimate the Null score distribution for each motif
           "min_train_n": 10000,  # Minimum number of training data points
           "KL_interval": 10000  # Number of datapoints accumulated before recomputing KL divergence
           }

# Default model parameters
# noinspection PyDictCreation
GLOBALS["model"] = {"A": np.array([[0.71020019, 0.28979981],
                                   [0.19677996, 0.80322004]])}

# Accepted input observation file extensions and related experimental assays
# Currently supported assays: ["shape", "pars"]
GLOBALS["extensions"] = {".shape": "shape",
                         ".pars": "pars",
                         ".dms": "shape"}

# Output file names
GLOBALS["output_name"] = {"scores": "scores.txt",
                          "unsorted_scores": "unsorted_scores.txt",
                          "h0_partial": "h0_partial.temp",
                          "h0": "h0.temp",
                          "p_scores": "p_scores.txt",
                          "posteriors": "posteriors.txt",
                          "viterbi": "viterbi.txt",
                          "training": "iterative_learning",
                          "model": "trained_model.pickle",
                          "fit_plot": "fit.svg",
                          "logL_plot": "logL.svg"}

# Memory configs
GLOBALS["dtypes"] = {"obs": np.float32,
                     "p": np.float64,
                     "path": np.int8}

# Allowed pairing pairs (with N considered valid pairings)
GLOBALS["pairing_table"] = {"A": ["T", "N"],
                            "T": ["A", "G", "N"],
                            "G": ["C", "T", "N"],
                            "C": ["G", "N"],
                            "N": ["A", "T", "G", "C", "N"]}

# Allowed pairing pairs (with N considered invalid pairings)
GLOBALS["pairing_table_no_N"] = {"A": ["T"],
                                 "T": ["A", "G"],
                                 "G": ["C", "T"],
                                 "C": ["G"],
                                 "N": []}

# User prompt answer compatibility list
GLOBALS["user_prompt"] = {"yes": ["y", "Y", "yes", "Yes"],
                          "no": ["n", "N", "no", "No"]}

if __name__ == '__main__':
    pass
