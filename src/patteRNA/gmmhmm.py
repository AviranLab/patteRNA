import numpy as np
import pickle
import logging
import multiprocessing
import os
import pygal
import warnings
import functools
from copy import deepcopy

from . import misclib
from . import rnalib
from . import globalbaz
from . import patternlib

# Initialize globals
N_TASKS = globalbaz.N_TASKS
SEQ_CONSTRAINTS = globalbaz.SEQ_CONSTRAINTS
VERBOSE = globalbaz.VERBOSE
OUTPUT = globalbaz.OUTPUT
DTYPES = globalbaz.DTYPES
NO_GMM = globalbaz.NO_GMM
PARS = globalbaz.PARS
NAN = globalbaz.NAN
MAXTASKPERCHILD = globalbaz.MAXTASKPERCHILD
OUTPUT_NAME = globalbaz.OUTPUT_NAME

LOCK = multiprocessing.Lock()  # Lock for parallel processes

# Initialize the logger
logger = logging.getLogger(__name__)


# noinspection PyTypeChecker,PyPep8Naming
class GMMHMM:
    """Full GMM-HMM model.

        Attributes:
            iter_cnt (int): EM iteration counter.
            histogram (dict): Contains bins for plotting a data histogram.
            train_set (rnalib.RNAset): Set of RNAs used for training the GMM-HMM.
            train_children (list): List of rnalib.RNA objects used for training
            test_children (list): List of rnalib.RNA objects used for testing.
            pattern (rnalib.Pattern): Pattern of interest.
            N (int): Number of states.
            states (np.array): Vector of unique possible states.
            pi (np.array): Initial probabilities.
            A (np.array): Transition probability matrix.
            logL (float): Log likelihood of the model.
            phi (np.array): Probabilities for NaNs.
            upsilon (np.array): Probabilities for 0 reactivities prior to log transform.
            K (int): Number of Gaussian components in the Gaussian Mixture model.
            w (np.array): Weights of the Gaussian components.
            w_min (float): Minimum weight of Gaussian components before dropping them.
            mu (np.array): Means of the Gaussian components.
            sigma (np.array): Variance of the Gaussian components.
            gmm_gamma (np.array): Overall state probabilities.

    """

    def __init__(self):
        """Initialize attributes."""

        # Misc
        self.iter_cnt = None
        self.histogram = None

        # Dataset
        self.train_set = None
        self.train_children = None
        self.test_children = None

        # Pattern of interest
        self.pattern = None

        # HMM states
        self.N = None
        self.states = None

        # HMM params
        self.pi = None
        self.A = None
        self.logL = None
        self.phi = None
        self.upsilon = None

        # GMM model
        self.K = None  # Nb of Gaussian components
        self.w = None  # Mixture coefficient
        self.w_min = None  # Minimum allowed mixture coefficient
        self.mu = None  # Gaussian component means
        self.sigma = None  # Gaussian component variances

        # HMM model
        self.gmm_gamma = None  # Overall state probabilities

    def import_data(self, train_set):
        """Import training data. Note that we are not spawning children GMMHMM_SingleObs at this point as the GMM
         is not yet initialized and we need to compute emission probabilities to spawn them.

        Args:
            train_set (rnalib.RNAset): Set of RNAs used for training the GMM-HMM.

        """

        self.train_set = train_set
        self.histogram = dict(self.train_set.histogram)  # Propagate the histogram bin information

    def spawn_children(self):
        """Spawn GMMHMM_SingleObs for each RNA in the training set."""

        self.train_children = []

        # Spawn a GMMHMM_SingleObs object for each RNA and initialize the emissions
        ix = 0
        while self.train_set.rnas:
            rna = self.train_set.rnas[0]
            rna_obj = GMMHMM_SingleObs(feed_attr=self.__dict__, rna=rna, ix=ix)
            rna_obj.get_b()
            self.train_children.append(rna_obj)
            self.train_set.rnas.pop(0)  # Remove that RNA from the original list (saves memory)
            ix += 1

        self.train_set.rnas = None  # Garbage collection

    def score(self, rnas, patterns, fp_pattern, is_GQ, fp_viterbi, fp_gammas):
        """Scoring phase. Parallelized with respect to RNAs.

        Args:
            rnas (list): List of RNAs used for prediction based on the trained GMM-HMM.
            patterns (list or Pattern obj): Patterns to be scored
            fp_pattern (str): Output score file
            is_GQ (bool): Are we scoring G-quadruplexes?
            fp_viterbi (str): If set then decode the Viterbi path and output to this file
            fp_gammas (str): If set then compute hidden state posteriors (gammas) and output to this file
        """

        # Set NaNs to current values of the last EM-step (i.e. no re-initialization of phi)
        global NAN
        NAN = True

        # Spawn a GMMHMM_SingleObs object for each RNA
        self.test_children = []
        ix = 0

        while rnas:
            rna = rnas[0]
            self.test_children.append(GMMHMM_SingleObs(feed_attr=self.__dict__, rna=rna, ix=ix))
            rnas.pop(0)  # Remove that RNA from the original list (saves memory)
            ix += 1

        # Run parallelized tasks across RNAs
        pool = pool_init()
        worker = functools.partial(self.score_worker,
                                   patterns=patterns,
                                   fp_pattern=fp_pattern,
                                   is_GQ=is_GQ,
                                   fp_viterbi=fp_viterbi,
                                   fp_gammas=fp_gammas)

        try:
            q = pool.imap_unordered(worker, self.test_children)
            pool.close()
            pool.join()
        except:
            pool.terminate()  # Ensures all children processes are killed if the process doesn't terminate
            raise

        for _ in q:
            pass

    @staticmethod
    def score_worker(rna, patterns, fp_pattern, is_GQ, fp_viterbi, fp_gammas):
        """Parallelized worker for the score function."""

        rna.get_b()  # Build emissions for this RNA
        rna.fwd_bkw()  # Forward-Backward pass

        if patterns is not None:
            rna.precompute_logB_ratios()  # Pre-compute single nucleotide emission ratios for pattern scoring
            if is_GQ:
                gquad_scorer(rna=rna,
                             fp=fp_pattern,
                             min_quartet=patterns[0],
                             max_quartet=patterns[1],
                             min_loop=patterns[2],
                             max_loop=patterns[3])
            else:
                pattern_scorer(rna=rna,
                               fp=fp_pattern,
                               patterns=patterns)

        if fp_viterbi is not None:
            LOCK.acquire()
            with open(fp_viterbi, "a") as f:
                rna.viterbi_decoding()  # Viterbi algorithm
                path = rna.viterbi_path["path"]
                path = [str(i) for i in path]
                f.write(">{}\n{}\n".format(rna.name, " ".join(path)))
            LOCK.release()

        if fp_gammas is not None:
            _, _, gamma_mix_sum = rna.E_step()  # Estimation step to get state posterior probabilities

            LOCK.acquire()
            with open(fp_gammas, "a") as f:
                f.write(">{}\n".format(rna.name))
                gamma_mix_sum /= np.sum(gamma_mix_sum, axis=0)[np.newaxis, :]
                for i in rna.states:
                    out = ["{:.3g}".format(g) for g in gamma_mix_sum[i, :]]
                    f.write("{}\n".format(" ".join(out)))
            LOCK.release()

        return rna

    def initialize_HMM(self, N, pi, A, phi, upsilon):
        """Initialize the HMM model parameters.

        Args:
            N (int): Number of states.
            pi (np.array): Initial probabilities.
            A (np.array): Transition probability matrix.
            phi (np.array): Probability vector for NaNs.
            upsilon (np.array): Probability vector for zeros.

        """

        # HMM params
        self.N = N
        self.states = np.arange(N)
        self.pi = np.repeat(1 / self.N, self.N) if pi is None else pi
        self.A = np.tile(1 / self.N, (self.N, self.N)) if A is None else A
        self.phi = np.repeat(np.sum(self.train_set.T_nan * (1 / self.N)) / self.train_set.T, self.N)
        if phi is not None:
            self.phi *= phi / np.sum(phi)
        self.upsilon = np.repeat(np.sum(self.train_set.T_0) / self.train_set.T, self.N)
        if upsilon is not None:
            self.upsilon *= upsilon / np.sum(upsilon)

    def initialize_GMM(self, K, mu, sigma, w, w_min):
        """Initialize the GMM model parameters.

        Args:
            K (int): Number of Gaussian components in the Gaussian Mixture model.
            mu (np.array): Means of the Gaussian components.
            sigma (np.array): Variance of the Gaussian components.
            w (np.array): Gaussian components weights.
            w_min (float): Minimum Gaussian component weight allowed.

        """

        self.K = K
        if w is None:
            self.w = np.tile(1.0, (self.N, self.K))
        else:
            self.w = np.array(w, dtype=float)
        self.w /= np.sum(self.w, axis=1).reshape([-1, 1])  # Ensures sum(weight) == 1 for each state

        if mu is None:
            if PARS:  # As PARS > 0 indicates pairing, we need to initialize the Gaussian means accordingly
                self.mu = np.array([misclib.rand_sample(self.K,
                                                        min_=self.train_set.min_obs,
                                                        max_=0),
                                    misclib.rand_sample(self.K,
                                                        min_=0,
                                                        max_=self.train_set.max_obs)],
                                   dtype=DTYPES["obs"])
            else:
                self.mu = np.array([misclib.rand_sample(self.K,
                                                        min_=self.train_set.median_obs,
                                                        max_=self.train_set.max_obs),
                                    misclib.rand_sample(self.K,
                                                        min_=self.train_set.min_obs,
                                                        max_=self.train_set.median_obs)],
                                   dtype=DTYPES["obs"])
        else:
            self.mu = mu
        self.sigma = np.tile(self.train_set.stdev_obs, (self.N, self.K)) if sigma is None else sigma

        self.w_min = w_min

    def dump(self, fp):
        """Save the trained model to a .pickle file."""
        save_dict = misclib.selective_dict_deepcopy(self.__dict__, included_keys=None,
                                                    exluded_keys=["train_set", "train_children"])

        with open(fp, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, fp):
        """Load a pickled trained model."""

        with open(fp, "rb") as f:
            pickled_dict = pickle.load(f)

        # Assign loaded attributes
        misclib.kwargs2attr_deep(self, pickled_dict)

    def take_snapshot(self, stdout=False, fp_fit=None, fp_logl=None):
        """Prints to logger a current snapshot of the model and generates output summary graphs.

        Args:
            stdout: Print current model state to logger?
            fp_fit: Plot fitting?
            fp_logl: Plot logL curve?

        """

        if self.gmm_gamma is None:
            p_st = np.repeat(1 / self.N, self.N)
        else:
            p_st = self.gmm_gamma

        # Print the current model state to the logger
        if stdout:
            logger.debug("\n"
                         "pi: \n{}\n"
                         "A: \n{}\n"
                         "w: \n{}\n"
                         "mu: \n{}\n"
                         "sigma: \n{}\n"
                         "phi: \n{}\n"
                         "upsilon: \n{}\n"
                         "P(States|y): \n{}\n"
                         "\n".format(self.pi,
                                     self.A,
                                     self.w,
                                     self.mu,
                                     self.sigma,
                                     self.phi,
                                     self.upsilon,
                                     p_st))

        # Plot the current fit
        if fp_fit is not None:

            # Define graph params
            palette = ["#909090"]  # Data histogram (grey)
            palette += ["#000000"]  # Complete GMM (black)
            palette += ["#ff0500"]  # State 0 (red)
            palette += ["#323299"]  # State 1 (blue)
            palette += self.K * ["#a3413f"]  # State 0 mixture components (redish)
            palette += self.K * ["#657fff"]  # State 1 mixture components (blueish)

            # noinspection PyUnresolvedReferences
            custom_style = pygal.style.Style(colors=palette,
                                             legend_font_size=10)
            curr_plt = pygal.Line(title="Iteration #{}".format(self.iter_cnt),
                                  x_title="Observation",
                                  y_title="Density",
                                  x_labels_major_every=int(np.ceil(self.histogram["n"] / 30)),
                                  show_minor_x_labels=False,
                                  x_label_rotation=60,
                                  style=custom_style,
                                  range=[0, np.max(self.histogram["dens"])],
                                  show_dots=False)
            curr_plt.x_labels = [str(np.round(v, decimals=2)) for v in self.histogram["bins"]]

            # Pre-compute GMM curves
            y_components = np.tile(0.0, (self.N, self.K, self.histogram["n"]))

            for i in self.states:
                for m in range(self.K):
                    if self.w[i, m] != 0:
                        y_components[i, m, :], self.w[i, m] = wnormpdf(self.histogram["bins"],
                                                                       self.mu[i, m],
                                                                       self.sigma[i, m],
                                                                       self.w[i, m], 0)
                y_components[i, :, :] *= p_st[i]
            y_states = np.sum(y_components, axis=1)
            y_gmm = np.sum(y_states, axis=0)

            # Make plot
            # Data histogram
            curr_plt.add("Data", self.histogram["dens"], fill=True)
            # Full GMM
            curr_plt.add("GMM", y_gmm, stroke_style={"width": 5})
            # Per state
            for i in self.states:
                curr_plt.add("i={}".format(i), y_states[i, :], stroke_style={"width": 3})
            # Per component
            if self.K > 12:
                curr_plt.show_legend = False

            for i in self.states:
                for m in range(self.K):
                    curr_plt.add("i={} k={}".format(i, m), y_components[i, m, :],
                                 stroke_style={"width": 2, "dasharray": "2, 2"})

            curr_plt.render_to_file(fp_fit)

        # Plot the logL curve
        if fp_logl is not None:
            curr_plt = pygal.Line(title="EM algorithm",
                                  x_title="# of iterations",
                                  x_labels_major_every=int(np.ceil(self.iter_cnt / 20)),
                                  x_label_rotation=60,
                                  show_minor_x_labels=False,
                                  y_title="log likelihood",
                                  show_legend=False)
            curr_plt.x_labels = map(str, np.arange(self.iter_cnt) + 1)
            curr_plt.add("logL", self.logL)
            curr_plt.render_to_file(fp_logl)

    # def drop_w(self):
    #     """Drop weights and associated GMM params if at least one w reached 0 in both states."""
    #
    #     w0 = self.w == 0
    #     w_tester = np.sum(w0, axis=1)
    #
    #     # If both states have at least 1 w=0
    #     if np.all(w_tester > 0):
    #         n_to_drop = int(np.min(w_tester))
    #         w_mask = np.tile(True, self.w.shape)  # Selection mask
    #         for i in self.states:
    #             w0_to_remove = np.where(w0[i, :])[0][0:n_to_drop]
    #             w_mask[i, w0_to_remove] = False  # Drop those value from the selection mask
    #
    #         # Use the selection mask to update GMM params matrices
    #         self.K -= n_to_drop
    #         self.mu = self.mu[w_mask].reshape([self.N, -1])
    #         self.sigma = self.sigma[w_mask].reshape([self.N, -1])
    #         self.w = self.w[w_mask].reshape([self.N, -1])
    #
    #         logger.info("Removed {} Gaussian components per state. New K={}".format(n_to_drop, self.K))

    def EM(self):
        """Baum-Welch EM algorithm with updates of the model's parameters.

        The EM step is multi-process parallelized for each RNAs.

        Returns:
            logL (float): Log likelihood of the model.

        """

        logL = []
        mu_bkp = None
        sigma_bkp = None
        w_bkp = None

        if NO_GMM:
            # Store a copy of the GMM parameters that will NOT be updated
            mu_bkp = deepcopy(self.mu)
            sigma_bkp = deepcopy(self.sigma)
            w_bkp = deepcopy(self.w)

        # Matrices/vectors holding parameter values across the entire dataset
        self.phi = np.zeros(self.N, dtype=DTYPES["p"])
        self.upsilon = np.zeros(self.N, dtype=DTYPES["p"])
        phi_upsilon_norm = 0.0
        self.A = np.tile(0.0, (self.N, self.N))
        A_norm = np.tile(0.0, (self.N, self.N))
        self.mu = np.tile(0.0, (self.N, self.K))
        self.sigma = np.tile(0.0, (self.N, self.K))
        mu_sigma_norm = np.tile(0.0, (self.N, self.K))
        self.w = np.tile(0.0, (self.N, self.K))
        w_norm = np.tile(0.0, (self.N, self.K))
        self.pi = np.zeros(self.N, dtype=DTYPES["p"])
        pi_norm = 0.0
        self.gmm_gamma = np.tile(np.nan, (self.N, self.train_set.n_rna))

        # Run the parallelized EM steps on each RNA of the training set
        pool = pool_init()
        try:
            q = pool.imap_unordered(self.EM_worker, self.train_children)
            pool.close()
            pool.join()
        except:
            pool.terminate()  # Ensures all children processes are killed if the process doesn't terminate
            raise

        # Run parallelized tasks
        for args_out in q:
            logL.append(args_out["logL"])

            # Sum parameters over observation vectors
            self.phi += args_out["phi"]
            self.upsilon += args_out["upsilon"]
            phi_upsilon_norm += args_out["phi_upsilon_norm"]
            self.A += args_out["A"]
            A_norm += args_out["A_norm"]
            self.mu += args_out["mu"]
            self.sigma += args_out["sigma"]
            mu_sigma_norm += args_out["mu_sigma_norm"]
            self.w += args_out["w"]
            w_norm += args_out["w_norm"]
            self.pi += args_out["pi"]
            pi_norm += args_out["pi_norm"]
            self.gmm_gamma[:, args_out["ix"]] = args_out["gmm_gamma"]

        # Handle w = 0. Implies that the nominator = 0 as well so we can just divide by 1 and we will get 0.
        mu_sigma_norm[mu_sigma_norm == 0] = 1
        w_norm[w_norm == 0] = 1

        # Normalize re-estimated parameters
        self.phi /= phi_upsilon_norm
        self.upsilon /= phi_upsilon_norm
        self.A /= A_norm
        self.mu /= mu_sigma_norm
        self.sigma /= mu_sigma_norm
        self.w /= w_norm
        self.pi /= pi_norm
        self.gmm_gamma = np.sum(self.gmm_gamma, axis=1) / np.sum(self.gmm_gamma, axis=(0, 1))

        if NO_GMM:
            # Reinstate the GMM parameters saved prior to the EM step
            self.mu = mu_bkp
            self.sigma = sigma_bkp
            self.w = w_bkp

        # Drop weights if at least 1 reached 0 in both states
        # self.drop_w()

        # Update parameters and probabilities for each RNAs after the training epoch
        for rna in self.train_children:
            rna.update_parameters(K=self.K,
                                  pi=self.pi,
                                  A=self.A,
                                  phi=self.phi,
                                  upsilon=self.upsilon,
                                  mu=self.mu,
                                  sigma=self.sigma,
                                  w=self.w)
            rna.get_b()

        return np.sum(logL)

    @staticmethod
    def EM_worker(rna):
        """Worker for the Baum-Welch expectation maximization algorithm."""

        ix = rna.ix

        logL = rna.fwd_bkw()  # Forward-Backward pass
        xi, gamma, gamma_mix_sum = rna.E_step()  # E-step
        # M-step
        params = rna.M_step(xi, gamma, gamma_mix_sum)
        output_dict = params

        # Add additional entries
        output_dict["ix"] = ix
        output_dict["logL"] = logL

        return output_dict

    def train(self, max_iter, epsilon):
        """Train the GMM-HMM using the Baum-Welch EM algorithm until convergence."""

        self.logL = []
        self.iter_cnt = 0
        did_converge = False

        # Spawn train set children
        self.spawn_children()

        # Print initial fit
        plot_dir = os.path.join(OUTPUT, OUTPUT_NAME["training"], "")
        misclib.make_dir(plot_dir)
        self.take_snapshot(stdout=VERBOSE,
                           fp_fit=os.path.join(plot_dir, "iter_{0:03d}.svg".format(self.iter_cnt)))

        # Iteration across the EM algorithm until convergence or maximum iterations
        while ~did_converge & (self.iter_cnt < max_iter):
            self.iter_cnt += 1
            curr_logL = self.EM()  # EM step
            self.logL.append(curr_logL)

            # Check if the model likelihood converged
            if self.iter_cnt >= 5:
                delta = np.diff(self.logL[-2:])
                delta = np.absolute(delta)
                convergence_criterion = np.absolute(epsilon * curr_logL)
                if delta <= convergence_criterion:
                    did_converge = True

            # Log the current model's log likelihood
            logger.info("iter #{} : logL {}".format(self.iter_cnt, np.round(curr_logL, 2)))

            self.take_snapshot(stdout=VERBOSE,
                               fp_fit=os.path.join(plot_dir, "iter_{0:03d}.svg".format(self.iter_cnt)))

        # Check if the likelihood converged
        if not did_converge:
            logger.warning("patteRNA did not converge within {} iterations.\n"
                           "Last 5 logL -> {}".format(max_iter,
                                                      np.round(self.logL[-5:], 2)))


# noinspection PyPep8Naming
class GMMHMM_SingleObs:
    """GMM-HMM model specific to a single observation vector.

        Attributes:
            N (int): Number of states.
            states (np.array): Vector of unique possible states.
            pi (np.array): Initial probabilities.
            A (np.array): Transition probability matrix.
            logL (float): Log likelihood of the model.
            phi (np.array): Probabilities for NaNs.
            upsilon (np.array): Probabilities for 0 reactivities prior to log transform.
            K (int): Number of Gaussian components in the Gaussian Mixture model.
            w (np.array): Weights of the Gaussian components.
            w_min (float): Minimum weight of Gaussian components before dropping them.
            mu (np.array): Means of the Gaussian components.
            sigma (np.array): Variance of the Gaussian components.
            name (str): RNA name.
            obs (np.array): Vector of observations.
            original_obs (np.array): Vector of observations before transformations.
            seq (np.array): RNA sequence.
            mask_nan (np.array): Mask for NaNs.
            mask_0 (np.array): Mask for 0s.
            T (int): RNA length.
            B (np.array): Emission probabilities.
            B_mixture (np.array): Emission probabilities per single Gaussian component.
            alpha (np.array): Forward pass probabilities.
            beta (np.array): Backward pass probabilities.
            c (np.array): Scaling factor.
            viterbi_path (dict): Viterbi path.
            nuc_logB_ratios: Log ratio of emissions at each nucletotides and for each states
            ix (int): Index for the RNA.
            states_inv (np.array): Inverted vector of unique possible states.

        """

    def __init__(self, feed_attr, rna, ix):
        """Inherit some GMMHMM class attributes and initialize single observation-specific attributes."""

        # Inherited attributes from GMMHMM
        self.N = None
        self.states = None
        self.pi = None
        self.A = None
        self.logL = None
        self.phi = None
        self.upsilon = None
        self.K = None
        self.w = None
        self.w_min = None
        self.mu = None
        self.sigma = None

        misclib.kwargs2attr_deep(self, feed_attr)  # Inherit objects

        # New attributes
        self.name = None
        self.obs = None
        self.original_obs = None
        self.seq = None
        self.mask_nan = None
        self.mask_0 = None
        self.T = None
        self.B = None
        self.B_mixture = None
        self.alpha = None
        self.beta = None
        self.c = None
        self.viterbi_path = None
        self.nuc_logB_ratios = None
        self.states_inv = self.states[::-1]

        # Add an index to the rna
        self.ix = ix

        misclib.kwargs2attr_deep(self, rna.__dict__)

    def update_parameters(self, K, pi, A, phi, upsilon, mu, sigma, w):
        """Update GMM-HMM parameters."""

        self.K = deepcopy(K)
        self.pi = deepcopy(pi)
        self.A = deepcopy(A)
        self.phi = deepcopy(phi)
        self.upsilon = deepcopy(upsilon)
        self.mu = deepcopy(mu)
        self.sigma = deepcopy(sigma)
        self.w = deepcopy(w)

    def get_b(self):
        """Compute the emission probabilities using a GMM model.

        Compute both the full emission probability matrix (B) and emission probabilities for each
        Gaussian component (B_mixture)

        """

        self.B_mixture = np.zeros([self.N, self.T, self.K], dtype=DTYPES["p"])

        for i in self.states:
            for m in range(self.K):
                if self.w[i, m] == 0:
                    r = np.zeros(self.T)
                else:
                    r, self.w[i, m] = wnormpdf(self.obs, self.mu[i, m], self.sigma[i, m], self.w[i, m], self.w_min)
                    r *= (1 - self.phi[i] - self.upsilon[i])

                self.B_mixture[i, :, m] = r

            # Add NaNs and Zeros probabilities
            if NAN:
                self.B_mixture[i, self.mask_nan, :] = self.phi[i] * self.w[i, :]
            else:
                # Assumes that NaNs represent "no information" - i.e. uniform distribution across states
                self.B_mixture[i, self.mask_nan, :] = (1 / self.N) * self.w[i, :]

            if PARS:
                # if this is a PARS dataset then 0 means "no information" - i.e. uniform distribution across states
                self.B_mixture[i, self.mask_0, :] = (1 / self.N) * self.w[i, :]
            else:
                self.B_mixture[i, self.mask_0, :] = self.upsilon[i] * self.w[i, :]

        self.B = np.sum(self.B_mixture, axis=2)

    def fwd_bkw(self):
        """Forward-Backward algorithm.
        Combine the forward algorithm (alpha-pass) with the backward algorithm (beta-pass) to compute
        alpha, beta and the log likelihood of the observation vector given the model parameters.

        alpha is scaled at each t using a constant c such that the sum(alpha[:, t]) == 1. beta is scaled using c
        computed for alpha so it doesn't usually sum to 1 at time t.

        Returns:
            alpha (np.array): alpha[i,t] is the scaled probability of state i at time t given previous observations.
            beta (np.array): beta[i, t] is the scaled probability of state i at time t given future observations.
            c (np.array): c[i,t] is the scaling factor used at each time step t.
            logL: log likelihood of the observation vector given the model parameters.

        """

        # Forward pass
        self.alpha = np.zeros(self.B.shape, dtype=DTYPES["p"])
        self.c = np.zeros(self.T, dtype=DTYPES["p"])
        self.logL = 0

        for t in range(self.T):
            if t == 0:
                self.alpha[:, t] = self.pi * self.B[:, t]
            else:
                self.alpha[:, t] = np.dot(self.A.T, self.alpha[:, t - 1]) * self.B[:, t]

            self.c[t] = 1 / np.sum(self.alpha[:, t])
            self.alpha[:, t] *= self.c[t]
            self.logL += -np.log(self.c[t])

        # Backward pass
        self.beta = np.zeros(self.B.shape, dtype=DTYPES["p"])
        self.beta[:, -1] = 1

        # noinspection PyTypeChecker
        for t in range(self.T - 1)[::-1]:
            self.beta[:, t] = np.dot(self.A, (self.B[:, t + 1] * self.beta[:, t + 1]))
            self.beta[:, t] *= self.c[t]

        return self.logL

    def E_step(self):
        """Estimation step (E-step).

        Scaled alpha and beta from the forward-backward algorithm are used to compute Xi and Gamma estimates.

        Returns:
            xi (np.array): N x N x T-1 array of the joint probability of state i at time t and state j at time t+1
            gamma (np.array): N x K x T array of posteriors - P(state | y, lambda)
            gamma_mix_sum (np.array): Gamma summed over Gaussian components. Shape is N x T.

        """

        # Compute joint event probabilities - Xi
        xi = np.zeros([self.N, self.N, self.T - 1], dtype=DTYPES["p"])
        for t in range(self.T - 1):
            for i in self.states:
                for j in self.states:
                    xi[i, j, t] = self.alpha[i, t] * self.A[i, j] * self.B[j, t + 1] * self.beta[j, t + 1]

        # Compute state posteriors - Gamma (and gammas for each Gaussian component)
        gamma_mix_sum = np.zeros([self.N, self.T], dtype=DTYPES["p"])
        gamma = np.zeros([self.N, self.K, self.T], dtype=DTYPES["p"])

        for i in self.states:
            for t in range(self.T):
                gamma_mix_sum[i, t] = (1 / self.c[t]) * self.alpha[i, t] * self.beta[i, t]
                for m in range(self.K):
                    if self.w[i, m] == 0:
                        mixture_ratio = 0
                    else:
                        mixture_ratio = self.B_mixture[i, t, m] / self.B[i, t]
                    gamma[i, m, t] = gamma_mix_sum[i, t] * mixture_ratio

        return xi, gamma, gamma_mix_sum

    def M_step(self, xi, gamma, gamma_mix_sum):
        """Maximization step (M-step).

        Maximize the likelihood of all GMM-HMM parameters.

        """

        params = {}

        # Exclude observations not emitted from the GMM
        gamma_nan = np.array(gamma)
        gamma_nan[:, :, (self.mask_nan | self.mask_0)] = np.nan

        # Pre-compute sums
        # Sum gammas over time steps
        gamma_nan_t_sum = np.nansum(gamma_nan, axis=2)
        params["gmm_gamma"] = np.nansum(gamma_nan_t_sum, axis=1)

        # Re-estimate P(NaNs) and P(0) - Phi and Upsilon
        params["phi"] = np.sum(self.mask_nan[np.newaxis, :] * gamma_mix_sum, axis=1)
        params["upsilon"] = np.sum(self.mask_0[np.newaxis, :] * gamma_mix_sum, axis=1)
        params["phi_upsilon_norm"] = np.sum(gamma_mix_sum, axis=(0, 1))

        # Re-estimate the numerator and denominator of A
        params["A"] = np.sum(xi, axis=2)  # Sum xi over time steps up to T-1
        params["A_norm"] = np.sum(gamma_mix_sum[:, :-1], axis=1)[:, np.newaxis]  # Sum gamma over time steps up to T-1

        # Re-estimate initial probabilities - Pi
        params["pi"] = gamma_mix_sum[:, 0]
        params["pi_norm"] = np.sum(gamma_mix_sum[:, 0])

        # Re-estimate Gaussian means - Mu
        params["mu"] = np.nansum(gamma_nan * self.obs[np.newaxis, np.newaxis, :], axis=2)
        params["mu_sigma_norm"] = np.array(gamma_nan_t_sum)

        # Re-estimate variances - Sigma
        sq_residual = (self.obs[np.newaxis, np.newaxis, :] - self.mu[:, :, np.newaxis]) ** 2
        params["sigma"] = np.nansum(gamma_nan * sq_residual, axis=2)

        # Re-estimate the weight of the Gaussian mixture components - w
        params["w"] = np.array(gamma_nan_t_sum)  # Excluding observations not emitted from the GMM
        params["w_norm"] = np.sum(gamma_nan_t_sum, axis=1)[:, np.newaxis]

        return params

    def viterbi_decoding(self):
        """Decode the most likely sequence of states given the observations using the Viterbi algorithm.

        The sum of log is used at each time steps instead of the product to ensure no numerical overflow.

        Returns:
            viterbi_path["path"]: Most likely sequence of states, i.e. optimal path.
            viterbi_path["log_p"]: Log probability of the optimal path at each step.
            viterbi_path["p"]: Probability at each step given the previous step only. This will be used later to
            score a predetermined path against the optimal path.

        """

        pi = elog(self.pi)
        A = elog(self.A)
        B = elog(self.B)

        trellis = np.zeros(self.B.shape, dtype=DTYPES["p"])
        backpt = np.ones(self.B.shape, dtype=DTYPES["path"]) * -1  # Back pointer

        self.viterbi_path = {"path": np.zeros(self.T, dtype=DTYPES["path"]),
                             "log_p": np.zeros(self.T, dtype=DTYPES["p"]),
                             "p": np.zeros(self.T, dtype=DTYPES["p"])}

        # t = 0
        trellis[:, 0] = pi + B[:, 0]

        # Run Viterbi
        for t in range(1, self.T):
            for i in self.states:
                p = trellis[:, t - 1] + A[:, i]
                backpt[i, t - 1] = np.argmax(p)
                trellis[i, t] = p[backpt[i, t - 1]] + B[i, t]

        # Find the most likely end state
        p = trellis[:, -1]
        i = np.argmax(p)
        self.viterbi_path["path"][-1] = i
        self.viterbi_path["log_p"][-1] = p[i]

        # Follow its backtrack
        for t in range(self.T - 2, -1, -1):
            self.viterbi_path["path"][t] = backpt[i, t]
            i = self.viterbi_path["path"][t]
            self.viterbi_path["log_p"][t] = trellis[i, t]

        # Realize the viterbi path to get probabilities at each states (independent of the previous observation).
        # This will be used to compare the optimal path to possible alternative paths.
        i = self.viterbi_path["path"][0]
        self.viterbi_path["p"][0] = self.pi[i] * self.B[i, 0]
        for t in range(1, self.T):
            j = self.viterbi_path["path"][t]
            self.viterbi_path["p"][t] = self.A[i, j] * self.B[j, t]
            i = j

    def precompute_logB_ratios(self):
        """Pre-compute log emission P ratios at each nucleotides and for each states."""

        self.nuc_logB_ratios = np.tile(np.nan, self.B.shape)
        self.nuc_logB_ratios[0, :] = np.log(self.B[0, :] / self.B[1, :])
        self.nuc_logB_ratios[1, :] = np.log(self.B[1, :] / self.B[0, :])

    def score_path(self, path):
        """Score a defined path.

        Scoring is done by taking the log joint probability ratio between a path and its inverse path. The joint
        probability refers to the probability of path (z) and the data (y) given the model (theta), i.e. P(z,y|theta).

        Args:
            path (dict): Contains the path's "start", "end" and the encoding in numerical states "path".

        Returns:
            score (float): Log score for the entire path.

        """

        score = 0
        t0 = path["start"]
        T = path["end"]

        # Consider the P of starting the path
        i = path["path"][t0]
        i_inv = self.states_inv[i]  # Inverse path
        score += np.log(self.alpha[i, t0] / self.alpha[i_inv, t0])

        # Sum the path log joint probability ratio
        for t in range(t0 + 1, T):
            j = path["path"][t]
            j_inv = self.states_inv[j]

            score += self.nuc_logB_ratios[j, t]  # Pre-computed emission ratios
            score += np.log(self.A[i, j] / self.A[i_inv, j_inv])  # Transitions

            i = j
            i_inv = j_inv

        # Consider the P of ending the path
        score += np.log(self.beta[i, T - 1] / self.beta[i_inv, T - 1])

        return score


def global_config(n_tasks, verbose, seq_constraints, output, no_gmm, pars, nan_flag):
    """Configure GLOBAL parameters."""

    # System config
    n_available_cpus = multiprocessing.cpu_count()  # Number of CPUs in the system
    global N_TASKS
    N_TASKS = n_available_cpus if n_tasks <= 0 else n_tasks  # Ensure a non negative number of CPUs
    global VERBOSE
    VERBOSE = verbose
    global SEQ_CONSTRAINTS
    SEQ_CONSTRAINTS = seq_constraints
    global OUTPUT
    OUTPUT = output
    global NO_GMM
    NO_GMM = no_gmm
    global PARS
    PARS = pars
    global NAN
    NAN = nan_flag

    logger.info("Running patteRNA using {} parallel processes.".format(N_TASKS))


def elog(x):
    """"Smart log implementation.

    Returns -inf for all values <= 0.

    Args:
        x (np.array): input vector.

    Returns:
        y (np.array): log transformed vector.

    """

    x = np.array(x, dtype=DTYPES["p"])

    if x.size == 1:
        if x <= 0:
            y = -np.inf
        else:
            y = np.log(x)
    else:
        valid_mask = x > 0
        y = np.empty_like(x)
        y[:] = -np.inf
        y[valid_mask] = np.log(x[valid_mask])

    return y


def path_repo2scores(fp, rna, path_repo):
    """Score all paths in a repository and write to output"""
    LOCK.acquire()
    with open(fp, "a") as f:
        for curr_path in path_repo:
            score = rna.score_path(curr_path)

            path_iv = range(curr_path["start"], curr_path["end"])
            path = [str(i) for i in curr_path["path"][path_iv]]

            f.write("{} {:d} {:d} {:.3g} {} {}\n".format(rna.name,
                                                         curr_path["start"],
                                                         curr_path["end"],
                                                         score,
                                                         "".join(path),
                                                         "".join(np.array(list(rna.seq))[path_iv])))
    LOCK.release()


def write_score_header(fp):
    """Write the output scoring file header."""

    with open(fp, "w") as f:
        header = "{} {} {} {} {} {}\n".format("transcript",
                                              "start",
                                              "end",
                                              "score",
                                              "path",
                                              "seq")
        f.write(header)


def wnormpdf(x, mean=0, var=1, w=1, w_min=0):
    """Weighted Normal PDF.

    Add a small value (1e-20) if likelihood=0 because a PDF should never generate a "true" 0.

    Args:
        x: Vector of input values to compute the density at.
        mean: Mean.
        var: Variance.
        w: Weight.
        w_min: Minimum weight allowed.

    Returns:
        y: Likelihoods.
        w: Updated weight.

    """

    stdev = np.sqrt(var)

    # Set w to 0 if either sigma or w reach threshold values.
    if (stdev < 0) | (w <= w_min):
        y = x
        w = 0
    else:
        # noinspection PyTypeChecker
        u = np.array((x - mean) / stdev, dtype=DTYPES["p"])
        y = np.exp(-u * u / 2) / (np.sqrt(2 * np.pi) * stdev)
        # noinspection PyTypeChecker
        if np.any(y == 0):
            y += 1e-20

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y *= w

    return y, w


def pool_init():
    """Initialize a pool for multiprocesses with a embedded lock."""
    pool = multiprocessing.Pool(processes=N_TASKS,
                                maxtasksperchild=MAXTASKPERCHILD)
    return pool


def gquad_scorer(rna, fp, min_quartet=1, max_quartet=5, min_loop=1, max_loop=20):
    """Score G_quadruplexes.

    Args:
        rna (GMMHMM_SingleObs obj): Current RNA to be processed
        fp (str): Pointer to the output file holding scores.
        min_quartet (int): Minimum number of quartets allowed.
        max_quartet (int): Maximum number of quartets allowed.
        min_loop (int): Minimum length of loops.
        max_loop (int): Maximum length of loops.

    """

    # Search for G-quadruplexes
    path_repo = patternlib.g_quadruplex_finder(seq=''.join(rna.seq),
                                               min_quartet=min_quartet,
                                               max_quartet=max_quartet,
                                               min_loop=min_loop,
                                               max_loop=max_loop)
    # Realize paths (can be empty)
    path_repo2scores(fp, rna, path_repo)

    return None


def pattern_scorer(rna, fp, patterns):
    """Score motif patterns.

    Args:
        rna (GMMHMM_SingleObs obj): Current RNA to be processed
        fp (str): Pointer to the output file holding scores.
        patterns (Patterns): All possible patterns given the dot-bracket RegEx.

    """

    path_repo = []  # Initialize a path repository
    seq = list(rna.seq)  # To avoid changing type to list at each pattern

    # Loop across putative patterns
    for pattern in patterns:
        # Loop across nucleotide starting positions
        for nuc_start in range(rna.T):
            nuc_stop = nuc_start + pattern.n
            pattern_flag = False  # Switch to determine if a path needs to be realized
            # Ensure we are still within the RNA length
            if nuc_stop <= rna.T:
                # Filter for patterns allowed by the RNA sequence if sequence constraint was used
                if SEQ_CONSTRAINTS:
                    if pattern.ensure_pairing(seq=seq[nuc_start:nuc_stop]):
                        pattern_flag = True
                else:
                    pattern_flag = True

                if pattern_flag:
                    # Build the path and store in the repository
                    # noinspection PyTypeChecker
                    path = np.repeat(-1, rna.T)
                    path[nuc_start:nuc_stop] = pattern.path
                    path_repo.append({"start": nuc_start,
                                      "end": nuc_stop,
                                      "path": path})

        # Realize all possible paths (can be empty) - Writing this after each patterns to avoid memory leaks
        path_repo2scores(fp, rna, path_repo)
        path_repo = []  # Flush the path repository

    return None


if __name__ == '__main__':
    pass
