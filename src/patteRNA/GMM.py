import numpy as np
from scipy.stats import norm


class GMM:
    def __init__(self):
        self.k = None
        self.w = None
        self.mu = None
        self.sigma = None
        self.phi = None
        self.nu = None
        self.gmm_pdfs = None
        self.n_params = None
        self.type = 'GMM'

        self._finite_obs = None
        self._state_props = None

    def set_params(self, config):

        params = {'k', 'w', 'mu', 'sigma', 'phi', 'nu', 'n_params'}
        self.__dict__.update((param, np.array(value)) for param, value in config.items() if param in params)

        self.setup_gmm_pdfs()

    def initialize(self, k, stats):

        self.k = k

        self.w = np.tile(1 / k, (2, k))  # Uniform weights
        # Symmetrically place initial distributions around the median, one higher than the other
        self.mu = np.vstack((np.linspace(stats['P60'], stats['P75'], k), np.linspace(stats['P25'], stats['P40'], k)))
        self.sigma = np.tile(stats['continuous_variance'], (2, k))  # Use overall variance
        self.phi = np.array((0.05, 0.05))  # P(NaN)
        self.nu = np.array((0.1, 0.1))  # P(zero)

        self._finite_obs = stats['finite_obs']
        self._state_props = np.array((0.5, 0.5))

        self.setup_gmm_pdfs()
        self.n_params = 6 * k + 2

    def setup_gmm_pdfs(self):
        self.gmm_pdfs = [[], []]
        for i in range(self.k):
            self.gmm_pdfs[0].append(norm(loc=self.mu[0, i], scale=np.sqrt(self.sigma[0, i])))
            self.gmm_pdfs[1].append(norm(loc=self.mu[1, i], scale=np.sqrt(self.sigma[1, i])))

    def compute_emissions(self, transcript, reference=False):

        # Peform computations for weighted kernels
        self.wnormpdfs(transcript)

        if reference:
            return

        # Correct kernel emissions for phi and nu
        # transcript.gamma_gmm_k *= (1 - self.nu[:, np.newaxis, np.newaxis] - self.phi[:, np.newaxis, np.newaxis])

        b = np.sum(transcript.gamma_gmm_k, axis=1)  # Sum emissions of all kernels
        b *= (1 - self.nu[:, np.newaxis] - self.phi[:, np.newaxis])

        # Assign emission likelihoods for discrete cases
        b[:, transcript.mask_0] = self.nu[:, np.newaxis]
        b[:, transcript.mask_nan] = self.phi[:, np.newaxis]

        transcript.B = b

    @staticmethod
    def post_process(transcript):
        transcript.gamma_gmm_k *= transcript.gamma[:, np.newaxis, :] / \
                                  np.sum(transcript.gamma_gmm_k, axis=1)[:, np.newaxis, :]

    def m_step(self, transcript):

        params = dict()

        gamma_gmm_k_sum = np.nansum(transcript.gamma_gmm_k, axis=2)

        # Overall unpaired proportion
        params["unpaired_prop"] = np.sum(transcript.gamma, axis=1)

        # Re-estimate phi and nu
        params["phi"] = np.sum(transcript.gamma[:, transcript.mask_nan], axis=1)
        params["nu"] = np.sum(transcript.gamma[:, transcript.mask_0], axis=1)
        params["phi_nu_norm"] = np.sum(transcript.gamma, axis=1)

        # Re-estimate GMM means and variances
        params["mu"] = np.nansum(transcript.gamma_gmm_k * transcript.obs[np.newaxis, np.newaxis, :], axis=2)
        sq_residual = (transcript.obs[np.newaxis, np.newaxis, :] - self.mu[:, :, np.newaxis]) ** 2
        params["sigma"] = np.nansum(transcript.gamma_gmm_k * sq_residual, axis=2)
        params["mu_sigma_norm"] = gamma_gmm_k_sum

        # Re-estimate the weight of the Gaussian mixture components - w
        params["w"] = gamma_gmm_k_sum  # Excluding observations not emitted from the GMM
        params["w_norm"] = np.sum(gamma_gmm_k_sum, axis=1)[:, np.newaxis]

        return params

    def update_from_pseudocounts(self, pseudocounts):
        """
        Update emission model parameters given the relevant pseudocount sums over all transcripts.
        """

        self.mu = pseudocounts['mu'] / pseudocounts['mu_sigma_norm']
        self.sigma = pseudocounts['sigma'] / pseudocounts['mu_sigma_norm']
        self.w = pseudocounts['w'] / pseudocounts['w_norm']
        self.phi = pseudocounts['phi'] / pseudocounts['phi_nu_norm']
        # self.phi = np.array((0.05, 0.05))
        self.nu = pseudocounts['nu'] / pseudocounts['phi_nu_norm']
        self._state_props = pseudocounts['unpaired_prop'] / pseudocounts['unpaired_prop'].sum()
        self.setup_gmm_pdfs()  # Construct new distribution objects

    def wnormpdfs(self, transcript):
        transcript.gamma_gmm_k = np.tile(np.nan, (2, self.k, len(transcript.obs)))
        for k in range(self.k):
            transcript.gamma_gmm_k[0, k, :] = self.w[0, k, np.newaxis] * self.gmm_pdfs[0][k].pdf(transcript.obs)
            transcript.gamma_gmm_k[1, k, :] = self.w[1, k, np.newaxis] * self.gmm_pdfs[1][k].pdf(transcript.obs)

        transcript.gamma_gmm_k[:, :, transcript.mask_finite] += 1e-20  # Zero likelihoods cause problems so we add a small buffer
        transcript.gamma_gmm_k[:, :, transcript.mask_0] = np.nan  # Override zeros (would have been mapped to -Infinity)

    def snapshot(self):
        text = ""
        text += "{}:\n{}\n".format('w', np.array2string(self.w))
        text += "{}:\n{}\n".format('mu', np.array2string(self.mu))
        text += "{}:\n{}\n".format('sigma', np.array2string(self.sigma))
        text += "{}:\n{}\n".format('phi', np.array2string(self.phi))
        text += "{}:\n{}\n".format('nu', np.array2string(self.nu))
        return text

    def serialize(self):
        """
        Return a dictionary containing all of the parameters needed to describe the emission model.
        """
        return {'type': self.type,
                'k': self.k,
                'w': self.w.tolist(),
                'mu': self.mu.tolist(),
                'sigma': self.sigma.tolist(),
                'phi': self.phi.tolist(),
                'nu': self.nu.tolist(),
                'n_params': self.n_params}

    def reset(self):
        self.k = None
        self.w = None
        self.mu = None
        self.sigma = None
        self.phi = None
        self.nu = None
        self.gmm_pdfs = None
        self.n_params = None
