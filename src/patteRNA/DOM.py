import numpy as np


class DOM:
    def __init__(self):
        self.k = None
        self.n_bins = None
        self.edges = None
        self.classes = None
        self.chi = None
        self.type = 'DOM'
        self.n_params = None

    def set_params(self, config):
        params = {'n_bins', 'edges', 'classes', 'chi', 'n_params'}
        self.__dict__.update((param, np.array(value)) for param, value in config.items() if param in params)

    def initialize(self, k, stats):

        k = k + 5

        qbin_sizes = 0.5 / k  # Quantile sizes
        qbin_edges = 0.25 + qbin_sizes*np.arange(0, k+1)  # Edge locations (in quantile terms)

        bin_edges = np.interp(qbin_edges, stats['quantile_basis'], stats['quantiles'])

        self.k = k
        self.n_bins = k + 2
        self.classes = list(range(1, self.n_bins + 2))
        self.edges = [-np.Inf] + [edge for edge in bin_edges] + [np.Inf]
        self.chi = np.zeros((2, self.n_bins + 1))

        dist = np.linspace(2, 1, self.n_bins)  # Bins captured by observations
        scaled_dist = 0.9 * dist / dist.sum()  # Scaling by 0.9 to allow for 0.1 emission prob of NaN
        self.chi[1, :-1] = scaled_dist  # Paired emission dist
        self.chi[0, :-1] = np.flip(scaled_dist)  # Unpaired emission dist
        self.chi[1, -1] = 0.1  # NaN observations
        self.chi[0, -1] = 0.1  # NaN observations

        self.n_params = 2*(self.n_bins-2)

    def discretize(self, transcript):
        """
        Compute the DOM class for all nucleotides in an RNA and save the resulting vector
        to Transcript.obs_dom.
        """

        # np.searchsorted is identical to the digitize call here, but marginally faster (especially
        # for a large number of bins and/or a large number of RNAs).
        # transcript.obs_dom = np.digitize(transcript.obs, bins=self.edges)

        transcript.obs_dom = np.searchsorted(self.edges, transcript.obs, side='left')

    def compute_emissions(self, transcript, reference=False):
        """
        Compute emission probabilities according to the discretized observation model.

        This amounts to simply accessing the correct indices of the DOM pdf matrix, chi.

        Args:
            transcript (src.patteRNA.Transcript.Transcript): Transcript to process
            reference (bool): Whether or not it's a reference transcript
        """
        if reference:
            pass
        transcript.B = self.chi[:, transcript.obs_dom-1]

    @staticmethod
    def post_process(transcript):
        pass

    def m_step(self, transcript):

        chi_0 = np.fromiter((transcript.gamma[0, transcript.obs_dom == dom_class].sum()
                             for dom_class in self.classes), float)
        chi_1 = np.fromiter((transcript.gamma[1, transcript.obs_dom == dom_class].sum()
                             for dom_class in self.classes), float)

        params = {'chi': np.vstack((chi_0, chi_1)),
                  'chi_norm': np.sum(transcript.gamma, axis=1)}

        return params

    def update_from_pseudocounts(self, pseudocounts, nan=False):
        self.chi = pseudocounts['chi'] / pseudocounts['chi_norm'][:, None]
        self.scale_chi(nan=nan)

    def scale_chi(self, nan=False):
        if nan:
            self.chi[:, :] = self.chi[:, :] / np.sum(self.chi[:, :], axis=1)[:, np.newaxis]
        else:
            self.chi[:, :-1] = 0.9 * self.chi[:, :-1] / np.sum(self.chi[:, :-1], axis=1)[:, np.newaxis]
            self.chi[:, -1] = 0.1  # NaN observations

    def snapshot(self):
        text = ""
        text += "{}:\n{}\n".format('chi', np.array2string(self.chi))
        return text

    def serialize(self):
        """
        Return a dictionary containing all of the parameters needed to describe the emission model.
        """
        return {'type': self.type,
                'n_bins': self.n_bins,
                'classes': self.classes,
                'edges': self.edges,
                'chi': self.chi.tolist(),
                'n_params': self.n_params}

    def reset(self):
        self.edges = None
        self.chi = None
        self.k = None
        self.n_bins = None
        self.classes = None
        self.n_params = None
