import numpy as np
from . import rnalib


class Transcript:
    def __init__(self, name, seq, obs):
        self.name = name
        self.seq = seq
        self.obs = obs
        self.T = len(obs)
        self.obs_dom = None
        self.ref = None

        self.alpha = None
        self.beta = None
        self.c = None
        self.B = None
        self.gamma = None
        self.gamma_gmm_k = None
        self.log_B_ratio = None

        self.mask_0 = self.obs <= 0
        self.mask_nan = np.isnan(self.obs)
        self.mask_finite = np.isfinite(self.obs)
        self.density = 1-np.sum(self.mask_nan)/self.T

        self.valid_sites = dict()
        self.nan_sites = dict()

    def log_transform(self):

        self.mask_finite = np.invert(self.mask_0 | self.mask_nan)

        self.obs[self.mask_finite] = np.log(self.obs[self.mask_finite])
        self.obs[self.mask_0] = -np.Inf

    def find_valid_sites(self, motif):
        self.valid_sites[motif] = set()
        pairing_table, _ = rnalib.compute_pairing_partners(motif)
        m = len(motif)
        for i in range(self.T - m + 1):
            if rnalib.is_valid_pairing(self.seq[i:i+m], pairing_table):
                self.valid_sites[motif].add(i)

    def find_nan_sites(self, length):
        self.nan_sites[length] = set()
        for i in range(self.T - length + 1):
            if np.all(self.mask_nan[i:i+length]):
                self.nan_sites[length].add(i)

    def compute_log_B_ratios(self):
        self.log_B_ratio = np.zeros((2, self.T), dtype=float)
        self.log_B_ratio[0, :] = np.log(self.B[0, :] / self.B[1, :])
        self.log_B_ratio[1, :] = -1 * self.log_B_ratio[0, :]
