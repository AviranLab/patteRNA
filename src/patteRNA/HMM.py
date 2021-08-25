import numpy as np


class HMM:
    def __init__(self):
        self.A = None
        self.pi = None
        self.n_params = None
        self.type = 'HMM'

    def set_params(self, config):
        params = {'A', 'pi'}
        self.__dict__.update((param, np.array(value)) for param, value in config.items() if param in params)

    def initialize(self):
        self.A = np.array(((0.7, 0.3), (0.3, 0.7)))  # Due to constraints of transition matrix, 2 params
        self.pi = np.array((0.5, 0.5))  # 1 param (pi[1] = 1 - pi[0]]
        self.n_params = 3

    @staticmethod
    def m_step(transcript):

        params = {"A": np.sum(transcript.xi, axis=2),  # Transition pseudocounts
                  "A_norm": np.sum(transcript.gamma[:, :-1], axis=1)[:, np.newaxis],  # Normalizing factor
                  "pi": transcript.gamma[:, 0],
                  "pi_norm": np.sum(transcript.gamma[:, 0])}

        return params

    def update_from_pseudocounts(self, pseudocounts):
        self.A = pseudocounts['A'] / pseudocounts['A_norm']
        self.pi = pseudocounts['pi'] / pseudocounts['pi_norm']
        # print('A', self.A)
        # print('pi', self.pi)

    def snapshot(self):
        text = ""
        text += "{}:\n{}\n".format('pi', np.array2string(self.pi, precision=2))
        text += "{}:\n{}\n".format('A', np.array2string(self.A, precision=2))
        return text

    def serialize(self):
        """
        Return a dictionary containing all of the parameters needed to describe the emission model.
        """
        return {'type': self.type,
                'A': self.A.tolist(),
                'pi': self.pi.tolist(),
                'n_params': self.n_params}

    def reset(self):
        self.A = None
        self.pi = None
