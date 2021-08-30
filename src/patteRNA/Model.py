import os
import numpy as np
from src.patteRNA import filelib
from src.patteRNA import version
from src.patteRNA.HMM import HMM
from src.patteRNA.DOM import DOM
from src.patteRNA.GMM import GMM

model_map = {'HMM': HMM,
             'DOM': DOM,
             'GMM': GMM}


class Model:
    def __init__(self, structure_model=None, emission_model=None, reference=False):
        self.structure_model = structure_model
        self.emission_model = emission_model
        self.reference = reference
        self.BIC = np.Inf
        self.states = 2

    def initialize(self, k, data_stats):
        self.structure_model.initialize()
        self.emission_model.initialize(k, data_stats)

    def overload(self, model):
        """
        Update all fields according to an input Model.
        """
        self.__dict__.update(model.__dict__)

    def take_snapshot(self):
        snapshot = "Current model parameters\n"
        snapshot += self.structure_model.snapshot()
        snapshot += self.emission_model.snapshot()
        return snapshot

    def e_step(self, transcript):

        self.emission_model.compute_emissions(transcript, reference=self.reference)

        logl = self.forward_backward(transcript)  # Compute alpha, beta, c for transcript

        # Compute joint event probabilities - xi
        transcript.xi = np.zeros((self.states, self.states, transcript.T), dtype=float)
        for t in range(transcript.T - 1):
            for i in range(self.states):
                for j in range(self.states):
                    transcript.xi[i, j, t] = transcript.alpha[i, t] * self.structure_model.A[i, j] \
                                             * transcript.B[j, t + 1] * transcript.beta[j, t + 1]

        # Compute state posteriors - gamma
        transcript.gamma = (1 / transcript.c[np.newaxis, :]) * transcript.alpha * transcript.beta

        self.emission_model.post_process(transcript)

        return logl

    def forward_backward(self, transcript):

        transcript.logl = 0

        # Initialization
        transcript.alpha = np.zeros((2, transcript.T), dtype=float)
        transcript.c = np.zeros(transcript.T, dtype=float)

        # Handle path initiation (pi: Probability of starting as state 0 / 1)
        transcript.alpha[:, 0] = self.structure_model.pi * transcript.B[:, 0]
        transcript.c[0] = 1 / np.sum(transcript.alpha[:, 0])
        transcript.alpha[:, 0] *= transcript.c[0]
        transcript.logl += -np.log(transcript.c[0])

        # Forward pass: compute alpha
        for t in range(1, transcript.T):
            transcript.alpha[:, t] = np.dot(self.structure_model.A.T, transcript.alpha[:, t - 1]) * transcript.B[:, t]

            transcript.c[t] = 1 / np.sum(transcript.alpha[:, t])
            transcript.alpha[:, t] *= transcript.c[t]
            transcript.logl += -np.log(transcript.c[t])

        # Backward pass: compute beta
        transcript.beta = np.zeros((2, transcript.T), dtype=float)
        transcript.beta[:, -1] = 1  # Set last state as equally favorable

        for t in range(transcript.T - 1)[::-1]:
            transcript.beta[:, t] = np.dot(self.structure_model.A, (transcript.B[:, t + 1] * transcript.beta[:, t + 1]))
            transcript.beta[:, t] *= transcript.c[t]

        return transcript.logl

    def m_step(self, transcript):

        params_transcript = dict()

        params_transcript.update(self.structure_model.m_step(transcript))
        params_transcript.update(self.emission_model.m_step(transcript))

        return params_transcript

    def update_from_pseudocounts(self, partial_pseudocounts, nan=False):

        params = partial_pseudocounts[0].keys()  # Get keys (parameter names)

        pseudocounts = dict().fromkeys(params, 0)  # Initialize total counts

        # Accumulate counts from all transcripts (each transcript gives a partial pseudocount)
        for params in partial_pseudocounts:
            for param in params.keys():
                pseudocounts[param] += params[param]

        self.structure_model.update_from_pseudocounts(pseudocounts)
        self.emission_model.update_from_pseudocounts(pseudocounts, nan=nan)

        return pseudocounts['logL']  # Total logL over all transcripts

    def compute_bic(self, logl, n_obs):
        self.BIC = - 2 * logl + (self.structure_model.n_params + self.emission_model.n_params) * np.log(n_obs)

    def viterbi_decoding(self, transcript):

        pi = np.log(self.structure_model.pi)
        A = np.log(self.structure_model.A)
        B = np.log(transcript.B)

        trellis = np.zeros(transcript.B.shape, dtype=float)
        backpt = np.ones(transcript.B.shape, dtype=int) * -1  # Back pointer

        viterbi_path = {"path": np.zeros(transcript.T, dtype=int),
                        "log_p": np.zeros(transcript.T, dtype=float),
                        "p": np.zeros(transcript.T, dtype=float)}

        # t = 0
        trellis[:, 0] = pi + B[:, 0]

        # Run Viterbi
        for t in range(1, transcript.T):
            for i in range(self.states):
                p = trellis[:, t - 1] + A[:, i]
                backpt[i, t - 1] = np.argmax(p)
                trellis[i, t] = p[backpt[i, t - 1]] + B[i, t]

        # Find the most likely end state
        p = trellis[:, -1]
        i = np.argmax(p)
        viterbi_path["path"][-1] = i
        viterbi_path["log_p"][-1] = p[i]

        # Follow its backtrack
        for t in range(transcript.T - 2, -1, -1):
            viterbi_path["path"][t] = backpt[i, t]
            i = viterbi_path["path"][t]
            viterbi_path["log_p"][t] = trellis[i, t]

        # Realize the viterbi path to get probabilities at each states (independent of the previous observation).
        # This will be used to compare the optimal path to possible alternative paths.
        i = viterbi_path["path"][0]
        viterbi_path["p"][0] = self.structure_model.pi[i] * transcript.B[i, 0]
        for t in range(1, transcript.T):
            j = viterbi_path["path"][t]
            viterbi_path["p"][t] = self.structure_model.A[i, j] * transcript.B[j, t]
            i = j
        return viterbi_path['path']

    def reset(self):

        self.structure_model.reset()
        self.emission_model.reset()
        self.BIC = np.Inf

    def serialize(self):
        return {'version': version.__version__,
                'structure_model': self.structure_model.serialize(),
                'emission_model': self.emission_model.serialize(),
                'training_type': 'reference' if self.reference else 'unsupervised'}

    def save(self, output_dir):
        filelib.save_model(self.serialize(), os.path.join(output_dir, 'trained_model.json'))

    def load(self, model_config):

        self.structure_model = model_map[model_config['structure_model']['type']]()
        self.structure_model.set_params(model_config['structure_model'])
        self.emission_model = model_map[model_config['emission_model']['type']]()
        self.emission_model.set_params(model_config['emission_model'])
