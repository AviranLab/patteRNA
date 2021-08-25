import numpy as np
from scipy.stats import entropy
from .Transcript import Transcript
from . import filelib



class Dataset:
    def __init__(self, fp_observations, fp_sequences=None, fp_references=None):

        self.fp_obs = fp_observations
        self.fp_fasta = fp_sequences
        self.fp_refs = fp_references

        self.rnas = dict()
        self.stats = dict()

    def load_rnas(self, log_flag=False):

        observations_dict = filelib.parse_observations(self.fp_obs)
        observations_rnas = set(observations_dict.keys())

        dataset_rnas = observations_rnas

        sequences_dict = None
        if self.fp_fasta:
            sequences_dict = filelib.parse_fasta(self.fp_fasta)
            sequences_rnas = set(sequences_dict.keys())

            # Cross reference input files to confirm all transcripts
            for rna in observations_rnas.difference(sequences_rnas):
                print('WARNING - No sequence found for RNA: {}'.format(rna))
                sequences_dict[rna] = ''.join(['N']*len(observations_dict[rna]))

            for rna in sequences_rnas.difference(observations_rnas):
                print('WARNING - No probing data found for RNA: {}'.format(rna))
                observations_dict[rna] = np.tile(np.nan, len(sequences_dict[rna]))

            dataset_rnas.update(sequences_rnas)

        for rna_name in dataset_rnas:
            if self.fp_fasta:
                self.rnas[rna_name] = Transcript(rna_name, sequences_dict[rna_name], observations_dict[rna_name])
            else:
                self.rnas[rna_name] = Transcript(rna_name, 'N'*len(observations_dict[rna_name]), observations_dict[rna_name])

        if log_flag:
            for rna in self.rnas:
                self.rnas[rna].log_transform()

        self.compute_stats()

    def compute_stats(self):
        """
        Parse all finite observations in the input file and compute some statistics on the data.

        These statistics are mostly used to initialize parameters of the emission model before training.
        """

        finite_obs = []
        total_obs = 0
        up_ref = 0
        p_ref = 0

        for rna in self.rnas:
            finite_obs.extend(self.rnas[rna].obs[np.isfinite(self.rnas[rna].obs)])
            total_obs += len(self.rnas[rna].obs)
            up_ref += int(np.sum(self.rnas[rna].ref == 0))
            p_ref += int(np.sum(self.rnas[rna].ref == 1))

        self.stats['quantile_basis'] = np.linspace(0, 1, 1000)
        self.stats['quantiles'] = np.quantile(finite_obs, self.stats["quantile_basis"])
        self.stats['P25'], self.stats['P75'] = np.percentile(finite_obs, (25, 75))
        self.stats['P40'], self.stats['P60'] = np.percentile(finite_obs, (40, 60))
        self.stats['n_obs'] = len(finite_obs)
        self.stats['up_ref'] = up_ref
        self.stats['p_ref'] = p_ref
        self.stats['total_obs'] = total_obs
        self.stats['continuous_variance'] = np.var(finite_obs)
        self.stats['minimum'] = np.min(finite_obs)
        self.stats['maximum'] = np.max(finite_obs)
        self.stats['finite_obs'] = finite_obs
        self.stats['histogram_bins'] = np.linspace(self.stats['minimum'], self.stats['maximum'], 20)
        self.stats['histogram'], _ = np.histogram(finite_obs,
                                               bins=self.stats['histogram_bins'],
                                               density=True)

    def spawn_training_set(self, kl_div):
        """
        Spawn a training set (smaller or equal size to overal data) based on KL divergence criteria.

        Transcripts are incrementally added to a training Dataset (high quality transcripts first) until
        the training set's KL divergence from the overall data falls below the provided threshold.
        """

        training_transcripts = []
        training_obs = []
        kl_div_set = 1.0

        # for rna in sorted(self.rnas.values(), key=lambda transcript: transcript.density, reverse=True):
        for rna in self.rnas.values():
            training_transcripts.append(rna.name)
            training_obs.extend(rna.obs[rna.mask_finite])
            training_histogram, _ = np.histogram(training_obs,
                                              bins=self.stats['histogram_bins'],
                                              density=True)
            kl_div_set = entropy(training_histogram, self.stats['histogram'])
            if kl_div_set < kl_div:
                break

        training_set = self.spawn_set(rnas=training_transcripts)
        training_set.compute_stats()

        return training_set, kl_div_set

    def pre_process(self, model, scoring=False):

        if model.emission_model.type == 'DOM':
            for rna in self.rnas:
                model.emission_model.discretize(self.rnas[rna])
        # if model.emission_model.type == 'GMM':
        #     for rna in self.rnas:
        #         model.emission_model.generate_discrete_masks(self.rnas[rna])

        if scoring:
            for rna in self.rnas.values():
                model.e_step(rna)
                rna.compute_log_B_ratios()

    def get_emissions(self, model):
        for rna in self.rnas:
            model.emission_model.compute_emissions(self.rnas[rna])

    def spawn_set(self, rnas):
        spawned_set = Dataset(fp_observations=None, fp_sequences=None, fp_references=None)
        spawned_set.rnas = {rna: self.rnas[rna] for rna in rnas}
        return spawned_set

    def spawn_reference_set(self):
        spawned_set = Dataset(fp_observations=None, fp_references=None, fp_sequences=None)
        references = [rna for rna in self.rnas if self.rnas[rna].ref is not -1]
        spawned_set.rnas = {rna: self.rnas[rna] for rna in references}
        spawned_set.compute_stats()
        return spawned_set

    def clear(self):
        self.rnas = None
        self.stats = None
