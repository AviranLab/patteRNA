import exrex
import logging
import os
import multiprocessing
import numpy as np
from scipy.stats import genlogistic
from scipy.ndimage.filters import median_filter, uniform_filter1d
from functools import partial
from patteRNA.LBC import LBC
from patteRNA import rnalib, filelib, timelib, misclib, viennalib
from tqdm import tqdm

LOCK = multiprocessing.Lock()
logger = logging.getLogger(__name__)
clock = timelib.Clock()


class ScoringManager:
    def __init__(self, model, run_config):
        self.model = model
        self.run_config = run_config
        self.mp_tasks = run_config['n_tasks']
        self.mp_pool = None
        self.motifs = []
        self.cscore_dists = None
        self.dataset = None
        self.no_vienna = run_config['no_vienna']
        self.lbc = LBC()

        if run_config['motif'] is not None:
            self.parse_motifs()

    def parse_motifs(self):

        expression = self.run_config['motif']
        expression = expression.replace('(', r'\(')
        expression = expression.replace('.', r'\.')
        expression = expression.replace(')', r'\)')
        motifs = exrex.generate(expression)
        self.motifs = list(filter(rnalib.valid_db, motifs))

    def import_data(self, dataset):
        self.dataset = dataset

    def execute_scoring(self):

        # Compile scoring configuration parameters
        scoring_config = {'posteriors': self.run_config['posteriors'],
                          'hdsl': self.run_config['HDSL'],
                          'spp': self.run_config['SPP'],
                          'viterbi': self.run_config['viterbi'],
                          'suppress_nan': True,
                          'fp_posteriors': os.path.join(self.run_config['output'], 'posteriors.txt'),
                          'fp_scores_pre': os.path.join(self.run_config['output'], 'scores_pre'),
                          'fp_scores': os.path.join(self.run_config['output'], 'scores.txt'),
                          'fp_hdsl': os.path.join(self.run_config['output'], 'hdsl.txt'),
                          'fp_spp': os.path.join(self.run_config['output'], 'spp.txt'),
                          'fp_viterbi': os.path.join(self.run_config['output'], 'viterbi.txt'),
                          'no_cscores': self.run_config['no_cscores'],
                          'min_cscores': self.run_config['min_cscores'],
                          'batch_size': self.run_config['batch_size'],
                          'motifs': self.motifs,
                          'path': self.run_config['path'],
                          'context': self.run_config['context'],
                          'cscore_dists': None,
                          'no_vienna': self.no_vienna,
                          'energy': ~np.any([self.no_vienna,
                                             self.run_config['no_cscores'],
                                             not viennalib.vienna_imported]),
                          'lbc': self.lbc,
                          'hdsl_params': self.run_config['hdsl_params']}

        self.pool_init()  # Initialize parallelized pool

        # Prepare score distributions for c-score normalization
        if not scoring_config['no_cscores']:
            logger.info('Sampling null sites for c-score normalization')
            clock.tick()

            self.cscore_dists = dict.fromkeys(self.motifs)

            cscore_batch = self.make_cscore_batch(scoring_config['min_cscores'])
            cscore_batch.pre_process(self.model, scoring=True)

            with tqdm(total=len(self.motifs),
                      leave=False,
                      unit='motif') as pb_samples:

                try:

                    if scoring_config['path']:
                        path = np.array(list(scoring_config['path']), dtype=int)
                    else:
                        path = None

                    worker = partial(self.sample_worker, path=path, batch=cscore_batch)
                    samples_pool = self.mp_pool.imap_unordered(worker, self.motifs)

                    for (motif, samples) in samples_pool:
                        params = genlogistic.fit(samples)
                        self.cscore_dists[motif] = genlogistic(c=params[0], loc=params[1], scale=params[2])
                        pb_samples.update()

                    self.mp_pool.close()
                    self.mp_pool.join()

                except Exception:
                    self.mp_pool.terminate()
                    raise

            scoring_config['cscore_dists'] = self.cscore_dists
            logger.info(' ... done in {}'.format(misclib.seconds_to_hms(clock.tock())))

        # Begin formal scoring phase by making batches to save on memory
        batches = self.make_batches(scoring_config['batch_size'])

        n_batches = len(self.dataset.rnas) // scoring_config['batch_size'] + 1  # Number of batches

        if self.motifs:
            header = "transcript\tstart score c-score BCE MEL Prob(motif) motif path seq\n"
            with open(scoring_config['fp_scores_pre'], 'w') as f:
                f.write(header)

        logger.info("Executing scoring")
        clock.tick()

        with tqdm(total=n_batches,
                  leave=False,
                  unit='batch',
                  desc='      Overall') as pbar_batches:

            # Process batches sequentially
            for i, batch in enumerate(batches):

                self.pool_init()
                batch.pre_process(self.model)

                with tqdm(total=len(batch.rnas),
                          leave=False,
                          unit="transcript",
                          desc="Current batch") as pbar_transcripts:

                    try:
                        worker = partial(self.score_worker, model=self.model, config=scoring_config)
                        jobs = self.mp_pool.imap_unordered(worker, batch.rnas.values())

                        for _ in jobs:
                            pbar_transcripts.update()
                        self.mp_pool.close()
                        self.mp_pool.join()

                    except Exception:
                        self.mp_pool.terminate()
                        raise

                batch.clear()
                pbar_batches.update()

        # Sort score file
        if self.motifs:
            scores = filelib.read_score_file(scoring_config['fp_scores_pre'])
            if not scores:
                os.rename(scoring_config['fp_scores_pre'], scoring_config['fp_scores'])
            else:
                if scoring_config['no_cscores']:
                    filelib.write_score_file(sorted(scores, key=lambda score: score['score'], reverse=True),
                                             scoring_config['fp_scores'])
                else:
                    if scoring_config['energy']:
                        filelib.write_score_file(sorted(scores, key=lambda score: score['Prob(motif)'], reverse=True),
                                                 scoring_config['fp_scores'])
                    else:
                        filelib.write_score_file(sorted(scores, key=lambda score: score['c-score'], reverse=True),
                                                 scoring_config['fp_scores'])
                os.remove(scoring_config['fp_scores_pre'])  # Clean-up
        logger.info(' ... done in {}'.format(misclib.seconds_to_hms(clock.tock())))

    @staticmethod
    def sample_worker(motif, path, batch):

        if path is None:
            path = rnalib.dot2states(motif)

        scores = []
        for transcript in batch.rnas.values():
            scores.extend(get_null_scores(transcript, motif, path))

        return motif, scores

    @staticmethod
    def score_worker(transcript, model, config):

        model.e_step(transcript)  # Apply model to transcripts
        outputs = compute_outputs(transcript, model, config)
        with LOCK as _:
            write_outputs(outputs, config)

    def make_cscore_batch(self, min_sample_size):
        """
        Scan through RNAs in provided data and determine how many are needed to sufficiently
        estimate null distributions for c-score normalization. Return a new Dataset with just
        the RNAs to use for score sampling.

        Args:
            min_sample_size: Minimum number of samples to estimate the null score distribution for a single motif.

        Returns:
            Dataset of RNAs which is a subset of the provided data and meets the criteria needed for score sampling.

        """

        motif_samples = {motif: 0 for motif in self.motifs}
        cscore_rnas = []

        for rna in self.dataset.rnas.values():
            cscore_rnas.append(rna.name)

            for motif in self.motifs:
                null_sites = count_null_sites(rna, motif)
                motif_samples[motif] += null_sites

            if np.all([motif_samples[motif] >= min_sample_size for motif in motif_samples]):
                break  # No more sites needed

        return self.dataset.spawn_set(rnas=cscore_rnas)

    def make_batches(self, size):

        rnas = list(self.dataset.rnas.keys())
        while rnas:
            rnas_batch = rnas[:size]
            rnas[:size] = []
            yield self.dataset.spawn_set(rnas=rnas_batch)

    def pool_init(self):
        self.mp_pool = multiprocessing.Pool(processes=self.mp_tasks,
                                            maxtasksperchild=1000)


def count_null_sites(transcript, motif):
    if motif not in transcript.valid_sites.keys():
        transcript.find_valid_sites(motif)
    if motif not in transcript.nan_sites.keys():
        transcript.find_nan_sites(len(motif))
    non_null_sites = transcript.nan_sites[len(motif)] | transcript.valid_sites[motif]
    count = transcript.T - len(motif) + 1 - len(non_null_sites)
    return count


def get_null_scores(transcript, motif, path):
    # Get sites which violate sequence constraints
    invalid_sites = np.where(~np.in1d(range(transcript.T - len(motif) + 1), transcript.valid_sites[motif]))[0]
    null_scores = list(filter(lambda score: ~np.isnan(score['score']),
                              map(lambda start: score_path(transcript, start, path, motif, None, lbc=False),
                                  invalid_sites)))
    return [null_score['score'] for null_score in null_scores]


def compute_cscores(scores, dists):
    list(map(lambda score: apply_cscore(score, dists[score['dot-bracket']]), scores))


def apply_cscore(score, dist):
    pv = dist.sf(score['score'])
    if pv == 0:
        log_c = np.Inf
    elif np.isnan(pv):
        log_c = np.nan
    else:
        log_c = -np.log10(pv)
    score['c-score'] = log_c


def score_path(transcript, start, path, motif, pt, lbc=True, context=40):
    m = len(path)
    end = start + m - 1

    bce = np.nan
    mel = np.nan

    if np.all(np.isnan(transcript.obs[start:end + 1])):
        score = np.nan
    else:
        score = 0
        score += np.log(transcript.alpha[path[0], start] / transcript.alpha[1 - path[0], start])
        score += np.sum((2 * path[1:-1] - 1) * transcript.log_B_ratio[1, start + 1:end])
        score += np.log(transcript.beta[path[-1], end] / transcript.beta[1 - path[-1], end])

        if lbc:
            rstart = int(np.max((0, start - context)))
            rend = int(np.min((len(transcript.seq), end + context)))
            start_shift = start - rstart
            hcs = rnalib.compile_motif_constraints(pt[0], pt[1], start_shift)
            lmfe = viennalib.fold(transcript.seq[rstart:rend])
            lcmfe = viennalib.hc_fold(transcript.seq[rstart:rend], hcs=hcs)
            mel = lmfe - lcmfe
            bce = bce_loss(transcript.gamma[1, start:end + 1], path)

    return {'score': score,
            'c-score': None,
            'start': start,
            'transcript': transcript.name,
            'dot-bracket': motif,
            'path': "".join([str(a) for a in path]),
            'BCE': bce,
            'MEL': mel,
            'Prob(motif)': np.nan,
            'seq': transcript.seq[start:start + m]}


def bce_loss(yhat, y):
    assert len(yhat) == len(y)
    return sum(
        -yi * np.log(yhi + 1e-20) if yi == 1 else -(1 - yi) * np.log(1 - yhi + 1e-20) for yhi, yi in zip(yhat, y))


def compute_outputs(transcript, model, config):
    outputs = {'name': transcript.name,
               'viterbi': '',
               'posteriors': '',
               'spp': '',
               'scores_pre': '',
               'hdsl': ''}  # Initialize outputs dictionary

    if config['viterbi']:
        vp = model.viterbi_decoding(transcript)  # Viterbi algorithm
        outputs['viterbi'] = "> {}\n{}\n".format(transcript.name, "".join([str(i) for i in vp]))

    # Posterior pairing probabilities
    if config['posteriors']:
        transcript.gamma /= np.sum(transcript.gamma, axis=0)[np.newaxis, :]
        outputs['posteriors'] = "> {}\n{}\n".format(transcript.name,
                                                    " ".join(["{:1.3f}".format(p) for p in transcript.gamma[0, :]]))

        # Smoothed P(paired) measure --> HDSL without augmentation
        if config['spp']:
            spp_tmp = transcript.gamma[1, :]  # Raw pairing probabilities
            spp_tmp = uniform_filter1d(spp_tmp, size=5)  # Local mean
            spp = median_filter(spp_tmp, size=15)  # Local median
            outputs['spp'] = "> {}\n{}\n".format(transcript.name,
                                                 " ".join(["{:1.3f}".format(p) for p in spp]))

    if config['motifs']:

        transcript.compute_log_B_ratios()

        scores = []

        for motif in config['motifs']:
            if config['path'] is not None:
                path = np.array(list(config['path']), dtype=int)
            else:
                path = rnalib.dot2states(motif)

            pt = transcript.find_valid_sites(motif)  # Returns motif base pairing list
            scores_tmp = list(map(lambda start: score_path(transcript, start, path, motif, pt, lbc=config['energy']),
                                  transcript.valid_sites[motif]))

            if config['suppress_nan']:
                scores_tmp = list(filter(lambda s: ~np.isnan(s['score']), scores_tmp))
            if config['cscore_dists'] is not None:
                compute_cscores(scores_tmp, config['cscore_dists'])
            scores += scores_tmp

        if config['energy']:
            config['lbc'].apply_classifier(scores)
        outputs['scores_pre'] = format_scores(scores)

        # Hairpin-derived structure level measure
        if config['hdsl']:
            hdsl_tmp = transcript.gamma[1, :]  # Pairing probabilities
            for score in scores:
                # Profile augmentation with hairpin scores
                if score['c-score'] > config['hdsl_params'][1]:
                    end = score['start'] + len(score['dot-bracket'])
                    boost = config['hdsl_params'][0] * (score['c-score'] - config['hdsl_params'][1])
                    hdsl_tmp[score['start']:end] += boost
            # Clipping to [0, 1]
            hdsl_tmp[hdsl_tmp < 0] = 0
            hdsl_tmp[hdsl_tmp > 1] = 1
            # Smoothing steps
            hdsl_tmp = uniform_filter1d(hdsl_tmp, size=5)  # Local mean
            hdsl = median_filter(hdsl_tmp, size=15)  # Local median
            outputs['hdsl'] = "> {}\n{}\n".format(transcript.name, " ".join(["{:1.3f}".format(p) for p in hdsl]))

    return outputs


def format_scores(scores):
    return "".join(["{} {} {:1.2f} {:1.2f} {:1.2f} {:1.2f} {:1.3g} {} {} {}\n".format(
        score['transcript'],
        score['start'] + 1,
        score['score'],
        score['c-score'],
        score['BCE'],
        score['MEL'],
        score['Prob(motif)'],
        score['dot-bracket'],
        score['path'],
        score['seq']) for score in scores])


def write_outputs(outputs, config):
    output_types = ['viterbi', 'posteriors', 'spp', 'scores_pre', 'hdsl']
    for output_type in output_types:
        if outputs[output_type]:
            with open(config[f'fp_{output_type}'], 'a') as f:
                f.write(outputs[output_type])
