import exrex
import logging
import os
import multiprocessing
import numpy as np
from scipy.stats import genlogistic
from scipy.ndimage.filters import median_filter, uniform_filter1d
from functools import partial
from . import rnalib, filelib, timelib, misclib
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
        self.motifs = None
        self.cscore_dists = None
        self.dataset = None

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
                          'viterbi': self.run_config['viterbi'],
                          'suppress_nan': True,
                          'fp_posteriors': os.path.join(self.run_config['output'], 'posteriors.txt'),
                          'fp_scores_pre': os.path.join(self.run_config['output'], 'scores_pre'),
                          'fp_scores': os.path.join(self.run_config['output'], 'scores.txt'),
                          'fp_hdsl': os.path.join(self.run_config['output'], 'hdsl.txt'),
                          'fp_viterbi': os.path.join(self.run_config['output'], 'viterbi.txt'),
                          'no_cscores': self.run_config['no_cscores'],
                          'min_cscores': self.run_config['min_cscores'],
                          'batch_size': self.run_config['batch_size'],
                          'motifs': self.motifs,
                          'cscore_dists': None}

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

                    worker = partial(self.sample_worker, batch=cscore_batch)
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

        header = "transcript start score c-score motif seq\n"
        open(scoring_config['fp_scores_pre'], 'w').write(header)

        logger.info("Executing scoring")
        clock.tick()

        with tqdm(total=n_batches,
                  leave=False,
                  unit='batch',
                  desc='      Overall') as pbar_batches:

            # Process batches sequentially
            for i, batch in enumerate(batches):

                # print("Batch {} of {}".format(i+1, n_batches))
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
        scores = filelib.read_score_file(scoring_config['fp_scores_pre'])
        if scoring_config['no_cscores']:
            filelib.write_score_file(sorted(scores, key=lambda score: score['score'], reverse=True),
                                     scoring_config['fp_scores'])
        else:
            filelib.write_score_file(sorted(scores, key=lambda score: score['c-score'], reverse=True),
                                     scoring_config['fp_scores'])
        os.remove(scoring_config['fp_scores_pre'])  # Clean-up
        logger.info(' ... done in {}'.format(misclib.seconds_to_hms(clock.tock())))

    @staticmethod
    def sample_worker(motif, batch):

        scores = []
        for transcript in batch.rnas.values():
            scores.extend(get_null_scores(transcript, motif))

        return motif, scores

    @staticmethod
    def score_worker(transcript, model, config):

        model.e_step(transcript)
        transcript.compute_log_B_ratios()

        if config['viterbi']:
            vp = model.viterbi_decoding(transcript)  # Viterbi algorithm
            path = "".join([str(i) for i in vp])
            LOCK.acquire()
            with open(config['fp_viterbi'], "a") as f:
                f.write(">{}\n{}\n".format(transcript.name, path))
            LOCK.release()

        if config['posteriors']:
            transcript.gamma /= np.sum(transcript.gamma, axis=0)[np.newaxis, :]
            LOCK.acquire()
            with open(config['fp_posteriors'], "a") as f:
                f.write("> {}\n".format(transcript.name))
                f.write("{}\n".format(" ".join(["{:1.3f}".format(p) for p in transcript.gamma[0, :]])))
            LOCK.release()

        if config['motifs'] is not None:

            scores = []
            for motif in config['motifs']:
                path = rnalib.dot2states(motif)
                transcript.find_valid_sites(motif)
                scores_tmp = list(map(lambda start: score_path(transcript, start, path, motif),
                                      transcript.valid_sites[motif]))
                if config['suppress_nan']:
                    scores_tmp = list(filter(lambda score: ~np.isnan(score['score']), scores_tmp))
                if config['cscore_dists'] is not None:
                    compute_cscores(scores_tmp, config['cscore_dists'])
                scores += scores_tmp

            LOCK.acquire()
            with open(config['fp_scores_pre'], "a") as f:
                output = "".join(["{} {} {:1.2f} {:1.2f} {} {}\n".format(
                    score['transcript'],
                    score['start'] + 1,
                    score['score'],
                    score['c-score'],
                    score['motif'],
                    transcript.seq[score['start']:score['start'] + len(score['motif'])]) for score in scores])
                f.write(output)
            LOCK.release()

            # Hairpin-derived structure level measure
            if config['hdsl']:
                hdsl_tmp = transcript.gamma[1, :]
                for score in scores:
                    # Profile augmentation with hairpin scores
                    if score['c-score'] > 0.5:
                        hdsl_tmp[score['start']:score['start'] + len(score['motif'])] += 0.2 * (score['c-score'] - 0.5)
                hdsl_tmp[hdsl_tmp < 0] = 0
                hdsl_tmp[hdsl_tmp > 1] = 1
                # Smoothing steps
                hdsl_tmp = uniform_filter1d(hdsl_tmp, size=5)  # Local mean
                hdsl = median_filter(hdsl_tmp, size=15)  # Local median

                LOCK.acquire()
                with open(config['fp_hdsl'], "a") as f:
                    f.write("> {}\n".format(transcript.name))
                    f.write("{}\n".format(" ".join(["{:1.3f}".format(p) for p in hdsl])))
                LOCK.release()

    def make_cscore_batch(self, min_sample_size):

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


def get_null_scores(transcript, motif):
    path = rnalib.dot2states(motif)
    # Get sites which violate sequence constraints
    invalid_sites = np.where(~np.in1d(range(transcript.T - len(motif) + 1), transcript.valid_sites[motif]))[0]
    null_scores = list(filter(lambda score: ~np.isnan(score['score']),
                              map(lambda start: score_path(transcript, start, path, motif), invalid_sites)))
    return [null_score['score'] for null_score in null_scores]


def compute_cscores(scores, dists):
    list(map(lambda score: apply_cscore(score, dists[score['motif']]), scores))


def apply_cscore(score, dist):
    pv = dist.sf(score['score'])
    if pv == 0:
        log_c = np.Inf
    elif np.isnan(pv):
        log_c = np.nan
    else:
        log_c = -np.log10(pv)
    score['c-score'] = log_c


def score_path(transcript, start, path, motif):
    m = len(path)
    end = start + m - 1

    if np.all(np.isnan(transcript.obs[start:end+1])):
        score = np.nan
    else:
        score = 0
        score += np.log(transcript.alpha[path[0], start] / transcript.alpha[1 - path[0], start])
        score += np.sum((2*path[1:-1] - 1)*transcript.log_B_ratio[1, start + 1:end])
        score += np.log(transcript.beta[path[-1], end] / transcript.beta[1 - path[-1], end])
    return {'score': score,
            'c-score': None,
            'start': start,
            'transcript': transcript.name,
            'motif': motif}
