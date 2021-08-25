import logging
import multiprocessing
import matplotlib
import numpy as np
import os
from functools import partial
from copy import deepcopy
from matplotlib.pyplot import figure

logger = logging.getLogger(__name__)  # Initialize logger
logging.getLogger('matplotlib').setLevel(logging.ERROR)

matplotlib.use('svg')
matplotlib.rcParams['font.family'] = 'Arial'

LOCK = multiprocessing.Lock()  # Lock for parallel processes


class TrainingManager:
    """
    High-level object for coordinating training.
    """

    def __init__(self, model, mp_tasks, output_dir, reference=False,
                 k=-1, maxiter=100, epsilon=1e-4):

        # Model and computing paramters
        self.model = model
        self.mp_tasks = mp_tasks
        self.mp_pool = None

        # Training parameters
        self.k = k
        self.maxiter = maxiter
        self.epsilon = epsilon

        # Training data
        self.training_set = None

        # Output directory
        self.output_dir = output_dir

        # Reference training flag
        self.reference = reference

    def import_data(self, training_set):
        self.training_set = training_set

    def execute_training(self):
        """
        Execute the training procedure arrive at a trained model, which is returned.

        If the number of kernels, k, has been specified, then training occurs once with exactly
        that number of kernels. If the number of kernels has not been specified (default), training
        will be repeated with increasing k (starting at k = 1) until the BIC starts to increase.
        """

        if self.k < 1:  # Automatically determine k
            k = 1
            prev_bic = np.Inf
            prev_model = None
            while self.model.BIC <= prev_bic:
                prev_bic = self.model.BIC
                prev_model = deepcopy(self.model)  # Save previous model state

                # Train next model size
                if self.reference:
                    self._train(k, maxiter=1, epsilon=np.Inf)  # Train next model size
                else:
                    self._train(k, self.maxiter, self.epsilon)  # Train next model size

                self.model.compute_bic(self.logL, self.training_set.stats['n_obs'])  # Model selection criteria
                logger.info(" >> BIC: {:.3e}".format(self.model.BIC))
                k += 1
            self.model.overload(prev_model)  # When while-loop breaks the last model was worst than the second to last,
            # so overload back to last model
            logger.info("Optimal k={:d}".format(self.model.emission_model.k))
        else:
            self._train(self.k, self.maxiter, self.epsilon)  # Train a specified model size

        fp_fit = os.path.join(self.output_dir, 'fit.svg')
        self.save_snapshot(fp_fit)  # Save visualization
        self.model.save(self.output_dir)  # Save model JSON for future use

        return self.model

    def _train(self, k, maxiter, epsilon):
        """
        Perform EM optimization of model parameters for a given model complexity, k.
        Args:
            k: Number of components comprising model.
            maxiter: Maximum number of iteration
            epsilon: Relative difference in log-likelihood between two iterations to use as criteria for convergence.

        Returns nothing, model parameters are set directly to the TrainingManager.model object.

        """

        self.model.reset()
        self.model.initialize(k, self.training_set.stats)

        logger.info("Training with k={:d}".format(self.model.emission_model.k))

        self.training_set.pre_process(self.model)

        converged = False
        iter_count = 0
        prev_logl = -np.Inf

        while iter_count <= maxiter:
            logger.debug("(k={}) EM iteration #{:d}".format(self.model.emission_model.k, iter_count + 1))
            print(" >> (k={}) EM iteration #{:d}".format(self.model.emission_model.k, iter_count + 1), end='\r')
            self.pool_init()
            # clock = timelib.Clock()
            # clock.tick()
            try:
                worker = partial(self.em_worker, model=self.model)
                partial_params = self.mp_pool.imap_unordered(worker, self.training_set.rnas.values())
                self.mp_pool.close()
                self.mp_pool.join()
            except Exception:
                self.mp_pool.terminate()
                raise

            # Execute pool, returning pseudocounts per transcript
            partial_pseudocounts = [partial_param for partial_param in partial_params]
            # logger.warning(clock.tock(pretty=True))

            # Update parameters from transcript-level pseudocounts
            self.logL = self.model.update_from_pseudocounts(partial_pseudocounts)
            logger.debug(self.model.take_snapshot())
            logger.debug('{}: logL = {:.1f}'.format(iter_count, self.logL))
            iter_count += 1

            if iter_count >= 5 or iter_count == maxiter:  # Check convergence after 5 iterations

                dlogl = self.logL - prev_logl

                if dlogl < epsilon * np.abs(self.logL):
                    converged = True
                    break

            prev_logl = self.logL

        if not converged:
            logger.warning("EM algorithm unable to converge. See log file in output directory for details or try "
                           "increasing --maxiter.")
        else:
            logger.info(" >> Log-likelihood of converged model with k={:d}: {:.3e}".format(self.model.emission_model.k,
                                                                                           self.logL))

    def pool_init(self):
        self.mp_pool = multiprocessing.Pool(processes=self.mp_tasks,
                                            maxtasksperchild=1000)

    @staticmethod
    def em_worker(transcript, model):

        logl = model.e_step(transcript)
        params = model.m_step(transcript)

        return {'logL': logl, **params}

    @staticmethod
    def em_worker_ref(transcript, model):

        logl = (transcript)
        params = model.m_step(transcript)

        return {'logL': logl, **params}

    def save_snapshot(self, fp):
        """

        Returns:

        """
        fig = figure(figsize=(7, 5))
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

        if self.model.emission_model.type == "DOM":
            x = np.arange(len(self.model.emission_model.chi[0, :]) - 1)

            ax.bar(x - 0.05 - 0.5, self.model.emission_model.chi[0, :-1], width=0.1, color='red', label='Unpaired',
                   zorder=10)
            ax.bar(x + 0.05 - 0.5, self.model.emission_model.chi[1, :-1], width=0.1, color='blue', label='Paired',
                   zorder=10)

            ax.set_xticks(np.arange(len(self.model.emission_model.chi[0, :]) - 2))
            ax.set_xticklabels(["{:1.2f}".format(edge) for edge in self.model.emission_model.edges[1:-1]])

            ax.set_xlabel("Observation")
            ax.set_ylabel("Probability")

            # ax.grid(which='major', axis='both')
            #
            textstr = '\n'.join((
                r'$P_{{NaN}}[0]={:1.2f}$'.format(self.model.emission_model.chi[0, -1]),
                r'$P_{{NaN}}[1]={:1.2f}$'.format(self.model.emission_model.chi[1, -1])))

            ax.text(1.05, 0.50, textstr, transform=ax.transAxes,
                    verticalalignment='top', bbox={'facecolor': 'none', 'edgecolor': 'none'})
            #
            # ax.legend(bbox_to_anchor=(1.02, 0.55), loc='lower left')
            # ax.title('Trained DOM Model: {}'.format(logger.__name__))

        if self.model.emission_model.type == "GMM":
            x = np.linspace(self.training_set.stats['minimum'], self.training_set.stats['maximum'], 100)
            pdf0 = np.zeros(100)
            pdf1 = np.zeros(100)
            for i in range(self.model.emission_model.k):
                pdf0 += self.model.emission_model.w[0, i]*self.model.emission_model.gmm_pdfs[0][i].pdf(x)
                pdf1 += self.model.emission_model.w[1, i]*self.model.emission_model.gmm_pdfs[1][i].pdf(x)
                # print(self.model.emission_model.w)
                # print(self.model.emission_model.mu)
                # print(self.model.emission_model.sigma)
                # print(pdf0)
            ax.plot(x, pdf0, color='red', label='Unpaired')
            ax.plot(x, pdf1, color='blue', label='Paired')

            ax.set_xlabel("Observation")
            ax.set_ylabel("Density")

            textstr = '\n'.join((
                r'$P_{{NaN}}[0]={:1.2f}$'.format(self.model.emission_model.nu[0]),
                r'$P_{{NaN}}[1]={:1.2f}$'.format(self.model.emission_model.nu[1]),
                r'$P_0[0]={:1.2f}$'.format(self.model.emission_model.phi[0]),
                r'$P_0[1]={:1.2f}$'.format(self.model.emission_model.phi[1])))

            ax.text(1.05, 0.50, textstr, transform=ax.transAxes,
                    verticalalignment='top', bbox={'facecolor': 'none', 'edgecolor': 'none'})

        ax.grid(which='major', axis='both')
        ax.legend(bbox_to_anchor=(1.02, 0.55), loc='lower left')
        fig.savefig(fp, format='svg')

    def terminate(self):
        """
        Clear TrainingManager object and perform clean up.
        """
        pass
