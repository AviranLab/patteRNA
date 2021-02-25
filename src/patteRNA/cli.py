"""Command line wrapper for patteRNA."""

import sys
import logging
from .Dataset import Dataset
from .TrainingManager import TrainingManager
from .ScoringManager import ScoringManager
from .Model import Model
from .DOM import DOM
from .GMM import GMM
from .HMM import HMM
from . import arglib, filelib, misclib, timelib, logger_config


def main():

    main_clock = timelib.Clock()
    main_clock.tick()  # Start task timer
    clock = timelib.Clock()

    # Parse command line arguments
    input_files, run_config = arglib.parse_cl_args(sys.argv[1:])

    # Prepare output folder, confirming any possible overwrites
    filelib.prepare_output_dir(run_config)

    # Set up logger and summarize configuration
    logger_config.setup_logging(run_config['output'], verbose=run_config['verbose'])
    logger = logging.getLogger(__name__)
    logger.info(arglib.summarize_job(input_files, run_config))
    logger.debug(arglib.summarize_config(input_files, run_config))

    fp_observations = input_files['probing']
    fp_sequences = input_files['fasta']
    fp_references = input_files['reference']

    # Parse input data
    data = Dataset(fp_observations, fp_sequences, fp_references)  # Initialize Dataset object

    logger.info("Loading input data")
    clock.tick()
    data.load_rnas(log_flag=run_config['log'])  # Load RNAs
    logger.info(" ... done in {}".format(misclib.seconds_to_hms(clock.tock())))

    if run_config['training']:
        # Initialize emission model for training
        if run_config['GMM']:
            em = GMM()
        else:
            em = DOM()

        # Initialize Model object with structure model (HMM) and emission model (DOM or GMM)
        model = Model(structure_model=HMM(), emission_model=em)

        # Spawn training set RNAs
        logger.info("Spawning training set")
        clock.tick()
        training_set, kl_div = data.spawn_training_set(kl_div=run_config['KL_div'])
        logger.info(" ... done in {}".format(misclib.seconds_to_hms(clock.tock())))

        logger.info("Training set summary: \n"
                    "       Passed QC                  {:d}\n"
                    "       Used transcripts           {:d}\n"
                    "       # Data points (in {})     {:d}\n"
                    "       KL div                     {:.3g}".format(len(data.rnas),
                                                                      len(training_set.rnas),
                                                                      model.emission_model.type,
                                                                      training_set.stats['n_obs'],
                                                                      kl_div))

        tm = TrainingManager(model=model, mp_tasks=run_config['n_tasks'], output_dir=run_config['output'], k=run_config['k'])
        tm.import_data(training_set)

        clock.tick()
        tm.execute_training()
        logger.info("Training phase done in {}".format(misclib.seconds_to_hms(clock.tock())))

    else:
        model = filelib.load_model(input_files['model'])  # Returns Model object
        logger.info("Using trained {}-{} model at {}".format(model.emission_model.type,
                                                             model.structure_model.type,
                                                             input_files['model']))

    if run_config['scoring']:
        logger.info("Initiating scoring phase")

        sm = ScoringManager(model, run_config)  # Scoring manager object

        logger.info("Scoring set summary: \n"
                    "       Passed QC                  {:d}\n"
                    "       Used transcripts           {:d}\n"
                    "       # Data points              {:d}\n"
                    "       # Motifs                   {:.3g}".format(len(data.rnas),
                                                                      len(data.rnas),
                                                                      data.stats['n_obs'],
                                                                      len(sm.motifs)))

        sm.import_data(data)

        clock.tick()
        sm.execute_scoring()
        logger.info("Scoring phase done in {}".format(misclib.seconds_to_hms(clock.tock())))

    logger.info("Process done in {}".format(misclib.seconds_to_hms(main_clock.tock())))
    logger.info("Output written to  -> {}".format(run_config['output']))


if __name__ == "__main__":
    main()
