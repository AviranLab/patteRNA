"""Command line interface for patteRNA."""

import logging
import sys
from patteRNA.Dataset import Dataset
from patteRNA.Model import Model
from patteRNA.ScoringManager import ScoringManager
from patteRNA.TrainingManager import TrainingManager
from patteRNA.DOM import DOM
from patteRNA.GMM import GMM
from patteRNA.HMM import HMM
from patteRNA import arglib, filelib, misclib, timelib, logger_config


def main(testcmd=None):
    """
    Main execution thread for patteRNA. Handles input arguments, reads input data,
    uses TrainingManager and ScoringManager objects to execute requested computations.

    Args:
        testcmd: Optional string input containing a mock input command syntax. Useful for running patteRNA
        for testing or via programmatic interfaces.

    """

    main_clock = timelib.Clock()
    main_clock.tick()  # Start task timer for overall process

    # Parse command line arguments if a manual test command is not provided
    if testcmd is None:
        input_files, run_config = arglib.parse_cl_args(sys.argv[1:])
    else:
        input_files, run_config = arglib.parse_cl_args(testcmd.split())
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

    # Initialize dataset and parse input data
    data = Dataset(fp_observations, fp_sequences, fp_references)  # Initialize Dataset object
    clock = timelib.Clock()  # Clock for timing individual steps
    logger.info("Loading input data")
    clock.tick()
    data.load_rnas(log_flag=run_config['log'])  # Load RNAs

    if input_files['reference'] is not None:
        reference_states, reference_seqs = filelib.read_dot_bracket(input_files['reference'])
        reference_rnas = list(reference_states.keys())
        if len(reference_rnas) > 0:
            run_config['reference'] = True
            for rna in reference_rnas:
                if reference_seqs[rna] != data.rnas[rna].seq:
                    logger.warning("Inconsistent sequence in reference file for RNA: {}\n"
                                   "Transcript will not be used for training.".format(rna))
                else:
                    data.rnas[rna].enforce_reference(reference_states[rna])

    logger.info(" ... done in {}".format(misclib.seconds_to_hms(clock.tock())))

    if run_config['training']:
        # Initialize emission model for training
        if run_config['GMM']:
            em = GMM()
        else:
            em = DOM()

        # Initialize Model object with structure model (HMM) and emission model (DOM or GMM)
        model = Model(structure_model=HMM(), emission_model=em, reference=run_config['reference'])

        if run_config['reference']:

            # Spawn training set of reference RNAs
            logger.info("Using reference set.")
            clock.tick()
            reference_set = data.spawn_reference_set()

            logger.info("Reference set summary: \n"
                        "       Passed QC                  {:d}\n"
                        "       Used transcripts           {:d}\n"
                        "       # Data points (in {})     {:d}\n"
                        "       # Unpaired Obs             {:d}\n"
                        "       # Paired Obs               {:d}\n".format(len(data.rnas),
                                                                          len(reference_set.rnas),
                                                                          model.emission_model.type,
                                                                          reference_set.stats['n_obs'],
                                                                          reference_set.stats['up_ref'],
                                                                          reference_set.stats['p_ref']))

            tm = TrainingManager(model=model, mp_tasks=run_config['n_tasks'], output_dir=run_config['output'],
                                 k=run_config['k'], reference=True, nan=run_config['nan'])
            tm.import_data(reference_set)

        else:

            # Spawn training set of RNAs
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

            tm = TrainingManager(model=model, mp_tasks=run_config['n_tasks'], output_dir=run_config['output'],
                                 k=run_config['k'], nan=run_config['nan'])
            tm.import_data(training_set)

        clock.tick()
        tm.execute_training()
        logger.info("Training phase done in {}".format(misclib.seconds_to_hms(clock.tock())))

    else:  # If not training, we must be loading a model for scoring

        model = Model()
        model.load(filelib.parse_model(input_files['model']))
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
