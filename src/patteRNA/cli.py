"""Command line wrapper for patteRNA."""

import os
import sys
import logging
import shutil
import numpy as np
import tqdm

from . import misclib
from . import logger_config, patternlib, rnalib, gmmhmm, globalbaz
from . import input_args

# Initialize globals
DTYPES = globalbaz.DTYPES
TEST_BATCH_SIZE = globalbaz.TEST_BATCH_SIZE
USER_PROMPT = globalbaz.USER_PROMPT
OUTPUT_NAME = globalbaz.OUTPUT_NAME


def main(testcmd=None):
    # Initialize parameters
    do_training = False
    do_scoring = False
    fp_out = None
    is_GQ = None
    main_time = misclib.timer_start()

    # Input arguments handling
    if testcmd:  # Means we are in testing mode with a mock CLI command
        testcmd = misclib.absolute_string(testcmd)  # Make paths absolute
        args = input_args.parse_cl_args(testcmd.split()[1:])
    else:
        args = input_args.parse_cl_args(sys.argv[1:])

    input_args.add_more_defaults(args)  # Add additional default parameters to args_cl
    args_yaml = input_args.parse_config_yaml(fp=args.config)  # Parse yaml config file arguments
    input_args.merge_args(args_yaml, args)  # Merge YAML arguments with command-line arguments

    # Create the output directory (if it already exists then it won't override the existing one)
    misclib.make_dir(args.output)

    # Initialize loggers
    logger_config.setup_logging(log_path=args.output, verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Parse the --pattern flag if provided
    patterns = None
    if args.pattern is not None:
        if args.pattern.split("[")[0] == "GQ":  # G-quadruplex
            # Check that an input sequence file was input
            if args.fasta is None:
                logger.error("No FASTA file provided. G-quaruplexes cannot be found without RNA sequences.")
                sys.exit()

            is_GQ = True
            patterns = args.pattern.split("[")[1].replace("]", "").split(",")
            patterns = [int(i) for i in patterns]
        else:  # Canonical motifs
            is_GQ = False
            patterns = patternlib.pattern_builder(args.pattern, args.seq, args.forbid_N_pairs)

    # Check if we need a training phase
    if args.model is None:
        do_training = True

    # Check if we need a scoring phase
    if args.gammas | args.viterbi | (args.pattern is not None):
        do_scoring = True

    # Check that no overwriting will occur
    overwrite_files = False
    delete_training = False

    if os.path.isdir(args.output):  # Output directory already exist
        if do_training:
            if os.path.isdir(os.path.join(args.output, OUTPUT_NAME["training"])):
                overwrite_files = True
                delete_training = True
            elif os.path.isfile(os.path.join(args.output, OUTPUT_NAME["model"])):
                overwrite_files = True

        if args.gammas and os.path.isfile(os.path.join(args.output, OUTPUT_NAME["gammas"])):
            overwrite_files = True
        if args.viterbi and os.path.isfile(os.path.join(args.output, OUTPUT_NAME["viterbi"])):
            overwrite_files = True
        if (args.pattern is not None) and os.path.isfile(os.path.join(args.output, OUTPUT_NAME["pattern"])):
            overwrite_files = True
    else:
        pass

    while True:
        if overwrite_files:
            if args.no_prompt:
                response = "Yes"
            else:
                response = input("Some output files already exist. Overwrite them? [yes/no] ")

            if response in USER_PROMPT["yes"]:
                if delete_training:
                    shutil.rmtree(os.path.join(args.output, OUTPUT_NAME["training"]), ignore_errors=False)
                break
            elif response in USER_PROMPT["no"]:
                sys.exit()
        else:
            break

    # # Check that w_min is valid (currently not supported)
    # if args.wmin >= (1 / args.k):
    #     logger.error("The current value of -wmin, {}, is too big. "
    #                  "All Gaussian mixture components could be dropped with this threshold.\n"
    #                  "Allowed values for -wmin must be smaller than 1/K, i.e <{}.".format(args.wmin, 1 / args.k))
    #     sys.exit()

    # Print configs to the log file
    config_summary = input_args.summarize_config(args)
    logger.debug(config_summary)

    logger.info("Target pattern -> {}".format(args.pattern))

    # Set mutable global system configs and initialize the model
    gmmhmm.global_config(n_tasks=args.n_tasks,
                         verbose=args.verbose,
                         seq_constraints=args.seq,
                         output=args.output,
                         no_gmm=args.nogmm,
                         pars=args.PARS,
                         nan_flag=args.NAN)
    hmm = gmmhmm.GMMHMM()  # Initialize a GMMHMM object

    # Load data
    try:
        dataset = os.path.basename(args.input)
        args.input = os.path.dirname(args.input)
        start_time = misclib.timer_start()
        logger.info("Loading input data ...")
        rna_set = rnalib.build_rnalib_from_files(fp_seq=args.fasta,
                                                 fp_obs=os.path.join(args.input, dataset))
    except FileNotFoundError:
        logger.error("WOOPS, couldn't find the input SP data.")
        sys.exit()

    # Log transform if required
    if args.log:
        rna_set.log_transform()

    # Do we need to train the model?
    train_set = None
    if do_training:

        # Prepare training set by filtering and building the RNA set
        train_set = rna_set.qc_and_build(min_density=args.min_density, n=args.n)

        logger.info("Loading ... done in {}".format(misclib.timer_stop(start_time)))
        logger.info("Transcript #: Initial {} | Train {}".format(len(rna_set.rnas),
                                                                 len(train_set.rnas)))

    # Garbage collection to free RAM if we do not need a scoring phase
    if do_scoring is False:
        rna_set = None

    # Train the model or load a pre-trained one
    if do_training:
        # Check that we have data to train the model
        if len(train_set.rnas) == 0:
            logger.error("No training data left after filtering, try decreasing --min-density.")
            sys.exit()

        train_set.compute_stats()
        logger.info("Training ...")
        start_time = misclib.timer_start()

        # Import the data and initialize the model
        hmm.import_data(train_set=train_set)
        hmm.initialize_HMM(N=2, pi=args.pi, rho=args.rho, A=args.A, phi=args.phi, upsilon=args.upsilon)
        hmm.initialize_GMM(K=args.k, mu=args.mu, sigma=args.sigma, w=args.w, w_min=args.wmin)

        # Train the model
        hmm.train(max_iter=args.maxiter, epsilon=args.epsilon)
        hmm.take_snapshot(stdout=True,
                          fp_fit=os.path.join(args.output, OUTPUT_NAME["fit_plot"]),
                          fp_logl=os.path.join(args.output, OUTPUT_NAME["logL_plot"]))

        # Save the model
        hmm.dump(fp=os.path.join(args.output, "trained_model.pickle"))
        logger.info("Training ... done in {}".format(misclib.timer_stop(start_time)))
    else:
        # Load the trained model
        logger.info("Using pre-trained model at {}".format(args.model))
        hmm.load(args.model)

    # Scoring phase (if required by the user)
    if do_scoring:
        start_time = misclib.timer_start()

        # Build the test set from the original RNA set
        logger.info("Building the scoring set ...")
        if args.filter_test:
            test_set = rna_set.qc_and_build(min_density=args.min_density, n=args.n)
        else:
            test_set = rna_set.qc_and_build(min_density=0, n=np.inf)

        # noinspection PyUnusedLocal
        rna_set = None  # Garbage collection to free RAM

        n_test = len(test_set.rnas)
        logger.info("Building the scoring set ... done")
        logger.info("Computing scores for {} transcripts ...".format(n_test))

        # Subset the test set such that a maximum of <TEST_BATCH_SIZE> RNAs are handled at each iterations
        batches_mask = misclib.make_batches(n=n_test,
                                            batch_size=TEST_BATCH_SIZE,
                                            stochastic=False)

        # Check if patterns are requested
        if patterns is not None:
            # Initialize pointers to the output score file
            fp_out = os.path.join(args.output, OUTPUT_NAME["pattern"])
            gmmhmm.write_score_header(fp=fp_out)  # Write header

        # Initialize pointers to output files
        if args.viterbi:
            fp_viterbi = os.path.join(args.output, OUTPUT_NAME["viterbi"])
        else:
            fp_viterbi = None

        if args.gammas:
            fp_gammas = os.path.join(args.output, OUTPUT_NAME["gammas"])
        else:
            fp_gammas = None

        # Initialize a progress bar and iterate over batches
        pbar = tqdm.tqdm(total=n_test, mininterval=1, disable=not args.verbose, leave=False)
        iter_cnt = 0

        # Iterations over batches (done to minimize RAM usage and handle very large datasets)
        for b in range(batches_mask.shape[1]):
            curr_batch = [i for (i, v) in zip(test_set.rnas, batches_mask[:, b]) if v]  # Make the batch
            n_batch = len(curr_batch)

            # Import the test data into the model and write the Viterbi path and Gamma posteriors if requested
            hmm.score(rnas=curr_batch,
                      patterns=patterns,
                      fp_pattern=fp_out,
                      is_GQ=is_GQ,
                      fp_viterbi=fp_viterbi,
                      fp_gammas=fp_gammas)

            pbar.update(n_batch)
            logger.debug("Computing scores... {}/{}".format(iter_cnt, n_test))
            iter_cnt += n_batch
        pbar.close()

        logger.info("Computing scores... done in {}".format(misclib.timer_stop(start_time)))
    logger.info("Process terminated in {}".format(misclib.timer_stop(main_time)))
    logger.info("Output written to -> {}".format(args.output))


if __name__ == "__main__":
    main()
