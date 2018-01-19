"""Command line wrapper for patteRNA."""

import os
import sys
import logging
import numpy as np
import tqdm

from . import misclib, logger_config, patternlib, rnalib, gmmhmm, globalbaz, file_handler, input_args

# Initialize globals and switches
GLOBALS = globalbaz.GLOBALS


def main(testcmd=None):
    # Initialize switches
    switch = {"do_training": False,
              "do_auto_k": False,
              "do_scoring": False,
              "do_scan": False,
              "is_motif": False,
              "is_gquad": False,
              "is_path": False,
              "do_seq": False}

    # Initialize parameters
    reference_set = None
    fp_out = None
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

    # Parse the motifs related options (--motif, --GQ and --path) and update SWITCHES
    motifs = None
    if (args.motif is not None) or (args.path is not None):
        switch["do_scoring"] = True
        switch["do_scan"] = True

        if args.motif is not None:
            switch["is_motif"] = True
            switch["do_seq"] = True

        if args.path is not None:
            switch["is_path"] = True

        motifs = patternlib.parse_motif(args)

    if args.GQ is not None:
        if switch["do_scan"]:
            logger.error("--GQ cannot be used in conjunction with options --motif and --path.")
            sys.exit()

        motifs = patternlib.parse_GQ(args)
        switch["is_gquad"] = True
        switch["do_seq"] = True
        switch["do_scoring"] = True
        switch["do_scan"] = True

    # LEGACY parse the --seq and --pattern flag if provided
    if args.pattern is not None:
        switch["do_seq"] = args.seq
        switch["do_scoring"] = True
        switch["do_scan"] = True
        motifs = patternlib.pattern_builder(motif_regex=args.pattern,
                                            seq_constraints=args.seq,
                                            forbid_N_pairs=args.forbid_N_pairs)

    # Check if we need a training phase
    if args.model is None:
        switch["do_training"] = True

    # Check if posteriors and viterbi options are switched on
    if args.posteriors | args.viterbi:
        switch["do_scoring"] = True

    # Check option compatibility
    if switch["do_seq"] and (args.fasta is None):
        logger.error("No FASTA file provided. Motifs cannot be found without RNA sequences. Use --path instead or "
                     "input a fasta file with -f/--fasta.")
        sys.exit()

    # Check if files/folders will be overwritten
    file_handler.check_overwrites(args, switch)

    # Print configs to the log file
    config_summary = input_args.summarize_config(args)
    logger.debug(config_summary)

    # Write UI message
    if switch["is_motif"]:
        logger.info("Target motif -> {}".format(args.motif))
    if switch["is_gquad"]:
        logger.info("Target GQ -> {}".format(args.GQ))
    if switch["is_path"]:
        logger.info("Target path -> {}".format(args.path))

    # Set mutable global system configs and initialize the model
    gmmhmm.global_config(n_tasks=args.n_tasks,
                         verbose=args.verbose,
                         seq_constraints=switch["do_seq"],
                         output=args.output,
                         no_gmm=args.nogmm,
                         pars=False,
                         nan=args.NAN)
    hmm = gmmhmm.GMMHMM()  # Initialize a GMMHMM object

    # Load data
    try:
        start_time = misclib.timer_start()
        logger.info("Loading input data ...")
        fps_obs, assays = input_args.check_obs_extensions([args.input])  # Check obs filename extensions
        if len(fps_obs) == 0:  # No files found
            logger.error("WOOPS, no input profiling data found.")
            sys.exit()
        else:
            if assays[0] == "pars":  # Check if the input is a PARS assay
                gmmhmm.global_config(pars=True)

            rna_set = rnalib.build_rnalib_from_files(fp_seq=args.fasta,
                                                     fp_obs=fps_obs[0],
                                                     fp_ref=args.reference)
    except FileNotFoundError:
        logger.error("WOOPS, no input profiling data found.")
        sys.exit()

    # Log transform if required
    if args.log:
        rna_set.log_transform()

    # Do we need to train the model?
    train_set = None
    if switch["do_training"]:

        # Prepare training set by filtering and building the RNA set
        train_set, reference_set = rna_set.qc_and_build(min_density=args.min_density, n=args.n)

        logger.info("Loading ... done in {}".format(misclib.timer_stop(start_time)))
        logger.info("Transcript #: Initial {} | Train {}".format(len(rna_set.rnas),
                                                                 len(train_set.rnas)))

        # Check if we need to determine K automatically
        if args.k <= 0:
            switch["do_auto_k"] = True

    # Garbage collection to free RAM if we do not need a scoring phase
    if switch["do_scoring"] is False:
        rna_set = None

    # Train the model or load a pre-trained one
    if switch["do_training"]:
        # Check that we have data to train the model
        if len(train_set.rnas) == 0:
            logger.error("No training data left after filtering, try decreasing --min-density.")
            sys.exit()

        train_set.compute_stats()

        # Import the data
        hmm.import_data(train_set=train_set)

        # Determine K if needed
        if switch["do_auto_k"]:
            logger.info("Finding an optimal K ...")
            start_time = misclib.timer_start()
            args.k, AICs = hmm.determine_k(N=2)
            logger.info("Optimal K is {} ... done in {}".format(args.k, misclib.timer_stop(start_time)))

        # Initialize the model
        logger.info("Training ...")
        start_time = misclib.timer_start()

        hmm.initialize_model(N=2, K=args.k, pi=args.pi, A=args.A,
                             mu=args.mu, sigma=args.sigma, w=args.w, w_min=args.wmin,
                             phi=args.phi, upsilon=args.upsilon, reference_set=reference_set)

        # Train the model
        hmm.train(max_iter=args.maxiter, epsilon=args.epsilon)
        hmm.take_snapshot(stdout=True,
                          fp_fit=os.path.join(args.output, GLOBALS["output_name"]["fit_plot"]),
                          fp_logl=os.path.join(args.output, GLOBALS["output_name"]["logL_plot"]))

        # Save the model
        hmm.dump(fp=os.path.join(args.output, "trained_model.pickle"))
        logger.info("Training ... done in {}".format(misclib.timer_stop(start_time)))
    else:
        # Load the trained model
        logger.info("Using pre-trained model at {}".format(args.model))
        hmm.load(args.model)

    # Scoring phase (if required by the user)
    fp_out_unsorted = None
    if switch["do_scoring"]:
        start_time = misclib.timer_start()

        # Build the test set from the original RNA set
        logger.info("Building the scoring set ...")
        if args.filter_test:
            test_set, _ = rna_set.qc_and_build(min_density=args.min_density, n=args.n)
        else:
            test_set, _ = rna_set.qc_and_build(min_density=0, n=np.inf)

        # noinspection PyUnusedLocal
        rna_set = None  # Garbage collection to free RAM

        n_test = len(test_set.rnas)
        logger.info("Building the scoring set ... done")
        logger.info("Computing scores for {} transcripts ...".format(n_test))

        # Subset the test set such that a maximum of <TEST_BATCH_SIZE> RNAs are handled at each iterations
        batches_mask = misclib.make_batches(n=n_test,
                                            batch_size=GLOBALS["test_batch_size"],
                                            stochastic=False)

        # Check if motifs are requested
        if switch["do_scan"]:
            # Initialize pointers to the output score file
            fp_out_unsorted = os.path.join(args.output, GLOBALS["output_name"]["unsorted_scores"])
            fp_out = os.path.join(args.output, GLOBALS["output_name"]["scores"])

        # Initialize pointers to output files
        if args.viterbi:
            fp_viterbi = os.path.join(args.output, GLOBALS["output_name"]["viterbi"])
        else:
            fp_viterbi = None

        if args.posteriors:
            fp_posteriors = os.path.join(args.output, GLOBALS["output_name"]["posteriors"])
        else:
            fp_posteriors = None

        # Initialize a progress bar and iterate over batches
        pbar = tqdm.tqdm(total=n_test, mininterval=1, disable=not args.verbose, leave=False)
        iter_cnt = 0

        # Iterations over batches (done to minimize RAM usage and handle very large datasets)
        for b in range(batches_mask.shape[1]):
            curr_batch = [i for (i, v) in zip(test_set.rnas, batches_mask[:, b]) if v]  # Make the batch
            n_batch = len(curr_batch)

            # Import the test data into the model and write the Viterbi path and Gamma posteriors if requested
            # noinspection PyUnboundLocalVariable
            hmm.score(rnas=curr_batch,
                      patterns=motifs,
                      fp_score=fp_out_unsorted,
                      is_GQ=switch["is_gquad"],
                      fp_viterbi=fp_viterbi,
                      fp_posteriors=fp_posteriors)

            pbar.update(n_batch)
            logger.debug("Computing scores... {}/{}".format(iter_cnt, n_test))
            iter_cnt += n_batch
        pbar.close()

        logger.info("Computing scores... done in {}".format(misclib.timer_stop(start_time)))

    # Sort the score file and delete the unsorted file
    if switch["do_scan"]:
        logger.info("Sorting scores...".format(misclib.timer_stop(main_time)))
        file_handler.sort_score_file(fp_out_unsorted, fp_out)
        logger.info("Sorting scores... done".format(misclib.timer_stop(main_time)))

    logger.info("Process terminated in {}".format(misclib.timer_stop(main_time)))
    logger.info("Output written to -> {}".format(args.output))


if __name__ == "__main__":
    main()
