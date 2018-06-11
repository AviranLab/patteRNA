"""Command line wrapper for patteRNA."""

import os
import sys
import logging
import numpy as np
import tqdm

from . import misclib
from . import logger_config
from . import patternlib
from . import rnalib
from . import gmmhmm
from . import globalbaz
from . import file_handler
from . import input_args

# Initialize globals and switches
GLOBALS = globalbaz.GLOBALS


# noinspection PyUnboundLocalVariable
def main(testcmd=None):
    # Initialize switches
    switch = {"do_training": False,
              "do_auto_k": False,
              "do_scoring": False,
              "do_scan": False,
              "is_motif": False,
              "is_gquad": False,
              "is_path": False,
              "do_seq": False,
              "do_pvalues": True}

    # UI
    ui_msg_padding = 27

    spinner_msg_pad = ui_msg_padding + 7  # This is to account for "INFO - "
    ui_msg = "{:<%d}" % ui_msg_padding
    ui_msg_done = "{:<%d}... done in {}" % ui_msg_padding

    # Initialize parameters
    reference_set = None
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

    if args.no_cscores:
        switch["do_pvalues"] = False

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

    # Remove any trailing temporary files
    fp_out_unsorted = None
    fp_out = None
    fp_p = None
    fp_h0_partial = None
    fp_h0 = None

    if switch["do_scan"]:
        # Initialize pointers to the output score file
        fp_out_unsorted = os.path.join(args.output, GLOBALS["output_name"]["unsorted_scores"])
        fp_out = os.path.join(args.output, GLOBALS["output_name"]["scores"])
        fp_p = os.path.join(args.output, GLOBALS["output_name"]["p_scores"])
        fp_h0_partial = os.path.join(args.output, GLOBALS["output_name"]["h0_partial"])
        fp_h0 = os.path.join(args.output, GLOBALS["output_name"]["h0"])

        # Cleanup - Making sure we remove any existing temporary file
        if os.path.exists(fp_out_unsorted):
            os.remove(fp_out_unsorted)
        if os.path.exists(fp_p):
            os.remove(fp_p)
        if os.path.exists(fp_h0_partial):
            os.remove(fp_h0_partial)
        if os.path.exists(fp_h0):
            os.remove(fp_h0)

    # Print configs to the log file
    config_summary = input_args.summarize_config(args)
    logger.debug(config_summary)

    # Set mutable global system configs and initialize the model
    gmmhmm.global_config(n_tasks=args.n_tasks,
                         verbose=args.verbose,
                         seq_constraints=switch["do_seq"],
                         output=args.output,
                         no_gmm=args.nogmm,
                         pars=False,
                         nan=args.NAN)
    hmm = gmmhmm.GMMHMM()  # Initialize a GMMHMM object

    # Write UI message
    if switch["is_motif"]:
        logger.info(ui_msg.format("Target motif  ->") + "{}".format(args.motif))
    if switch["is_gquad"]:
        logger.info(ui_msg.format("Target GQ  ->") + "{}".format(args.GQ))
    if switch["is_path"]:
        logger.info(ui_msg.format("Target path  ->") + "{}".format(args.path))

    # Load data
    try:
        task_time = misclib.timer_start()
        logger.info(ui_msg.format("Loading input data"))
        fps_obs, assays = input_args.check_obs_extensions([args.input])  # Check obs filename extensions
        if len(fps_obs) == 0:  # No files found
            logger.error("WOOPS, no input profiling data found.")
            sys.exit()
        else:
            if assays[0] == "pars":  # Check if the input is a PARS assay
                gmmhmm.global_config(pars=True)

            rna_set, initial_n_rnas = rnalib.build_rnalib_from_files(fp_seq=args.fasta,
                                                                     fp_obs=fps_obs[0],
                                                                     fp_ref=args.reference,
                                                                     min_density=args.min_density)
    except FileNotFoundError:
        logger.error("WOOPS, no input profiling data found.")
        sys.exit()

    # Log transform if required
    if args.log:
        rna_set.log_transform()

    logger.info(ui_msg_done.format("Loading input data", misclib.timer_stop(task_time)))

    # Do we need to train the model?
    train_set = None
    if switch["do_training"]:
        logger.info(ui_msg.format("Building the training set"))

        # Prepare training set by filtering and building the RNA set
        task_time = misclib.timer_start()
        train_set, reference_set = rna_set.qc_and_build(is_training=True, KL_threshold=args.KL_div)

        logger.info(ui_msg_done.format("Building the training set", misclib.timer_stop(task_time)))
        logger.info("Training set summary - \n"
                    "       Initial transcripts        {:d}\n"
                    "       Passed QC                  {:d}\n"
                    "       Used transcripts           {:d}\n"
                    "       # Data points (in GMM)     {:d}\n"
                    "       KL div                     {:.3g}".format(initial_n_rnas,
                                                                      len(rna_set.rnas),
                                                                      len(train_set.rnas),
                                                                      train_set.T_continuous,
                                                                      train_set.KL_div))

    # Check if we need to determine K automatically
    if args.k <= 0:
        args.k = 1
        switch["do_auto_k"] = True

    # Garbage collection to free RAM if we do not need a scoring phase
    if switch["do_scoring"] is False:
        rna_set = None

    # Train the model or load a pre-trained one
    if switch["do_training"]:

        # Check if reference structures were provided
        if len(reference_set.rnas) != 0:
            switch["do_auto_k"] = False

        # Check that we have data to train the model
        if len(train_set.rnas) == 0:
            logger.error("No training data found.")
            sys.exit()

        train_set.build_continuous_obs()
        train_set.build_histogram()

        task_time = misclib.timer_start()

        found_optimal_k = False
        prev_bic = np.inf
        while not found_optimal_k:

            # Reset the model
            hmm.reset()

            # Import the data
            train_set.compute_stats(args.k)
            hmm.import_data(train_set=train_set)

            # Initialize the model
            hmm.initialize_model(N=2, K=args.k, pi=args.pi, A=args.A,
                                 mu=args.mu, sigma=args.sigma, w=args.w, w_min=-1,
                                 phi=args.phi, upsilon=args.upsilon, reference_set=reference_set)

            # Train the model
            hmm.train(max_iter=args.maxiter, epsilon=args.epsilon)

            if not switch["do_auto_k"]:
                found_optimal_k = True

            # logger.info("Model's BIC = {:.3g}".format(hmm.bic))
            if prev_bic <= hmm.bic:  # Means the previous model had a better AIC
                found_optimal_k = True
                # Load the previous model
                hmm.load(fp=os.path.join(args.output, "trained_model.pickle"))

                # Import the data
                hmm.import_data(train_set=train_set)
                train_set.compute_stats(args.k)

            else:
                # Save the current model
                hmm.dump(fp=os.path.join(args.output, "trained_model.pickle"))
                prev_bic = hmm.bic
                args.k += 1

        train_set.continuous_obs = None  # Garbage collection

        hmm.take_snapshot(stdout=True,
                          fp_fit=os.path.join(args.output, GLOBALS["output_name"]["fit_plot"]),
                          fp_logl=os.path.join(args.output, GLOBALS["output_name"]["logL_plot"]))

        if switch["do_auto_k"]:
            logger.info(ui_msg.format("Optimal K  ->") + "{:d}".format(hmm.K))

        # Check if the likelihood converged
        if not hmm.did_converge:
            logger.warning("patteRNA did not converge within {} iterations. Try increasing --maxiter.\n"
                           "Last 5 logL -> {}".format(args.maxiter,
                                                      np.round(hmm.logL[-5:], 2)))

        # Check if the model had to invert pairing states
        if hmm.did_invert:
            logger.warning("Oups! The model inverted paired/unpaired states. Parameters were adjusted "
                           "to resolve this. Please check the output to make sure the model is appropriate.")

        logger.info(ui_msg_done.format("Training", misclib.timer_stop(task_time)))
    else:
        # Load the trained model
        logger.info("Using pre-trained model at {}".format(args.model))
        hmm.load(args.model)

    # Scoring phase (if required by the user)
    if switch["do_scoring"]:
        task_time = misclib.timer_start()

        logger.info(ui_msg.format("Building the scoring set"))
        # Build the test set from the original RNA set
        test_set, _ = rna_set.qc_and_build()

        n_test = len(test_set.rnas)

        logger.info(ui_msg_done.format("Building the scoring set", misclib.timer_stop(task_time)))
        logger.info("Scoring set summary - \n"
                    "       Initial transcripts        {:d}\n"
                    "       Passed QC                  {:d}\n"
                    "       Used transcripts           {:d}\n"
                    "       # Data points              {:d}".format(initial_n_rnas,
                                                                    len(rna_set.rnas),
                                                                    len(test_set.rnas),
                                                                    test_set.T - test_set.T_nan))

        # noinspection PyUnusedLocal
        rna_set = None  # Garbage collection to free RAM

        # Subset the test set such that a maximum of <TEST_BATCH_SIZE> RNAs are handled at each iterations
        batches_mask = misclib.make_batches(n=n_test,
                                            batch_size=GLOBALS["test_batch_size"],
                                            stochastic=False)

        # Check if motifs are requested
        if switch["do_scan"]:

            # Write the header
            gmmhmm.write_score_header(fp_out_unsorted)

        # Initialize pointers to output files
        if args.viterbi:
            fp_viterbi = os.path.join(args.output, GLOBALS["output_name"]["viterbi"])
        else:
            fp_viterbi = None

        if args.posteriors:
            fp_posteriors = os.path.join(args.output, GLOBALS["output_name"]["posteriors"])
        else:
            fp_posteriors = None

        logger.info(ui_msg.format("Computing outputs"))
        # Initialize a progress bar and iterate over batches
        pbar = tqdm.tqdm(total=n_test, mininterval=1, disable=not args.verbose, leave=False,
                         desc="Scoring dataset", unit=" transcripts")
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
                      do_pvalues=switch["do_pvalues"],
                      fp_viterbi=fp_viterbi,
                      fp_posteriors=fp_posteriors,
                      fp_h0_partial=fp_h0_partial,
                      fp_h0=fp_h0)

            pbar.update(n_batch)
            if not args.verbose:
                logger.debug("Computing outputs ... {}/{}".format(iter_cnt, n_test))
            iter_cnt += n_batch
        pbar.update(n_test)
        pbar.close()
        if not args.verbose:
            logger.debug("Computing outputs ... {}/{}".format(iter_cnt, n_test))
        logger.info(ui_msg_done.format("Computing outputs", misclib.timer_stop(task_time)))

    if switch["do_scan"]:
        # Compute p_values
        if switch["do_pvalues"]:
            task_time = misclib.timer_start()
            logger.info(ui_msg.format("Computing c-scores"))
            file_handler.h0_merge(fp=fp_h0,
                                  fp_partial=fp_h0_partial)
            os.remove(fp_h0_partial)
            gmmhmm.compute_and_write_p_values(fp_h0, fp_out_unsorted, fp_p)
            os.remove(fp_h0)
            os.rename(fp_p, fp_out_unsorted)
            logger.info(ui_msg_done.format("Computing c-scores", misclib.timer_stop(task_time)))

        # Sort the score file and delete the unsorted file
        logger.info(ui_msg.format("Sorting scores"))
        task_time = misclib.timer_start()
        if switch["do_pvalues"]:  # sort by c-values
            file_handler.sort_score_file(fp_out_unsorted, fp_out, column_ix=4, decreasing=True)
        else:  # sort by scores
            file_handler.sort_score_file(fp_out_unsorted, fp_out, column_ix=3, decreasing=True)
        os.remove(fp_out_unsorted)
        logger.info(ui_msg_done.format("Sorting scores", misclib.timer_stop(task_time)))

    logger.info(ui_msg_done.format("Task", misclib.timer_stop(main_time)))
    logger.info(ui_msg.format("Output written to  ->") + "{}".format(args.output))


if __name__ == "__main__":
    main()
