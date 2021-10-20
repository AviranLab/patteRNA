import logging

logger = logging.getLogger(__name__)

vienna_imported = False

try:
    # Following comment suppresses unresolved reference warnings related to the fact that RNA
    # will not be declared if the import statement fails within this try block
    # noinspection PyUnresolvedReferences
    import RNA

    fc = RNA.fold_compound('GCGCGCAAAGCGCGC')
    mfe, _ = fc.mfe()
    if mfe == '((((((...))))))':
        vienna_imported = True
    else:
        logger.warning('WARNING - ViennaRNA Python interface was imported, but did not behave as '
                       'expected. Check results or use --no-vienna to run patteRNA without NNTM folding.')
except ModuleNotFoundError:
    logger.debug('ViennaRNA Python interface not detected.')  # Debug level log message
except ImportError as e:
    logger.warning('WARNING - ViennaRNA Python interface was found, but could not be imported successfully. '
                   'Check that you are using the same verison of Python that was configured with the interface. '
                   'Check results or use --no-vienna to run patteRNA without NNTM folding. '
                   'See error below:\n{}'.format(repr(e)))


def fold(seq):
    """
    Compute the minimum free energy of a given sequence.

    Args:
        seq (str): RNA sequence to fold

    Returns:
        mfe (float): Minimum free energy of folded structure

    """
    return RNA.fold(seq)[1]


def hc_fold(seq, hcs):
    """
    Compute the minimum free energy of a given sequence subject to hard constraints (either
    base-pairs or unpaired nucleotides).

    Args:
        seq (str): RNA sequence to fold
        hcs (list): List of base pairing constraints

    Returns:
        mfe (float): Minimum free energy of folded structure

    """
    rna = RNA.fold_compound(seq)
    add_hcs(rna, hcs)
    return rna.mfe()[1]


def add_hcs(rna, hcs):
    """
    Add hard constraints to a ViennaRNA fold_compound object.
    Args:
        rna (RNA.fold_compound): Fold compound object from ViennaRNA
        hcs (list): List of hard constraints
    """
    for hc in hcs:
        if hc[1] >= 0:
            rna.hc_add_bp(hc[0] + 1, hc[1] + 1)
        else:
            rna.hc_add_up(hc[0] + 1)
