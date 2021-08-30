vienna_imported = False
try:
    import RNA

    fc = RNA.fold_compound('GCGCGCAAAGCGCGC')
    mfe, _ = fc.mfe()
    if mfe == '((((((...))))))':
        vienna_imported = True
    else:
        raise RuntimeWarning('WARNING - ViennaRNA Python interface was imported, but did not behave as '
                             'expected. Check results or use --no-vienna to run patteRNA without NNTM folding.')
except ModuleNotFoundError:
    pass


def fold(seq):
    return RNA.fold(seq)[1]


def hc_fold(seq, hcs):
    rna = RNA.fold_compound(seq)
    add_hcs(rna, hcs)
    return rna.mfe()[1]


def add_hcs(rna, hcs):
    for hc in hcs:
        if hc[1] >= 0:
            rna.hc_add_bp(hc[0] + 1, hc[1] + 1)
        else:
            rna.hc_add_up(hc[0] + 1)
