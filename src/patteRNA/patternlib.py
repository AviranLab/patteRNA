import matplotlib
matplotlib.use("agg")

import numpy as np
import exrex
import regex
import itertools
import logging
import sys
from scipy.stats import beta, pearson3, exponnorm, genlogistic
import matplotlib.pyplot as plt

from . import globalbaz

# Initialize logger
logger = logging.getLogger(__name__)

PAIRING_TABLE = globalbaz.GLOBALS["pairing_table"]  # Set the default pairing table


class Pattern:
    def __init__(self, dot, path_str):
        """Initialize attributes."""
        self.dot = dot
        self.path = np.array(list(path_str), dtype=globalbaz.GLOBALS["dtypes"]["path"])

        if self.dot is None:
            self.n = len(self.path)
            self.dot = "-" * self.n
        else:
            self.n = len(self.dot)

        if len(self.dot) != len(self.path):
            logger.error("--motif and --path are incompatible.")
            sys.exit()

        # Get some characteristics of the pattern
        dot = np.array(list(self.dot))
        self.left_partner = np.where(dot == "(")[0]
        self.right_partner = np.where(dot == ")")[0][::-1]  # Invert right partner
        self.n_left = len(self.left_partner)
        self.n_right = len(self.right_partner)
        self.is_symmetrical = self.n_left == self.n_right
        self.pairing_table = self.compute_pairing_table()

    def valid_dot(self):
        """Ensure dot strings are valid."""

        valid = False
        if self.dot.count("(") == self.dot.count(")"):
            valid = True

        return valid

    def compute_pairing_table(self):

        table = [[], []]

        n = self.n_left

        dot_init = str(self.dot)
        working_dot = dot_init

        if self.valid_dot():

            while not len(table[0]) == n:

                open_bracket = 0

                for s, i in zip(working_dot, range(len(dot_init))):

                    if s == "(":

                        open_bracket = 1

                        for si, j in zip(working_dot[i+1:], range(len(working_dot[i+1:]))):

                            if si == "(":
                                open_bracket += 1
                                continue
                            if si == ")":
                                open_bracket -= 1

                            if si == ".":
                                continue

                            if open_bracket == 0:

                                open_index = i
                                close_index = i+j+1

                                table[0].append(open_index)
                                table[1].append(close_index)

                                working_dot = working_dot[:open_index] + \
                                              'C' + working_dot[open_index+1:close_index] + \
                                              'C' + working_dot[close_index+1:]
                                break
                        break

        return table

    def ensure_pairing(self, seq):
        """Ensures that the structure given by the dot-bracket can form based on the underlying sequence.

        Args:
            seq (list): Underlying RNA sequence as a list

        Returns: True if pairing is ensured, False otherwise.

        """

        pairing_ensured = True

        for i in range(len(self.pairing_table[0])):
            if not seq[self.pairing_table[0][i]] in PAIRING_TABLE[seq[self.pairing_table[1][i]]]:
                pairing_ensured = False
                break

        # pairing_ensured = True
        #
        # if self.is_symmetrical:
        #     seq = list(seq)  # string to list
        #     for ix in range(self.n_left):
        #         if not seq[self.right_partner[ix]] in PAIRING_TABLE[seq[self.left_partner[ix]]]:
        #             pairing_ensured = False
        #             break
        # else:
        #     pairing_ensured = False

        return pairing_ensured


def parse_motif(args):
    """Parse the input --motif and --path option simultaneously"""

    motifs = pattern_builder(motif_regex=args.motif,
                             path_regex=args.path,
                             forbid_N_pairs=args.forbid_N_pairs)

    return motifs


def parse_GQ(args):
    """Parse the input --GQ option"""

    motifs = args.GQ.split("[")[1].replace("]", "").split(",")
    motifs = [int(i) for i in motifs]

    return motifs


# noinspection PyPep8Naming
def pattern_builder(motif_regex=None, path_regex=None, seq_constraints=False, forbid_N_pairs=False):
    """Generate all possible state sequences given a dot-bracket pattern RegEx.

    Args:
        motif_regex (str): Input motif dot-bracket regex
        path_regex (str): Path regex applied as mask to input motif_regex
        seq_constraints (bool): Apply sequence constraints? (LEGACY)
        forbid_N_pairs (bool): Are N-N paired considered invalid?

    """
    global PAIRING_TABLE

    # Check if we consider N-N base pairings invalid and switch to the proper pairing table
    if forbid_N_pairs:
        PAIRING_TABLE = globalbaz.GLOBALS["pairing_table_no_N"]

    dots = None
    if motif_regex is not None:
        dots = dot_regex2substrings(motif_regex)

    paths = None
    if path_regex is not None:
        paths = path_regex2substrings(path_regex)

    putative_patterns = []
    if (dots is not None) and (paths is not None):
        for dot, path in zip(dots, paths):
            putative_patterns.append(Pattern(dot=dot, path_str=path))
    elif dots is not None:
        for dot in dots:
            putative_patterns.append(Pattern(dot=dot, path_str=dot2states(dot, as_string=True)))
    elif paths is not None:
        for path in paths:
            putative_patterns.append(Pattern(dot=None, path_str=path))

    patterns = []
    if seq_constraints or (motif_regex is not None):
        for pattern in putative_patterns:
            if pattern.valid_dot():
                patterns.append(pattern)
    else:
        patterns = putative_patterns

    return patterns


def dot_regex2substrings(regex_in):
    """Generate all possible dot-bracket based on a motif regex"""

    # add character literals
    regex_in = regex_in.replace("(", "\(")
    regex_in = regex_in.replace(")", "\)")
    regex_in = regex_in.replace(".", "\.")

    return list(exrex.generate(regex_in))


def path_regex2substrings(regex_in):
    """Generate all possible paths based on a path regex."""

    return list(exrex.generate(regex_in))


def g_quadruplex_finder(seq, min_quartet, max_quartet, min_loop, max_loop):
    """Finds putative G-quadruplexes in a RNA sequence.

    G-quadruplex paths are encoded as: 0 = loops / 1 = paired G / -1 = outside the G-quadruplex.
    Loop refers to the spacing outside of quartets, i.e. between columns of stacked Gs.
    Bulges between quartets are not currently supported.

    Args:
        seq (str): RNA sequence.
        min_quartet (int): Minimum number of quartets allowed.
        max_quartet (int): Maximum number of quartets allowed.
        min_loop (int): Minimum length of loops.
        max_loop (int): Maximum length of loops.

    Returns:
        quad_repo (list of dict): G-quadruplex repository. Each entry contains start/end positions and the taken path.

    """

    # noinspection PyPep8Naming
    T = len(seq)
    quad_repo = []  # Initialize the repository of G-quadruplexes

    # Scan across quartet sizes
    for nG in range(min_quartet, max_quartet + 1):

        # Find all possible Gs columns
        pattern = nG * "G"
        max_size = 4 * nG + 3 * max_loop
        pattern = regex.compile(pattern)
        found_pattern = []

        for m in pattern.finditer(seq, overlapped=True):
            found_pattern.append(m.span(0)[0])

        # Find all allowed G-quads
        potential_pattern = np.array(found_pattern)
        found_pattern = np.array(found_pattern)

        # Scan each possible initial Gs columns
        for start in found_pattern:
            # We can remove the first one as this cannot be simultaneously the start and the next G column
            potential_pattern = np.delete(potential_pattern, 0)

            selected_ix = []

            # Check all subsequent Gs columns
            # noinspection PyTypeChecker
            for curr_g in potential_pattern:

                # Check that we are still scanning for G-quads of allowed size
                total_size = curr_g - start + nG
                if total_size <= max_size:
                    selected_ix.append(curr_g)
                else:
                    break

            # Find all possible G-quads with the current initial Gs column
            if len(selected_ix) >= 3:
                # Combinatorics step with G columns in ascending order only
                ixs = set(list(itertools.combinations(selected_ix, 3)))

                # Loop across possible combinations
                for ix in ixs:
                    # Check that all three internal loops are of allowed length
                    # noinspection PyTypeChecker
                    loop_size = np.array(ix[0] - start - nG)  # First loop length
                    loop_size = np.append(loop_size, np.diff(ix) - nG)  # Append loop 2 and 3 lengths
                    mask = (loop_size >= min_loop) & (loop_size <= max_loop)

                    if np.all(mask):
                        # This combination passed checks and is a putative G-quadruplex, lets get its path
                        # noinspection PyTypeChecker
                        path = np.repeat(-1, T)
                        path[start:(start + nG)] = 1  # Paired - G-column

                        # Build the path
                        prev_ix = start
                        for j in range(0, 3):
                            path[(prev_ix + nG):ix[j]] = 0  # Unpaired - Loop
                            path[ix[j]:(ix[j] + nG)] = 1  # Paired - G-column
                            prev_ix = ix[j]

                        # Add this path to the repository
                        end = prev_ix + nG
                        quad_repo.append({"start": start,
                                          "end": end,
                                          "path": np.array(path, dtype=globalbaz.GLOBALS["dtypes"]["path"]),
                                          "dot": "-" * (end-start)})

    return quad_repo


def h0_collect_scores(h0_set, path_repo, rna):
    """Score paths for the Null score distribution based on valid scored motifs.

    Args:
        h0_set (dict): Repository of Null scores per motifs
        path_repo (list[dict]): Repository of valid motifs, starts, stops, and dot-brackets
        rna (GMMHMM_SingleObs): Object holding RNA sequence and model

    Returns:
        h0_set (dict): Updated repository of Null scores per motifs

    """

    h0_paths = {}  # Will return an empty h0_path dict if the path_repo is empty
    if path_repo:
        for path in path_repo:

            frame = range(path["start"], path["end"])  # Get start/end index
            motif = "".join([str(i) for i in path["path"][frame]])  # Get pattern (to use as key)

            #  Index h0_paths by unique motifs
            if motif not in h0_paths.keys():
                h0_paths[motif] = []

            h0_paths[motif].append(path["start"])

        # Score null paths
        for h0_path, starts in h0_paths.items():
            t = len(h0_path)

            if h0_path not in h0_set.keys():
                h0_set[h0_path] = []

            if h0_set[h0_path] is not None:  # If false, means we have enough h0 scores for this path (i.e. skip it)

                # Get random starts, omitting locations where a possible motif was found
                mask = np.repeat(True, rna.T - t + 1)
                mask[starts] = False
                rstarts = np.argwhere(mask).reshape(-1)

                # If no unrealizable paths are available, then resort to sampling available paths
                # todo optimize as we do not need to compute score again (i.e. we can retrieve already scored paths)
                if len(rstarts) == 0:
                    rstarts = np.array(starts)

                rends = rstarts + t  # Get path's ends based on starts

                num_path = np.array(list(h0_path), dtype=int)  # String to numerical path

                for rstart, rend in zip(rstarts, rends):

                    if len(h0_set[h0_path]) < globalbaz.GLOBALS["h0_max_size"]:  # More scores are needed for the h0
                        full_path = np.pad(num_path, (rstart, rna.T - rend), 'constant', constant_values=-1)
                        curr_score, rejected = rna.score_path({"start": rstart, "end": rend, "path": full_path})

                        if not rejected:
                            h0_set[h0_path].append(curr_score)

                    else:  # We collected enough h0 so we can go to the next path
                        break

    return h0_set


def h0_fit(h0_path_set):
    """Fit a Gaussian distribution for each motifs and RNA.

    Args:
        h0_path_set (dict): Repository of Null scores per motifs

    Returns:
        h0_params (dict): Null distribution parameters for each motif

    """

    h0_params = {}
    for path, scores in h0_path_set.items():
        h0_params[path] = {}
        if len(scores) >= globalbaz.GLOBALS["h0_min_size"]:
            # x_bins = np.linspace(min(scores), max(scores), 100)
            # plt.hist(scores, x_bins, normed=1)
            # plt.show()
            # # mean, std = norm.fit(np.array(scores))
            # h0_params[path]["mean"] = mean
            # h0_params[path]["std"] = std

            path_scores = np.array(scores)
            floc = np.min(path_scores)-1
            fscale = 2*((np.max(path_scores)+1)-(np.min(path_scores)-1))

            a, b, floc, fscale = beta.fit(path_scores, floc=floc, fscale=fscale)
            h0_params[path]["a"] = a
            h0_params[path]["b"] = b
            h0_params[path]["loc"] = floc
            h0_params[path]["scale"] = fscale

        else:
            # h0_params[path]["mean"] = np.nan
            # h0_params[path]["std"] = np.nan

            logger.warning("Insufficient null sites to build a null distribution for path: '{}'\n"
                           "Scores for path '{}' will not be normalized.".format(path, path))

            h0_params[path]["a"] = np.nan
            h0_params[path]["b"] = np.nan
            h0_params[path]["loc"] = np.nan
            h0_params[path]["scale"] = np.nan

    return h0_params


def dot2states(dot, as_string=False):
    """Translate a dot-bracket string in a sequence of numerical states"""

    dot = dot.replace(".", "0")  # Unpaired
    dot = dot.replace("(", "1")  # Paired
    dot = dot.replace(")", "1")  # Paired
    dot = dot.replace(">", "1")  # Paired (ct2dot symbols)
    dot = dot.replace("<", "1")  # Paired (ct2dot symbols)
    dot = dot.replace("{", "1")  # Paired (ct2dot symbols)
    dot = dot.replace("}", "1")  # Paired (ct2dot symbols)

    if as_string:
        dotl = dot
    else:
        dotl = np.array(list(dot), dtype=globalbaz.GLOBALS["dtypes"]["path"])

    return dotl


if __name__ == '__main__':
    pass
