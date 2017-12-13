import numpy as np
import exrex
import regex
import itertools

from . import globalbaz

PAIRING_TABLE = globalbaz.PAIRING_TABLE
DTYPES = globalbaz.DTYPES


# CANONICAL STRUCTURAL MOTIFS
class Pattern:
    def __init__(self, dot):
        """Initialize attributes."""
        self.dot = dot
        self.path = dot2states(dot)
        self.n = len(dot)

        # Get some characteristics of the pattern
        dot = np.array(list(dot))
        self.left_partner = np.where(dot == "(")[0]
        self.right_partner = np.where(dot == ")")[0][::-1]  # Invert right partner
        self.n_left = len(self.left_partner)
        self.n_right = len(self.right_partner)
        self.is_symmetrical = self.n_left == self.n_right

    def ensure_pairing(self, seq):
        """Ensures that the structure given by the dot-bracket can form based on the underlying sequence.

        Args:
            seq (list): Underlying RNA sequence as a list

        Returns: True if pairing is ensured, False otherwise.

        """

        pairing_ensured = True

        if self.is_symmetrical:
            seq = list(seq)  # string to list
            for ix in range(self.n_left):
                if not seq[self.right_partner[ix]] in PAIRING_TABLE[seq[self.left_partner[ix]]]:
                    pairing_ensured = False
                    break
        else:
            pairing_ensured = False

        return pairing_ensured


def pattern_builder(pattern_regex, seq_constraints, forbid_N_pairs):
    """Generate all possible state sequences given a dot-bracket pattern RegEx."""

    # Check if we consider N-N base pairings invalid and switch to the proper pairing table
    if forbid_N_pairs:
        global PAIRING_TABLE
        PAIRING_TABLE = globalbaz.PAIRING_TABLE_NO_N

    patterns = []
    dots = dot_regex2substrings(pattern_regex, seq_constraints)
    for dot in dots:
        # Encode the dot-bracket to a numerical state sequence
        patterns.append(Pattern(dot))

    return patterns


def dot_regex2substrings(pattern, seq_constraints):
    """Generate all possible dot-bracket based on a pattern.

    Args:
        pattern (str): Regex for the motif
        seq_constraints (str): Apply sequence constraints in the future?

    Returns:
        dots (list): All possible substrings based on the pattern

    """

    # add character literals
    pattern = pattern.replace("(", "\(")
    pattern = pattern.replace(")", "\)")
    pattern = pattern.replace(".", "\.")

    dots = []
    for dot in list(exrex.generate(pattern)):
        if seq_constraints:
            # generate all possible VALID dot-bracket structures
            if dot.count("(") == dot.count(")"):
                dots.append(dot)
        else:
            dots.append(dot)

    return dots


# G-QUADRUPLEXES
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
                        quad_repo.append({"start": start,
                                          "end": prev_ix + nG,
                                          "path": np.array(path, dtype=DTYPES["path"])})

    return quad_repo


# GENERAL
def dot2states(dot):
    """Translate a dot-bracket string in a sequence of numerical states"""

    dot = dot.replace(".", "0")  # Unpaired
    dot = dot.replace("(", "1")  # Paired
    dot = dot.replace(")", "1")  # Paired

    dotl = np.array(list(dot), dtype=DTYPES["path"])
    return dotl


if __name__ == '__main__':
    pass
