import string
import numpy as np

BASES = ('A', 'T', 'G', 'C', 'U', 'N')

PAIRING_TABLE = {'A': ('U', 'T'),
                 'U': ('A', 'G'),
                 'G': ('C', 'U', 'T'),
                 'C': ('G',),
                 'T': ('A', 'G'),
                 'N': ()}

BRACKETS = ['()', '<>', '{}', '[]']  # Standard brackets, followed by alphabet (A<->a, B<->b, C<->c, etc.)
BRACKETS.extend([''.join((lb, rb)) for lb, rb in zip(string.ascii_uppercase, string.ascii_lowercase)])

LEFT_BRACKETS = set([pair[0] for pair in BRACKETS])
RIGHT_BRACKETS = set([pair[1] for pair in BRACKETS])
VALID_DB_CHAR = LEFT_BRACKETS | RIGHT_BRACKETS | set('.')

# Partner map is used to get both characters of a bracket when you have one
PARTNER_MAP = {lb: (lb, rb) for lb, rb in BRACKETS}  # Set up left brackets
PARTNER_MAP.update({rb: (lb, rb) for lb, rb in BRACKETS})  # Right brackets


def verify_sequence(seq, name):
    for char in seq:
        if char not in BASES:
            print('WARNING - Invalid nucleotide character detected in: {}\n'
                  '          Valid characters are ACGUT'.format(name))
            return False
    return True


def valid_db(db):
    # Check characters are all valid and note which brackets are present
    brackets = set()
    for s in db:
        if s not in VALID_DB_CHAR:
            return False
        if s == '.':
            continue
        else:
            brackets.add(''.join(PARTNER_MAP[s]))

    # Check validity of all bracket characters
    for lb, rb in brackets:

        # Skip bracket characters that aren't present
        if db.count(lb) == 0 and db.count(rb) == 0:
            continue

        # First, do a quick check if the number of brackets match up
        # (can detect most, but not all, invalid db with this)
        if db.count(lb) != db.count(rb):
            return False

        # Then, check that every right bracket is preceded by an available left bracket
        bracket_mask = np.array([(s == lb, s == rb) for s in db], dtype=bool)
        bracket_mask_cumsum = bracket_mask.cumsum(axis=0)

        # Return False if at any point there are more right brackets than left brackets
        if np.any(bracket_mask_cumsum[:, 1] > bracket_mask_cumsum[:, 0]):
            return False

    return True


def check_sequence_constraints(seq, db):
    partners, _ = compute_pairing_partners(db)
    return is_valid_pairing(seq, partners)


def is_valid_pairing(seq, partners):
    """
    Ensures that the structure given by the dot-bracket can form based on the underlying sequence.

    Args:
        seq (str): Underlying RNA sequence as a list
        partners (list): List of (i, j) tuples of base pair partners

    Returns: True if pairing is valid, False otherwise.

    """

    valid = True
    for i, j in partners:
        if seq[j] not in PAIRING_TABLE[seq[i]]:
            return False
    return valid


def compute_pairing_partners(db, ignore_invalid=False):
    partners = []
    unpaired = []

    if not ignore_invalid:
        if not valid_db(db):
            raise AssertionError('An invalid dot-bracket structure was provided.')

    db = [str(s) for s in db]  # Reformat if needed

    ob = {lb: [] for lb in LEFT_BRACKETS}
    # Open brackets, a dict for keeping track of partner-less open brackets while parsing

    for j, s in enumerate(db):
        if s in LEFT_BRACKETS:
            ob[s].append(j)
        elif s in RIGHT_BRACKETS:
            partners.append((ob[PARTNER_MAP[s][0]].pop(-1), j))
        else:
            unpaired.append((j, -1))

    return sorted(partners, key=lambda bp: bp[0]), unpaired


def dot2states(dot):
    """Translate a dot-bracket string in a sequence of numerical states"""

    dot = dot.replace(".", "0")  # Unpaired
    dot = dot.replace("(", "1")  # Paired
    dot = dot.replace(")", "1")  # Paired

    return np.array(list(dot), dtype=int)
