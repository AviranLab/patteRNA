import sys
import os
import numpy as np
import pathlib
import json
from patteRNA import rnalib


HEADER_COL_ORDER = {'transcript': 0,
                    'start': 1,
                    'end': 1.1,
                    'score': 2,
                    'c-score': 3,
                    'bce': 3.1,
                    'BCE': 3.11,
                    'deltaG': 3.2,
                    'MEL': 3.21,
                    'lbc-prob': 3.3,
                    'Prob(motif)': 3.3,
                    'dot-bracket': 4,
                    'motif': 4.1,
                    'path': 4.1,
                    'seq': 5}

user_prompts = {"yes": ["y", "Y", "yes", "Yes"],
                "no": ["n", "N", "no", "No"]}

DB_MAP = {'.': 0,
          '1': 1,
          '0': 0}

DB_MAP.update([(bracket, 1) for brackets in rnalib.BRACKETS for bracket in brackets])


def save_model(jstr, fp_out):
    with open(fp_out, 'w') as f:
        json.dump(jstr, f, indent=2)


def parse_model(fp):
    return json.loads(open(fp, 'r').read())


def parse_observations(fp):
    """
        Returns dictionary of SHAPE reactivities for each transcript in a .shape file.

            Parameters:
                fp (str): SHAPE file

            Returns:
                out_dict (dict): dictionary of SHAPE vectors (np.array) for each transcript
    """

    out_dict = dict()

    with open(fp, 'r') as f:
        while True:
            name_row = f.readline()
            if not name_row:  # End of file
                break
            name = name_row.split('>')[1].strip()  # Parse name
            obs = f.readline().strip().split(' ')  # Observations text
            obs_array = np.array([parse_shape_str(ob) for ob in obs])
            out_dict[name] = obs_array

    return out_dict


def parse_fasta(fp):
    """
        Returns dictionary of sequences for each transcript in a .fa/.fasta file.

            Parameters:
                fp (str): FASTA-like file

            Returns:
                out_dict (dict): dictionary of sequences (str) for each transcript
    """

    out_dict = dict()

    with open(fp, 'r') as f:
        while True:
            name_row = f.readline()
            if not name_row:  # End of file
                break
            name = name_row.split('>')[1].strip()  # Parse name
            seq = f.readline().strip().upper()  # Sequence text
            rnalib.verify_sequence(seq, name)
            out_dict[name] = seq

    return out_dict


def read_dot_bracket(fp):
    """
        Returns dictionaries of structures and sequences, if the with_sequence flag is set to True)
        for each transcript in a .dot file.

            Parameters:
                fp (str): Dot file

            Returns:
                db_dict (dict): Dictionary of dot-bracket structures
                seq_dict (dict): Dictionary of nucleotide sequences
    """

    db_dict = dict()
    seq_dict = dict()

    with open(fp, 'r') as f:

        entry_name = None

        for line in f.readlines():

            if line[0] == '>':
                entry_name = line.split('>')[1].strip()
                continue

            if entry_name is not None:
                if line[0].upper() in rnalib.BASES:
                    seq = line.strip()
                    rnalib.verify_sequence(seq, entry_name)
                    seq_dict[entry_name] = seq
                    continue
                if line[0] in rnalib.VALID_DB_CHAR:
                    if rnalib.valid_db(line.strip()):
                        db_dict[entry_name] = np.array([DB_MAP[d] for d in line.strip()], dtype=int)
                        entry_name = None
                else:
                    if line[0] in ('1', '0'):
                        db_dict[entry_name] = np.array([DB_MAP[d] for d in line.strip()], dtype=int)
                        entry_name = None
                    else:
                        raise Exception(
                            'Reference structure information for transcript ""{}"" is invalid'.format(entry_name))

    return db_dict, seq_dict


def parse_shape_str(shape_str):
    if not shape_str or '-999' in shape_str or shape_str in ('nan', 'NaN', 'NAN', 'Nan', 'na', 'NA', 'None', 'none'):
        return np.nan
    return float(shape_str)


def read_score_file(fp):
    scores = []
    with open(fp, "r") as f:
        header = f.readline()
        columns = header.strip().split()
        for line in f:
            line = line.rstrip().split()
            d = {col: format_val(val, col) for col, val in zip(columns, line)}
            scores.append(d)
    return scores


def write_score_file(scores, fp_out):
    """
        Inputs a list of scores dictionaries for each transcript and writes to output file.
        Automatically infers column order.

            Parameters:
                scores (list): List of dictionaries
                fp_out (str): Scores file
    """
    with open(fp_out, 'w') as f:
        header = recover_score_header(scores)
        f.write(header)
        f.write('\n')
        cols = header.split()
        for score in scores:
            f.write(" ".join([format_col(score, col) for col in cols]))
            f.write("\n")


def format_val(val, col):
    if col in ('start', 'end'):
        return int(val)
    if col in ('transcript', 'dot-bracket', 'motif', 'path', 'seq'):
        return val
    return float(val)


def format_col(score, col):
    if col in ('start', 'end'):
        return "{:d}".format(score[col])  # Handles integers
    if col in ('transcript', 'dot-bracket', 'motif', 'path', 'seq'):
        return "{}".format(score[col])  # Handles strings
    if col in ('score', 'c-score'):
        return "{:.2f}".format(float(score[col]))  # Handles NaNs
    return "{}".format(score[col])


def recover_score_header(scores):
    return " ".join(sorted(scores[0].keys(), key=lambda c: HEADER_COL_ORDER[c]))


def prepare_output_dir(run_config):
    path = run_config['output']
    make_dir(path)

    overwite_files = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file == 'trained_model.json' and run_config['training']:
            overwite_files.append(file_path)
        if file == 'fit.svg' and run_config['training']:
            overwite_files.append(file_path)
        if file == 'posteriors.txt' and run_config['posteriors']:
            overwite_files.append(file_path)
        if file == 'viterbi.txt' and run_config['viterbi']:
            overwite_files.append(file_path)
        if file == 'scores_pre' and run_config['scoring']:
            overwite_files.append(file_path)
        if file == 'scores.txt' and run_config['scoring']:
            overwite_files.append(file_path)
        if file == 'hdsl.txt' and run_config['HDSL']:
            overwite_files.append(file_path)
        if file == 'spp.txt' and run_config['SPP']:
            overwite_files.append(file_path)

    if overwite_files:
        print("WARNING: the output directory contains previous outputs. "
              "The following files will be overwritten:\n")
        print("\n".join(overwite_files) + "\n")

        response = None

        while not response:
            response_text = input("Enter yes [y/yes] to confirm overwrite, or no [n/no] to cancel run: ")
            response = interpret_response(response_text)

        if response == 'yes':
            for file_path in overwite_files:
                os.remove(file_path)
        elif response == 'no':
            sys.exit()


def interpret_response(response_text):
    for valid_reponse in user_prompts:
        if response_text in user_prompts[valid_reponse]:
            return valid_reponse
    return None


def make_dir(path):
    """
    Create a directory. Spawn parents if needed.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
