import sys
import os
import numpy as np
import pathlib
import json
from . import rnalib
from .HMM import HMM
from .DOM import DOM
from .GMM import GMM
from .Model import Model


HEADER_COL_ORDER = {'transcript': 0,
                    'start': 1,
                    'end': 1.1,
                    'score': 2,
                    'c-score': 3,
                    'motif': 4,
                    'path': 4.1,
                    'seq': 5}

user_prompts = {"yes": ["y", "Y", "yes", "Yes"],
                "no": ["n", "N", "no", "No"]}


def save_model(model, fp_out):
    model_dict = model.serialize()
    with open(fp_out, 'w') as f:
        json.dump(model_dict, f, indent=2)


def load_model(fp_in):
    structure_model = None
    emission_model = None

    input_dict = json.loads(open(fp_in, 'r').read())  # Read input data

    if input_dict['structure_model']['type'] == 'HMM':
        structure_model = HMM()
        structure_model.set_params(transitions=input_dict['structure_model']['A'],
                                   pi=input_dict['structure_model']['pi'])
    if input_dict['emission_model']['type'] == 'DOM':
        emission_model = DOM()
        emission_model.set_params(n_bins=input_dict['emission_model']['n_bins'],
                                  edges=input_dict['emission_model']['edges'],
                                  classes=input_dict['emission_model']['classes'],
                                  chi=input_dict['emission_model']['chi'])
    if input_dict['emission_model']['type'] == 'GMM':
        emission_model = GMM()
        emission_model.set_params(k=input_dict['emission_model']['k'],
                                  w=input_dict['emission_model']['w'],
                                  mu=input_dict['emission_model']['mu'],
                                  sigma=input_dict['emission_model']['sigma'],
                                  phi=input_dict['emission_model']['phi'],
                                  nu=input_dict['emission_model']['nu'],
                                  n_params=input_dict['emission_model']['n_params'])

    return Model(structure_model=structure_model, emission_model=emission_model)


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

    if overwite_files:
        print("WARNING: the output directory contains previous outputs. "
              "The following files will be overwritten:\n")
        print("\n".join(overwite_files)+"\n")

        response = None

        while not response:
            response_text = input("Enter yes [yes/y] to confirm overwrite or no [no/n] to cancel run: ")
            response = interpret_response(response_text)

        if response == 'yes':
            for file_path in overwite_files:
                os.remove(file_path)
            pass
        else:
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
