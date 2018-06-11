"""Root library.

This module contains multiple base functions for general applications.

No __main__, just defining classes and functions.

"""

import os
import subprocess
import time
from datetime import timedelta
import numpy as np
from copy import deepcopy


def make_dir(path):
    """Create a directory. Spawn parents if needed."""

    if not os.path.exists(path):
        os.makedirs(path)


def print_attr(obj, attr=None):
    """Print all or selected attributes of an object to stdout. None prints all attributes."""

    for key in sorted(obj.__dict__):
        if attr is None:
            print(key.ljust(len(key) + 1, "\t"), " --> ", obj.__dict__[key])
        elif key in attr:
            print(key.ljust(len(key) + 1, "\t"), " --> ", obj.__dict__[key])


def rename_attribute(obj, old_name, new_name):
    """Rename a object attribute."""

    obj.__dict__[new_name] = obj.__dict__.pop(old_name)


def kwargs2attr(obj, kwargs):
    """Shallow copy attributes values based on keyword arguments. Assign only if the attribute already exists in obj."""
    if kwargs is None:
        return

    try:
        assert isinstance(kwargs, dict)
    except AssertionError:
        print("Attempted to set arguments not using a dictionary.")
        return

    attr = obj.__dict__.keys()
    for key in kwargs.keys():
        if key in attr:
            setattr(obj, key, kwargs[key])


def kwargs2attr_deep(obj, kwargs):
    """Deep copy attributes values based on keyword arguments. Assign only if the attribute already exists in obj."""
    if kwargs is None:
        return

    try:
        assert isinstance(kwargs, dict)
    except AssertionError:
        print("Attempted to set arguments not using a dictionary.")
        return

    attr = obj.__dict__.keys()
    for key in kwargs.keys():
        if key in attr:
            setattr(obj, key, deepcopy(kwargs[key]))


def file_length(fp):
    """Determine the number of rows in a file using linux wc -l."""

    cmd = subprocess.Popen(['wc', '-l', fp],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    result, err = cmd.communicate()
    if cmd.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def seconds_to_hms(t):
    """Formats a datetime.timedelta object to a HH:MM:SS string."""

    hours, remainder = divmod(t.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    if hours < 10:
        hours = "0%s" % int(hours)
    if minutes < 10:
        minutes = "0%s" % minutes
    if seconds < 10:
        seconds = "0%s" % seconds
    return "%s:%s:%s" % (hours, minutes, seconds)


def timer_start():
    """Start a timer."""

    return time.time()


def timer_stop(t0):
    """Stop a timer and return time as a formatted string."""

    t = timedelta(seconds=time.time() - t0)
    return seconds_to_hms(t)


def make_batches(n, batch_size, stochastic=False):
    """Creates a matrix for selecting batches.

    Args:
        n (int): Size of the dataset.
        batch_size (int): Size of each batch.
        stochastic (bool): Shuffle the data before making batch?

    Returns:
        sel (np.array): Array with each column being a boolean mask for selecting a single batch.

    """

    n = int(n)
    n_batches = int(np.ceil(n / batch_size))
    # noinspection PyTypeChecker
    sel = np.tile(False, (n, n_batches))

    # Make batch selectors
    ix = 0
    for i in range(n_batches):
        sel[ix:(ix + batch_size), i] = True
        ix += batch_size

    if stochastic:
        # Randomly shuffle the rows in the array
        np.random.shuffle(sel)

    return sel


def selective_dict_deepcopy(input_dict, included_keys=None, exluded_keys=None):
    """Deep copy a dictionary using either keys to be included or keys to be excluded. If both are None, the entire
    dictionary is copied.

    Args:
        input_dict (dict): Input dictionary.
        included_keys (list): List of keys to include in the output dictionary.
        exluded_keys (list): List of keys to exclude in the output dictionary.

    Returns:
        Deep copy of the input dictionary with selected keys.

    """

    new_dict = {}
    keys_list = list(input_dict.keys())

    if included_keys is not None:
        keys_list = included_keys

    elif exluded_keys is not None:
        for exluded_key in list(exluded_keys):
            keys_list.remove(exluded_key)

    for key in keys_list:
        new_dict[key] = deepcopy(input_dict[key])

    return new_dict


def rand_sample(n, min_=float(0), max_=float(1)):
    """Sample random values within a defined range.

    Args:
        n (np.array): Shape of the array.
        min_ (float): Minimum sampled values.
        max_ (float): Maximum sampled values.

    Returns:
        r (np.array): Array of the same shape of the input array filled with random values between min_ and max_.

    """

    r = (min_ - max_) * np.random.random_sample(n) + max_
    return r


def absolute_string(txt):
    """Replace paths within a string with absolute paths"""

    txt = txt.replace("~", os.path.expanduser("~"))
    txt = txt.replace(" .", " " + os.path.abspath("."))
    txt = txt.replace(" ..", " " + os.path.abspath(".."))

    return txt


if __name__ == '__main__':
    pass
