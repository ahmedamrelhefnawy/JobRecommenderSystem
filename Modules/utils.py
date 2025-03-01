import numpy as np


def rename_key(dictionary, old_key, new_key):
    if old_key in dictionary:
        dictionary[new_key] = dictionary.pop(old_key)

def normalize(array: np.array):
    if np.any(array):
        return (array - np.min(array)) / (np.max(array) - np.min(array))
    return array