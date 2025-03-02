import numpy as np

def rename_key(dictionary, old_key, new_key):
    if old_key in dictionary:
        dictionary[new_key] = dictionary.pop(old_key)

def normalize(array: np.array):
    max_value = np.max(array)
    min_value = np.min(array)
    
    if min_value != max_value: 
        return (array - np.min(array)) / (np.max(array) - np.min(array))
    return np.zeros_like(array)

