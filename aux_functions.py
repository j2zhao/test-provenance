"""
aux_functions.py contains auxillary functions for tracking spatial provenance

"""

import numpy as np
import time
import os
import uuid
import random
# from numpy.core.numeric import allclose

def reset_array_prov(array, id = None):
    if id == None:
        id = uuid.uuid1()

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i,j].set_provenance((id,(i,j))) 
    return array

def save_array_prov(array, path):
    prov = np.empty(array.shape, dtype=object)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            prov[i, j] = array[i, j].provenance
    path = os.path.join(path, str(time.time()))
    np.save(path, prov)

# arr = np.load('logs/1626725745.618419.npy', allow_pickle=True)
