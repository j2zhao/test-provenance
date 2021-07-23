
from tracked_object import *

from aux_functions import *
import random
import time

def set_array(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i,j] = TrackedObj(random.random(), (i,j))

def meta_test(function, inputs = [(100, 100)]):
    arrs = []
    for input in inputs:
        arr = np.empty(input, dtype= object)
        arrs.append(set_array(arr))

    