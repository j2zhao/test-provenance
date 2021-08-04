"""
example_functions.py contains example functions to run.
"""
# 1 or 2 matrix
# different types of operations
# just try repetition
import numpy as np

def test1(arr):
    # basic test
    for i in range(1):    
        arr = np.negative(arr)
    return arr

def test2(arr, arr2):
    # test 2 arrays
    for i in range(1):    
        k = arr + arr2
    return arr

def test3(arr):
    # tests filters
    out = arr < 0.5
    return out

def test4(arr):
    # tests copy
    out = np.copy(arr)
    return out

def test4(arr):
    #tests reduction/long provenance
    arr = np.sum(arr, axis = 0)
    return arr

def test5(arr):
    #tests duplication
    arr = np.tile(arr, (2, 2))
    return arr