import numpy as np
import numpy.core.tracked_float as tf

def test1(arr_size = (10, 1000000), ):
    # basic test
    provenance = []
    for i in range(arr_size[0]):
        for j in range(arr_size[1]):
            prov = (((1, i, j)), ((2, i, j)))
            provenance.append(prov)
    return provenance

def test2(arr = (10, 1000000), arr2 = (10, 1000000)):
    # test 2 arrays
    provenance = []
    for i in range(arr[0]):
        for j in range(arr[1]):
            prov = (((1, i, j)), (2, i, j), (3, i, j))
            provenance.append(prov)
    return provenance

def test3(arr = (10, 1000000)):
    # test reduction
    provenance = []
    for i in range(arr[0]):
        out = (1, i, 1)
        prov = []
        for j in range(arr[1]):
            prov.append((2, j, 1))
        pr = ((out), tuple(prov))
        provenance.append(pr)
    return provenance

def test4(arr = (10, 1000000)):
    #tests duplication (2x2)
    prov = []
    l = arr[0]
    w = arr[1]
    for i in range(arr[0]):
        for j in range(arr[1]):
            pr = (((1, i, j)), ((2, i, j), (2, i + l, j), (2, i, j + w), (2, i + l, j + w)))
            prov.append(pr)
    return pr

def test5(arr = (10, 1000000)):
    #tests duplication (3 x 1 )
    prov = []
    l = arr[0]
    w = arr[1]
    for i in range(arr[0]):
        for j in range(arr[1]):
            pr = (((1, i, j)), ((2, i, j), (2, i + l, j), (2, i + l*2, j)))
            prov.append(pr)
    return arr

def test6(arr = (1000, 1000), arr2 = (1000, 1000)):
    #test matmul
    prov = []
    
    for j in range(arr[0]):
        for i in range(arr2[1]):
            pr = []
            for k in range(arr[1]):
                pr.append((2, j, k))
                pr.append((3, k, i))
                prov2 = ((1, j, i), tuple(pr))
                prov.append(prov2)
    return prov

def test7(arr = (10000000, 1), arr2 = (10000000, 1)):
    # test vector*vector
    prov = []
    pr = []
    for i in range(arr[0]):
        pr.append((2, i, 1))
        pr.append((3, i, 1))
    prov2 = ((1, 1, 1), tuple(pr))
    prov.append(prov2)
    return prov

def test8(arr = (1000, 1000), arr2 = (1000, 1)):
    # test matrix*vector
    prov = []
    
    for j in range(arr[0]):
        pr = []
        for i in range(arr[1]):
            pr.append((2, j, i))
            pr.append((3, i, 1))
        prov2 = ((1, j, 1), tuple(pr))
        prov.append(prov2)

    return prov

def test9(prov_arr, arr = (10,000,000, 1)):
    # tests filters
    prov = []
    for i in range(arr[0]):
        if prov_arr.n != 0:
            pr = ((0, i, 1), (2, i, 1))
            prov.append(pr)
    return prov    

def test10(prov_arr, arr = (10,000,000, 1)):
    prov = []
    for i in range(arr[0]):
        if prov_arr.n != 0:
            pr = ((0, i, 1), (2, i, 1))
            prov.append(pr)
    return prov 