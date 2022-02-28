import numpy as np
import numpy.core.tracked_float as tf

def test1(arr_size = (10, 1000000)):
    # basic test
    arr = np.random.random(arr_size).astype(tf.tracked_float)
    arr = np.negative(arr)
    return arr

def test2(arr = (10, 1000000), arr2 = (10, 1000000)):
    # test 2 arrays
    arr = np.random.random(arr).astype(tf.tracked_float)
    arr2 = np.random.random(arr2).astype(tf.tracked_float)
    arr = arr + arr2
    return arr

def test3(arr = (10, 1000000)):
    # test reduction
    arr = np.random.random(arr).astype(tf.tracked_float)
    arr = np.sum(arr, axis = 1, initial=None)
    return arr

def test4(arr = (10, 1000000)):
    #tests duplication
    arr = np.random.random(arr).astype(tf.tracked_float)
    arr = np.tile(arr, (2, 2))
    return arr

def test5(arr = (10, 1000000)):
    #tests duplication
    arr = np.random.random(arr).astype(tf.tracked_float)
    arr = np.tile(arr, (3, 1))
    return arr

def test6(arr = (1000, 1000), arr2 = (1000, 1000)):
    #test matmul
    arr = np.random.random(arr).astype(tf.tracked_float)
    arr2 = np.random.random(arr2).astype(tf.tracked_float)
    arr = np.matmul(arr, arr2)
    return arr

def test7(arr = (1000000, 1), arr2 = (1000000, 1)):
    # test vector*vector
    arr = np.random.random(arr).astype(tf.tracked_float)
    arr2 = np.random.random(arr2).astype(tf.tracked_float)
    np.reshape(arr, (1000000, ))
    np.reshape(arr2, (1000000, ))
    arr = np.dot(arr, arr2)
    return arr

def test8(arr = (1000, 1000), arr2 = (1000, 1)):
    # test vector*vector
    arr = np.random.random(arr).astype(tf.tracked_float)
    arr2 = np.random.random(arr2).astype(tf.tracked_float)
    np.reshape(arr2, (1000, ))
    arr = np.dot(arr, arr2)
    return arr

def test9(arr = (10,000,000, 1)):
    # tests filters
    arr = np.random.random(arr).astype(tf.tracked_float)
    out = np.zeros(arr.shape).astype(tf.tracked_float)
    out[arr < 0.5] = arr[arr < 0.5]
    return out    

def test10(arr = 10000000):
    x = np.arange(arr)
    std = arr/5
    y = np.random.normal(x, std)
    y = np.reshape(y, (arr,1)).astype(tf.tracked_float)
    
    out = np.zeros(y.shape).astype(tf.tracked_float)
    out[y < 0.5] = arr[y < 0.5]
    return out
