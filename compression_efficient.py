import re
import numpy as np
import copy
import os
import pickle
import numpy.core.tracked_float as tf
import time

prov_types = ['abs', 'rel0', 'rel1']

def sort_(prov):
    s = ''
    for p in prov:
        s = s + str(p)
    return int(s)

def compress_input(prov_list):
    prov_list.sort(key=sort_)
    compressed_col = {}
    temp_start = -1
    last_value = -1
    cur_row = -1
    # don't miss last one
    for prov in prov_list:
        if temp_start == -1:
            temp_start = prov[1]
            last_value = prov[1]
            cur_row = prov[0]
        elif cur_row != prov[0]:
            if (temp_start, last_value) not in compressed_col:
                compressed_col[(temp_start, last_value)] = []
            compressed_col[(temp_start, last_value)].append(cur_row)
            temp_start = prov[1]
            last_value = prov[1]
            cur_row = prov[0]

        elif last_value == prov[1] -1:
            last_value = prov[1]
        else:
            if (temp_start, last_value) not in compressed_col:
                compressed_col[(temp_start, last_value)] = []
            compressed_col[(temp_start, last_value)].append(cur_row)
            temp_start = prov[1]
            last_value = prov[1]
    
    if temp_start != -1 and (temp_start, last_value) not in compressed_col:
        compressed_col[(temp_start, last_value)] = []
    compressed_col[(temp_start, last_value)].append(cur_row)

    compressed = []
    for col in compressed_col:
        temp_start = -1
        last_value = -1
        for row in compressed_col[col]:
            if temp_start == -1:
                temp_start = row
                last_value = row
            elif last_value == row - 1:
                last_value = row
            else:
                compressed.add(((temp_start, last_value), col))
                temp_start = row
                last_value = row
    
        compressed.append(((temp_start, last_value), col))
    
    return compressed

# to fix
def prov_eq(prov1, prov2):
    if len(prov1) != len(prov2):
        return ()
    prov = []
    for i in range(len(prov1)):
        prov.append({})
        found = False
        for t in prov1[i]:
            if t in prov2[i] and prov2[i][t] == prov1[i][t]:
                prov[i][t] = prov2[i][t]
                found = True
        if not found:
            return ()
    return tuple(prov)

def convert_to_relative(prov_interval, row, col):
    abs0 = []
    abs1 = []

    rel00 = []
    rel01 = []

    rel10 = []
    rel11 = []

    for ((start0, end0), (start1, end1)) in prov_interval:
        abs0.append((start0, end0))
        rel00.append((start0 - row, end0 - row))
        rel01.append((start0- col, end0 - col))

        abs1.append((start1, end1))
        rel10.append((start1 - row, end1 - row))
        rel11.append((start1 - col, end1 - col))

    return ({'abs': abs0, 'rel0': rel00, 'rel1':rel01}, {'abs': abs1, 'rel0': rel10, 'rel1':rel11})
    #return {'abs': {0: abs0, 1: abs1}, 'rel0': {0:rel00, 1:rel10}, 'rel1': {0:rel01, 1:rel11}}
    

#to fix
def compress_output(prov_arr, id, relative = True):
    '''
    we compress by specified id -> requires id field
    '''
    compressed = []
    prev_compressed_col = None
    for row in range(prov_arr.shape[0]):
        temp_start = -1
        last_value = -1
        prov1 = -1
        compressed_col = []
        for col in range(prov_arr.shape[1]):
            # if previous interval is empty
            if prov1 == -1:
                if id in prov_arr[row, col]:
                    temp_start = col
                    last_value = col
                    prov1 = convert_to_relative(prov_arr[row, col][id])
                # need to check alternative??
            else:
                # check for provenance and match
                if id in prov_arr[row, col]:
                    if relative:
                        prov2 = prov_eq(prov1, convert_to_relative(prov_arr[row, col][id]))
                    else:
                        if prov1 == convert_to_relative(prov_arr[row, col][id]):
                            prov2 = prov1
                        else:
                            prov2 = ()
                elif id not in prov_arr[row, col]:
                    prov2 = -1
                # compression step
                if prov2 == -1:
                    compressed_col.append(((row, row), (temp_start, last_value), prov1))
                    temp_start = -1
                    last_value = -1
                    prov1 = prov2
                elif len(prov2) == 0:
                    compressed_col.append(((row, row), (temp_start, last_value), prov1))
                    temp_start = col
                    last_value = col
                    prov1 = convert_to_relative(prov_arr[row, col][id])
 
                else:
                    last_value = col
                    prov1 = prov2
        if prov1 != -1:
            compressed_col.append(((row, row), (temp_start, last_value), prov1))

        if prev_compressed_col == None:
            prev_compressed_col = compressed_col

        else:
            prev_index = 0
            new_compressed_col = []
            for interval in compressed_col:
                while True:
                    if prev_index >= len(prev_compressed_col):
                        new_compressed_col.append(interval)
                        break
                    prev = prev_compressed_col[prev_index]
                    if prev[1][0] < interval[1][0]:
                        compressed.append(prev)
                        prev_index +=1
                    elif prev[1][0] == interval[1][0]:
                        if relative:
                            prov = prov_eq(prev[2], interval[2])
                        else:
                            if prev[2] == interval[2]:
                                prov = prev[2]
                            else:
                                prov = ()
                        if prev[1][1] == interval[1][1] and len(prov) != 0:
                            new_compressed_col.append(((prev[0][0], interval[0][1]), interval[1], prov))
                            prev_index +=1
                        else:
                            compressed.append(prev)
                            prev_index +=1
                            new_compressed_col.append(interval)
                        break
                    else:
                        new_compressed_col.append(interval)
                        break
            if len(prev_compressed_col) > prev_index:
                for i in range(prev_index, len(prev_compressed_col)):
                    compressed.append(prev_compressed_col[i])
            prev_compressed_col = new_compressed_col

    if prev_compressed_col != None:
        compressed += prev_compressed_col
    return compressed

            

def divide_by_id(prov):
    prov_dict = {}
    for p in prov:
        if p[0] not in prov_dict:
            prov_dict[p[0]] = []
        prov_dict[p[0]].append((p[1], p[2]))
    return prov_dict


def compression(prov_arr, relative = True):
    '''
    separate_by_ids: if True, try to compress by different ids
    separate_by_ids: if False, try to compress all input arrays in separate lineage
    '''
    # change this format based on whatever
    # need to change to -1 for compression
    # merge inputs
    if len(prov_arr.shape) == 1:
        prov_arr = np.reshape(prov_arr, (prov_arr.shape[0], 1))
    ids = set()
    cell_prov = np.zeros(prov_arr.shape, dtype=object)
    start = time.time()
    for row in range(prov_arr.shape[0]):
        for col in range(prov_arr.shape[1]):
            prov_dict = divide_by_id(prov_arr[row, col].provenance)
            compress = {}
            for id in prov_dict:
                compress[id] = compress_input(prov_dict[id])
                ids.add(id)
            cell_prov[row, col] = compress
    # print(start)
    end = time.time()
    print('cell level compression: {}'.format(end - start))
    # convert to relative -> only do this by dimension and id, not by interval

    start = time.time()
    output = {}
    for id in ids:
        output[id] = compress_output(cell_prov, id = id, relative = relative)
    end = time.time()
    print('output compression: {}'.format(end - start))
    return output


def generate_array(size = (1000, 1000)):
    # arr = np.random.random(size).astype(tf.tracked_float)
    # tf.initialize(arr, 0)
    # arr2 = np.random.random(size).astype(tf.tracked_float)
    # tf.initialize(arr2, 2)
    # arr = np.dot(arr, arr2)

    arr = np.random.random(arr).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr2 = np.random.random(arr2).astype(tf.tracked_float)
    tf.initialize(arr, 2)
    np.reshape(arr, (10000000, ))
    np.reshape(arr2, (10000000, ))
    arr = np.dot(arr, arr2)
    
    return arr


if __name__ == '__main__':


    # base = './logs'
    # prov = np.load('logs/1632433790.421791.npy', allow_pickle=True)

    prov = generate_array()
    compressed = compression(prov)
    print(compressed)
