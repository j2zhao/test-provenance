import numpy as np
import copy
import os
import pickle
import numpy.core.tracked_float as tf
import time


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

    compressed = set()
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
    
        compressed.add(((temp_start, last_value), col))
    
    return compressed

prov_types = ('absabs', 'absrel', 'relabs', 'relrel')
def prov_eq_simple(prov1, prov2):
    prov = {}
    for pt in prov_types:
        if pt in prov1 and pt in prov2:
            if prov1[pt] == prov2[pt]:
                prov[pt] = copy.deepcopy(prov1[pt]) # maybe don't need deepcopy
    return prov

def prov_eq_ids(prov1, prov2):
    prov = {}
    if len(prov1) != len(prov2):
        return {}
    for id in prov1:
        if id not in prov2:
            return {}
        else:
            pr = prov_eq_simple(prov1[id], prov2[id])
            if len(pr) == 0:
                return {}
            else:
                prov[id] = pr
    return prov

def prov_eq(prov1, prov2, merge_id = False):
    if not merge_id:
        return prov_eq_simple(prov1, prov2)
    else:
        return prov_eq_ids(prov1, prov2)

def compress_output(prov_arr, merge_id = False, id = None):
    '''
    if merge_id == True -> we compress with all ids
    else: we compress by specified id -> requires id field
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
                if id != None and id in prov_arr[row, col]:
                    temp_start = col
                    last_value = col
                    prov1 = prov_arr[row, col][id]
                elif len(prov_arr[row, col]) != 0:
                    temp_start = col
                    last_value = col
                    prov1 = prov_arr[row, col]
            else:
                # check for provenance and match
                if id != None and id in prov_arr[row, col]:
                    prov2 = prov_eq(prov1, prov_arr[row, col][id], merge_id)
                elif id != None and id not in prov_arr[row, col]:
                    prov2 = -1
                elif len(prov_arr[row, col]) == 0:
                    prov2 = -1
                else:
                    prov2 = prov_eq(prov1, prov_arr[row, col], merge_id)
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
                    if id != None:
                        prov1 = prov_arr[row, col][id]
                    else:
                        prov1 = prov_arr[row, col]
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
                        prov = prov_eq(prev[2], interval[2], merge_id)
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

def convert_to_relative(prov_interval, row, col):
    absabs = []
    absrel = []
    relabs = []
    relrel = []
    for ((start0, end0), (start1, end1)) in prov_interval:
        absabs.append(((start0, end0), (start1, end1)))
        absrel.append(((start0, end0), (start1 - col, end1 - col)))
        relabs.append(((start0 - row, end0 - row), (start1, end1)))
        relrel.append(((start0 - row, end0 - row), (start1 - col, end1 -col)))
    result = {'absabs': absabs, 'absrel': absrel, 'relabs': relabs, 'relrel': relrel}
    return result
    
def compression(prov_arr, separate_by_ids = True):
    '''
    separate_by_ids: if True, try to compress by different ids
    separate_by_ids: if False, try to compress all input arrays in separate lineage
    '''
    # change this format based on whatever
    # need to change to -1 for compression
    # merge inputs
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
    
    # attempt to alto
    start = time.time()
    for row in range(prov_arr.shape[0]):
        for col in range(prov_arr.shape[1]):
            for id in cell_prov[row, col]:
                cell_prov[row, col][id] = convert_to_relative(cell_prov[row, col][id], row, col)
    
    #  print(start)
    end = time.time()
    print('conversion to relational: {}'.format(end - start))

    start = time.time()
    # print(start)
    # merge outputs
    if separate_by_ids:
        output = {}
        for id in ids:
            output[id] = compress_output(cell_prov, merge_id = False, id = id)
    else:
        output = compress_output(cell_prov, merge_id = True)
    
    end = time.time()
    print('output compression: {}'.format(end - start))
    return output


def generate_array(size = (1000, 1000)):
    arr = np.random.random(size).astype(tf.tracked_float)
    tf.initialize(arr, 0)
    return arr


if __name__ == '__main__':


    # base = './logs'
    # prov = np.load('logs/1632433790.421791.npy', allow_pickle=True)

    prov = generate_array((1000000, 1))
    compressed = compression(prov, separate_by_ids = True)
    print(compressed)
