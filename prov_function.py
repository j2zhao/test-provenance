from prov_check import FunctionProvenance
import csv
import numpy as np
import sklearn
import random
import constants

srange = [0, 10000]

# run function and saves to update later
def prov_function():
    pass


# run function and updates right away
def prov_function2():
    pass

# generates a list of function given arguments 
def iterate_parameters(arr_num, other_args):
    arrays = []
    for _ in range(arr_num):
        d1, d2 = random.randrange(srange[0], srange[1])
        arrays.append(np.random.rand(d1, d2))

    other = {}
    for key, val in other_args:
        index = random.randrange(len(val))
        other[key] = val[index]
    return arrays, other
        
def generate_new_array(old_arrays):
    arrays = []
    for arr in range(len(old_arrays)):
        d1, d2 = arr.shape
        arrays.append(np.random.rand(d1, d2))
    return arrays

# run one run of experiments
def run(provenance, func, arrays, args, repetition, log):
    prov_obj = FunctionProvenance(log = log, saved_prov=provenance)
    for i in range(repetition):
        print('Function {} Run {}'.format(func.__name__, i))
        if i != 0:
            arrays = generate_new_array(arrays)
        arg_dic, arr_tup, output, prov, t1, t2 = prov_obj.prov_function(func, arrays, args)
        if t2 == 0:
            print('type: general')
        elif t2 == 1:
            print('type: specfic')
        else:
            print('type: None')
        
        if t1 == 0:
            print('No Match')
        elif t1 == 1:
            print('Not enough iterations')
        elif t1 == 2:
            print('Pattern Unknown')
        elif t1 == 3:
            print('Pattern Found')
        
        if prov == None:
            provenance, compressed = prov_obj.compress_function(output)
            if not compressed:
                print('Not Compressed')
                provenance = constants.UNKNOWN
            else:
                print('Compressed')
            
            prov_obj.add_prov(provenance, func.__name__, arg_dic, arr_tup)
            prov_obj.add_log(0, 0, func.__name__, arg_dic, arr_tup, provenance)

# given a csv of functions, generates a list of functions runs
def run_functions(file = 'functions.csv', num = 5, rep = 3):
    with open(file, 'w') as f:
        reader = csv.reader(f)
        for row in reader:
            n = len(row)
            nfunc = row[0]
            arr_num = int(row[1])
            if n > 2:
                other_args = row[2:]
            else:
                other_args = []
            print(n)
            func = getattr(np, nfunc)
            print(nfunc)
            raise ValueError()
            provenance = None
            for i in range(num):
                arrays, args = iterate_parameters(arr_num, other_args)
                run(provenance, func, arrays, args, rep, './log/test.txt')


if __name__ == '__main__':
    run_functions()