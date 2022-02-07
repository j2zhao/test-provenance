from numpy.lib.arraysetops import isin
from numpy.lib.npyio import save
from compression import compression
import numpy as np
import numpy.core.tracked_float as tf
import inspect
import json
import copy
import sys
import constants


def match_provenance(prov1, prov2, arr_tup, arr_tup):
    for i in range(prov)




# deal with list of arrays -> TODO later
# arguments can't change type
class FunctionProvenance():
    """
    provenance tuple:
    (args_list, provenance, number of passes)
    
    """
    def __init__(self, log, n = 2, m = 2, ratio = 0.1, saved_prov = None):
        # if saved_function:
        #     with open(saved_function, 'r') as f:
        #         self.prov = json.load(f)
        # else:
        #     self.prov = {}

        if saved_prov:
            self.prov = saved_prov
        else:
            self.prov = {}
        
        self.log = log
        self.ratio = ratio
        self.n = n
        self.m = m

    def _run_func(self, func, args, kwargs):

        # initialize inputs
        oargs = []
        i = 0
        for arr in enumerate(args):
            if isinstance(arr, np.array):
                oargs.append(tf.initialize(arr.astype(tf.tracked_float), i))
                i += 1
            else:
                oargs.append(arr)

        okwargs = {}
        for k, v in kwargs:
            if isinstance(v, np.array):
                okwargs[k] = tf.initialize(v.astype(tf.tracked_float), i)
                i += 1
            else:
                okwargs[k] = v 

        # run function

        output = func(*oargs, **okwargs)
        return output


    def _new_function(self, func):
        args = inspect.getfullargspec(func)
        nfunc = func.__name__
        self.prov[nfunc] = {}
        self.prov[nfunc]['args'] = args
        self.prov[nfunc]['provs'] = []
        self.prov[nfunc]['val_args'] = []
        self.prov[nfunc]['arb_args'] = []

        # size array related
        self.prov[nfunc]['arr_args'] = []
        # related to current program
        # provenance tuple:
        #{ARRAY_ARGS: [(args_list, provenance, number of passes)]}
        self.prov[nfunc]['cur_provs'] = {}

    def _get_prov(self, nfunc, args, kwargs):
        # update with shape
        oargs = []
        for arr in enumerate(args):
            if isinstance(arr, np.array):
                oargs.append(arr.shape)
            else:
                oargs.append(arr)

        fargs = {}
        dargs = self.prov[nfunc]['args'].args
        vargs = self.prov[nfunc]['args'].varargs
        vkwargs = self.prov[nfunc]['args'].varkw
        defargs = self.prov[nfunc]['args'].default
        kwdargs = self.prov[nfunc]['args'].kwonlyargs
        defkwargs = self.prov[nfunc]['args'].kwonlydefaults

        used_kw = []

        if defargs != None:
            default_start = len(dargs) - len(defargs)
        else:
            default_start = -1
        
        for i, arg in iter(dargs.args):
            if i < len(oargs):
                fargs[arg] = oargs[i]
            elif arg in kwargs:
                fargs[arg] = kwargs[arg]
                used_kw.append(arg)
            elif i >= default_start:
                index = i - default_start
                fargs[arg] = defargs[index]
            else:
                raise NameError('arguments do not match function')
        
        if len(oargs) > len(dargs):
            if vargs == None:
                raise NameError('arguments do not match function')
            else:
                fargs[vargs] = oargs[len(dargs):]
        else:
            fargs[vargs] = []

        for k in kwdargs:
            if k in kwargs:
                fargs[k] = kwargs[k]
                used_kw.append(k)
            elif k in defkwargs:
                fargs[k] = defkwargs[k]
            else:
                raise NameError('arguments do not match function')
        
        fargs[vkwargs] = {}
        for k in kwargs:
            if k not in used_kw:
                fargs[vkwargs][k] = kwargs[k]
        
        arg_arrs = self.prov[nfunc]['arr_args']
        arrs_shape = []
        for key, val in fargs:
            if isinstance(val, np.ndarray):
                if key not in arg_arrs:
                    arg_arrs.append(key)
                arrs_shape.append((key, val.shape))
                fargs[key] = constants.ARRAY
        return fargs, tuple(arrs_shape)
    
    def prov_function(self, func, args, kwargs):
        ''''
        Returns: output of function, compressed provenance (if existing), version in database
        0: not existing
        1: existing but less than min passes
        2: existing but stated as unknown
        3: existing and returned

        0 -> general
        1 -> specific
        '''
        type = -1
        #initialize new function dictionary
        nfunc = func.__name__
        if nfunc not in self.prov:
            self._new_function(func)

        # get argument and set with arbitrary values
        arg_dic, arr_tup = self._get_prov(nfunc, args, kwargs)
        
        provs = self.prov[nfunc]['provs']
        
        aarg_dic = copy.deepcopy(arg_dic)
        arbs = self.prov[nfunc]['arb_args']
        for a in arbs:
            if a in aarg_dic:
                aarg_dic[a] = constants.ARB
        
        # try to find match in general provenance (linear right now -> probably good enough)
        provenance = None
        num_pass = 0
        for prov in provs:
            if prov[0] == aarg_dic:
                provenance = prov[1]
                num_pass = prov[2]
        
        if provenance != None:
            type = 0
        # try to find match in specific provenance
        else:
            cur_provs = self.prov[nfunc]['cur_provs']
            if arr_tup in cur_provs:
                for prov in cur_provs[arr_tup]:
                    if prov[0] == arg_dic:
                        provenance = prov[1]
                        num_pass = prov[2]
            if provenance != None:
                type = 1

        # if there isn't a match, we should run the full program
        if provenance == None:
            output = self._run_func(self, func, args, kwargs)
            return arg_dic, arr_tup, output, None, 0, type

        # if there is a match but the number of times we've seen a program is < n, we should run the number program
        elif num_pass < self.n:
            output = self._run_func(self, func, args, kwargs)
            return arg_dic, arr_tup, output, None, 1, type

        # if there is a match but we couldn't find a pattern
        elif provenance == constants.UNKNOWN:
            output = self._run_func(self, func, args, kwargs)
            return arg_dic, arr_tup, output, None, 2, type
        
        # return just the function if there is a match
        else:
            output = func(*args, **kwargs)
            return arg_dic, arr_tup, output, provenance, 3, type

    # test if getsize of works
    def compress_function(self, prov):
        compress_prov = compression(prov)
        if sys.getsizeof(compress_prov) > self.ratio*sys.getsizeof(prov):
            return compress_prov, True
        else:
            return prov, False

    def _gen_tup(self, tup, dict_arr):
        a, b = tup
        if a in dict_arr and b in dict_arr:
            return (dict_arr[a], dict_arr[b])
        elif a in dict_arr:
            return (dict_arr[a], b)
        elif b in dict_arr:
            return (a, dict_arr[b])
        else:
            return (a, b)
    
    def size_function(self, provenance, arr_tup):
        dict_arr = {}
        
        for tup in arr_tup:
            key, shape = tup
            dict_arr[shape[0]] = key + '0'
            dict_arr[shape[1]] = key + '1'

        for id in provenance:
            for i, val in enumerate(provenance[id]):
                w, h, prov = val 
                w = self._gen_tup(w, dict_arr)
                h = self._gen_tup(h, dict_arr)
                for key in prov:
                    for i, val in enumerate(prov[key]):
                        a, b = val
                        a = self._gen_tup(a, dict_arr)
                        b = self._gen_tup(b, dict_arr)
                        prov[key][i] = (a, b)
                provenance[id][i] = (w, h, prov)
        return provenance


    def add_prov(self, provenance, nfunc, args, arr_args, add_arb = True):
        provs = self.prov[nfunc]['provs']
        arb_vals = self.prov[nfunc]['arb_args']
        cur_provs = self.prov[nfunc]['cur_provs']
        found = False
        
        arb_args = {}
        for arg in args:
            if arg in arb_vals:
                arb_args[arg] = constants.ARB
            else:
                arb_args[arg] = args[arg]
        rel_prov = self.size_function(copy.deepcopy(provenance), arr_args)
        for prov in provs:
            if prov[0] == arb_args:
                if prov[1] != rel_prov:
                    prov[1] = constants.UNKNOWN
                prov[2] += 1
                found == True
                break
        if not found:
            full_provenance = (arb_args, rel_prov, 1)
            self.prov[nfunc]['provs'].append(full_provenance)

        found_2 = False
        if arr_args in cur_provs:
            for prov in cur_provs[arr_args]:
                if prov[0] == args:
                    if prov[1] != provenance:
                        prov[1] = constants.UNKNOWN
                    prov[2] += 1
                    found_2 == True
                    break
            if not found_2:
                full_provenance = (arb_args, provenance, 1)
                self.prov[nfunc]['cur_provs'][arr_args].append(full_provenance)
        else:
            full_provenance = (arb_args, provenance, 1)
            self.prov[nfunc]['cur_provs'][arr_args] = [full_provenance]
        
        if add_arb:
            self.update_arbitrary(self, nfunc, arb_args, provenance)
        return arb_args, found
    
    def add_log(self, line, arrays, nfunc, args, arr_tup, prov):
        ### NEED TO DEAL WITH ARRAYS MAINLY
        # if isinstance(prov, np.array):
        #     array_name = ''
        #     np.save(array_name, prov) #need to generate new array with provenance?
        #     with open(self.log) as f:
        #         tup = (line, arrays, nfunc, args, array_name, 0)
        #         f.write(str(tup)) # does this work?  i might need to convert customly?
        with open(self.log) as f:
            tup = (line, arrays, nfunc, args, arr_tup, prov, 1)
            f.write(str(tup)) # does this work?  i might need to convert customly??
    
    def save(self, file_name):
        with open(file_name) as f:
            json.dump(self.prov, f)
    

    def _update_arbitrary(self, nfunc, arb_list):
        '''this function checks it with all the other ones
        this is O(prov^2) because of the fact that arguments can be mutable -> preserve functionality over speed '''
        provs = self.prov[nfunc]['provs']        
        #find a list of qualified values
        failed_args = []
        
        # check every argument
        for arg in arb_list:
            failed = False
            #for every provenance convert to arbitrary
            new_provs = []
            for prov in provs:
                if arg in prov[0]:
                    unique = True
                    prov[arg] = constants.ARB
                    # this is the expensive check?
                    for prov2 in new_provs:
                        if prov2[0] == prov[0]:
                            unique = False
                            if prov2[1] != prov[1]:
                                failed_args.append(arg)
                                failed = True
                                break
                    if failed:
                        break
                    if unique:
                        new_provs.append(prov)

        # find new list of arbitrary args
        success = []
        for arg in arb_list:
            if arg not in failed_args:
                success.append(arg)
        
        new_provs = []
        for prov in provs:
            for arg in prov[0]:
                if arg in success:
                    prov[0][arg] = constants.ARB
            
            dup = False
            for prov2 in new_provs:
                if prov[0] == prov2[0]:
                    dup = True
                    break
            if not dup:
                new_provs.append(prov)
        
        self.prov[nfunc]['provs'] = new_provs
        self.prov[nfunc]['arb_args'].append(success)
        return new_provs, success

    def update_arbitrary(self, nfunc, args, provenance):
        ''' '''
        provs = self.prov[nfunc]['provs']
        val_args = self.prov[nfunc]['val_args']
        arb_list = self.prov[nfunc]['arb_args']
        arr_list = self.prov[nfunc]['arrs_args']
        arg_count = {}
        for prov in provs:
            if prov[1] == provenance:
                for k in args:
                    if k in val_args or k in arb_list or k in arr_list:
                        continue
                    if k in prov[0] and args[k] != prov[0][k]:
                        if k in arg_count:
                            arg_count[k] += 1
                        else:
                            arg_count[k] = 1

        arb_list = []
        for arg in arg_count:
            if arg_count[arg] >= self.m:
                arb_list.append(arg)
        
        arb_list = self._update_arbitrary(nfunc, arb_list)




    


