
import numpy as np
import operator
import functools
import copy 
from aux_functions import *

def add_provenance(orig_func):
    @functools.wraps(orig_func)
    def funct(ref, *args):
        args2 = []
        provenance = copy.copy(ref.provenance) # need to copy? potentially expensive
        for arg in args:
            if hasattr(arg, 'provenance') and hasattr(arg, 'value'):
                provenance += arg.provenance
                args2.append(arg.value)
            else:
                args2.append(arg.value)
        print(*args2)
        value = orig_func(ref,*args2)
        output = ref.__class__(value, provenance)
        return output
    return funct



class TrackedFloat(object):
    
    
    def __init__(self, value, id):
        self.value = value
        
        self.id = id
        
        if id == None:
            self.provenance = []
        elif isinstance(id, list):
            self.provenance = id
        else:
            self.provenance = [id]
    
    
    def reset(self, id):
        self.id = id
        if id == None:
            self.provenance = []
        elif isinstance(id, list):
            self.provenance = id
        else:
            self.provenance = [id]

    @add_provenance
    def __add__(self, other):
        return self.value + other

    # def __radd__(self, other):
    #     if hasattr(other, 'provenance'):
    #         return TrackedFloat(self.value + other.value, self.provenance + other.provenance)
    #     else:
    #         return TrackedFloat(self.value + other.value, self.provenance)
    
    # def __iadd__(self, other):
    #     if hasattr(other, 'provenance'):
    #         self.provenance = self.provenance + other.provenance
    #         self.value = self.value + other.value
    #         return self
    #     else:
    #         self.value = self.value + other.value
    #         return self

    # def __sub__(self, other):
    #     if hasattr(other, 'provenance'):
    #         return TrackedFloat(self.value - other.value, self.provenance + other.provenance)
    #     else:
    #         return TrackedFloat(self.value - other.value, self.provenance)

    # def __rsub__(self, other):
    #     if hasattr(other, 'provenance'):
    #         return TrackedFloat(other.value - self.value, self.provenance + other.provenance)
    #     else:
    #         return TrackedFloat(other.value - self.value, self.provenance)
    
    # def __isub__(self, other):
    #     if hasattr(other, 'provenance'):
    #         self.provenance = self.provenance + other.provenance
    #         self.value = self.value - other.value
    #         return self
    #     else:
    #         self.value = self.value - other.value
    #         return self
    
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return str(self.value)

# why get and set? -> set might make sense

arr = np.empty((3, 1), dtype=object)
arr[0] = TrackedFloat(0, None)
arr[1] = TrackedFloat(5, None)
arr[2] = TrackedFloat(8, None)
reset_array_prov(arr)
print(arr.sum().provenance)
save_array_prov(arr, './logs')
