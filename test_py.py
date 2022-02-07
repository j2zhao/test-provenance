
import inspect

def fun(a, b, c = 'tets', *args, k, **kwargs):
    
    print(a)
    print(b)
    print(c)
    print(args)
    print(kwargs)


fun(3, k = 4, d = 3, b=4)
print(inspect.getfullargspec(fun))