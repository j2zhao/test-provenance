#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <math.h>
#include "common.h"
#include "ndarraytypes.h"


/* Uncomment the following line to work around a bug in numpy */
/* #define ACQUIRE_GIL */

static void
set_overflow(void) {
#ifdef ACQUIRE_GIL
    /* Need to grab the GIL to dodge a bug in numpy */
    PyGILState_STATE state = PyGILState_Ensure();
#endif
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_OverflowError,
                "overflow in (tracked) float arithmetic");
    }
#ifdef ACQUIRE_GIL
    PyGILState_Release(state);
#endif
}

// static void
// set_zero_divide(void) {
// #ifdef ACQUIRE_GIL
//     /* Need to grab the GIL to dodge a bug in numpy */
//     PyGILState_STATE state = PyGILState_Ensure();
// #endif
//     if (!PyErr_Occurred()) {
//         PyErr_SetString(PyExc_ZeroDivisionError,
//                 "zero divide in (tracked) float arithmetic");
//     }
// #ifdef ACQUIRE_GIL
//     PyGILState_Release(state);
// #endif
// }



typedef struct {
    npy_int id;
    /*interval for indices */
    /* starts at + 1 */
    npy_int start_0;
    npy_int start_1;
    /*overflow data*/
} provenance;


typedef struct {
    /* numerator */
    npy_float64 n;
    provenance p;
    npy_int size;
    npy_int prov_size;
    provenance* overflow;
    } tracked_float;

static NPY_INLINE int
tfloat_nonzero(tracked_float x) {
    return x.n!=0;
}

static NPY_INLINE int
sign(npy_float64 n) {
    return n<0?-1:n==0?0:1;
}

static NPY_INLINE tracked_float
make_tfloat(npy_float64 n, npy_int id, npy_int start_0, npy_int start_1) {
    provenance prov = {id, start_0, start_1};
    if (start_0 == -1) {
        tracked_float r = {n, prov, 0, 0, NULL};
        return r;
    }
    else {
        tracked_float r = {n, prov, 1, 0, NULL};
        return r;
    }
}

/* overflows don't change between programs */
static NPY_INLINE tracked_float
make_tfloat_prov1(npy_float64 n, tracked_float a) {
    a.n = n;
    return a;
}

static NPY_INLINE tracked_float
make_tfloat_prov2(npy_float64 n, tracked_float a, tracked_float b) {
    npy_int size0 = a.size;
    npy_int size1 = b.size;
    npy_int size = size0 + size1;
    size_t p_size = sizeof(provenance);
    /* we have history */
    if (size == 0) {
        a.n = n;
        return a;
    } else {
        if (size0 == 0) {
            a.size = b.size;
            a.prov_size = b.prov_size;
            a.p = b.p;
            a.overflow = b.overflow;
            a.n = n;
            return a;
        } else {
            if (a.prov_size - size0 + 1 >= b.size) {
                memcpy(a.overflow + size0 - 1, &a.p, p_size);
                memcpy(a.overflow + size0, &a.overflow, p_size);
                a.size = size;
                a.n = n;
                return a;
            } else {
                int new_size = (size1 + size0 - 1)*2;
                provenance* overflow;
                overflow = (provenance*) malloc((new_size)*p_size);
                memcpy(overflow, a.overflow, (size0 - 1)*p_size);
                free(a.overflow);
                memcpy(overflow + size0 - 1, b.p, p_size);
                if (size1 > 1) {
                    memcpy(overflow + size0 - 1, b.p, (size1 - 1)*p_size);
                }
                a.prov_size = new_size;
                a.size = size;
                a.n = n;
                return a;
            }
        }
    }
        
}

static NPY_INLINE int 
append_prov(provenance* p, provenance* of, tracked_float* a) {
    npy_int s = a -> size;
    if (s == 0) {
        return s;
    }
    memcpy(p, &(a -> p), sizeof(provenance));
    if (s > 1) {
        memcpy(of, a -> overflow, sizeof(provenance)*(s - 1));
    }
    return s;
}

static NPY_INLINE int 
append_prov_of(provenance* of, tracked_float* a) {
    npy_int s = a -> size;
    if (s == 0) {
        return s;
    }
    memcpy(of, &(a-> p), sizeof(provenance));
    if (s > 1) {
        memcpy(of + 1, a -> overflow, sizeof(provenance)*(s - 1));
    }
    return s;
}


/* Expose tracked_float to Python as a numpy scalar */


typedef struct {
    PyObject_HEAD
    tracked_float f;
} PyTFloat;

static PyTypeObject PyTFloat_Type;


static NPY_INLINE int
PyTFloat_Check(PyObject* object) {
    return PyObject_IsInstance(object,(PyObject*)&PyTFloat_Type);
}

static PyObject*
PyTFloat_FromTFloat(tracked_float x) {
    PyTFloat* p = (PyTFloat*)PyTFloat_Type.tp_alloc(&PyTFloat_Type, 0);
    if (p) {
        p->f = x;
    }
    return (PyObject*)p;
}

/*
 * Returns Py_NotImplemented on most conversion failures, or raises an
 * overflow error for too long ints
 */
#define AS_DOUBLE(dst,object) \
    { \
        if (PyTFloat_Check(object)) { \
            dst = ((PyTFloat*)object)->f.n; \
        } \
        else if (PyFloat_Check(object)){ \
            dst = PyFloat_AsDouble(object); \
        }  else if (PyLong_Check(object)) { \
            dst = PyLong_AsDouble(object); \
        } else { \
            Py_INCREF(Py_NotImplemented); \
            return Py_NotImplemented; \
        } \
    } \

#define AS_DOUBLE_2(dst,object) \
    { \
        if (PyTFloat_Check(object)) { \
            dst = ((PyTFloat*)object)->f.n; \
        } \
        else if (PyFloat_Check(object)){ \
            dst = PyFloat_AsDouble(object); \
        }  else if (PyLong_Check(object)) { \
            dst = PyLong_AsDouble(object); \
        } else { \
            Py_INCREF(Py_NotImplemented); \
            return -1; \
        } \
    } \

// no error checking
// only make one type for now
//initialize with only one provenance
static PyObject*
pytfloat_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    npy_float64 n = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 0));
    npy_int id = PyLong_AsLong(PyTuple_GET_ITEM(args, 1));
    npy_int start_0 = PyLong_AsLong(PyTuple_GET_ITEM(args, 2));
    npy_int start_1 = PyLong_AsLong(PyTuple_GET_ITEM(args, 3));
    tracked_float r = make_tfloat(n, id, start_0, start_1);
    return PyTFloat_FromTFloat(r);
}

static PyObject*
pytfloat_richcompare(PyObject* a, PyObject* b, int op) {
    npy_float64 x;
    npy_float64 y;
    AS_DOUBLE(x, a)
    AS_DOUBLE(y, b)
    int result;
    
    #define OP(py, op) case py: \
        result = x op y; \
        break; 
    
    switch (op) {
        OP(Py_LT, <)
        OP(Py_LE, <=)
        OP(Py_EQ, ==)
        OP(Py_NE, !=)
        OP(Py_GT, >)
        OP(Py_GE, >=)
    };
    #undef OP
    return PyBool_FromLong(result);
}

//does not work -> TODO think about fixing
// static char*
// provenance_repr(provenance p, provenance* overflow) {
//     int rsize;
//     if (overflow != NULL) {
//         rsize = sizeof(*overflow)/sizeof(provenance) + 1;
//     }
//     else {
//         rsize = 1;
//     }
//     char output[rsize*50];
//     int offset = 0;
//     provenance q = p;
//     int x;
//     for (x = 0; x < rsize; x ++) {
//         offset += sprintf(output + offset, "[(%d,%d,%d,%d,%d,%d)]", 
//             q.id, q.start_0, q.start_1, q.end_0, q.end_1, q.type);
//         if (x > 0 && x < rsize -1 ) {
//             offset += sprintf(output + offset, " , ");
//         }
//         if (x != rsize - 1) {
//             q = overflow[x];
//         }
//     }
//     return output;
    
// }

// static PyObject*
// pytfloat_repr(PyObject* self) {
//     tracked_float x = ((PyTFloat*)self)->f;
//     const char* c = provenance_repr(x.p, x.overflow);
//     return PyUnicode_FromFormat("%d , %V", x.n, c);
// }

// static PyObject*
// pytfloat_str(PyObject* self) {
//     return pytfloat_repr(self);
// }

static npy_hash_t
pytfloat_hash(PyObject* self) {
    tracked_float x = ((PyTFloat*)self)->f;
    /* Use a extremely weak hash as Python expects ?*/
    long h = 131071*x.n+524287*x.size;
    /* Never return the special error value -1 */
    return h==-1?2:h;
}

#define TFLOAT_BINOP(name, exp) \
    static PyObject* \
    pytfloat_##name(PyObject* a, PyObject* b) { \
        tracked_float z;\
        int a_tf = PyTFloat_Check(a); \
        int b_tf = PyTFloat_Check(b); \
        npy_float64 x; \
        AS_DOUBLE(x, a);\
        npy_float64 y; \
        AS_DOUBLE(y, b);\
        npy_float64 n = (npy_float64) exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        if (a_tf && b_tf) { \
            z = make_tfloat_prov2(n, ((PyTFloat*) a) ->f, ((PyTFloat*) b) ->f); \
        } else if (a_tf) { \
            z = make_tfloat_prov1(n, ((PyTFloat*) a) -> f); \
        } else if (b_tf) { \
            z = make_tfloat_prov1(n, ((PyTFloat*) b) -> f); \
        } else { \
            return PyFloat_FromDouble((double) n); \
        } \
        return PyTFloat_FromTFloat(z); \
    }

TFLOAT_BINOP(add, x + y)
TFLOAT_BINOP(subtract, x - y)
TFLOAT_BINOP(multiply, x*y)
TFLOAT_BINOP(divide, x/y)
TFLOAT_BINOP(floor_divide, (int)floor(x / y) )

#define TFLOAT_UNOP(name, exp) \
    static PyObject* \
    pytfloat_##name(PyObject* self) { \
        tracked_float x = ((PyTFloat*)self)->f; \
        npy_float64 y = x.n; \
        npy_float64 z = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        tracked_float tf = make_tfloat_prov1(z, x); \
        return PyTFloat_FromTFloat(tf); \
    }

#define TFLOAT_UNOP_2(name, type, exp, convert) \
    static PyObject* \
    pytfloat_##name(PyObject* self) { \
        tracked_float x = ((PyTFloat*)self)->f; \
        npy_float64 y = x.n; \
        type z = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        return convert(z); \
    }

TFLOAT_UNOP(negative, -y)
TFLOAT_UNOP(absolute,fabs(y))

TFLOAT_UNOP_2(int, long, (long)y, PyLong_FromLong)
TFLOAT_UNOP_2(float,double, (double)y,PyFloat_FromDouble)

static PyObject*
pytfloat_positive(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static int
pytfloat_nonzero(PyObject* self) {
    tracked_float x = ((PyTFloat*)self)->f;
    return tfloat_nonzero(x);
}

static PyNumberMethods pytfloat_as_number = {
    pytfloat_add,          /* nb_add */
    pytfloat_subtract,     /* nb_subtract */
    pytfloat_multiply,     /* nb_multiply */
    0,                       /* nb_remainder */
    0,                       /* nb_divmod */
    0,                       /* nb_power  -> maybe? */
    pytfloat_negative,      /* nb_negative */
    pytfloat_positive,     /* nb_positive */
    pytfloat_absolute,     /* nb_absolute */
    pytfloat_nonzero,      /* nb_nonzero */
    0,                       /* nb_invert */
    0,                       /* nb_lshift */
    0,                       /* nb_rshift */
    0,                       /* nb_and */
    0,                       /* nb_xor */
    0,                       /* nb_or */
    pytfloat_int,          /* nb_int */
    0,                       /* reserved */
    pytfloat_float,        /* nb_float */

    0,                       /* nb_inplace_add */
    0,                       /* nb_inplace_subtract */
    0,                       /* nb_inplace_multiply */
    0,                       /* nb_inplace_remainder */
    0,                       /* nb_inplace_power */
    0,                       /* nb_inplace_lshift */
    0,                       /* nb_inplace_rshift */
    0,                       /* nb_inplace_and */
    0,                       /* nb_inplace_xor */
    0,                       /* nb_inplace_or */

    pytfloat_floor_divide, /* nb_floor_divide */
    pytfloat_divide,       /* nb_true_divide */
    0,                       /* nb_inplace_floor_divide */
    0,                       /* nb_inplace_true_divide */
    0,                       /* nb_index */
};

static PyObject*
pytfloat_float_get(PyObject* self, void* closure) {
    return PyFloat_FromDouble(((PyTFloat*)self)->f.n);
}

static PyObject*
pytfloat_psize_get(PyObject* self, void* closure) {
    return PyLong_FromLong(((PyTFloat*)self)->f.size);
}

static int
pytfloat_float_set(PyObject* self, PyObject* value, void* closure) {
    npy_float64 x = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return 0;
    }
    ((PyTFloat*)self)->f.n = x;
    return 1;
}


static PyObject*
pytfloat_provenance_get(PyObject* self, void* closure) {
    tracked_float f = ((PyTFloat*)self)->f;
    npy_int size = f.size;
    int i;
    provenance p = f.p;

    Py_ssize_t x = PyLong_AsSsize_t(PyLong_FromLong(size));
    PyObject* plist = PyList_New(x);
    if (size == 0) {
        return plist;
    }
    PyObject* prov = PyTuple_Pack(3, PyLong_FromLong(p.id), PyLong_FromLong(p.start_0), 
        PyLong_FromLong(p.start_1));
    PyList_SetItem(plist, 0, prov);

    for (i = 1; i < size; i++) {
        p = f.overflow[i - 1];
        PyObject* prov = PyTuple_Pack(3, PyLong_FromLong(p.id), PyLong_FromLong(p.start_0), 
            PyLong_FromLong(p.start_1));
        PyList_SetItem(plist, i, prov);
    }
    return plist;
}

static PyGetSetDef pytfloat_getset[] = {
    {(char*)"n", pytfloat_float_get, pytfloat_float_set, (char*)"float", 0},
    {(char*)"provenance",pytfloat_provenance_get,0,(char*)"list of provenance",0},
    {(char*)"pnum",pytfloat_psize_get,0,(char*)"number of predecessors",0},
    {0} /* sentinel */
};

static void pytfloat_dealloc(PyObject* self){
    if (((PyTFloat*)self)->f.overflow != NULL) {
        free(((PyTFloat*)self)->f.overflow);
    }
    Py_TYPE(self)->tp_free(self);
}

static PyTypeObject PyTFloat_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "numpy.core.tracked_float.tracked_float",  /* tp_name */
    sizeof(PyTFloat),                       /* tp_basicsize */
    0,                                        /* tp_itemsize -> am not sure if array counts as dynamic?? maybe not, but it might be more than a new object -> then again, we don't really see it*/
    pytfloat_dealloc,                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                        /* TODO: FIX tp_repr */
    &pytfloat_as_number,                    /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    pytfloat_hash,                          /* tp_hash */
    0,                                        /* tp_call */
    0,                           /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Floating number with tracking ",         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    pytfloat_richcompare,                   /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    pytfloat_getset,                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    pytfloat_new,                           /* tp_new */
    0,                                        /* tp_free */
    0,                                        /* tp_is_gc */
    0,                                        /* tp_bases */
    0,                                        /* tp_mro */
    0,                                        /* tp_cache */
    0,                                        /* tp_subclasses */
    0,                                        /* tp_weaklist */
    0,                                        /* tp_del */
    0,                                        /* tp_version_tag */
};

/* NumPy support */
static PyObject*
npytfloat_getitem(void* data, void* arr) {
    tracked_float r;
    memcpy(&r,data, sizeof(tracked_float)); // do we really need to copy? -> do we need to copy the array too?
    return PyTFloat_FromTFloat(r);
}

// do we change the whole item or the value -> whole item 
// what do we do with lineage??? -> how will we ever set the function? i suppose with custom function, 
// but that won't be vectorized
static int
npytfloat_setitem(PyObject* item, void* data, void* arr) {
    tracked_float r;
    if (PyTFloat_Check(item)) {
        r = ((PyTFloat*)item)->f;
    }
    else {
        npy_float64 n;
        AS_DOUBLE_2(n, item);
        if (error_converting(n)) {
            return -1;
        }
        r = make_tfloat(n, -1, -1, -1) ;
    }
    memcpy(data, &r, sizeof(tracked_float)); 
    return 0;
}

static NPY_INLINE void
byteswap(npy_int32* x) {
    char* p = (char*)x;
    size_t i;
    for (i = 0; i < sizeof(*x)/2; i++) {
        size_t j = sizeof(*x)-1-i;
        char t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
}

static NPY_INLINE void
byteswap_float(npy_float64* x) {
    char* p = (char*)x;
    size_t i;
    for (i = 0; i < sizeof(*x)/2; i++) {
        size_t j = sizeof(*x)-1-i;
        char t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
}

static NPY_INLINE void
byteswap_provenance(provenance* x) {
    byteswap(&(x -> id));
    byteswap(&(x -> start_0));
    byteswap(&(x -> start_1));
}

/* Note: do we have to copy over the pointer????  -> rn i say yes */

static void
npytfloat_copyswap(void* dst, void* src, int swap, void* arr) {
    tracked_float* r;
    if (!src) {
        return;
    }
    r = (tracked_float*)dst;
    memcpy(r,src,sizeof(tracked_float));
    
    if (r -> overflow != NULL) {
        int t = r-> prov_size;
        provenance* of;
        of = (provenance*) malloc((t)*sizeof(provenance));
        memcpy(of, (r -> overflow), sizeof(provenance)*t);
        r -> overflow = of;
    }
    
    if (swap) {
        byteswap_float(&r->n);
        byteswap_provenance(&r->p);
        byteswap(&r->size);
        byteswap(&r->prov_size);
        if (r -> overflow != NULL) {
            provenance* of = r -> overflow;
            int t = r -> size;
            int i;
            for(i = 0; i < t -1; i++) {
                byteswap_provenance(&of[i]);
            }
        }
    }
}

static void
npytfloat_copyswapn(void* dst_, npy_intp dstride, void* src_,
        npy_intp sstride, npy_intp n, int swap, void* arr) {
    char *dst = (char*)dst_, *src = (char*)src_;
    npy_intp i;
    if (!src) {
        return;
    }
    if (!swap && dstride == sizeof(tracked_float) && sstride == sizeof(tracked_float)) {
        memcpy(dst, src, n*sizeof(tracked_float));
        for (i = 0; i < n; i++) {
            tracked_float* r = (tracked_float*)(dst+dstride*i);
            if (r -> overflow != NULL) {
                int t = r-> prov_size;
                provenance* of;
                of = (provenance*) malloc((t)*sizeof(provenance));
                memcpy(of, r->overflow, sizeof(provenance)*(r -> size - 1));
                r -> overflow = of;
            }
        }
    }
    else {
        for (i = 0; i < n; i++) {
            npytfloat_copyswap(dst+dstride*i, src+sstride*i, swap, arr);
        }
    }
}

static int
npytfloat_compare(const void* d0, const void* d1, void* arr) {
    tracked_float x = *(tracked_float*)d0,
             y = *(tracked_float*)d1;
    npy_float64 a = x.n, b = y.n;

    return (a < b)?-1:(a=b)?0:1;
}

#define FIND_EXTREME(name, op) \
    static int \
    npytfloat_##name(void* data_, npy_intp n, \
            npy_intp* max_ind, void* arr) { \
        const tracked_float* data; \
        npy_intp best_i; \
        npy_float64 x, y; \
        npy_intp i; \
        if (!n) { \
            return 0; \
        } \
        data = (tracked_float*)data_; \
        best_i = 0; \
        y = data[0].n; \
        for (i = 1; i < n; i++) { \
            x = data[i].n; \
            if (op) { \
                best_i = i; \
                y = data[i].n; \
            } \
        } \
        *max_ind = best_i; \
        return 0; \
    }

FIND_EXTREME(argmin, x < y)
FIND_EXTREME(argmax, x > y)




// TODO: this questionably works, need testing
static void
npytfloat_dot(void* ip0_, npy_intp is0, void* ip1_, npy_intp is1,
        void* op, npy_intp n, void* arr) {
    npy_float64 r = 0;
    tracked_float tf;
    const char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
    npy_intp i;
    npy_int prov_size = 0;
    //size of provenance
    for (i = 0; i < n; i++) {
        prov_size += ((tracked_float*)ip0) -> size;
        prov_size += ((tracked_float*)ip1) -> size;
        r += (((tracked_float*)ip0) -> n) * (((tracked_float*)ip1) -> n);
        ip0 += is0;
        ip1 += is1;
    }

    if (prov_size > 1) {
        provenance* of;
        of = (provenance*) malloc((prov_size - 1)*sizeof(provenance));
        tf.overflow = of;
    }

    tf.n = r;
    tf.size = prov_size;
    int j = 0; // temporary pointer in provenance
    ip0 = (char*)ip0_;
    ip1 = (char*)ip1_;
    for (i = 0; i < n; i++) {
        if (!j){
            j += append_prov(&(tf.p), tf.overflow, (tracked_float*)ip0);
        } else {
            j += append_prov_of(tf.overflow, (tracked_float*)ip0);
        }
        if (!j){
            j += append_prov(&(tf.p), tf.overflow, (tracked_float*)ip1);
        } else {
            j += append_prov_of(tf.overflow, (tracked_float*)ip1);
        }
        ip0 += is0;
        ip1 += is1;
    }

    *(tracked_float*)op = tf;
}

static npy_bool
npytfloat_nonzero(void* data, void* arr) {
    tracked_float r;
    memcpy(&r,data,sizeof(r));
    return (r.n == 0)?NPY_FALSE:NPY_TRUE;
}

static int
npytfloat_fill(void* data_, npy_intp length, void* arr) {
    tracked_float* data = (tracked_float*)data_;
    npy_float64 delta = (data[1].n) - (data[0].n);
    tracked_float r;
    npy_intp i;
    
    for (i = 2; i < length; i++) {
        memcpy(&r, &data[i - 1], sizeof(tracked_float));
        r.n += delta;
        data[i] = r;
    }
    return 0;
}

static int
npytfloat_fillwithscalar(void* buffer_, npy_intp length,
        void* value, void* arr) {
    tracked_float r = *(tracked_float*)value;
    tracked_float* buffer = (tracked_float*)buffer_;
    npy_intp i;
    for (i = 0; i < length; i++) {
        memcpy(&r, &buffer[i], sizeof(tracked_float));
        buffer[i] = r;
    }
    return 0;
}

// only support two dimensions for now
// I don't know how PyArrayObject works with PyLong


static PyArray_ArrFuncs npytfloat_arrfuncs;

typedef struct { char c; tracked_float r; } align_test;

//I'm going to straight up have memory management issues sigh...
PyArray_Descr npytfloat_descr = {
    PyObject_HEAD_INIT(0)
    &PyTFloat_Type,       /* typeobj */
    'V',                    /* kind */
    'r',                    /* type */
    '=',                    /* byteorder */
    /*
     * For now, we need NPY_NEEDS_PYAPI in order to make numpy detect our
     * exceptions.  This isn't technically necessary,
     * since we're careful about thread safety, and hopefully future
     * versions of numpy will recognize that.
     */
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM , /* hasobject */
    0,                      /* type_num */
    sizeof(tracked_float),       /* elsize */
    offsetof(align_test, r), /* alignment */
    0,                      /* subarray */
    0,                      /* fields */
    0,                      /* names */
    &npytfloat_arrfuncs,  /* f */
};

#define DEFINE_CAST(From,To,statement) \
    static void \
    npycast_##From##_##To(void* from_, void* to_, npy_intp n, \
                          void* fromarr, void* toarr) { \
        const From* from = (From*)from_; \
        To* to = (To*)to_; \
        npy_intp i; \
        for (i = 0; i < n; i++) { \
            From x = from[i]; \
            statement \
            to[i] = y; \
        } \
    }

//TODO: none of the casting/initialization functions initialize provenance
//We have to have that as a separate step
#define DEFINE_CAST2(type) \
    DEFINE_CAST(type,tracked_float,tracked_float y = make_tfloat((npy_float64) x, -1, -1, -1);) \
    DEFINE_CAST(tracked_float, type ,npy_float64 z = x.n; \
                type y = z; if (y != z) set_overflow();)

DEFINE_CAST2(npy_int8)
DEFINE_CAST2(npy_int16)
DEFINE_CAST2(npy_int32)
DEFINE_CAST2(npy_int64)
DEFINE_CAST2(npy_float16)
DEFINE_CAST2(npy_float32)
DEFINE_CAST2(npy_float64)
DEFINE_CAST2(npy_uint8)
DEFINE_CAST2(npy_uint16)
DEFINE_CAST2(npy_uint32)
DEFINE_CAST2(npy_uint64)
DEFINE_CAST2(npy_bool)

#define TFLOAT_BINARY_UFUNC(name, exp) \
    void tfloat_ufunc_##name(char** args, npy_intp const *dimensions, \
              npy_intp const *steps, void* data) { \
        npy_intp is0 = steps[0], is1 = steps[1], \
            os = steps[2], n = *dimensions; \
        char *i0 = args[0], *i1 = args[1], *o = args[2]; \
        int k; \
        npy_float64 a, b; \
        for (k = 0; k < n; k++) { \
            tracked_float x = *(tracked_float*)i0; \
            tracked_float y = *(tracked_float*)i1; \
            a = x.n; \
            b = y.n; \
            npy_float64 n = (npy_float64) exp;\
            *(tracked_float*) o = make_tfloat_prov2(n, x, y);\
            i0 += is0; i1 += is1; o += os; \
        } \
    }

#define TFLOAT_BINARY_UFUNC_GEN(name, outtype, exp) \
    void tfloat_ufunc_##name(char** args, npy_intp const *dimensions, \
              npy_intp const *steps, void* data) { \
        npy_intp is0 = steps[0], is1 = steps[1], \
            os = steps[2], n = *dimensions; \
        char *i0 = args[0], *i1 = args[1], *o = args[2]; \
        int k; \
        npy_float64 a, b; \
        for (k = 0; k < n; k++) { \
            tracked_float x = *(tracked_float*)i0; \
            tracked_float y = *(tracked_float*)i1; \
            a = x.n; \
            b = y.n; \
            *(outtype*)o = exp; \
            i0 += is0; i1 += is1; o += os; \
        } \
    }

//figure out how to create equivalent of gcd and etc. -> custom functions for this
TFLOAT_BINARY_UFUNC(add, a + b)

TFLOAT_BINARY_UFUNC(subtract,a - b)
TFLOAT_BINARY_UFUNC(multiply, a*b)
TFLOAT_BINARY_UFUNC(divide,a/b)
TFLOAT_BINARY_UFUNC(floor_divide,(int) floor(a/ b))
PyUFuncGenericFunction tfloat_ufunc_true_divide = tfloat_ufunc_divide;
TFLOAT_BINARY_UFUNC(minimum, a<b?a:b) // how we define lineage in these cases?
TFLOAT_BINARY_UFUNC(maximum, a< b?b:a)
TFLOAT_BINARY_UFUNC_GEN(equal,npy_bool,a = b)
TFLOAT_BINARY_UFUNC_GEN(not_equal,npy_bool, a!= b)
TFLOAT_BINARY_UFUNC_GEN(less,npy_bool, a < b)
TFLOAT_BINARY_UFUNC_GEN(greater,npy_bool,a > b)
TFLOAT_BINARY_UFUNC_GEN(less_equal, npy_bool, a <= b)
TFLOAT_BINARY_UFUNC_GEN(greater_equal, npy_bool, a >= b)

#define TFLOAT_UNARY_UFUNC(name,exp) \
    void tfloat_ufunc_##name(char** args, npy_intp const *dimensions, \
                               npy_intp const *steps, void* data) { \
        npy_intp is = steps[0], os = steps[1], n = *dimensions; \
        char *i = args[0], *o = args[1]; \
        int k; \
        for (k = 0; k < n; k++) { \
            tracked_float x = *(tracked_float*)i; \
            npy_float64 a = x.n;\
            npy_float64 n = (npy_float64) exp;\
            *(tracked_float*)o = make_tfloat_prov1(n, x);\
            i += is; o += os; \
        } \
    }

TFLOAT_UNARY_UFUNC(absolute,fabs(a))
TFLOAT_UNARY_UFUNC(floor, floor(a))
TFLOAT_UNARY_UFUNC(ceil, ceil(a)) 
TFLOAT_UNARY_UFUNC(trunc,trunc(a))
TFLOAT_UNARY_UFUNC(square, a*a)
TFLOAT_UNARY_UFUNC(rint,rint(a))
TFLOAT_UNARY_UFUNC(sign,sign(a))
TFLOAT_UNARY_UFUNC(negative, -a)


/*UNARY_UFUNC(reciprocal,tracked_float,inverse(x)) -> TODO: add more functions as needed*/
//get provenance as string NEED TO REGISTER AS NEW UFUNC???
//UNARY_UFUNC(provenance_str,char*,provenance_repr(x.p, x.overflow))

static NPY_INLINE void
tfloat_matrix_multiply(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    /* pointers to data for input and output arrays */
    char *ip1 = args[0];
    char *ip2 = args[1];
    char *op = args[2];

    /* lengths of core dimensions */
    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    /* striding over core dimensions */
    npy_intp is1_m = steps[0];
    npy_intp is1_n = steps[1];
    npy_intp is2_n = steps[2];
    npy_intp is2_p = steps[3];
    npy_intp os_m = steps[4];
    npy_intp os_p = steps[5];

    /* core dimensions counters */
    npy_intp m, p;

    /* calculate dot product for each row/column vector pair */
    for (m = 0; m < dm; m++) {
        for (p = 0; p < dp; p++) {
            npytfloat_dot(ip1, is1_n, ip2, is2_n, op, dn, NULL);

            /* advance to next column of 2nd input array and output array */
            ip2 += is2_p;
            op  +=  os_p;
        }

        /* reset to first column of 2nd input array and output array */
        ip2 -= is2_p * p;
        op -= os_p * p;

        /* advance to next row of 1st input array and output array */
        ip1 += is1_m;
        op += os_m;
    }
}

static void
tfloat_gufunc_matrix_multiply(char **args, npy_intp const *dimensions,
                                npy_intp const *steps, void *NPY_UNUSED(func))
{
    /* outer dimensions counter */
    npy_intp N_;

    /* length of flattened outer dimensions */
    npy_intp dN = dimensions[0];

    /* striding over flattened outer dimensions for input and output arrays */
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];

    /*
     * loop through outer dimensions, performing matrix multiply on
     * core dimensions for each loop
     */
    for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
        tfloat_matrix_multiply(args, dimensions+1, steps+3);
    }
}

static PyObject*
npytfloat_initialize(PyObject* self, PyObject *args) {
    PyArrayObject* array = (PyArrayObject*)PyTuple_GET_ITEM(args, 0);
    long id = PyLong_AsLong(PyTuple_GET_ITEM(args, 1));
    npy_intp* shape = PyArray_SHAPE(array);
    npy_int i;
    npy_int j;
    tracked_float *r;

    for (i = 0; i < shape[0]; i ++) {
        for (j = 0; j < shape[1]; j ++){
            provenance p = {id, i, j};
            r = (tracked_float*) PyArray_GETPTR2(array, i, j);
            tracked_float tf = {r -> n, p, 1, 0, NULL};
            if (r -> overflow != NULL && r -> size != 0) {
                free(((PyTFloat*)self)->f.overflow);
            }
            memcpy(r, &tf, sizeof(tracked_float));
        } 
    }
    return PyLong_FromLong(1);
}

static PyObject*
npytfloat_free(PyObject* self, PyObject *args) {
    PyArrayObject* array = (PyArrayObject*)PyTuple_GET_ITEM(args, 0);
    npy_intp* shape = PyArray_SHAPE(array);
    npy_int i;
    npy_int j;
    tracked_float *r;
    for (i = 0; i < shape[0]; i ++) {
        for (j = 0; j < shape[1]; j ++){
            r = (tracked_float*) PyArray_GETPTR2(array, i, j);
            if (r -> overflow != NULL && r -> size != 0) {
                free(((PyTFloat*)self)->f.overflow);
            }
            free(r)
        } 
    }
    return PyLong_FromLong(1);
}

PyMethodDef module_methods[] = {
    {"initialize", (PyCFunction) npytfloat_initialize, METH_VARARGS, ""},
    {"free", (PyCFunction) npytfloat_free, METH_VARARGS, ""},
    {0} /* sentinel */
};



static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "tracked_float",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_tracked_float(void) {
    PyObject *m = NULL;
    PyObject* numpy_str;
    PyObject* numpy;
    int npy_tfloat;

    import_array();
    if (PyErr_Occurred()) {
        goto fail;
    }
    import_umath();
    if (PyErr_Occurred()) {
        goto fail;
    }
    numpy_str = PyUnicode_FromString("numpy");
    if (!numpy_str) {
        goto fail;
    }
    numpy = PyImport_Import(numpy_str);
    Py_DECREF(numpy_str);
    if (!numpy) {
        goto fail;
    }

    /* Can't set this until we import numpy */
    PyTFloat_Type.tp_base = &PyGenericArrType_Type;

    /* Initialize tracked_float type object */
    if (PyType_Ready(&PyTFloat_Type) < 0) {
        goto fail;
    }

    /* Initialize  descriptor */
    PyArray_InitArrFuncs(&npytfloat_arrfuncs);
    npytfloat_arrfuncs.getitem = npytfloat_getitem;
    npytfloat_arrfuncs.setitem = npytfloat_setitem;
    npytfloat_arrfuncs.copyswapn = npytfloat_copyswapn;
    npytfloat_arrfuncs.copyswap = npytfloat_copyswap;
    npytfloat_arrfuncs.compare = npytfloat_compare;
    npytfloat_arrfuncs.argmin = npytfloat_argmin;
    npytfloat_arrfuncs.argmax = npytfloat_argmax;
    npytfloat_arrfuncs.dotfunc = npytfloat_dot;
    npytfloat_arrfuncs.nonzero = npytfloat_nonzero;
    npytfloat_arrfuncs.fill = npytfloat_fill;
    npytfloat_arrfuncs.fillwithscalar = npytfloat_fillwithscalar;
    /* Left undefined: scanfunc, fromstr, sort, argsort */
    Py_SET_TYPE(&npytfloat_descr, &PyArrayDescr_Type);
    npy_tfloat = PyArray_RegisterDataType(&npytfloat_descr);
    
    if (npy_tfloat<0) {
        goto fail;
    }

    /* Support dtype(tfloat) syntax */
    if (PyDict_SetItemString(PyTFloat_Type.tp_dict, "dtype",
                             (PyObject*)&npytfloat_descr) < 0) {
        goto fail;
    }

    /* Register casts to and from tfloat */
    #define REGISTER_CAST(From,To,from_descr,to_typenum, safe) { \
            PyArray_Descr* from_descr_##From##_##To = (from_descr); \
            if (PyArray_RegisterCastFunc(from_descr_##From##_##To, \
                                         (to_typenum), \
                                         npycast_##From##_##To) < 0) { \
                goto fail; \
            } \
            if (safe && PyArray_RegisterCanCast(from_descr_##From##_##To, \
                                                (to_typenum), \
                                                NPY_NOSCALAR) < 0) { \
                goto fail; \
            } \
        }
    #define REGISTER_INT_CASTS(bits) \
        REGISTER_CAST(npy_int##bits, tracked_float, \
                      PyArray_DescrFromType(NPY_INT##bits), npy_tfloat, 1) \
        REGISTER_CAST(tracked_float, npy_int##bits, &npytfloat_descr, \
                      NPY_INT##bits, 0)
    
    #define REGISTER_FLOAT_CASTS(bits, safe1, safe2) \
        REGISTER_CAST(npy_float##bits, tracked_float, \
                      PyArray_DescrFromType(NPY_FLOAT##bits), npy_tfloat, safe1) \
        REGISTER_CAST(tracked_float, npy_float##bits, &npytfloat_descr, \
                      NPY_FLOAT##bits, safe1)
    
    #define REGISTER_UINT_CASTS(bits) \
        REGISTER_CAST(npy_uint##bits, tracked_float, \
                      PyArray_DescrFromType(NPY_UINT##bits), npy_tfloat, 1) \
        REGISTER_CAST(tracked_float, npy_uint##bits, &npytfloat_descr, \
                      NPY_UINT##bits, 0)

    REGISTER_INT_CASTS(8)
    REGISTER_INT_CASTS(16)
    REGISTER_INT_CASTS(32)
    REGISTER_INT_CASTS(64)

    REGISTER_FLOAT_CASTS(16, 1, 0)
    REGISTER_FLOAT_CASTS(32, 1, 0)
    REGISTER_FLOAT_CASTS(64, 1, 1)

    REGISTER_UINT_CASTS(8)
    REGISTER_UINT_CASTS(16)
    REGISTER_UINT_CASTS(32)
    REGISTER_UINT_CASTS(64)

    REGISTER_CAST(npy_bool, tracked_float, PyArray_DescrFromType(NPY_BOOL), npy_tfloat, 1) 
    REGISTER_CAST(tracked_float, npy_bool, &npytfloat_descr, NPY_BOOL, 0)

    /* Register ufuncs */
#define REGISTER_UFUNC(name,...) { \
        PyUFuncObject* ufunc = \
            (PyUFuncObject*)PyObject_GetAttrString(numpy, #name); \
        int _types[] = __VA_ARGS__; \
        if (!ufunc) { \
            goto fail; \
        } \
        if (sizeof(_types)/sizeof(int)!=ufunc->nargs) { \
            PyErr_Format(PyExc_AssertionError, \
                         "ufunc %s takes %d arguments, our loop takes %lu", \
                         #name, ufunc->nargs, (unsigned long) \
                         (sizeof(_types)/sizeof(int))); \
            Py_DECREF(ufunc); \
            goto fail; \
        } \
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc, npy_tfloat, \
                tfloat_ufunc_##name, _types, 0) < 0) { \
            Py_DECREF(ufunc); \
            goto fail; \
        } \
        Py_DECREF(ufunc); \
    }

    #define REGISTER_UFUNC_BINARY_TFLOAT(name) \
        REGISTER_UFUNC(name, {npy_tfloat, npy_tfloat, npy_tfloat})
        
    #define REGISTER_UFUNC_BINARY_COMPARE(name) \
        REGISTER_UFUNC(name, {npy_tfloat, npy_tfloat, NPY_BOOL})
        
    #define REGISTER_UFUNC_UNARY(name) \
        REGISTER_UFUNC(name, {npy_tfloat, npy_tfloat})
    /* Binary */
    REGISTER_UFUNC_BINARY_TFLOAT(add)
    REGISTER_UFUNC_BINARY_TFLOAT(subtract)
    REGISTER_UFUNC_BINARY_TFLOAT(multiply)
    REGISTER_UFUNC_BINARY_TFLOAT(divide)
    REGISTER_UFUNC_BINARY_TFLOAT(true_divide)
    REGISTER_UFUNC_BINARY_TFLOAT(floor_divide)
    REGISTER_UFUNC_BINARY_TFLOAT(minimum)
    REGISTER_UFUNC_BINARY_TFLOAT(maximum)
    /* Comparisons */
    REGISTER_UFUNC_BINARY_COMPARE(equal)
    REGISTER_UFUNC_BINARY_COMPARE(not_equal)
    REGISTER_UFUNC_BINARY_COMPARE(less)
    REGISTER_UFUNC_BINARY_COMPARE(greater)
    REGISTER_UFUNC_BINARY_COMPARE(less_equal)
    REGISTER_UFUNC_BINARY_COMPARE(greater_equal)
    /* Unary */
    REGISTER_UFUNC_UNARY(negative)
    REGISTER_UFUNC_UNARY(absolute)
    REGISTER_UFUNC_UNARY(floor)
    REGISTER_UFUNC_UNARY(ceil)
    REGISTER_UFUNC_UNARY(trunc)
    REGISTER_UFUNC_UNARY(rint)
    REGISTER_UFUNC_UNARY(square)
    REGISTER_UFUNC_UNARY(sign)

    /* Create module */
    m = PyModule_Create(&moduledef);

    if (!m) {
        goto fail;
    }

    /* Add tfloat type */
    Py_INCREF(&PyTFloat_Type);
    PyModule_AddObject(m, "tracked_float",(PyObject*)&PyTFloat_Type);

    /* Create matrix multiply generalized ufunc */
    {
        int types2[3] = {npy_tfloat,npy_tfloat,npy_tfloat};
        PyObject* gufunc = PyUFunc_FromFuncAndDataAndSignature(0,0,0,0,2,1,
            PyUFunc_None,(char*)"matrix_multiply",
            (char*)"return result of multiplying two matrices of tfloats",
            0,"(m,n),(n,p)->(m,p)");
        if (!gufunc) {
            goto fail;
        }
        if (PyUFunc_RegisterLoopForType((PyUFuncObject*)gufunc, npy_tfloat,
                tfloat_gufunc_matrix_multiply, types2, 0) < 0) {
            goto fail;
        }
        PyModule_AddObject(m,"matrix_multiply",(PyObject*)gufunc);
    }

    return m;

fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load tracked_float module.");
    }
    if (m) {
        Py_DECREF(m);
        m = NULL;
    }
    return m;
}