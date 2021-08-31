#define NPY_NO_DEPRECATED_API NPY_API_VERSION

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

// static void
// set_overflow(void) {
// #ifdef ACQUIRE_GIL
//     /* Need to grab the GIL to dodge a bug in numpy */
//     PyGILState_STATE state = PyGILState_Ensure();
// #endif
//     if (!PyErr_Occurred()) {
//         PyErr_SetString(PyExc_OverflowError,
//                 "overflow in rational arithmetic");
//     }
// #ifdef ACQUIRE_GIL
//     PyGILState_Release(state);
// #endif
// }

// static void
// set_zero_divide(void) {
// #ifdef ACQUIRE_GIL
//     /* Need to grab the GIL to dodge a bug in numpy */
//     PyGILState_STATE state = PyGILState_Ensure();
// #endif
//     if (!PyErr_Occurred()) {
//         PyErr_SetString(PyExc_ZeroDivisionError,
//                 "zero divide in rational arithmetic");
//     }
// #ifdef ACQUIRE_GIL
//     PyGILState_Release(state);
// #endif
// }



typedef struct {
    long id;
    /*interval for indices */
    /* starts at + 1 */
    long start_0;
    long start_1;
    long end_0;
    long end_1;
    /* type */
    long type;
    /*overflow data*/
} provenance;


typedef struct {
    /* numerator */
    npy_float64 n;
    provenance p;
    long size;
    provenance* overflow;
    } tracked_float;

static NPY_INLINE int
tfloat_nonzero(tracked_float x) {
    return x.n!=0;
}


static NPY_INLINE provenance
make provenance(long id, long start_0, long start_1, long end_0, long end_1, long type) {
    provenance prov = {id, start_0, start_1, end_0, end_1, type};
    return prov
}

static NPY_INLINE tracked_float
make_tfloat_full(npy_float64 n, long id, long start_0, long start_1, long end_0, long end_1, long type, long size, provenance *overflow) {
    provenance prov = {id, start_0, start_1, end_0, end_1, type};
    tracked_float r = {n, prov, size, overflow};
    return r;
}

static NPY_INLINE tracked_float
make_tfloat_start(npy_float64 n, long id, long start_0, long start_1) {
    provenance prov = {id, start_0, start_1, -1, -1, 0};
    tracked_float r = {n, prov, 1,  NULL};
    return r;
}

static NPY_INLINE tracked_float
make_prov_start(long id, long start_0, long start_1) {
    provenance prov = {id, start_0, start_1, 0, 0, 0};
    return prov;
}



/* overflows don't change between programs */
static NPY_INLINE tracked_float
make_tfloat_prov1(npy_float64 n, tracked_float a) {
    long size = a.size;
    size_t p_size = size(provenance);
    tracked_float r;
    if (size - 1) > 0 {
        provenance overflow[size - 1];
        provenance p;
        memcpy(overflow, a.overflow, p_size*(size - 1));
        memcpy(&p, &a.p, p_size);
        r = {n, p, size, overflow};
        return r;
    }
    else {
        provenance p;
        memcpy(&p, &a.p, p_size);
        r = {n, p, size, NULL};
        return r;
    }
}

static NPY_INLINE tracked_float
make_tfloat_prov2(np_float64 n, tracked_float a, tracked_float b) {
    long size0 = a.size
    long size1 = b.size
    long size = size0 + size1
    /* we have history */
    if (size >= 1) {
        provenance p;
        provenance overflow[size - 1];
        int offset = -1;
        size_t p_size = sizeof(provenance);
        tracked_float r;
        /*a has history*/
        if (size0 > 0) {
            memcpy(&p, &a.p, p_size);
            x = true;
            if (size0 > 1) {
                memcpy(overflow, a.overflow, (size0 - 1)*p_sizes);
            }
            offset = size0 - 1;
        }
        if (size1 > 0) {
            if (offset == -1) {
                memcpy(&p, &b.p, p_size);
                memcpy(overflow, b.overflow, (size1 - 1)*p_size);
            }
            else {
                memcpy(overflow + offset, b.p, p_size);
                memcpy(overflow + offset + 1, b.p, p_size*(size1 - 1));
            }
        }
        r = {n, p, size, overflow};
        return r;
    }
    else {
        provenance p = {-1, -1, -1, -1, -1, -1};
        tracked_float r = {n, p , 0, NULL};
        return r;
    }
        
}

static NPY_INLINE int 
append_prov(provenance* p, provenance* of, tracked_float* a) {
    long s = a -> size;
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
append_prov_of(provenance* of, tracked_float a) {
    long s = a.size;
    if (s == 0) {
        return s;
    }
    memcpy(of, &a.p, sizeof(provenance));
    if (s > 1) {
        memcpy(of + 1, a.overflow, sizeof(provenance)*(s - 1));
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
        return 0; \
    } \

// no error checking
// only make one type for now
//initialize with only one provenance
static PyObject*
pytfloat_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    npy_float64 n = PyTuple_GET_ITEM(args, 0);
    long id = PyTuple_GET_ITEM(args, 1);
    long start_0 = PyTuple_GET_ITEM(args, 2);
    long start_1 = PyTuple_GET_ITEM(args, 3)
    tracked_float r = make_tfloat_start(n, int id, int start_0, int start_1);
    return PyTFloat_FromTFFloat(r);
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

static char*
provenance_repr(provenance p, provenance* overflow) {
    size_t rsize;
    if (overflow != NULL) {
        rsize = sizeof(overflow)/sizeof(provenance) + 1;
    }
    else {
        rsize = 1;
    }
    char* output[rsize*50];
    int offset = 0;
    provenance q = p;
    int x;
    for (x = 0, x < rsize, x ++) {
        offset += sprint(output + offset, "[(%d,%d,%d,%d,%d,%d)]", 
            q.id, q.start_0, q.start_1, q.start_2,x.end_0, q.end_1, q.type);
        if (x > 0 && x < rsize -1 ) {
            offset += sprint(output + offset, " , ");
        }
        if (rsize != 1) {
            q = overflow[x];
        }
    }

    return output;
    
}

static PyObject*
pytfloat_repr(PyObject* self) {
    tracked_float x = ((PyTFloat*)self)->f;
    const char* c = provenance_repr(x.p, x.overflow);
    return PyUnicode_FromFormat("%d ; %V", x.n, c);
}

static PyObject*
pytfloat_str(PyObject* self) {
    return pytfloat_rep(self);
}

static npy_hash_t
pytfloat_hash(PyObject* self) {
    tracked x = ((PyTFloat*)self)->f;
    /* Use a extremely weak hash as Python expects ?*/
    long h = 131071*x.n+524287*x.size;
    /* Never return the special error value -1 */
    return h==-1?2:h;
}

#define TFLOAT_BINOP_2(name, exp) \
    static PyObject* \
    pytfloat_##name(PyObject* a, PyObject* b) { \
        tracked_float z;\
        int a_tf = PyTFloat_Check(a); \
        int b_tf = PyTFloat_Check(b); \
        npy_float64 x = AS_DOUBLE(a);\
        npy_float64 y = AS_DOUBLE(b);\
        np_float64 n = (np_float64) exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        if (a_tf && b_tf) { \
            z = make_tfloat_prov2(n, a->f, b ->f); \
        } else if (a_tf) { \
            z = make_tfloat_prov1(n, a -> f); \
        } else if (b_tf) { \
            z = make_tfloat_prov1(n, b -> f); \
        } else { \
            return PyFloat_FromDouble((double) n); \
        } \
        return PyTFloat_FromTFloat(z); \
    }
#define TFLOAT_BINOP(name, exp) TFLOAT_BINOP_2(name, x exp y))

TFLOAT_BINOP(add, +)
TFLOAT_BINOP(subtract, -)
TFLOAT_BINOP(multiply, *)
TFLOAT_BINOP(divide, /)
TFLOAT_BINOP_2(floor_divide, (int)floor(x / y) )

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

TFLOAT_UNOP(negative, -x)
TFLOAT_UNOP(absolute,fabs(x))

TFLOAT_UNOP_2(int, long, (long)y, PyLong_FromLong)
TFLOAT_UNOP_2(float,double, (double)y,PyFloat_FromDouble)

static PyObject*
pytfloat_positive(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static int
pytfloat_nonzero(PyObject* self) {
    tfloat x = ((PyTFloat*)self)->f;
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
pytfloat_float_get(PyObject* self) {
    return PyFloat_FromDouble(((PyTFloat*)self)->f.n);
}

static PyObject*
pytfloat_psize_get(PyObject* self) {
    return PyLong_FromLong(((PyTFloat*)self)->f.size);
}

static int
pytfloat_float_set(PyObject* self, PyObject* value, void* closure) {
    npy_float64 x = PyFloat_AsDouble(((PyFloat*)value);
    if (PyErr_Occurred()) {
        return 0;
    }
    ((PyTFloat*)self)->f.n = x;
    return 1;
}


static PyObject*
pytfloat_provenance_get(PyObject* self, void* closure) {
    tracked_float f = ((PyTFloat*)self)->f;
    long size = f.size;
    int i;
    provenance p = f.p;

    PyListObject plist = PyList_New(PyLong_AsSsize_t(PyLong_FromLong(size)));
    PyTupleObject prov = PyTuple_Pack(6, PyLong_FromLong(p.id), PyLong_FromLong(p.start_0), 
        PyLong_FromLong(p.start_1), PyLong_FromLong(p.end_0), PyLong_FromLong(p.end_1), 
        PyLong_FromLong(p.type));
    PyList_SetItem(plist, 0, prov);

    for (i = 1; i < size, i++) {
        provenance p = f.overflow[i];
        PyTupleObject prov = PyTuple_Pack(6, PyLong_FromLong(p.id), PyLong_FromLong(p.start_0), 
            PyLong_FromLong(p.start_1), PyLong_FromLong(p.end_0), PyLong_FromLong(p.end_1), 
            PyLong_FromLong(p.type));
        PyList_SetItem(plist, 0, prov);
    }
    return plist;
}

static PyGetSetDef pytfloat_getset[] = {
    {(char*)"n", pytfloat_float, pytfloat_float_set, (char*)"float", 0},
    {(char*)"provenance",pytfloat_provenance_get,0,(char*)"list of provenance",0},
    {(char*)"pnum",pytfloat_psize_get,0,(char*)"number of predecessors",0},
    {0} /* sentinel */
};

static void pytfloat_dealloc(PyObject* self){
    if ((PyTFloat*)self->f.overflow != NULL) {
        free(((PyTFloat*)self)->f.overflow);
    }
    Py_TYPE(obj)->tp_free(obj);
}

static PyTypeObject PyTFloat_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "numpy.core.provenance.tfloat",  /* tp_name */
    sizeof(PyTFloat),                       /* tp_basicsize */
    0,                                        /* tp_itemsize -> am not sure if array counts as dynamic?? maybe not, but it might be more than a new object -> then again, we don't really see it*/
    pytfloat_dealloc,                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    pytfloat_repr,                          /* tp_repr */
    &pytfloat_as_number,                    /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    pytfloat_hash,                          /* tp_hash */
    0,                                        /* tp_call */
    pytfloat_str,                           /* tp_str */
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
        double n = AS_DOUBLE(item);
        int eq;
        if (error_converting(n)) {
            return -1;
        }
        r = make_tfloat_start(n, -1, -1, -1) ;
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
    byteswap(&(x -> end_0));
    byteswap(&(x -> end_1));
    byteswap(&(x -> type));
}

/* Note: do we have to copy over the pointer????  -> rn i say yes */

static void
npytfloat_copyswap(void* dst, void* src, int swap, void* arr) {
    tracked_float* r;
    if (!src) {
        return;
    }
    r = (tfloat*)dst;
    memcpy(r,src,sizeof(tfloat));
    
    if (r -> overflow != NULL) {
        size_t t = sizeof((r -> overflow))/sizeof(provenance)
        provenance of[t];
        memcpy(of, &(r -> overflow), sizeof(provenance)*t);
        r -> overflow = &of;
    }
    
    if (swap) {
        byteswap_float(&r->n);
        byteswap_provenance(&r->p);
        byteswap(&r->size);
        if (r.overflow != NULL) {
            provenance of[] = r -> overflow;
            size_t t = sizeof(r -> overflow)/sizeof(provenance);
            int i;
            for(i = 0; i < t; i++) {
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
            rational* r = (rational*)(dst+dstride*i);
            if (r -> overflow != NULL) {
                size_t t = sizeof((r -> overflow))/sizeof(provenance)
                provenance of[t];
                memcpy(of, &(s.overflow), sizeof(provenance)*(s.size - 1));
                r -> overflow = &of;
            }
        }
    }
    else {
        for (i = 0; i < n; i++) {
            npyrational_copyswap(dst+dstride*i, src+sstride*i, swap, void* arr)
        }
    }
}


npytfloat_compare(const void* d0, const void* d1, void* arr) {
    tfloat x = *(tfloat*)d0,
             y = *(tfloat*)d1;
    npy_float64 a = x.n, b = y.n;

    return rational_lt(x,y)?-1:rational_eq(x,y)?0:1;
}

#define FIND_EXTREME(name, op) \
    static int \
    npyrational_##name(void* data_, npy_intp n, \
            npy_intp* max_ind, void* arr) { \
        const tfloat* data; \
        npy_intp best_i; \
        npy_float64 x, y; \
        npy_intp i; \
        if (!n) { \
            return 0; \
        } \
        data = (rational*)data_; \
        best_i = 0; \
        y = data[0].n; \
        for (i = 1; i < n; i++) { \
            x = data[i].n; \
            if (op) { \
                best_i = i; \
                y = data[i]; \
            } \
        } \
        *max_ind = best_i; \
        return 0; \
    }

FIND_EXTREME(argmin, x < y)
FIND_EXTREME(argmax, x > y)




// TODO: We can make an optimization here when compressing -> but how is yet unclear
static void
npyrational_dot(void* ip0_, npy_intp is0, void* ip1_, npy_intp is1,
        void* op, npy_intp n, void* arr) {
    npy_float64 r = 0;
    tracked_float tf;
    const char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
    npy_intp i;
    long prov_size = 0;
    //size of provenance
    for (i = 0; i < n; i++) {
        prov_size += ((tfloat*)ip0) -> size;
        prov_size += ((tfloat*)ip1) -> size;
        r += (((tfloat*)ip0) -> n) * (((tfloat*)ip1) -> n);
        ip0 += is0;
        ip1 += is1;
    }

    if (prov_size > 1) {
        provenance of[prov_size - 1];
        tf.overflow = &of;
    }

    tf.n = r;
    tf.size = prov_size;
    int j = 0; // temporary pointer in provenance
    ip0 = (char*)ip0_;
    ip1 = (char*)ip1_;
    for (i = 0; i < n; i++) {
        if (!j){
            j += append_prov(&(tf.p), tf.overflow, (tfloat*)ip01);
        } else {
            j += append_prov(tf.overflow, (tfloat*)ip01);
        }
        if (!j){
            j += append_prov(&(tf.p), tf.overflow, (tfloat*)ip11);
        } else {
            j += append_prov(tf.overflow, (tfloat*)ip11);
        }
        ip01 += is0;
        ip11 += is1;
    }

    *(tracked_float*)op = tf;
}

static npy_bool
npyrational_nonzero(void* data, void* arr) {
    tracked_float r;
    memcpy(&r,data,sizeof(r));
    return (r.n == 0)?NPY_FALSE:NPY_TRUE;
}

static int
npyrational_fill(void* data_, npy_intp length, void* arr) {
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
npyrational_fillwithscalar(void* buffer_, npy_intp length,
        void* value, void* arr) {
    tracked_float r = *(tracked_float*)value;
    tracked_float* buffer = (tracked_float*)buffer_;
    npy_intp i;
    for (i = 0; i < length; i++) {
        buffer[i] = r;
    }
    return 0;
}

// only support two dimensions for now
static int
npyrational_reintialize(PyArrayObject* array, long id) {
    npy_intp* shape = PyArray_SHAPE(array);
    long i;
    long j;
    tracked_float *r;

    for (i = 0; i < shape[0]; i ++) {
        for (j = 0; j < shape[0]; j ++){
            r = (tracked_float*) PyArray_GETPTR2(array, i, j);
            r -> p = {id, i, j, 0, 0, 0};
            r -> size = 1;
            r -> overflow = NULL;
        } 
    }
    return 0;
}

static PyArray_ArrFuncs npytfloat_arrfuncs;

typedef struct { char c; tracked_float r; } align_test;

PyArray_Descr npyrational_descr = {
    PyObject_HEAD_INIT(0)
    &PyTFloat_Type,       /* typeobj */
    'f',                    /* kind */
    'r',                    /* type */
    '=',                    /* byteorder */
    /*
     * For now, we need NPY_NEEDS_PYAPI in order to make numpy detect our
     * exceptions.  This isn't technically necessary,
     * since we're careful about thread safety, and hopefully future
     * versions of numpy will recognize that.
     */
    NPY_ITEM_IS_POINTER | NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM , /* hasobject */
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