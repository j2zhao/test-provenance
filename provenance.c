#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <math.h>
#include "common.h"

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



static NPY_INLINE int* 
typedef struct {
    int id;
    /*interval for indices */
    /* starts at + 1 */
    int start_0;
    int start_1;
    int end_0;
    int end_1;
    /* type */
    int type;
    /*overflow data*/
} provenance;


typedef struct {
    /* numerator */
    npy_float64 n;
    provenance p;
    int size;
    provenance* overflow;
    } tracked_float;

static NPY_INLINE int
tfloat_nonzero(tracked_float x) {
    return x.n!=0;
}


static NPY_INLINE provenance
make provenance(int id, int start_0, int start_1, int end_0, int end_1, int type) {
    provenance prov = {id, start_0 + 1, start_1 + 1, end_0 + 1, end_1 + 1, type};
    return prov
}

static NPY_INLINE tracked_float
make_tfloat_full(npy_float64 n, int id, int start_0, int start_1, int end_0, int end_1, int type,int size, provenance *overflow) {
    provenance prov = {id, start_0 + 1, start_1 + 1, end_0 + 1, end_1 + 1, type};
    tracked_float r = {n, prov, size, overflow};
    return r;
}

static NPY_INLINE tracked_float
make_tfloat_start(npy_float64 n, int id, int start_0, int start_1) {
    provenance prov = {id, start_0 + 1, start_1 + 1, 0, 0, 0};
    tracked_float r = {n, prov, 1,  NULL};
    return r
}

/* overflows don't change between programs */
static NPY_INLINE tracked_float
make_tfloat_prov1(npy_float64 n, tracked_float a) {
    int size = a.size;
    size_t p_size = size(provenance);
    tracked_float r;
    if (size - 1) > 0:
        provenance overflow[size - 1];
        memcpy(overflow, a.overflow, p_size*(size - 1))
        provenance p;
        memcpy(p, a.p, p_size)
        r = {n, p, size, overflow}
        return r
    else:
        provenance p;
        memcpy(p, a.p, p_size);
        r = {n, p, size, NULL};
        return r
}

static NPY_INLINE tracked_float
make_tfloat_prov2(np_float64 n, tracked_float a, tracked_float b) {
    int size0 = a.size
    int size1 = b.size
    
    int size = size0 + size1
    /* we have history */
    if size >= 1:
        provenance p;
        provenance overflow[size - 1];
        int offset = -1;
        size_t p_size = sizeof(provenance);
        /*a has history*/
        if size0 > 0:
            memcpy(p, a.p, p_size)
            x = true;
            if size0 > 1:
                memcpy(overflow, a.overflow, (size0 - 1)*p_sizes)
            offset = size0 - 1;
        
        if size1 > 0:
            if offset == -1:
                memcpy(p, b.p, p_size)
                memcpy(overflow, b.overflow, (size1 - 1)*p_size)
            else:
                memcpy(overflow + offset, b.p, p_size)
                memcpy(overflow + offset + 1, b.p, p_size*(size1 - 1))
        tracked_float r = {n, p, size, overflow}
        return r
    else:
        tracked_float r = {n, NULL, 0, NULL}
        return r
        
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
        if (PyRational_Check(object)) { \
            dst = ((PyRational*)object)->f.n; \
        } \
        else if (PyFloat_Check(object)){ \
            dst = PyFloat_AsDouble(object); \
        }  else if (PyLong_Check(object)) { \
            dst = PyLong_AsLong(object); \
        } else { \
            Py_INCREF(Py_NotImplemented); \
            return Py_NotImplemented; \
        } \
    } \

// no error checking
// only make one type for now
//initialize with only one provenance
static PyObject*
pyrational_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    npy_float64 n = PyTuple_GET_ITEM(args, 0);
    int id = PyTuple_GET_ITEM(args, 1);
    int start_0 = PyTuple_GET_ITEM(args, 2);
    int start_1 = PyTuple_GET_ITEM(args, 3)
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
        if rat1 && rat2: \
            result = x op y; \
        else if rat1: \
            result =  x op b; \
        else if rat2: \
            result = a op y; \
        else: \
            result = a op b; \
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
provenance_repr(provence p, provenance* overflow) {
    size_t rsize;
    if overflow != NULL:
        rsize = sizeof(overflow)/sizeof(provenance) + 1;
    else
        rsize = 1;
    char* output[rsize*50];
    int offset = 0;
    provenance q = p;

    for (int x = 0, x < rsize, x ++) {
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
pyrational_repr(PyObject* self) {
    tracked_float x = ((PyRational*)self)->f;
    const char* c = provenance_repr(x.p, x.overflow);
    return PyUnicode_FromFormat("%d ; %V", x.n, c);
}

static PyObject*
pyrational_str(PyObject* self) {
    return pyrational_rep(self);
}

static npy_hash_t
pyrational_hash(PyObject* self) {
    tracked x = ((PyRational*)self)->f;
    /* Use a extremely weak hash as Python expects ?*/
    long h = 131071*x.n+524287*x.size;
    /* Never return the special error value -1 */
    return h==-1?2:h;
}

#define RATIONAL_BINOP_2(name, exp) \
    static PyObject* \
    pytfloat_##name(PyObject* a, PyObject* b) { \
        npy_float64 x, y; \
        tracked_float z;\
        int a_tf = PyTFloat_Check(a); \
        int b_tf = PyTFloat_Check(b); \
        if (a_tf) \
            x = a.n; \
        else: \
            x = (npy_float64) a; \
        if (b_tf) \
            y = b.n; \
        else: \
            y = (npy_float64) b; \
        np_float64 n = (np_float64) exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        if (a_tf && b_tf) { \
            z = make_tfloat_prov2(n, a ->f, b ->f); \
        } else if (a_tf) { \
            z = make_tfloat_prov1(n, a -> f); \
        } else if (b_tf) { \
            z = make_tfloat_prov1(n, b -> f); \
        } else { \
            return PyFloat_FromDouble((double) n); \
        } \
        return PyTFloat_FromTFloat(z); \
    }
#define RATIONAL_BINOP(name, exp) RATIONAL_BINOP_2(name, x exp y))

RATIONAL_BINOP(add, +)
RATIONAL_BINOP(subtract, -)
RATIONAL_BINOP(multiply, *)
RATIONAL_BINOP(divide, /)
RATIONAL_BINOP_2(floor_divide, (int)floor(x / y) )

#define RATIONAL_UNOP(name, exp) \
    static PyObject* \
    pyrational_##name(PyObject* self) { \
        tracked_float x = ((PyTFloat*)self)->f; \
        npy_float64 y = x.n; \
        npy_float64 z = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        tracked_float tf = make_tfloat_prov1(z, x); \
        return PyTFloat_FromTFloat(tf); \
    }

#define RATIONAL_UNOP_2(name, type, exp, convert) \
    static PyObject* \
    pyrational_##name(PyObject* self) { \
        tracked_float x = ((PyTFloat*)self)->f; \
        npy_float64 y = x.n; \
        type z = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        return convert(z); \
    }

RATIONAL_UNOP(negative, -x)
RATIONAL_UNOP(absolute,fabs(x))

RATIONAL_UNOP_2(int, long, (int)x, PyLong_FromLong)
RATIONAL_UNOP_2(float,double, (double)x,PyFloat_FromDouble)

static PyObject*
pytfloat_positive(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static int
pytfloat_nonzero(PyObject* self) {
    tfloat x = ((PyTFloat*)self)->f;
    return rational_nonzero(x);
}

static PyNumberMethods pyrational_as_number = {
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
    pyrational_float,        /* nb_float */

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

    pyrational_floor_divide, /* nb_floor_divide */
    pyrational_divide,       /* nb_true_divide */
    0,                       /* nb_inplace_floor_divide */
    0,                       /* nb_inplace_true_divide */
    0,                       /* nb_index */
};

static PyObject*
pytfloat_float(PyObject* self) {
    return PyLong_FromLong(((PyRational*)self)->r.n);
}

static PyObject*
pytfloat_float(PyObject* self, PyObject* value, void* closure) {

    npy_float64 n = PyFloat_AsDouble(((PyRational*)value);
}


static int
pytfloat_set_provenance_field(mytype* self, PyObject* start_0, PyObject* start_1, PyObject* end_0, PyObject* end_1, PyObject* type, void* closure)
{
    float f = PyFloat_AsFloat(value);
    if (PyErr_Occurred()) {
        return -1;
    }
    self->float_field = f;
    return 0;
}

static PyGetSetDef pytfloat_getset[] = {
    {(char*)"n", pytfloat_float, 0,(char*)"numerator",0},
    {(char*)"d",pyrational_d,0,(char*)"denominator",0},
    {0} /* sentinel */
};


PyArray_Descr npyrational_descr = {
    PyObject_HEAD_INIT(0)
    &PyRational_Type,       /* typeobj */
    'V',                    /* kind */
    'r',                    /* type */
    '=',                    /* byteorder */
    /*
     * For now, we need NPY_NEEDS_PYAPI in order to make numpy detect our
     * exceptions.  This isn't technically necessary,
     * since we're careful about thread safety, and hopefully future
     * versions of numpy will recognize that.
     */
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM, /* hasobject */
    0,                      /* type_num */
    sizeof(rational),       /* elsize */
    offsetof(align_test,r), /* alignment */
    0,                      /* subarray */
    0,                      /* fields */
    0,                      /* names */
    &npyrational_arrfuncs,  /* f */
};

