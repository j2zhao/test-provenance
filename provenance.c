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
    int *overflow;
} provenance;


typedef struct {
    /* numerator */
    npy_float64 n;
    } tracked_float;

static NPY_INLINE tracked_float
make_tfloat_full(npy_float64 n, int id, int start_0, int start_1, int end_0, int end_1, int type, int *overflow) {
    tracked_float r = {n, id, start_0 + 1, start_1 + 1, end_0 + 1, end_1 + 1, type, overflow};
    return r;
}

static NPY_INLINE tracked_float
make_tfloat_start(npy_float64 n, int start_0, int start_1) {
    tracked_float r = {n, start_0 + 1, start_1 + 1, 0, 0, 0, NULL};
    return r
}

/* overflows don't change between programs */
static NPY_INLINE tracked_float
make_tfloat_prov1(npy_float64 n, tracked_float a) {
    int size = sizeof(a -> overflow)
    int overflow[size];
    memcpy(overflow, a -> overflow, size)
    overflow = memcpy()
    tracked_float r = {n, a.start_0, a.start_1, a.end_0, a.end_1, a.type, overflow}
    return n
}

make_tfloat_prov2(np_float64 n, tracked_float a, tracked_float b) {
    

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
PyTFloat_FromTFloat(PyTFloat x) {
    PyTFloat* p = (PyTFloat*)PyTFloat_Type.tp_alloc(&PyTFloat_Type,0);
    if (p) {
        p->f = x;
    }
    return (PyObject*)p;
}


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

