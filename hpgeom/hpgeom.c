#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <stdio.h>

//#include "healpix_utils.h"
#include "healpix_geom.h"


// add radians option...

static PyObject *angle_to_pixel(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    int64_t nside;
    PyObject *a_obj=NULL, *b_obj=NULL;
    PyObject *a_arr=NULL, *b_arr=NULL;
    PyObject *pix_obj=NULL;
    int lonlat=1;
    int nest=1;
    static char *kwlist[] = {"nside", "a", "b", "lonlat", "nest", NULL};

    npy_intp n_a, n_b;
    npy_intp dims[1];
    int64_t *pixels;
    int i;
    int status;
    double *a_data, *b_data;
    double theta, phi;
    hpx_info hpx;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lOO|pp", kwlist,
                                     &nside, &a_obj, &b_obj,
                                     &lonlat, &nest))
        return NULL;

    a_arr = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (a_arr == NULL) return NULL;
    b_arr = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (b_arr == NULL) goto fail;

    /* fix these; they crash on single*/
    n_a = PyArray_DIM((PyArrayObject *)a_arr, 0);
    n_b = PyArray_DIM((PyArrayObject *)b_arr, 0);

    /* check that n_a == n_b */
    /* also that only 1d? */

    if ((pixels = (int64_t *) calloc(n_a, sizeof(int64_t))) == NULL) {
        // raise
        goto fail;
    }

    a_data = (double *) PyArray_DATA((PyArrayObject *)a_arr);
    b_data = (double *) PyArray_DATA((PyArrayObject *)b_arr);

    enum Scheme scheme;
    if (nest) {
        scheme = NEST;
    } else {
        scheme = RING;
    }
    hpx = hpx_info_from_nside(nside, scheme);

    for (i=0; i<n_a; i++) {
        /* I think that here should be the conversion from lon/lat to theta/phi */
        if (lonlat) {
            if (!hpix_lonlat_degrees_to_thetaphi_radians(a_data[i], b_data[i], &theta, &phi)) {
                // raise
                goto fail;
            }
        } else {
            theta = a_data[i];
            phi = b_data[i];
        }
        /* check ranges somewhere around here.*/
        pixels[i] = ang2pix(hpx, theta, phi);
    }

    dims[0] = (npy_intp) n_a;
    pix_obj = PyArray_SimpleNewFromData(1, dims, NPY_INT64, pixels);

    /* do I free the memory from pixels? */

    Py_DECREF(a_arr);
    Py_DECREF(b_arr);

    //dims[0] = (npy_intp) nlon;
    //pix_obj = PyArray_ZEROS(1, dims, NPY_INT64, 0);

    return pix_obj;

 fail:
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);

    // Should raise.
    Py_RETURN_NONE;
}


static PyMethodDef hpgeom_methods[] = {
    {"angle_to_pixel", (PyCFunction)(void(*)(void))angle_to_pixel,
     METH_VARARGS | METH_KEYWORDS,
     "Convert angles to pixels."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef hpgeom_module = {
    PyModuleDef_HEAD_INIT,
    "_hpgeom",
    NULL,
    -1,
    hpgeom_methods
};


PyMODINIT_FUNC
PyInit__hpgeom(void)
{
    import_array();
    return PyModule_Create(&hpgeom_module);
}
