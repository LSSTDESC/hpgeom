#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <stdio.h>

#include "healpix_geom.h"
#include "hpgeom_utils.h"


static PyObject *angle_to_pixel(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    int64_t nside;
    PyObject *a_obj=NULL, *b_obj=NULL;
    PyObject *a_arr=NULL, *b_arr=NULL;
    PyObject *pix_obj=NULL;
    int lonlat=1;
    int nest=1;
    int degrees=1;
    static char *kwlist[] = {"nside", "a", "b", "lonlat", "nest", "degrees", NULL};

    int64_t *pixels;
    int i;
    double *a_data, *b_data;
    double theta, phi;
    healpix_info hpx;
    char err[ERR_SIZE];

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lOO|ppp", kwlist,
                                     &nside, &a_obj, &b_obj,
                                     &lonlat, &nest, &degrees))
        return NULL;

    a_arr = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (a_arr == NULL) return NULL;
    b_arr = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (b_arr == NULL) goto fail;

    int ndim_a = PyArray_NDIM((PyArrayObject *)a_arr);
    int ndim_b = PyArray_NDIM((PyArrayObject *)b_arr);

    if (ndim_a != ndim_b) {
        fprintf(stdout, "Mismatched dims!\n");
        goto fail;
    }
    bool is_scalar = (ndim_a == 0);
    if (is_scalar) {
        fprintf(stdout, "scalar!\n");
    }

    npy_intp a_size = PyArray_SIZE((PyArrayObject *)a_arr);
    npy_intp b_size = PyArray_SIZE((PyArrayObject *)b_arr);

    if (a_size != b_size) {
        fprintf(stdout, "Mismatched sizes!\n");
        goto fail;
    }

    if ((pixels = (int64_t *) calloc(a_size, sizeof(int64_t))) == NULL) {
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
    if (!check_nside(nside, scheme, err)) {
        // raise with error string
        fprintf(stderr, "%s", err);
        goto fail;
    }
    hpx = healpix_info_from_nside(nside, scheme);

    for (i=0; i<a_size; i++) {
        if (lonlat) {
            if (!lonlat_to_thetaphi(a_data[i], b_data[i], &theta, &phi, (bool) degrees, err)) {
                // raise with err string
                goto fail;
            }
        } else {
            theta = a_data[i];
            phi = b_data[i];
        }
        /* check ranges somewhere around here.*/
        pixels[i] = ang2pix(hpx, theta, phi);
    }

    if (is_scalar) {
        pix_obj = PyLong_FromLongLong(pixels[0]);
    } else {
        pix_obj = PyArray_SimpleNewFromData(1, PyArray_DIMS(a_arr), NPY_INT64, pixels);
    }

    /* do I free the memory from pixels? */

    Py_DECREF(a_arr);
    Py_DECREF(b_arr);

    return pix_obj;

 fail:
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);

    // Should raise.
    Py_RETURN_NONE;
}

PyDoc_STRVAR(angle_to_pixel_doc,
             "angle_to_pixel(nside, a, b, nest=True, lonlat=True, degrees=True)\n"
             "--\n\n"
             "Convert angles to pixels.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "nside : `int`\n"
             "    HEALPix nside.  Must be power of 2 for nest ordering.\n"
             "a : `np.ndarray` (N,)\n"
             "    Longitude (radians if lonlat=False, degrees if lonlat=True and degrees=True)\n"
             "b : `np.ndarray` (N,)\n"
             "    Latitude or co-latitude\n"
             "nest : `bool`, optional\n"
             "    Use nest ordering scheme?  Default is True.\n"
             "lonlat : `bool`, optional\n"
             "    Use longitude/latitude instead of longitude/co-latitude (radians).  Default is True.\n"
             "degrees : `bool`, optional\n"
             "    If lonlat is True then this sets if the units are degrees or radians.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "pix : `np.ndarray` (N,)\n"
             "    HEALPix pixel numbers.\n"
             );

static PyMethodDef hpgeom_methods[] = {
    {"angle_to_pixel", (PyCFunction)(void(*)(void))angle_to_pixel,
     METH_VARARGS | METH_KEYWORDS,
    angle_to_pixel_doc},
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
