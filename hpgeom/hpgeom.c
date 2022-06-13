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

    int64_t *pixels = NULL;
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
        PyErr_SetString(PyExc_ValueError, "a and b arrays have mismatched dimensions.");
        goto fail;
    }
    bool is_scalar = (ndim_a == 0);

    npy_intp a_size = PyArray_SIZE((PyArrayObject *)a_arr);
    npy_intp b_size = PyArray_SIZE((PyArrayObject *)b_arr);

    if (a_size != b_size) {
        PyErr_SetString(PyExc_ValueError, "a and b arrays have mismatched sizes.");
        goto fail;
    }

    if ((pixels = (int64_t *) calloc(a_size, sizeof(int64_t))) == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory for pixels.");
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
        PyErr_SetString(PyExc_ValueError, err);
        goto fail;
    }
    hpx = healpix_info_from_nside(nside, scheme);

    for (i=0; i<a_size; i++) {
        if (lonlat) {
            if (!lonlat_to_thetaphi(a_data[i], b_data[i], &theta, &phi, (bool) degrees, err)) {
                PyErr_SetString(PyExc_ValueError, err);
                goto fail;
            }
        } else {
            if (!check_theta_phi(a_data[i], b_data[i], err)) {
                PyErr_SetString(PyExc_ValueError, err);
                goto fail;
            }
            theta = a_data[i];
            phi = b_data[i];
        }
        pixels[i] = ang2pix(hpx, theta, phi);
    }

    if (is_scalar) {
        pix_obj = PyLong_FromLongLong(pixels[0]);
    } else {
        npy_intp *dims = PyArray_DIMS((PyArrayObject *)a_arr);
        pix_obj = PyArray_SimpleNewFromData(1, dims, NPY_INT64, pixels);
    }

    /* do I free the memory from pixels? */

    Py_DECREF(a_arr);
    Py_DECREF(b_arr);

    return pix_obj;

 fail:
    // free memory from pixels if it's set
    if (pixels != NULL) {
        free(pixels);
    }
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);

    return NULL;
}

static PyObject *pixel_to_angle(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    int64_t nside;
    PyObject *pix_obj=NULL;
    PyObject *pix_arr=NULL;
    PyObject *a_obj=NULL, *b_obj=NULL;
    int lonlat=1;
    int nest=1;
    int degrees=1;
    static char *kwlist[] = {"nside", "pix", "lonlat", "nest", "degrees", NULL};

    int64_t *pix_data;
    int i;
    double *as = NULL, *bs = NULL;
    double theta, phi;
    healpix_info hpx;
    char err[ERR_SIZE];

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lO|ppp", kwlist,
                                     &nside, &pix_obj,
                                     &lonlat, &nest, &degrees))
        return NULL;

    pix_arr = PyArray_FROM_OTF(pix_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (pix_arr == NULL) return NULL;

    int ndim_pix = PyArray_NDIM((PyArrayObject *)pix_arr);

    bool is_scalar = (ndim_pix == 0);

    npy_intp pix_size = PyArray_SIZE((PyArrayObject *)pix_arr);

    if ((as = (double *) calloc(pix_size, sizeof(double))) == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory for a array.");
        goto fail;
    }
    if ((bs = (double *) calloc(pix_size, sizeof(double))) == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory for b array.");
        goto fail;
    }

    pix_data = (int64_t *) PyArray_DATA((PyArrayObject *)pix_arr);

    enum Scheme scheme;
    if (nest) {
        scheme = NEST;
    } else {
        scheme = RING;
    }
    if (!check_nside(nside, scheme, err)) {
        PyErr_SetString(PyExc_ValueError, err);
        goto fail;
    }
    hpx = healpix_info_from_nside(nside, scheme);

    for (i=0; i<pix_size; i++) {
        if (!check_pixel(hpx, pix_data[i], err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
        pix2ang(hpx, pix_data[i], &theta, &phi);
        if (lonlat) {
            if (!thetaphi_to_lonlat(theta, phi, &as[i], &bs[i], (bool) degrees, err)) {
                PyErr_SetString(PyExc_ValueError, err);
                goto fail;
            }
        } else {
            as[i] = theta;
            bs[i] = phi;
        }
    }

    if (is_scalar) {
        a_obj = PyFloat_FromDouble(as[0]);
        b_obj = PyFloat_FromDouble(bs[0]);
    } else {
        npy_intp *dims = PyArray_DIMS((PyArrayObject *)pix_arr);
        a_obj = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, as);
        b_obj = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, bs);
    }

    /* do I free the memory from as, bs? Or is that grabbed by the object?*/
    Py_DECREF(pix_arr);

    PyObject* retval = PyTuple_New(2);
    PyTuple_SET_ITEM(retval, 0, a_obj);
    PyTuple_SET_ITEM(retval, 1, b_obj);

    return retval;

 fail:
    if (as != NULL) {
        free(as);
    }
    if (bs != NULL) {
        free(bs);
    }
    Py_XDECREF(pix_arr);

    return NULL;
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
             "    Longitude or theta (radians if lonlat=False, degrees if lonlat=True and degrees=True)\n"
             "b : `np.ndarray` (N,)\n"
             "    Latitude or phi\n"
             "nest : `bool`, optional\n"
             "    Use nest ordering scheme?\n"
             "lonlat : `bool`, optional\n"
             "    Use longitude/latitude instead of longitude/co-latitude (radians).\n"
             "degrees : `bool`, optional\n"
             "    If lonlat is True then this sets if the units are degrees or radians.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "pix : `np.ndarray` (N,)\n"
             "    HEALPix pixel numbers.\n"
             );

PyDoc_STRVAR(pixel_to_angle_doc,
             "pixel_to_angle(nside, pix, nest=True, lonlat=True, degrees=True)\n"
             "--\n\n"
             "Convert pixels to angles.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "nside : `int`\n"
             "    HEALPix nside.  Must be power of 2 for nest ordering.\n"
             "pix : `np.ndarray` (N,)\n"
             "    Pixel numbers\n"
             "nest : `bool`, optional\n"
             "    Use nest ordering scheme?\n"
             "lonlat : `bool`, optional\n"
             "    Output longitude/latitude instead of longitude/co-latitude (radians).\n"
             "degrees : `bool`, optional\n"
             "    If lonlat is True then this sets if the units are degrees or radians.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "a : `np.ndarray` (N,)\n"
             "    Longitude or theta (radians if lonlat=False, degrees if lonlat=True and degrees=True)\n"
             "b : `np.ndarray` (N,)\n"
             "    Latitude or phi\n"
             );


static PyMethodDef hpgeom_methods[] = {
    {"angle_to_pixel", (PyCFunction)(void(*)(void))angle_to_pixel,
     METH_VARARGS | METH_KEYWORDS,
     angle_to_pixel_doc},
    {"pixel_to_angle", (PyCFunction)(void(*)(void))pixel_to_angle,
     METH_VARARGS | METH_KEYWORDS,
     pixel_to_angle_doc},
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
