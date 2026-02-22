/*
 *  Copyright (C) 2022 LSST DESC
 *  Author: Eli Rykoff
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <stdio.h>

#include "thread_compat.h"
#include "healpix_geom.h"
#include "hpgeom_stack.h"
#include "hpgeom_utils.h"

#define NSIDE_DOC_PAR                      \
    "nside : `int` or `np.ndarray` (N,)\n" \
    "    HEALPix nside.  Must be power of 2 for nest ordering.\n"
#define AB_DOC_DESCR                                                                 \
    "    Longitude/latitude (if lonlat=True) or Co-latitude(theta)/longitude(phi)\n" \
    "    (if lonlat=False). Longitude/latitude will be in degrees if degrees=True\n" \
    "    and in radians if degrees=False. Theta/phi are always in radians.\n"

#define AB_DOC_PAR "a, b : `float` or `np.ndarray` (N,)\n" AB_DOC_DESCR
#define NEST_DOC_PAR            \
    "nest : `bool`, optional\n" \
    "    Use nest ordering scheme?\n"
#define LONLAT_DOC_PAR            \
    "lonlat : `bool`, optional\n" \
    "    Use longitude/latitude for a, b instead of co-latitude/longitude.\n"
#define DEGREES_DOC_PAR            \
    "degrees : `bool`, optional\n" \
    "    If lonlat=True then this sets if the units are degrees or radians.\n"
#define PIX_DOC_PAR                         \
    "pixels : `int` or `np.ndarray` (N,)\n" \
    "    HEALPix pixel numbers.\n"
#define FACT_DOC_PAR                                                         \
    "fact : `int`, optional\n"                                               \
    "    Only used when inclusive=True. The overlap test is performed at\n"  \
    "    a resolution fact*nside. For nest ordering, fact must be a power\n" \
    "    of 2, and nside*fact must always be <= 2**29.  For ring ordering\n" \
    "    fact may be any positive integer.\n"
#define RETURN_PIXEL_RANGES_PAR                                                  \
    "return_pixel_ranges : `bool`, optional\n"                                   \
    "    Return an array of pixel ranges instead of a list of pixels.\n"         \
    "    The ranges will be sorted, and each range is of the form [lo, high).\n" \
    "    This option is only compatible with nest ordering.\n"
#define N_THREADS_PAR               \
    "n_threads : `int`, optional\n" \
    "    Number of threads to use. Any number < 1 will use 1 thread.\n"

PyDoc_STRVAR(angle_to_pixel_doc,
             "angle_to_pixel(nside, a, b, nest=True, lonlat=True, degrees=True)\n"
             "--\n\n"
             "Convert angles to pixels.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR AB_DOC_PAR NEST_DOC_PAR LONLAT_DOC_PAR
                 DEGREES_DOC_PAR N_THREADS_PAR
             "\n"
             "Returns\n"
             "-------\n" PIX_DOC_PAR
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If angles are out of range, or arrays cannot be broadcast"
             "    together.\n");

// Core processing logic - used by both single and multi-threaded paths
static bool angle_to_pixel_iteration(NpyIter *iter, int lonlat, int nest, int degrees,
                                     npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    double theta, phi;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;
    enum Scheme scheme = nest ? NEST : RING;

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        double *a = (double *)dataptrarray[1];
        double *b = (double *)dataptrarray[2];
        int64_t *outpix = (int64_t *)dataptrarray[3];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, scheme, err)) {
                return false;
            }
            hpx = healpix_info_from_nside(*nside, scheme);
            last_nside = *nside;
            started = true;
        }

        // Convert angles
        if (lonlat) {
            if (!hpgeom_lonlat_to_thetaphi(*a, *b, &theta, &phi, (bool)degrees, err)) {
                return false;
            }
        } else {
            if (!hpgeom_check_theta_phi(*a, *b, err)) {
                return false;
            }
            theta = *a;
            phi = *b;
        }

        *outpix = ang2pix(&hpx, theta, phi);

    } while (iternext(iter));

    return true;
}

// Worker function for each thread
static void *angle_to_pixel_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !angle_to_pixel_iteration(td->iter, td->lonlat, td->nest, td->degrees,
                                           td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *angle_to_pixel(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *a_obj = NULL, *b_obj = NULL;
    PyObject *nside_arr = NULL, *a_arr = NULL, *b_arr = NULL;
    PyObject *pix_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int lonlat = 1;
    int nest = 1;
    int degrees = 1;
    int n_threads = 1;
    static char *kwlist[] = {"nside", "a",       "b",         "lonlat",
                             "nest",  "degrees", "n_threads", NULL};

    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|pppi", kwlist, &nside_obj, &a_obj,
                                     &b_obj, &lonlat, &nest, &degrees, &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    a_arr = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (a_arr == NULL) goto fail;
    b_arr = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (b_arr == NULL) goto fail;

    // The input arrays are nside_arr (int64_t), a_arr (double), b_arr (double).
    // The output array is pix_arr (int64_t).
    PyArrayObject *op[4];
    npy_uint32 op_flags[4];
    PyArray_Descr *op_dtypes[4];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)a_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;
    op[2] = (PyArrayObject *)b_arr;
    op_flags[2] = NPY_ITER_READONLY;
    op_dtypes[2] = NULL;
    op[3] = NULL;
    op_flags[3] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[3] = PyArray_DescrFromType(NPY_INT64);

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |=
            NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(4, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags,
                            op_dtypes);

    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, a, b arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(pthread_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].lonlat = lonlat;
            thread_data[t].nest = nest;
            thread_data[t].degrees = degrees;
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], angle_to_pixel_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                // We can set the PyErr here since we have the GIL.
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path: set to the full range 0, iter_size.
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !angle_to_pixel_iteration(iter, lonlat, nest, degrees, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    // The reference to the automatically generated output array is owned
    // by the iterator, so we must explicitly increase the reference
    // count to keep it after deallocating the iterator. This is also the
    // case for all uses following.
    pix_arr = (PyObject *)NpyIter_GetOperandArray(iter)[3];
    Py_INCREF(pix_arr);

    Py_DECREF(nside_arr);
    Py_DECREF(a_arr);
    Py_DECREF(b_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return PyArray_Return((PyArrayObject *)pix_arr);

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);
    Py_XDECREF(pix_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(pixel_to_angle_doc,
             "pixel_to_angle(nside, pix, nest=True, lonlat=True, degrees=True)\n"
             "--\n\n"
             "Convert pixels to angles.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR PIX_DOC_PAR NEST_DOC_PAR
                 LONLAT_DOC_PAR DEGREES_DOC_PAR N_THREADS_PAR
             "\n"
             "Returns\n"
             "-------\n" AB_DOC_PAR
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If pixel values are out of range, or arrays cannot be broadcast"
             "    together.\n");

// Core processing logic for pixel_to_angle - used by both single and multi-threaded paths
static bool pixel_to_angle_iteration(NpyIter *iter, int lonlat, int nest, int degrees,
                                     npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    double theta, phi;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;
    enum Scheme scheme = nest ? NEST : RING;

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        int64_t *pix = (int64_t *)dataptrarray[1];
        double *outa = (double *)dataptrarray[2];
        double *outb = (double *)dataptrarray[3];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, scheme, err)) {
                return false;
            }
            hpx = healpix_info_from_nside(*nside, scheme);
            last_nside = *nside;
            started = true;
        }

        if (!hpgeom_check_pixel(&hpx, *pix, err)) {
            return false;
        }

        pix2ang(&hpx, *pix, &theta, &phi);

        if (lonlat) {
            // We can skip error checking since theta/phi will always be
            // within range on output.
            hpgeom_thetaphi_to_lonlat(theta, phi, outa, outb, (bool)degrees, false, err);
        } else {
            *outa = theta;
            *outb = phi;
        }

    } while (iternext(iter));

    return true;
}

// Worker function for each thread
static void *pixel_to_angle_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !pixel_to_angle_iteration(td->iter, td->lonlat, td->nest, td->degrees,
                                           td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *pixel_to_angle(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *pix_obj = NULL;
    PyObject *nside_arr = NULL, *pix_arr = NULL;
    PyObject *a_arr = NULL, *b_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int lonlat = 1;
    int nest = 1;
    int degrees = 1;
    int n_threads = 1;
    static char *kwlist[] = {"nside", "pix", "lonlat", "nest", "degrees", "n_threads", NULL};

    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pppi", kwlist, &nside_obj, &pix_obj,
                                     &lonlat, &nest, &degrees, &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    pix_arr = PyArray_FROM_OTF(pix_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (pix_arr == NULL) goto fail;

    // The input arrays are nside_arr (int64_t), pix_arr (int64_t).
    // The output arrays are a_arr (double), b_arr (double).
    PyArrayObject *op[4];
    npy_uint32 op_flags[4];
    PyArray_Descr *op_dtypes[4];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)pix_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;
    op[2] = NULL;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);
    op[3] = NULL;
    op_flags[3] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[3] = PyArray_DescrFromType(NPY_DOUBLE);

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(4, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, pix arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].lonlat = lonlat;
            thread_data[t].nest = nest;
            thread_data[t].degrees = degrees;
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], pixel_to_angle_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path: set to the full range 0, iter_size.
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !pixel_to_angle_iteration(iter, lonlat, nest, degrees, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    a_arr = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(a_arr);
    b_arr = (PyObject *)NpyIter_GetOperandArray(iter)[3];
    Py_INCREF(b_arr);

    Py_DECREF(nside_arr);
    Py_DECREF(pix_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    PyObject *retval = PyTuple_New(2);
    PyTuple_SET_ITEM(retval, 0, PyArray_Return((PyArrayObject *)a_arr));
    PyTuple_SET_ITEM(retval, 1, PyArray_Return((PyArrayObject *)b_arr));

    return retval;

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(pix_arr);
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

static PyObject *create_query_return_arr(struct i64rangeset *pixset, int return_pixel_ranges,
                                         int convert, healpix_info *hpx) {
    // Convenience routine to share code between query returns.

    PyObject *return_arr;

    if (return_pixel_ranges) {
        npy_intp dims[2];
        dims[0] = pixset->stack->size / 2;
        dims[1] = 2;

        return_arr = PyArray_SimpleNew(2, dims, NPY_INT64);
        if (return_arr == NULL) goto fail;
        int64_t *range_data = (int64_t *)PyArray_DATA((PyArrayObject *)return_arr);

        memcpy(range_data, pixset->stack->data, pixset->stack->size * sizeof(int64_t));
    } else {
        size_t npix = i64rangeset_npix(pixset);
        npy_intp dims[1];
        dims[0] = (npy_intp)npix;

        return_arr = PyArray_SimpleNew(1, dims, NPY_INT64);
        if (return_arr == NULL) goto fail;
        int64_t *pix_data = (int64_t *)PyArray_DATA((PyArrayObject *)return_arr);

        i64rangeset_fill_buffer(pixset, npix, pix_data);

        if (convert) {
            // Convert from nest to ring
            for (size_t i = 0; i < npix; i++) pix_data[i] = nest2ring(hpx, pix_data[i]);

            // And sort the pixels, as is expected.
            PyArray_Sort((PyArrayObject *)return_arr, 0, NPY_QUICKSORT);
        }
    }

    return return_arr;

fail:
    Py_XDECREF(return_arr);

    return NULL;
}

PyDoc_STRVAR(query_circle_doc,
             "query_circle(nside, a, b, radius, inclusive=False, fact=4, nest=True, "
             "lonlat=True, degrees=True)\n"
             "--\n\n"
             "Returns pixels whose centers lie within the circle defined by a, b\n"
             "([lon, lat] if lonlat=True otherwise [theta, phi]) and radius (in \n"
             "degrees if lonlat=True and degrees=True, otherwise radians) if\n"
             "inclusive is False, or which overlap with this circle (if inclusive\n"
             "is True).\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR "a, b : `float`\n" AB_DOC_DESCR
             "radius : `float`\n"
             "    The radius of the circle. Degrees if degrees=True otherwise radians.\n"
             "inclusive : `bool`, optional\n"
             "    If False, return the exact set of pixels whose pixel centers lie\n"
             "    within the circle. If True, return all pixels that overlap with\n"
             "    the circle. This is an approximation and may return a few extra\n"
             "    pixels.\n" FACT_DOC_PAR NEST_DOC_PAR LONLAT_DOC_PAR
                 DEGREES_DOC_PAR RETURN_PIXEL_RANGES_PAR
             "\n"
             "Returns\n"
             "-------\n"
             "pixels : `np.ndarray` (N,)\n"
             "    Array of pixels (`np.int64`) which cover the circle.\n"
             "    (if return_pixel_ranges is False) or\n"
             "pixel_ranges : `np.ndarray` (M, 2)\n"
             "    Array of pixel ranges, [lo, high), which cover the circle.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If position or radius are out of range, or fact is not allowed.\n"
             "RuntimeError\n"
             "    If query_circle has an internal error.\n"
             "\n"
             "Notes\n"
             "-----\n"
             "This method is more efficient with ring ordering.\n"
             "For inclusive=True, the algorithm may return some pixels which do not overlap\n"
             "with the circle. Higher fact values result in fewer false positives at the\n"
             "expense of increased run time.\n");

static PyObject *query_circle(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    int64_t nside;
    double a, b, radius;
    int inclusive = 0;
    long fact = 4;
    int nest = 1;
    int lonlat = 1;
    int degrees = 1;
    int return_pixel_ranges = 0;
    static char *kwlist[] = {"nside", "a",    "b",      "radius",  "inclusive",
                             "fact",  "nest", "lonlat", "degrees", "return_pixel_ranges",
                             NULL};

    char err[ERR_SIZE];
    int status = 1;
    i64rangeset *pixset = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Lddd|plpppp", kwlist, &nside, &a, &b,
                                     &radius, &inclusive, &fact, &nest, &lonlat, &degrees,
                                     &return_pixel_ranges))
        goto fail;

    if (return_pixel_ranges & ~nest) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Can only use return_pixel_ranges with nest ordering.");
        goto fail;
    }

    double theta, phi;
    if (lonlat) {
        if (!hpgeom_lonlat_to_thetaphi(a, b, &theta, &phi, (bool)degrees, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
        if (degrees) {
            radius *= HPG_D2R;
        }
    } else {
        if (!hpgeom_check_theta_phi(a, b, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
        theta = a;
        phi = b;
    }

    if (!hpgeom_check_radius(radius, err)) {
        PyErr_SetString(PyExc_ValueError, err);
        goto fail;
    }

    enum Scheme scheme;
    if (nest) {
        scheme = NEST;
    } else {
        scheme = RING;
    }
    if (!hpgeom_check_nside(nside, scheme, err)) {
        PyErr_SetString(PyExc_ValueError, err);
        goto fail;
    }
    healpix_info hpx = healpix_info_from_nside(nside, scheme);

    pixset = i64rangeset_new(&status, err);
    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    if (!inclusive) {
        fact = 0;
    } else {
        if (!hpgeom_check_fact(&hpx, fact, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

    NPY_BEGIN_ALLOW_THREADS

    query_disc(&hpx, theta, phi, radius, fact, pixset, &status, err);

    NPY_END_ALLOW_THREADS

    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    PyObject *return_arr = create_query_return_arr(pixset, return_pixel_ranges, 0, &hpx);

    i64rangeset_delete(pixset);

    return PyArray_Return((PyArrayObject *)return_arr);

fail:
    i64rangeset_delete(pixset);

    return NULL;
}

PyDoc_STRVAR(query_polygon_doc,
             "query_polygon(nside, a, b, inclusive=False, fact=4, nest=True, lonlat=True, "
             "degrees=True)\n"
             "--\n\n"
             "Returns pixels whose centers lie within the convex polygon defined by the "
             "points in a, b\n"
             "([lon, lat] if lonlat=True, otherwise [theta, phi]) if inclusive is False, or "
             "which overlap\n"
             "with this polygon (if inclusive is True).\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR "a, b : `np.ndarray` (N,)\n" AB_DOC_DESCR
             "inclusive : `bool`, optional\n"
             "    If False, return the exact set of pixels whose pixel centers lie\n"
             "    within the polygon. If True, return all pixels that overlap with\n"
             "    the polygon. This is an approximation and may return a few extra\n"
             "    pixels.\n" FACT_DOC_PAR NEST_DOC_PAR LONLAT_DOC_PAR
                 DEGREES_DOC_PAR RETURN_PIXEL_RANGES_PAR
             "\n"
             "Returns\n"
             "-------\n"
             "pixels : `np.ndarray` (N,)\n"
             "    Array of pixels (`np.int64`) which cover the polygon.\n"
             "    (if return_pixel_ranges is False) or\n"
             "pixel_ranges : `np.ndarray` (M, 2)\n"
             "    Array of pixel ranges, [lo, high), which cover the polygon.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If vertices are out of range.\n"
             "RuntimeError\n"
             "    If polygon does not have at least 3 vertices, or polygon is not convex,\n"
             "    or polygon has degenerate corners, or there is an internal error.\n"
             "\n"
             "Notes\n"
             "-----\n"
             "This method is more efficient with nest ordering.\n"
             "For inclusive=True, the algorithm may return some pixels which do not overlap\n"
             "with the polygon. Higher fact values result in fewer false positives at the\n"
             "expense of increased run time.\n");

static PyObject *query_polygon_meth(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    int64_t nside;
    PyObject *a_obj = NULL, *b_obj = NULL;
    PyObject *a_arr = NULL, *b_arr = NULL;
    int inclusive = 0;
    long fact = 4;
    int nest = 1;
    int lonlat = 1;
    int degrees = 1;
    int return_pixel_ranges = 0;
    static char *kwlist[] = {"nside", "a",      "b",       "inclusive",           "fact",
                             "nest",  "lonlat", "degrees", "return_pixel_ranges", NULL};
    char err[ERR_SIZE];
    int status = 1;
    i64rangeset *pixset = NULL;
    pointingarr *vertices = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "LOO|plpppp", kwlist, &nside, &a_obj,
                                     &b_obj, &inclusive, &fact, &nest, &lonlat, &degrees,
                                     &return_pixel_ranges))
        goto fail;

    if (return_pixel_ranges & ~nest) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Can only use return_pixel_ranges with nest ordering.");
        goto fail;
    }

    a_arr = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (a_arr == NULL) goto fail;
    b_arr = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (b_arr == NULL) goto fail;

    if (PyArray_NDIM((PyArrayObject *)a_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "a array must be 1D.");
        goto fail;
    }
    if (PyArray_NDIM((PyArrayObject *)b_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "b array must be 1D.");
        goto fail;
    }

    npy_intp nvert = PyArray_DIM((PyArrayObject *)a_arr, 0);
    if (PyArray_DIM((PyArrayObject *)b_arr, 0) != nvert) {
        PyErr_SetString(PyExc_ValueError, "a and b arrays must be the same length.");
        goto fail;
    }
    if (nvert < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Polygon must have at least 3 vertices.");
        goto fail;
    }

    vertices = pointingarr_new(nvert, &status, err);
    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    enum Scheme scheme;
    if (nest) {
        scheme = NEST;
    } else {
        scheme = RING;
    }
    if (!hpgeom_check_nside(nside, scheme, err)) {
        PyErr_SetString(PyExc_ValueError, err);
        goto fail;
    }
    healpix_info hpx = healpix_info_from_nside(nside, scheme);

    pixset = i64rangeset_new(&status, err);
    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    if (!inclusive) {
        fact = 0;
    } else {
        if (!hpgeom_check_fact(&hpx, fact, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

    double *a_data = (double *)PyArray_DATA((PyArrayObject *)a_arr);
    double *b_data = (double *)PyArray_DATA((PyArrayObject *)b_arr);

    double theta, phi;
    for (npy_intp i = 0; i < nvert; i++) {
        if (lonlat) {
            if (!hpgeom_lonlat_to_thetaphi(a_data[i], b_data[i], &theta, &phi, (bool)degrees,
                                           err)) {
                PyErr_SetString(PyExc_ValueError, err);
                goto fail;
            }
        } else {
            if (!hpgeom_check_theta_phi(a_data[i], b_data[i], err)) {
                PyErr_SetString(PyExc_ValueError, err);
                goto fail;
            }
            theta = a_data[i];
            phi = b_data[i];
        }
        vertices->data[i].theta = theta;
        vertices->data[i].phi = phi;
    }

    // Check for a closed polygon with a small double-precision delta.
    double delta_theta = fabs(vertices->data[nvert - 1].theta - vertices->data[0].theta);
    double delta_phi = fabs(vertices->data[nvert - 1].phi - vertices->data[0].phi);
    if ((delta_theta < 1e-14) && (delta_phi < 1e-14)) {
        // Skip last coord
        vertices->size--;
    }

    NPY_BEGIN_ALLOW_THREADS

    query_polygon(&hpx, vertices, fact, pixset, &status, err);

    NPY_END_ALLOW_THREADS

    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    PyObject *return_arr = create_query_return_arr(pixset, return_pixel_ranges, 0, &hpx);

    Py_DECREF(a_arr);
    Py_DECREF(b_arr);
    i64rangeset_delete(pixset);
    pointingarr_delete(vertices);

    return PyArray_Return((PyArrayObject *)return_arr);

fail:
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);
    i64rangeset_delete(pixset);
    pointingarr_delete(vertices);

    return NULL;
}

PyDoc_STRVAR(query_ellipse_doc,
             "query_ellipse(nside, a, b, semi_major, semi_minor, alpha, inclusive=False, "
             "fact=4, nest=True, lonlat=True, degrees=True)\n"
             "--\n\n"
             "Returns pixels whose centers lie within an ellipse if inclusive is False,\n"
             "or which overlap with this ellipse if inclusive is True. The ellipse is\n"
             "defined by a center a, b ([lon, lat] if lonlat=True otherwise [theta, phi]\n"
             "and semi-major and semi-minor axes (in degrees if lonlat=True and\n"
             "degrees=True, otherwise radians). The inclination angle alpha is defined\n"
             "East of North, and is in degrees if lonlat=True and degrees=True,\n"
             "otherwise radians. The shape of the ellipse is defined by the set\n"
             "of points where the sum of the distances from a point to each of the\n"
             "foci add up to less than twice the semi-major axis.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR "a, b : `float`\n" AB_DOC_DESCR
             "semi_major, semi_minor : `float`\n"
             "    The semi-major and semi-minor axes of the ellipse. Degrees if degrees=True\n"
             "    and lonlat=True, otherwise radians. The semi-major axis must be >= the\n"
             "    semi-minor axis.\n"
             "alpha : `float`\n"
             "    Inclination angle, counterclockwise with respect to North. Degrees if\n"
             "    degrees=True and lonlat=True, otherwise radians.\n"
             "inclusive : `bool`, optional\n"
             "    If False, return the exact set of pixels whose pixel centers lie\n"
             "    within the ellipse. If True, return all pixels that overlap with\n"
             "    the ellipse. This is an approximation and may return a few extra\n"
             "    pixels.\n" FACT_DOC_PAR NEST_DOC_PAR LONLAT_DOC_PAR
                 DEGREES_DOC_PAR RETURN_PIXEL_RANGES_PAR
             "\n"
             "Returns\n"
             "-------\n"
             "pixels : `np.ndarray` (N,)\n"
             "    Array of pixels (`np.int64`) which cover the ellipse.\n"
             "    (if return_pixel_ranges is False) or\n"
             "pixel_ranges : `np.ndarray` (M, 2)\n"
             "    Array of pixel ranges, [lo, high), which cover the ellipse.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If position or semi-major/minor axes are out of range, or fact is \n"
             "    not allowed.\n"
             "RuntimeError\n"
             "    If query_ellipse has an internal error.\n"
             "\n"
             "Notes\n"
             "-----\n"
             "This method runs natively only with nest ordering. If called with ring\n"
             "ordering then a ResourceWarning is emitted and the pixel numbers will be\n"
             "converted to ring and sorted before output.\n"
             "For inclusive=True, the algorithm may return some pixels which do not overlap\n"
             "with the ellipse. Higher fact values result in fewer false positives at the\n"
             "expense of increased run time.\n");

static PyObject *query_ellipse_meth(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    int64_t nside;
    double a, b, semi_major, semi_minor, alpha;
    int inclusive = 0;
    long fact = 4;
    int nest = 1;
    int lonlat = 1;
    int degrees = 1;
    int return_pixel_ranges = 0;
    static char *kwlist[] = {
        "nside", "a",    "b",      "semi_major", "semi_minor",          "alpha", "inclusive",
        "fact",  "nest", "lonlat", "degrees",    "return_pixel_ranges", NULL};

    char err[ERR_SIZE];
    int status = 1;
    i64rangeset *pixset = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Lddddd|plpppp", kwlist, &nside, &a, &b,
                                     &semi_major, &semi_minor, &alpha, &inclusive, &fact,
                                     &nest, &lonlat, &degrees, &return_pixel_ranges))
        goto fail;

    if (return_pixel_ranges & ~nest) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Can only use return_pixel_ranges with nest ordering.");
        goto fail;
    }

    double theta, phi;
    if (lonlat) {
        if (!hpgeom_lonlat_to_thetaphi(a, b, &theta, &phi, (bool)degrees, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
        if (degrees) {
            semi_major *= HPG_D2R;
            semi_minor *= HPG_D2R;
            alpha *= HPG_D2R;
        }
    } else {
        if (!hpgeom_check_theta_phi(a, b, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
        theta = a;
        phi = b;
    }

    if (!hpgeom_check_semi(semi_major, semi_minor, err)) {
        PyErr_SetString(PyExc_ValueError, err);
        goto fail;
    }

    if (!nest) {
        PyErr_WarnEx(PyExc_ResourceWarning,
                     "query_ellipse natively supports nest ordering.  Result will be "
                     "converted from nest->ring and sorted",
                     0);
    }

    if (!hpgeom_check_nside(nside, NEST, err)) {
        PyErr_SetString(PyExc_ValueError, err);
        goto fail;
    }
    healpix_info hpx = healpix_info_from_nside(nside, NEST);

    pixset = i64rangeset_new(&status, err);
    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    if (!inclusive) {
        fact = 0;
    } else {
        if (!hpgeom_check_fact(&hpx, fact, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

    NPY_BEGIN_ALLOW_THREADS

    query_ellipse(&hpx, theta, phi, semi_major, semi_minor, alpha, fact, pixset, &status, err);

    NPY_END_ALLOW_THREADS

    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    PyObject *return_arr = create_query_return_arr(pixset, return_pixel_ranges, !nest, &hpx);

    i64rangeset_delete(pixset);

    return PyArray_Return((PyArrayObject *)return_arr);

fail:
    i64rangeset_delete(pixset);

    return NULL;
}

PyDoc_STRVAR(
    query_box_doc,
    "query_box(nside, a0, a1, b0, b1, inclusive=False, fact=4, nest=True, lonlat=True, "
    "degrees=True)\n"
    "--\n\n"
    "Returns pixels whose centers lie within a box if inclusive is False,\n"
    "or which overlap with this box if inclusive is True. The box is defined\n"
    "by all the points within [a0, a1] and [b0, b1] ([lon, lat] if lonlat=True\n"
    "otherwise [theta, phi] (in degrees if lonlat=True and degrees=True, otherwise\n"
    "radians). The box will have boundaries in constant longitude/latitude, rather\n"
    "than great circle boundaries as with query_polygon. If a0 > a1 then the box will\n"
    "wrap around 360 degrees. If a0 == 0.0 and a1 == 360.0 then the box will contain\n"
    "points at all longitudes. If b0 == 90.0 or -90.0 then the box will be an arc\n"
    "of a circle with the center at the north/south pole.\n"
    "\n"
    "Parameters\n"
    "----------\n" NSIDE_DOC_PAR "a0, a1, b0, b1 : `float`\n" AB_DOC_DESCR
    "inclusive : `bool`, optional\n"
    "    If False, return the exact set of pixels whose pixel centers lie\n"
    "    within the box. If True, return all pixels that overlap with\n"
    "    the box. This is an approximation and may return a few extra\n"
    "    pixels.\n" FACT_DOC_PAR NEST_DOC_PAR LONLAT_DOC_PAR
        DEGREES_DOC_PAR RETURN_PIXEL_RANGES_PAR
    "\n"
    "Returns\n"
    "-------\n"
    "pixels : `np.ndarray` (N,)\n"
    "    Array of pixels (`np.int64`) which cover the box.\n"
    "    (if return_pixel_ranges is False) or\n"
    "pixel_ranges : `np.ndarray` (M, 2)\n"
    "    Array of pixel ranges, [lo, high), which cover the box.\n"
    "\n"
    "Raises\n"
    "------\n"
    "ValueError\n"
    "    If positions are out of range, or fact is not allowed.\n"
    "RuntimeError\n"
    "    If query_box has an internal error.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This method runs natively only with nest ordering. If called with ring\n"
    "ordering then a ResourceWarning is emitted and the pixel numbers will be\n"
    "converted to ring and sorted before output.\n"
    "For inclusive=True, the algorithm may return some pixels which do not overlap\n"
    "with the box. Higher fact values result in fewer false positives at the\n"
    "expense of increased run time.\n");

static PyObject *query_box_meth(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    int64_t nside;
    double a0, a1, b0, b1;
    int inclusive = 0;
    long fact = 4;
    int nest = 1;
    int lonlat = 1;
    int degrees = 1;
    int return_pixel_ranges = 0;
    static char *kwlist[] = {"nside",
                             "a0",
                             "a1",
                             "b0",
                             "b1",
                             "inclusive",
                             "fact",
                             "nest",
                             "lonlat",
                             "degrees",
                             "return_pixel_ranges",
                             NULL};

    char err[ERR_SIZE];
    int status = 1;
    i64rangeset *pixset = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Ldddd|plpppp", kwlist, &nside, &a0, &a1,
                                     &b0, &b1, &inclusive, &fact, &nest, &lonlat, &degrees,
                                     &return_pixel_ranges))
        goto fail;

    if (return_pixel_ranges & ~nest) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Can only use return_pixel_ranges with nest ordering.");
        goto fail;
    }

    double theta0, theta1, phi0, phi1;
    bool full_lon = false;
    if (lonlat) {
        if (a0 == 0.0 && a1 == 360.0) full_lon = true;
        if (b0 > b1) {
            PyErr_SetString(PyExc_ValueError, "b1/lat1 must be >= b0/lat0.");
            goto fail;
        }
        // Swap theta ordering.
        if (!hpgeom_lonlat_to_thetaphi(a0, b0, &theta1, &phi0, (bool)degrees, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
        if (!hpgeom_lonlat_to_thetaphi(a1, b1, &theta0, &phi1, (bool)degrees, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    } else {
        if (b0 == 0.0 && b1 == HPG_TWO_PI) full_lon = true;
        if (a0 > a1) {
            PyErr_SetString(PyExc_ValueError, "a1/colatitude1 must be <= a0/colatitude0.");
            goto fail;
        }
        if (!hpgeom_check_theta_phi(a0, b0, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
        theta0 = a0;
        phi0 = b0;
        if (!hpgeom_check_theta_phi(a1, b1, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
        theta1 = a1;
        phi1 = b1;
    }

    if (!nest) {
        PyErr_WarnEx(PyExc_ResourceWarning,
                     "query_box natively supports nest ordering.  Result will be "
                     "converted from nest->ring and sorted",
                     0);
    }

    if (!hpgeom_check_nside(nside, NEST, err)) {
        PyErr_SetString(PyExc_ValueError, err);
        goto fail;
    }
    healpix_info hpx = healpix_info_from_nside(nside, NEST);

    pixset = i64rangeset_new(&status, err);
    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    if (!inclusive) {
        fact = 0;
    } else {
        if (!hpgeom_check_fact(&hpx, fact, err)) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

    NPY_BEGIN_ALLOW_THREADS

    query_box(&hpx, theta0, theta1, phi0, phi1, full_lon, fact, pixset, &status, err);

    NPY_END_ALLOW_THREADS

    if (!status) {
        PyErr_SetString(PyExc_RuntimeError, err);
        goto fail;
    }

    PyObject *return_arr = create_query_return_arr(pixset, return_pixel_ranges, !nest, &hpx);

    i64rangeset_delete(pixset);

    return PyArray_Return((PyArrayObject *)return_arr);

fail:
    i64rangeset_delete(pixset);

    return NULL;
}

PyDoc_STRVAR(nest_to_ring_doc,
             "nest_to_ring(nside, pix)\n"
             "--\n\n"
             "Convert pixel number from nest to ring ordering.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR
             "pix : `int` or `np.ndarray` (N,)\n"
             "    The pixel numbers in nest scheme.\n"
             N_THREADS_PAR
             "\n"
             "Returns\n"
             "-------\n"
             "pix : `int` or `np.ndarray` (N,)\n"
             "    The pixel numbers in ring scheme.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    Pixel or nside values are out of range.\n");

// Core processing logic for nest_to_ring - used by both single and multi-threaded paths
static bool nest_to_ring_iteration(NpyIter *iter, npy_intp start_idx, npy_intp end_idx, 
                                    char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        int64_t *nest_pix = (int64_t *)dataptrarray[1];
        int64_t *ring_pix = (int64_t *)dataptrarray[2];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, NEST, err)) {
                return false;
            }
            hpx = healpix_info_from_nside(*nside, NEST);
            last_nside = *nside;
            started = true;
        }

        if (!hpgeom_check_pixel(&hpx, *nest_pix, err)) {
            return false;
        }

        *ring_pix = nest2ring(&hpx, *nest_pix);

    } while (iternext(iter));

    return true;
}

// Worker function for each thread
static void *nest_to_ring_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !nest_to_ring_iteration(td->iter, td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *nest_to_ring(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *nest_pix_obj = NULL;
    PyObject *nside_arr = NULL, *nest_pix_arr = NULL;
    PyObject *ring_pix_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int n_threads = 1;
    static char *kwlist[] = {"nside", "pix", "n_threads", NULL};

    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i", kwlist, &nside_obj, &nest_pix_obj,
                                     &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    nest_pix_arr =
        PyArray_FROM_OTF(nest_pix_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nest_pix_arr == NULL) goto fail;

    // The input arrays are nside_arr (int64_t) and nest_pix_arr (int64_t).
    // The output array is ring_pix_arr (int64_t).
    PyArrayObject *op[3];
    npy_uint32 op_flags[3];
    PyArray_Descr *op_dtypes[3];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)nest_pix_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;
    op[2] = NULL;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[2] = PyArray_DescrFromType(NPY_INT64);

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(3, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, pix arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], nest_to_ring_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path: set to the full range 0, iter_size.
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !nest_to_ring_iteration(iter, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    ring_pix_arr = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(ring_pix_arr);

    Py_DECREF(nside_arr);
    Py_DECREF(nest_pix_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return PyArray_Return((PyArrayObject *)ring_pix_arr);

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(nest_pix_arr);
    Py_XDECREF(ring_pix_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(ring_to_nest_doc,
             "ring_to_nest(nside, pix)\n"
             "--\n\n"
             "Convert pixel number from ring to nest ordering.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR
             "pix : `int` or `np.ndarray` (N,)\n"
             "    The pixel numbers in ring scheme.\n"
             N_THREADS_PAR
             "\n"
             "Returns\n"
             "-------\n"
             "pix : `int` or `np.ndarray` (N,)\n"
             "    The pixel numbers in nest scheme.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    Pixel or nside values are out of range.\n");

// Core processing logic for ring_to_nest - used by both single and multi-threaded paths
static bool ring_to_nest_iteration(NpyIter *iter, npy_intp start_idx, npy_intp end_idx,
                                    char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        int64_t *ring_pix = (int64_t *)dataptrarray[1];
        int64_t *nest_pix = (int64_t *)dataptrarray[2];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, NEST, err)) {
                return false;
            }
            hpx = healpix_info_from_nside(*nside, NEST);
            last_nside = *nside;
            started = true;
        }

        if (!hpgeom_check_pixel(&hpx, *ring_pix, err)) {
            return false;
        }

        *nest_pix = ring2nest(&hpx, *ring_pix);

    } while (iternext(iter));

    return true;
}

// Worker function for each thread (reuses ThreadData)
static void *ring_to_nest_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !ring_to_nest_iteration(td->iter, td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *ring_to_nest(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *ring_pix_obj = NULL;
    PyObject *nside_arr = NULL, *ring_pix_arr = NULL;
    PyObject *nest_pix_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int n_threads = 1;
    static char *kwlist[] = {"nside", "pix", "n_threads", NULL};

    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i", kwlist, &nside_obj, &ring_pix_obj,
                                     &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    ring_pix_arr =
        PyArray_FROM_OTF(ring_pix_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (ring_pix_arr == NULL) goto fail;

    // The input arrays are nside_arr (int64_t) and ring_pix_arr (int64_t).
    // The output array is nest_pix_arr (int64_t).
    PyArrayObject *op[3];
    npy_uint32 op_flags[3];
    PyArray_Descr *op_dtypes[3];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)ring_pix_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;
    op[2] = NULL;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[2] = PyArray_DescrFromType(NPY_INT64);

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(3, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, pix arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            // These fields aren't used by ring_to_nest_iteration, but we need to
            // initialize them to avoid potential issues
            thread_data[t].lonlat = 0;
            thread_data[t].nest = 0;
            thread_data[t].degrees = 0;
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], ring_to_nest_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path: set to the full range 0, iter_size.
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !ring_to_nest_iteration(iter, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    nest_pix_arr = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(nest_pix_arr);

    Py_DECREF(nside_arr);
    Py_DECREF(ring_pix_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return PyArray_Return((PyArrayObject *)nest_pix_arr);

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(ring_pix_arr);
    Py_XDECREF(nest_pix_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(boundaries_doc,
             "boundaries(nside, pix, step=1, nest=True, lonlat=True, degrees=True)\n"
             "--\n\n"
             "Returns an array containing lon/lat or colatitude/longitude to the\n"
             "boundary of the given pixel(s).\n"
             "\n"
             "The returned arrays have the shape (4*step) or (npixel, 4*step).\n"
             "In order to get coordinates for just the corners, specify step=1.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR PIX_DOC_PAR
             "step : `int`, optional\n"
             "    Number of steps for each side of the pixel.\n" NEST_DOC_PAR
                 LONLAT_DOC_PAR DEGREES_DOC_PAR N_THREADS_PAR
             "\n"
             "Returns\n"
             "-------\n"
             "a, b : `np.ndarray` (4*step,) or (N, 4*step,)\n" AB_DOC_DESCR
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If pixel values are out of range, or nside, pix arrays are not\n"
             "    compatible, or step is not positive.\n");

// Core processing logic for boundaries - used by both single and multi-threaded paths
static bool boundaries_iteration(NpyIter *iter, int lonlat, int nest, int degrees,
                                 long step, double *as, double *bs,
                                 npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;
    enum Scheme scheme = nest ? NEST : RING;
    pointingarr *ptg_arr = NULL;
    int status;

    // Allocate pointingarr for this thread
    ptg_arr = pointingarr_new(4 * step, &status, err);
    if (!status) {
        return false;
    }

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        pointingarr_delete(ptg_arr);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        pointingarr_delete(ptg_arr);
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        int64_t *pix = (int64_t *)dataptrarray[1];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, scheme, err)) {
                pointingarr_delete(ptg_arr);
                return false;
            }
            hpx = healpix_info_from_nside(*nside, scheme);
            last_nside = *nside;
            started = true;
        }

        if (!hpgeom_check_pixel(&hpx, *pix, err)) {
            pointingarr_delete(ptg_arr);
            return false;
        }

        boundaries(&hpx, *pix, step, ptg_arr, &status);
        if (!status) {
            snprintf(err, ERR_SIZE, "boundaries() failed");
            pointingarr_delete(ptg_arr);
            return false;
        }

        for (size_t i = 0; i < ptg_arr->size; i++) {
            size_t index = ptg_arr->size * NpyIter_GetIterIndex(iter) + i;
            if (lonlat) {
                // We can skip error checking since theta/phi will always be
                // within range on output.
                hpgeom_thetaphi_to_lonlat(ptg_arr->data[i].theta, ptg_arr->data[i].phi,
                                          &as[index], &bs[index], (bool)degrees, false,
                                          err);
            } else {
                as[index] = ptg_arr->data[i].theta;
                bs[index] = ptg_arr->data[i].phi;
            }
        }

    } while (iternext(iter));

    pointingarr_delete(ptg_arr);
    return true;
}

// Worker function for each thread
static void *boundaries_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !boundaries_iteration(td->iter, td->lonlat, td->nest, td->degrees,
                                       td->step, (double *) td->data0, (double *) td->data1,
                                       td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *boundaries_meth(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *pix_obj = NULL;
    PyObject *nside_arr = NULL, *pix_arr = NULL;
    PyObject *a_arr = NULL, *b_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int lonlat = 1;
    int nest = 1;
    int degrees = 1;
    long step = 1;
    int n_threads = 1;
    static char *kwlist[] = {"nside", "pix", "step", "lonlat", "nest", "degrees", "n_threads", NULL};

    double *as = NULL, *bs = NULL;
    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Lpppi", kwlist, &nside_obj, &pix_obj,
                                     &step, &lonlat, &nest, &degrees, &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    pix_arr = PyArray_FROM_OTF(pix_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (pix_arr == NULL) goto fail;

    if (step < 1) {
        PyErr_SetString(PyExc_ValueError, "step must be positive.");
        goto fail;
    }

    if (PyArray_NDIM((PyArrayObject *)pix_arr) > 1) {
        PyErr_SetString(PyExc_ValueError, "pix array must be at most 1D.");
        goto fail;
    }

    // The input arrays are nside_arr (int64_t) and pix_arr (int64_t).
    // We are allocating our own output arrays.
    PyArrayObject *op[2];
    npy_uint32 op_flags[2];
    PyArray_Descr *op_dtypes[2];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)pix_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(2, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, pix arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);
    int ndims = NpyIter_GetNDim(iter);

    // Allocate output arrays
    if (ndims == 0) {
        npy_intp dims[1];
        dims[0] = 4 * step;
        a_arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
        if (a_arr == NULL) goto fail;
        b_arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
        if (b_arr == NULL) goto fail;
    } else {
        npy_intp dims[2];
        dims[0] = iter_size;
        dims[1] = 4 * step;
        a_arr = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
        if (a_arr == NULL) goto fail;
        b_arr = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
        if (b_arr == NULL) goto fail;
    }
    as = (double *)PyArray_DATA((PyArrayObject *)a_arr);
    bs = (double *)PyArray_DATA((PyArrayObject *)b_arr);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < (MIN_CHUNK_SIZE / step)) {
            n_threads = iter_size / (MIN_CHUNK_SIZE / step);
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].lonlat = lonlat;
            thread_data[t].nest = nest;
            thread_data[t].degrees = degrees;
            thread_data[t].step = step;
            thread_data[t].data0 = (void *) as;
            thread_data[t].data1 = (void *) bs;
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], boundaries_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !boundaries_iteration(iter, lonlat, nest, degrees, step, as, bs,
                                           0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    Py_DECREF(nside_arr);
    Py_DECREF(pix_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    PyObject *retval = PyTuple_New(2);
    PyTuple_SET_ITEM(retval, 0, PyArray_Return((PyArrayObject *)a_arr));
    PyTuple_SET_ITEM(retval, 1, PyArray_Return((PyArrayObject *)b_arr));

    return retval;

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(pix_arr);
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(vector_to_pixel_doc,
             "vector_to_pixel(nside, x, y, z, nest=True)\n"
             "--\n\n"
             "Convert vectors to pixels.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR
             "x : `float` or `np.ndarray` (N,)\n"
             "    x coordinates for vectors.\n"
             "y : `float` or `np.ndarray` (N,)\n"
             "    y coordinates for vectors.\n"
             "z : `float` or `np.ndarray` (N,)\n"
             "    z coordinates for vectors.\n" NEST_DOC_PAR N_THREADS_PAR
             "\n"
             "Returns\n"
             "-------\n" PIX_DOC_PAR);

// Core processing logic for vector_to_pixel - used by both single and multi-threaded paths
static bool vector_to_pixel_iteration(NpyIter *iter, int nest,
                                      npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;
    enum Scheme scheme = nest ? NEST : RING;
    vec3 vec;

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        double *x = (double *)dataptrarray[1];
        double *y = (double *)dataptrarray[2];
        double *z = (double *)dataptrarray[3];
        int64_t *outpix = (int64_t *)dataptrarray[4];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, scheme, err)) {
                return false;
            }
            hpx = healpix_info_from_nside(*nside, scheme);
            last_nside = *nside;
            started = true;
        }

        vec.x = *x;
        vec.y = *y;
        vec.z = *z;
        *outpix = vec2pix(&hpx, &vec);

    } while (iternext(iter));

    return true;
}

// Worker function for each thread (reuses ThreadData)
static void *vector_to_pixel_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !vector_to_pixel_iteration(td->iter, td->nest,
                                            td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *vector_to_pixel(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *x_obj = NULL, *y_obj = NULL, *z_obj = NULL;
    PyObject *nside_arr = NULL, *x_arr = NULL, *y_arr = NULL, *z_arr = NULL;
    PyObject *pix_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int nest = 1;
    int n_threads = 1;
    static char *kwlist[] = {"nside", "x", "y", "z", "nest", "n_threads", NULL};

    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|pi", kwlist, &nside_obj, &x_obj,
                                     &y_obj, &z_obj, &nest, &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    x_arr = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (x_arr == NULL) goto fail;
    y_arr = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (y_arr == NULL) goto fail;
    z_arr = PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (z_arr == NULL) goto fail;

    // The input arrays are nside_arr (int64_t), x_arr (double), y_arr (double),
    // z_arr (double).
    // The output array is pix_arr (int64_t).
    PyArrayObject *op[5];
    npy_uint32 op_flags[5];
    PyArray_Descr *op_dtypes[5];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)x_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;
    op[2] = (PyArrayObject *)y_arr;
    op_flags[2] = NPY_ITER_READONLY;
    op_dtypes[2] = NULL;
    op[3] = (PyArrayObject *)z_arr;
    op_flags[3] = NPY_ITER_READONLY;
    op_dtypes[3] = NULL;
    op[4] = NULL;
    op_flags[4] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[4] = PyArray_DescrFromType(NPY_INT64);

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(5, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, x, y, z arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].lonlat = 0;  // Unused
            thread_data[t].nest = nest;
            thread_data[t].degrees = 0;  // Unused
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], vector_to_pixel_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path: set to the full range 0, iter_size.
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !vector_to_pixel_iteration(iter, nest, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    pix_arr = (PyObject *)NpyIter_GetOperandArray(iter)[4];
    Py_INCREF(pix_arr);

    Py_DECREF(nside_arr);
    Py_DECREF(x_arr);
    Py_DECREF(y_arr);
    Py_DECREF(z_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return PyArray_Return((PyArrayObject *)pix_arr);

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(x_arr);
    Py_XDECREF(y_arr);
    Py_XDECREF(z_arr);
    Py_XDECREF(pix_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(pixel_to_vector_doc,
             "pixel_to_vector(nside, pix, nest=True)\n"
             "--\n\n"
             "Convert pixels to vectors.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR PIX_DOC_PAR NEST_DOC_PAR N_THREADS_PAR
             "\n"
             "Returns\n"
             "-------\n"
             "x : `float` or `np.ndarray` (N,)\n"
             "    x coordinates for vectors.\n"
             "y : `float` or `np.ndarray` (N,)\n"
             "    y coordinates for vectors.\n"
             "z : `float` or `np.ndarray` (N,)\n"
             "    z coordinates for vectors.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If pixel values are out of range, or nside and pix arrays are not\n"
             "    compatible.\n");

// Core processing logic for pixel_to_vector - used by both single and multi-threaded paths
static bool pixel_to_vector_iteration(NpyIter *iter, int nest,
                                      npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;
    enum Scheme scheme = nest ? NEST : RING;
    vec3 vec;

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        int64_t *pix = (int64_t *)dataptrarray[1];
        double *x = (double *)dataptrarray[2];
        double *y = (double *)dataptrarray[3];
        double *z = (double *)dataptrarray[4];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, scheme, err)) {
                return false;
            }
            hpx = healpix_info_from_nside(*nside, scheme);
            last_nside = *nside;
            started = true;
        }

        if (!hpgeom_check_pixel(&hpx, *pix, err)) {
            return false;
        }

        vec = pix2vec(&hpx, *pix);
        *x = vec.x;
        *y = vec.y;
        *z = vec.z;

    } while (iternext(iter));

    return true;
}

// Worker function for each thread (reuses ThreadData)
static void *pixel_to_vector_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !pixel_to_vector_iteration(td->iter, td->nest,
                                            td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *pixel_to_vector(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *pix_obj = NULL;
    PyObject *nside_arr = NULL, *pix_arr = NULL;
    PyObject *x_arr = NULL, *y_arr = NULL, *z_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int nest = 1;
    int n_threads = 1;
    static char *kwlist[] = {"nside", "pix", "nest", "n_threads", NULL};

    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pi", kwlist, &nside_obj, &pix_obj,
                                     &nest, &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    pix_arr = PyArray_FROM_OTF(pix_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (pix_arr == NULL) goto fail;

    // The input arrays are nside_arr (int64_t), pix_arr (int64_t).
    // The output arrays are x_arr (double), y_arr (double), z_arr (double).
    PyArrayObject *op[5];
    npy_uint32 op_flags[5];
    PyArray_Descr *op_dtypes[5];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)pix_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;
    op[2] = NULL;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);
    op[3] = NULL;
    op_flags[3] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[3] = PyArray_DescrFromType(NPY_DOUBLE);
    op[4] = NULL;
    op_flags[4] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[4] = PyArray_DescrFromType(NPY_DOUBLE);

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(5, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, pix arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].lonlat = 0;  // Unused
            thread_data[t].nest = nest;
            thread_data[t].degrees = 0;  // Unused
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], pixel_to_vector_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path: set to the full range 0, iter_size.
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !pixel_to_vector_iteration(iter, nest, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    x_arr = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(x_arr);
    y_arr = (PyObject *)NpyIter_GetOperandArray(iter)[3];
    Py_INCREF(y_arr);
    z_arr = (PyObject *)NpyIter_GetOperandArray(iter)[4];
    Py_INCREF(z_arr);

    Py_DECREF(nside_arr);
    Py_DECREF(pix_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    PyObject *retval = PyTuple_New(3);
    PyTuple_SET_ITEM(retval, 0, PyArray_Return((PyArrayObject *)x_arr));
    PyTuple_SET_ITEM(retval, 1, PyArray_Return((PyArrayObject *)y_arr));
    PyTuple_SET_ITEM(retval, 2, PyArray_Return((PyArrayObject *)z_arr));

    return retval;

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(x_arr);
    Py_XDECREF(y_arr);
    Py_XDECREF(z_arr);
    Py_XDECREF(pix_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(neighbors_doc,
             "neighbors(nside, pix, nest=True)\n"
             "--\n\n"
             "Return 8 nearest neighbors for given pixels.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR PIX_DOC_PAR NEST_DOC_PAR N_THREADS_PAR
             "\n"
             "Returns\n"
             "-------\n"
             "neighbor_pixels : `np.ndarray` (8,) or (N, 8)\n"
             "    Pixel numbers of the SW, W, NW, N, NE, E, SE, and S neighbors.\n"
             "    If a neighbor does not exist (as can be the case for W, N, E, and S)\n"
             "    the corresponding pixel number will be -1.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If pixel is out of range, or nside, pix arrays are not compatible.\n");

// Core processing logic for neighbors - used by both single and multi-threaded paths
static bool neighbors_iteration(NpyIter *iter, int nest, int64_t *neighbor_pixels,
                                npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;
    enum Scheme scheme = nest ? NEST : RING;
    i64stack *neigh = NULL;
    int status;

    // Allocate i64stack for this thread
    neigh = i64stack_new(8, &status, err);
    if (!status) {
        return false;
    }
    i64stack_resize(neigh, 8, &status, err);
    if (!status) {
        i64stack_delete(neigh);
        return false;
    }

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        i64stack_delete(neigh);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        i64stack_delete(neigh);
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        int64_t *pix = (int64_t *)dataptrarray[1];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, scheme, err)) {
                i64stack_delete(neigh);
                return false;
            }
            hpx = healpix_info_from_nside(*nside, scheme);
            last_nside = *nside;
            started = true;
        }

        if (!hpgeom_check_pixel(&hpx, *pix, err)) {
            i64stack_delete(neigh);
            return false;
        }

        neighbors(&hpx, *pix, neigh, &status, err);
        if (!status) {
            i64stack_delete(neigh);
            return false;
        }

        for (size_t i = 0; i < neigh->size; i++) {
            size_t index = neigh->size * NpyIter_GetIterIndex(iter) + i;
            neighbor_pixels[index] = neigh->data[i];
        }

    } while (iternext(iter));

    i64stack_delete(neigh);
    return true;
}

// Worker function for each thread
static void *neighbors_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !neighbors_iteration(td->iter, td->nest, (int64_t *) td->data0,
                                      td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *neighbors_meth(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *pix_obj = NULL;
    PyObject *nside_arr = NULL, *pix_arr = NULL;
    PyObject *neighbor_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int nest = 1;
    int n_threads = 1;
    static char *kwlist[] = {"nside", "pix", "nest", "n_threads", NULL};

    int64_t *neighbor_pixels = NULL;
    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pi", kwlist, &nside_obj, &pix_obj,
                                     &nest, &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    pix_arr = PyArray_FROM_OTF(pix_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (pix_arr == NULL) goto fail;

    if (PyArray_NDIM((PyArrayObject *)pix_arr) > 1) {
        PyErr_SetString(PyExc_ValueError, "pix array must be at most 1D.");
        goto fail;
    }
    if (PyArray_NDIM((PyArrayObject *)nside_arr) > 1) {
        PyErr_SetString(PyExc_ValueError, "nside array must be at most 1D.");
        goto fail;
    }

    // The input arrays are nside_arr (int64_t) and pix_arr (int64_t).
    // We are allocating our own output arrays.
    PyArrayObject *op[2];
    npy_uint32 op_flags[2];
    PyArray_Descr *op_dtypes[2];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)pix_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(2, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, pix arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);
    int ndims = NpyIter_GetNDim(iter);

    // Allocate output array
    if (ndims == 0) {
        npy_intp dims[1];
        dims[0] = 8;
        neighbor_arr = PyArray_SimpleNew(1, dims, NPY_INT64);
    } else {
        npy_intp dims[2];
        dims[0] = iter_size;
        dims[1] = 8;
        neighbor_arr = PyArray_SimpleNew(2, dims, NPY_INT64);
    }
    if (neighbor_arr == NULL) goto fail;
    neighbor_pixels = (int64_t *)PyArray_DATA((PyArrayObject *)neighbor_arr);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].nest = nest;
            thread_data[t].data0 = (void *) neighbor_pixels;
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], neighbors_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !neighbors_iteration(iter, nest, neighbor_pixels, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    Py_DECREF(nside_arr);
    Py_DECREF(pix_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return PyArray_Return((PyArrayObject *)neighbor_arr);

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(pix_arr);
    Py_XDECREF(neighbor_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(max_pixel_radius_doc,
             "max_pixel_radius(nside, degrees=True)\n"
             "--\n\n"
             "Compute maximum angular distance between any pixel center and its corners.\n"
             "\n"
             "Parameters\n"
             "----------\n" NSIDE_DOC_PAR N_THREADS_PAR
             "degrees : `bool`, optional\n"
             "    If True, returns pixel radius in degrees, otherwise radians.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "radii : `np.ndarray` (N, ) or `float`\n"
             "    Angular distance(s) (in degrees or radians).\n");

// Core processing logic for max_pixel_radius - used by both single and multi-threaded paths
static bool max_pixel_radius_iteration(NpyIter *iter, int degrees,
                                       npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        double *pixrad = (double *)dataptrarray[1];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, RING, err)) {
                return false;
            }
            hpx = healpix_info_from_nside(*nside, RING);
            last_nside = *nside;
            started = true;
        }

        *pixrad = max_pixrad(&hpx);
        if (degrees) *pixrad *= HPG_R2D;

    } while (iternext(iter));

    return true;
}

// Worker function for each thread (reuses ThreadData)
static void *max_pixel_radius_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !max_pixel_radius_iteration(td->iter, td->degrees,
                                             td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *max_pixel_radius(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL;
    PyObject *nside_arr = NULL;
    PyObject *pixrad_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int degrees = 1;
    int n_threads = 1;
    static char *kwlist[] = {"nside", "degrees", "n_threads", NULL};

    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|pi", kwlist, &nside_obj, &degrees,
                                     &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;

    // The input array is nside_arr (int64_t).
    // The output array is pixrad_arr (double).
    PyArrayObject *op[2];
    npy_uint32 op_flags[2];
    PyArray_Descr *op_dtypes[2];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = NULL;
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[1] = PyArray_DescrFromType(NPY_DOUBLE);

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(2, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside array could not be processed.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].lonlat = 0;   // Unused
            thread_data[t].nest = 0;     // Unused
            thread_data[t].degrees = degrees;
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], max_pixel_radius_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path: set to the full range 0, iter_size.
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !max_pixel_radius_iteration(iter, degrees, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    pixrad_arr = (PyObject *)NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(pixrad_arr);

    Py_DECREF(nside_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return PyArray_Return((PyArrayObject *)pixrad_arr);

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(pixrad_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(
    get_interpolation_weights_doc,
    "get_interpolation_weights(nside, a, b, nest=True, lonlat=True, degrees=True)\n"
    "--\n\n"
    "Return the 4 closest pixels and weights to perform bilinear interpolation along\n"
    "latitude and longitude.\n"
    "\n"
    "Parameters\n"
    "----------\n" NSIDE_DOC_PAR AB_DOC_PAR NEST_DOC_PAR LONLAT_DOC_PAR DEGREES_DOC_PAR N_THREADS_PAR
    "\n"
    "Returns\n"
    "-------\n"
    "pixels : `np.ndarray` (N, 4)\n"
    "    Array of pixels (`np.int64`), each set of 4 can be used to do bilinear\n"
    "    interpolation of a map.\n"
    "weights : `np.ndarray` (N, 4)\n"
    "    Array of weiaghts (`np.float64`), each set of 4 corresponds with the pixels.\n");

// Core processing logic for get_interpolation_weights - used by both single and multi-threaded paths
static bool get_interpolation_weights_iteration(NpyIter *iter, int lonlat, int nest, int degrees,
                                                int64_t *pixels, double *weights,
                                                npy_intp start_idx, npy_intp end_idx, char *err) {
    NpyIter_IterNextFunc *iternext;
    char **dataptrarray;
    char *errmsg;
    double theta, phi;
    healpix_info hpx;
    int64_t last_nside = -1;
    bool started = false;
    enum Scheme scheme = nest ? NEST : RING;

    // For ranged iteration, reset to the specified range
    if (NpyIter_ResetToIterIndexRange(iter, start_idx, end_idx, &errmsg) != NPY_SUCCEED) {
        snprintf(err, ERR_SIZE, "%s", errmsg);
        return false;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        snprintf(err, ERR_SIZE, "Failed to get iterator next function");
        return false;
    }

    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        int64_t *nside = (int64_t *)dataptrarray[0];
        double *a = (double *)dataptrarray[1];
        double *b = (double *)dataptrarray[2];

        // Re-initialize healpix_info when nside changes
        if ((!started) || (*nside != last_nside)) {
            if (!hpgeom_check_nside(*nside, scheme, err)) {
                return false;
            }
            hpx = healpix_info_from_nside(*nside, scheme);
            last_nside = *nside;
            started = true;
        }

        // Convert angles
        if (lonlat) {
            if (!hpgeom_lonlat_to_thetaphi(*a, *b, &theta, &phi, (bool)degrees, err)) {
                return false;
            }
        } else {
            if (!hpgeom_check_theta_phi(*a, *b, err)) {
                return false;
            }
            theta = *a;
            phi = *b;
        }

        // Get current index and compute interpolation weights
        size_t index = 4 * NpyIter_GetIterIndex(iter);
        get_interpol(&hpx, theta, phi, &pixels[index], &weights[index]);

    } while (iternext(iter));

    return true;
}

// Worker function for each thread
static void *get_interpolation_weights_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    td->failed = !get_interpolation_weights_iteration(td->iter, td->lonlat, td->nest, td->degrees,
                                                      (int64_t *) td->data0, (double *) td->data1,
                                                      td->start_idx, td->end_idx, td->err);
    return NULL;
}

static PyObject *get_interpolation_weights(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *nside_obj = NULL, *a_obj = NULL, *b_obj = NULL;
    PyObject *nside_arr = NULL, *a_arr = NULL, *b_arr = NULL;
    PyObject *pix_arr = NULL;
    PyObject *wgt_arr = NULL;

    NpyIter *iter = NULL;
    thread_handle_t *threads = NULL;
    ThreadData *thread_data = NULL;

    int lonlat = 1;
    int nest = 1;
    int degrees = 1;
    int n_threads = 1;
    static char *kwlist[] = {"nside", "a", "b", "lonlat", "nest", "degrees", "n_threads", NULL};

    int64_t *pixels = NULL;
    double *weights = NULL;
    char err[ERR_SIZE];
    bool loop_failed = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|pppi", kwlist, &nside_obj, &a_obj,
                                     &b_obj, &lonlat, &nest, &degrees, &n_threads))
        goto fail;

    nside_arr =
        PyArray_FROM_OTF(nside_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (nside_arr == NULL) goto fail;
    a_arr = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (a_arr == NULL) goto fail;
    b_arr = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (b_arr == NULL) goto fail;

    if (PyArray_NDIM((PyArrayObject *)a_arr) > 1) {
        PyErr_SetString(PyExc_ValueError, "a array must be at most 1D.");
        goto fail;
    }
    if (PyArray_NDIM((PyArrayObject *)a_arr) != PyArray_NDIM((PyArrayObject *)b_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "a and b arrays must have same number of dimensions.");
        goto fail;
    }

    // The input arrays are nside_arr (int64_t), a_arr (double), and b_arr (double).
    // We are allocating our own output arrays.
    PyArrayObject *op[3];
    npy_uint32 op_flags[3];
    PyArray_Descr *op_dtypes[3];

    op[0] = (PyArrayObject *)nside_arr;
    op_flags[0] = NPY_ITER_READONLY;
    op_dtypes[0] = NULL;
    op[1] = (PyArrayObject *)a_arr;
    op_flags[1] = NPY_ITER_READONLY;
    op_dtypes[1] = NULL;
    op[2] = (PyArrayObject *)b_arr;
    op_flags[2] = NPY_ITER_READONLY;
    op_dtypes[2] = NULL;

    npy_uint32 iter_flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_RANGED | NPY_ITER_BUFFERED;
    if (n_threads > 1) {
        iter_flags |= NPY_ITER_DELAY_BUFALLOC | NPY_ITER_GROWINNER;
    }

    iter = NpyIter_MultiNew(3, op, iter_flags, NPY_KEEPORDER, NPY_NO_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "nside, a, b arrays could not be broadcast together.");
        goto fail;
    }

    npy_intp iter_size = NpyIter_GetIterSize(iter);
    int ndims = NpyIter_GetNDim(iter);

    // Allocate output arrays
    if (ndims == 0) {
        npy_intp dims[1];
        dims[0] = 4;
        pix_arr = PyArray_SimpleNew(1, dims, NPY_INT64);
        if (pix_arr == NULL) goto fail;
        wgt_arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
        if (wgt_arr == NULL) goto fail;
    } else {
        npy_intp dims[2];
        dims[0] = iter_size;
        dims[1] = 4;
        pix_arr = PyArray_SimpleNew(2, dims, NPY_INT64);
        if (pix_arr == NULL) goto fail;
        wgt_arr = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
        if (wgt_arr == NULL) goto fail;
    }
    pixels = (int64_t *)PyArray_DATA((PyArrayObject *)pix_arr);
    weights = (double *)PyArray_DATA((PyArrayObject *)wgt_arr);

    if (iter_size == 0) {
        goto cleanup;
    }

    if (n_threads > 1) {
        // Don't use threading if chunks would be too small
        if (iter_size / n_threads < MIN_CHUNK_SIZE) {
            n_threads = iter_size / MIN_CHUNK_SIZE;
        }
    }

    if (n_threads > 1) {
        // Multi-threaded path
        if (n_threads > iter_size) {
            n_threads = iter_size;
        }

        threads = malloc(n_threads * sizeof(thread_handle_t));
        thread_data = malloc(n_threads * sizeof(ThreadData));

        if (threads == NULL || thread_data == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
            goto fail;
        }

        // Create iterator copies (with GIL held)
        thread_data[0].iter = iter;
        for (int t = 1; t < n_threads; t++) {
            thread_data[t].iter = NpyIter_Copy(iter);
            if (thread_data[t].iter == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to copy iterator for threading");
                goto fail;
            }
        }

        // Divide work among threads
        npy_intp chunk_size = iter_size / n_threads;
        npy_intp remainder = iter_size % n_threads;

        for (int t = 0; t < n_threads; t++) {
            thread_data[t].start_idx = t * chunk_size + (t < remainder ? t : remainder);
            thread_data[t].end_idx =
                thread_data[t].start_idx + chunk_size + (t < remainder ? 1 : 0);
            thread_data[t].lonlat = lonlat;
            thread_data[t].nest = nest;
            thread_data[t].degrees = degrees;
            thread_data[t].data0 = (void *) pixels;
            thread_data[t].data1 = (void *) weights;
            thread_data[t].failed = false;
            thread_data[t].err[0] = '\0';
        }

        NPY_BEGIN_ALLOW_THREADS

        for (int t = 0; t < n_threads; t++) {
            if (thread_create(&threads[t], get_interpolation_weights_worker, &thread_data[t]) != 0) {
                thread_data[t].failed = true;
                snprintf(thread_data[t].err, ERR_SIZE, "Failed to create thread %d", t);
                threads[t] = NULL;
            }
        }

        for (int t = 0; t < n_threads; t++) {
            if (threads[t] != NULL) thread_join(threads[t]);
        }

        NPY_END_ALLOW_THREADS

        // Check for errors
        for (int t = 0; t < n_threads; t++) {
            if (thread_data[t].failed) {
                PyErr_SetString(PyExc_ValueError, thread_data[t].err);
                goto fail;
            }
        }

    } else {
        // Single-threaded path
        NPY_BEGIN_ALLOW_THREADS

        loop_failed = !get_interpolation_weights_iteration(iter, lonlat, nest, degrees,
                                                           pixels, weights, 0, iter_size, err);

        NPY_END_ALLOW_THREADS

        if (loop_failed) {
            PyErr_SetString(PyExc_ValueError, err);
            goto fail;
        }
    }

cleanup:
    Py_DECREF(nside_arr);
    Py_DECREF(a_arr);
    Py_DECREF(b_arr);
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        iter = NULL;
        goto fail;
    }
    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    PyObject *retval = PyTuple_New(2);
    PyTuple_SET_ITEM(retval, 0, PyArray_Return((PyArrayObject *)pix_arr));
    PyTuple_SET_ITEM(retval, 1, PyArray_Return((PyArrayObject *)wgt_arr));

    return retval;

fail:
    Py_XDECREF(nside_arr);
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);
    Py_XDECREF(pix_arr);
    Py_XDECREF(wgt_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    if (thread_data != NULL) {
        for (int t = 1; t < n_threads; t++) {
            if (thread_data[t].iter != NULL) {
                NpyIter_Deallocate(thread_data[t].iter);
            }
        }
        free(thread_data);
    }
    if (threads != NULL) free(threads);

    return NULL;
}

PyDoc_STRVAR(pixel_ranges_to_pixels_doc,
             "pixel_ranges_to_pixels(pixel_ranges, inclusive=False)\n"
             "--\n\n"
             "Convert (M, 2) array of pixel ranges to an array of pixels.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "pixel_ranges : `np.ndarray` (M, 2)\n"
             "    Array of pixel ranges, [lo, high) (if inclusive=False) or\n"
             "    [lo, high] (if inclusive=True).\n"
             "\n"
             "Returns\n"
             "-------\n"
             "pixels : `np.ndarray` (N,)\n"
             "    Array of pixels.\n");

static PyObject *pixel_ranges_to_pixels(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *pixel_ranges_obj = NULL;
    PyObject *pixel_ranges_arr = NULL;
    PyObject *pix_arr = NULL;
    int inclusive = 0;
    static char *kwlist[] = {"pixel_ranges", "inclusive", NULL};
    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext;
    char **dataptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist, &pixel_ranges_obj,
                                     &inclusive))
        goto fail;

    pixel_ranges_arr = PyArray_FROM_OTF(pixel_ranges_obj, NPY_INT64,
                                        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (pixel_ranges_arr == NULL) goto fail;

    if ((PyArray_NDIM((PyArrayObject *)pixel_ranges_arr) != 2) ||
        (PyArray_DIM((PyArrayObject *)pixel_ranges_arr, 1) != 2)) {
        PyErr_SetString(PyExc_ValueError, "pixel_ranges must be 2D, with shape (M, 2).");
        goto fail;
    }

    if (PyArray_SIZE((PyArrayObject *)pixel_ranges_arr) == 0) {
        npy_intp dims[1];
        dims[0] = 0;

        pix_arr = PyArray_SimpleNew(1, dims, NPY_INT64);
        if (pix_arr == NULL) goto fail;

        goto succeed;
    }

    iter = NpyIter_New((PyArrayObject *)pixel_ranges_arr,
                       NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX, NPY_KEEPORDER, NPY_NO_CASTING,
                       NULL);
    if (iter == NULL) goto fail;
    // We don't want to iterate over the second axis.
    if (NpyIter_RemoveAxis(iter, 1) == NPY_FAIL) goto fail;

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) goto fail;

    dataptr = NpyIter_GetDataPtrArray(iter);

    // We first loop over to count the number of pixels in the output,
    // and check that the pixel_ranges are valid.
    npy_intp dims[1];
    dims[0] = 0;

    // Check for zero-size before entering loop.
    if (NpyIter_GetIterSize(iter) > 0) {
        do {
            int64_t *data = (int64_t *)*dataptr;

            if (*(data + 1) < *data) {
                PyErr_SetString(PyExc_ValueError,
                                "pixel_ranges[:, 0] must all be <= pixel_ranges[:, 1]");
                goto fail;
            }

            dims[0] += (*(data + 1) - *data) + inclusive;
        } while (iternext(iter));
    }

    // Create the output array
    pix_arr = PyArray_SimpleNew(1, dims, NPY_INT64);
    if (pix_arr == NULL) goto fail;

    int64_t *pix_data = (int64_t *)PyArray_DATA((PyArrayObject *)pix_arr);

    // Reset the iterator and loop to expand pixels.
    if (NpyIter_Reset(iter, NULL) == NPY_FAIL) goto fail;

    size_t counter = 0;

    // Check for zero-size before entering loop.
    if (NpyIter_GetIterSize(iter) > 0) {
        do {
            int64_t *data = (int64_t *)*dataptr;

            for (int64_t pix = *data; pix < (*(data + 1) + inclusive); pix++) {
                pix_data[counter++] = pix;
            }
        } while (iternext(iter));
    }

succeed:

    Py_DECREF(pixel_ranges_arr);
    if (iter != NULL) NpyIter_Deallocate(iter);

    return PyArray_Return((PyArrayObject *)pix_arr);

fail:
    Py_XDECREF(pixel_ranges_arr);
    if (iter != NULL) NpyIter_Deallocate(iter);
    Py_XDECREF(pix_arr);

    return NULL;
}

static PyMethodDef hpgeom_methods[] = {
    {"angle_to_pixel", (PyCFunction)(void (*)(void))angle_to_pixel,
     METH_VARARGS | METH_KEYWORDS, angle_to_pixel_doc},
    {"pixel_to_angle", (PyCFunction)(void (*)(void))pixel_to_angle,
     METH_VARARGS | METH_KEYWORDS, pixel_to_angle_doc},
    {"query_circle", (PyCFunction)(void (*)(void))query_circle, METH_VARARGS | METH_KEYWORDS,
     query_circle_doc},
    {"query_polygon", (PyCFunction)(void (*)(void))query_polygon_meth,
     METH_VARARGS | METH_KEYWORDS, query_polygon_doc},
    {"query_ellipse", (PyCFunction)(void (*)(void))query_ellipse_meth,
     METH_VARARGS | METH_KEYWORDS, query_ellipse_doc},
    {"query_box", (PyCFunction)(void (*)(void))query_box_meth, METH_VARARGS | METH_KEYWORDS,
     query_box_doc},
    {"nest_to_ring", (PyCFunction)(void (*)(void))nest_to_ring, METH_VARARGS | METH_KEYWORDS,
     nest_to_ring_doc},
    {"ring_to_nest", (PyCFunction)(void (*)(void))ring_to_nest, METH_VARARGS | METH_KEYWORDS,
     ring_to_nest_doc},
    {"boundaries", (PyCFunction)(void (*)(void))boundaries_meth, METH_VARARGS | METH_KEYWORDS,
     boundaries_doc},
    {"vector_to_pixel", (PyCFunction)(void (*)(void))vector_to_pixel,
     METH_VARARGS | METH_KEYWORDS, vector_to_pixel_doc},
    {"pixel_to_vector", (PyCFunction)(void (*)(void))pixel_to_vector,
     METH_VARARGS | METH_KEYWORDS, pixel_to_vector_doc},
    {"neighbors", (PyCFunction)(void (*)(void))neighbors_meth, METH_VARARGS | METH_KEYWORDS,
     neighbors_doc},
    {"max_pixel_radius", (PyCFunction)(void (*)(void))max_pixel_radius,
     METH_VARARGS | METH_KEYWORDS, max_pixel_radius_doc},
    {"get_interpolation_weights", (PyCFunction)(void (*)(void))get_interpolation_weights,
     METH_VARARGS | METH_KEYWORDS, get_interpolation_weights_doc},
    {"pixel_ranges_to_pixels", (PyCFunction)(void (*)(void))pixel_ranges_to_pixels,
     METH_VARARGS | METH_KEYWORDS, pixel_ranges_to_pixels_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef hpgeom_module = {PyModuleDef_HEAD_INIT, "_hpgeom", NULL, -1,
                                           hpgeom_methods};

PyMODINIT_FUNC PyInit__hpgeom(void) {
    import_array();
    return PyModule_Create(&hpgeom_module);
}
