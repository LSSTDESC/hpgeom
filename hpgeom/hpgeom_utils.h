/*
 * Copyright 2022 LSST DESC
 * Author: Eli Rykoff
 *
 * This product includes software developed by the
 * LSST DESC (https://www.lsstdesc.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef _HPGEOM_UTILS_H
#define _HPGEOM_UTILS_H

#include <numpy/arrayobject.h>

#include "healpix_geom.h"
#include <stdbool.h>
#include <stdint.h>

#define ERR_SIZE 256
#define MIN_CHUNK_SIZE 5000

typedef struct {
    NpyIter *iter;
    npy_intp start_idx;
    npy_intp end_idx;
    int lonlat;
    int nest;
    int degrees;
    char err[ERR_SIZE];
    bool failed;
} ThreadData;

int hpgeom_check_nside(int64_t nside, Scheme scheme, char *err);
int hpgeom_check_theta_phi(double theta, double phi, char *err);
int hpgeom_check_pixel(healpix_info *hpx, int64_t pix, char *err);
int hpgeom_lonlat_to_thetaphi(double lon, double lat, double *theta, double *phi, bool degrees,
                              char *err);
int hpgeom_thetaphi_to_lonlat(double theta, double phi, double *lon, double *lat, bool degrees,
                              bool check_status, char *err);
int hpgeom_check_fact(healpix_info *hpx, long fact, char *err);
int hpgeom_check_radius(double radius, char *err);
int hpgeom_check_semi(double semi_major, double semi_minor, char *err);

#endif
