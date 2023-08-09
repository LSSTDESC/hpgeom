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

#ifndef _HEALPIX_GEOM_H
#define _HEALPIX_GEOM_H

#include "hpgeom_stack.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#define HPG_PI 3.141592653589793238462643383279502884197           /* pi */
#define HPG_TWO_PI 6.283185307179586476925286766559005768394       /* 2*pi */
#define HPG_INV_TWOPI 1 / HPG_TWO_PI                               /* 1/(2*pi) */
#define HPG_FOURPI 12.5663706143591729538505735331180115367        /* 4*pi */
#define HPG_HALFPI 1.570796326794896619231321691639751442099       /* pi/2 */
#define HPG_INV_HALFPI 0.6366197723675813430755350534900574        /* 2/pi */
#define HPG_INV_SQRT4PI 0.2820947917738781434740397257803862929220 /* sqrt(4*pi) */

#define HPG_ONETHIRD 1.0 / 3.0
#define HPG_TWOTHIRD 2.0 / 3.0
#define HPG_FOURTHIRD 4.0 / 3.0

#define HPG_D2R HPG_PI / 180.0
#define HPG_R2D 180.0 / HPG_PI

#define HPG_EPSILON 2.220446049250313e-16

#define MAX_ORDER 29
#define MAX_NSIDE (int64_t)(1) << MAX_ORDER

typedef enum Scheme { RING, NEST } Scheme;

typedef struct healpix_info {
    int order;
    int64_t nside;
    int64_t npface;
    int64_t ncap;
    int64_t npix;
    double fact2;
    double fact1;
    Scheme scheme;
} healpix_info;

healpix_info healpix_info_from_order(int order, enum Scheme scheme);
healpix_info healpix_info_from_nside(int64_t nside, enum Scheme scheme);

int64_t isqrt(int64_t i);
int ilog2(int64_t arg);
int64_t imodulo(int64_t v1, int64_t v2);
double fmodulo(double v1, double v2);

int64_t ang2pix(healpix_info *hpx, double theta, double phi);
int64_t loc2pix(healpix_info *hpx, double z, double phi, double sth, bool hav_sth);
int64_t xyf2nest(healpix_info *hpx, int ix, int iy, int face_num);
int64_t xyf2ring(healpix_info *hpx, int ix, int iy, int face_num);

double ring2z(healpix_info *hpx, int64_t ring);
void pix2zphi(healpix_info *hpx, int64_t pix, double *z, double *phi);
void pix2xyf(healpix_info *hpx, int64_t pix, int *ix, int *iy, int *face_num);
int64_t xyf2pix(healpix_info *hpx, int ix, int iy, int face_num);

void pix2ang(healpix_info *hpx, int64_t pix, double *theta, double *phi);
void pix2loc(healpix_info *hpx, int64_t pix, double *z, double *phi, double *sth,
             bool *have_sth);
void nest2xyf(healpix_info *hpx, int64_t pix, int *ix, int *iy, int *face_num);
void ring2xyf(healpix_info *hpx, int64_t pix, int *ix, int *iy, int *face_num);
int64_t nest2ring(healpix_info *hpx, int64_t pix);
int64_t ring2nest(healpix_info *hpx, int64_t pix);
vec3 pix2vec(healpix_info *hpx, int64_t pix);
int64_t vec2pix(healpix_info *hpx, vec3 *vec);

int64_t spread_bits64(int v);
int compress_bits64(int64_t v);

int64_t ring_above(healpix_info *hpx, double z);
void get_ring_info_small(healpix_info *hpx, int64_t ring, int64_t *startpix, int64_t *ringpix,
                         bool *shifted);

double max_pixrad(healpix_info *hpx);
bool check_pixel_ring(healpix_info *hpx1, healpix_info *hpx2, int64_t pix, int64_t nr,
                      int64_t ipix1, int fct, double cz, double cphi, double cosrp2,
                      int64_t cpix);
void check_pixel_nest(int o, int order_, int omax, int zone, struct i64rangeset *pixset,
                      int64_t pix, struct i64stack *stk, bool inclusive, int *stacktop,
                      int *status, char *err);

void query_disc(healpix_info *hpx, double ptg_theta, double ptg_phi, double radius, int fact,
                struct i64rangeset *pixset, int *status, char *err);
void query_ellipse(healpix_info *hpx, double ptg_theta, double ptg_phi, double semi_major,
                   double semi_minor, double alpha, int fact, struct i64rangeset *pixset,
                   int *status, char *err);
void query_box(healpix_info *hpx, double ptg_theta0, double ptg_theta1, double ptg_phi0,
               double ptg_phi1, bool full_lon, int fact, struct i64rangeset *pixset,
               int *status, char *err);

void xyf2loc(double x, double y, int face, double *z, double *phi, double *sth,
             bool *have_sth);
void locToVec3(double z, double phi, double sth, bool have_sth, vec3 *vec);
void boundaries(healpix_info *hpx, int64_t pix, size_t step, pointingarr *out, int *status);
void neighbors(healpix_info *hpx, int64_t pix, i64stack *result, int *status, char *err);
void query_multidisc(healpix_info *hpx, vec3arr *norm, double *rad, int fact,
                     i64rangeset *pixset, int *status, char *err);
void query_polygon(healpix_info *hpx, pointingarr *vertex, int fact, i64rangeset *pixset,
                   int *status, char *err);

void get_ring_info2(healpix_info *hpx, int64_t ring, int64_t *startpix, int64_t *ringpix,
                    double *theta, bool *shifted);
void get_interpol(healpix_info *hpx, double ptg_theta, double ptg_phi, int64_t *pixels,
                  double *weights);

#endif
