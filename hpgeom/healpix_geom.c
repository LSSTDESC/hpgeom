/*
 * This file is modified from Healpix_cxx/healpix_base.cc by
 * Eli Rykoff, Matt Becker, Erin Sheldon.
 * Copyright (C) 2022 LSST DESC
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

/*
 *  Original healpix_cxx code:
 *  Copyright (C) 2003-2016 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <stdio.h>
#include <stdlib.h>

#include "healpix_geom.h"
#include "hpgeom_stack.h"
#include "hpgeom_utils.h"

static const uint16_t utab[] = {
#define Z(a) 0x##a##0, 0x##a##1, 0x##a##4, 0x##a##5
#define Y(a) Z(a##0), Z(a##1), Z(a##4), Z(a##5)
#define X(a) Y(a##0), Y(a##1), Y(a##4), Y(a##5)
    X(0), X(1), X(4), X(5)
#undef X
#undef Y
#undef Z
};

static const uint16_t ctab[] = {
#define Z(a) a, a + 1, a + 256, a + 257
#define Y(a) Z(a), Z(a + 2), Z(a + 512), Z(a + 514)
#define X(a) Y(a), Y(a + 4), Y(a + 1024), Y(a + 1028)
    X(0), X(8), X(2048), X(2056)
#undef X
#undef Y
#undef Z
};

static const int jrll[] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
static const int jpll[] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};

int64_t isqrt(int64_t i) { return sqrt(((double)(i)) + 0.5); }

int ilog2(int64_t arg) {
    int res = 0;
    while (arg > 0x0000FFFF) {
        res += 16;
        arg >>= 16;
    }

    if (arg > 0x000000FF) {
        res |= 8;
        arg >>= 8;
    }
    if (arg > 0x0000000F) {
        res |= 4;
        arg >>= 4;
    }
    if (arg > 0x00000003) {
        res |= 2;
        arg >>= 2;
    }
    if (arg > 0x00000001) {
        res |= 1;
    }
    return res;
}

int64_t imodulo(int64_t v1, int64_t v2) {
    return (v1 >= 0) ? ((v1 < v2) ? v1 : (v1 % v2)) : ((v1 % v2) + v2);
}

static inline int64_t i64max(int64_t v1, int64_t v2) { return v1 > v2 ? v1 : v2; }
static inline int64_t i64min(int64_t v1, int64_t v2) { return v1 < v2 ? v1 : v2; }
static inline int intmin(int v1, int v2) { return v1 < v2 ? v1 : v2; }

/*
static inline double dblmax(double v1, double v2) {
  return v1 > v2 ? v1 : v2;
}
*/
static inline double dblmin(double v1, double v2) { return v1 < v2 ? v1 : v2; }

static inline int64_t special_div(int64_t a, int64_t b) {
    int64_t t = (a >= (b << 1));
    a -= t * (b << 1);
    return (t << 1) + (a >= b);
}

static inline double safe_atan2(double y, double x) {
    return ((x == 0.) && (y == 0.)) ? 0.0 : atan2(y, x);
}

double fmodulo(double v1, double v2) {
    if (v1 >= 0) return (v1 < v2) ? v1 : fmod(v1, v2);
    double tmp = fmod(v1, v2) + v2;
    return (tmp == v2) ? 0. : tmp;
}

healpix_info healpix_info_from_order(int order, enum Scheme scheme) {
    healpix_info hpx;

    hpx.order = order;
    hpx.nside = (int64_t)(1) << order;
    hpx.npface = hpx.nside << order;
    hpx.ncap = (hpx.npface - hpx.nside) << 1;
    hpx.npix = 12 * hpx.npface;
    hpx.fact2 = 4. / hpx.npix;
    hpx.fact1 = (hpx.nside << 1) * hpx.fact2;
    hpx.scheme = scheme;

    return hpx;
}

healpix_info healpix_info_from_nside(int64_t nside, enum Scheme scheme) {
    healpix_info hpx;

    if ((nside) & (nside - 1)) {
        hpx.order = -1;
    } else {
        hpx.order = ilog2(nside);
    }
    hpx.nside = nside;
    hpx.npface = nside * nside;
    hpx.ncap = (hpx.npface - nside) << 1;
    hpx.npix = 12 * hpx.npface;
    hpx.fact2 = 4. / hpx.npix;
    hpx.fact1 = (nside << 1) * hpx.fact2;
    hpx.scheme = scheme;

    return hpx;
}

int64_t ang2pix(healpix_info *hpx, double theta, double phi) {
    if ((theta < 0.01) || (theta > 3.14159 - 0.01)) {
        return loc2pix(hpx, cos(theta), phi, 0.0, false);
    } else {
        return loc2pix(hpx, cos(theta), phi, sin(theta), true);
    }
}

int64_t vec2pix(healpix_info *hpx, vec3 *vec) {
    double xl = 1. / vec3_length(vec);
    double phi = safe_atan2(vec->y, vec->x);
    double nz = vec->z * xl;
    if (fabs(nz) > 0.99) {
        return loc2pix(hpx, nz, phi, sqrt(vec->x * vec->x + vec->y * vec->y) * xl, true);
    } else {
        return loc2pix(hpx, nz, phi, 0, false);
    }
}

void pix2ang(healpix_info *hpx, int64_t pix, double *theta, double *phi) {
    double z, sth;
    bool have_sth;
    pix2loc(hpx, pix, &z, phi, &sth, &have_sth);
    if (have_sth) {
        *theta = atan2(sth, z);
    } else {
        *theta = acos(z);
    }
}

vec3 pix2vec(healpix_info *hpx, int64_t pix) {
    double z, phi, sth;
    bool have_sth;
    vec3 res;
    pix2loc(hpx, pix, &z, &phi, &sth, &have_sth);
    if (have_sth) {
        res.x = sth * cos(phi);
        res.y = sth * sin(phi);
        res.z = z;
    } else {
        sth = sqrt((1 - z) * (1 + z));
        res.x = sth * cos(phi);
        res.y = sth * sin(phi);
        res.z = z;
    }
    return res;
}

void pix2zphi(healpix_info *hpx, int64_t pix, double *z, double *phi) {
    bool dum_b;
    double dum_d;
    pix2loc(hpx, pix, z, phi, &dum_d, &dum_b);
}

void pix2xyf(healpix_info *hpx, int64_t pix, int *ix, int *iy, int *face_num) {
    (hpx->scheme == RING) ? ring2xyf(hpx, pix, ix, iy, face_num)
                          : nest2xyf(hpx, pix, ix, iy, face_num);
}

int64_t xyf2pix(healpix_info *hpx, int ix, int iy, int face_num) {
    return (hpx->scheme == RING) ? xyf2ring(hpx, ix, iy, face_num)
                                 : xyf2nest(hpx, ix, iy, face_num);
}

int64_t nest2ring(healpix_info *hpx, int64_t pix) {
    int ix, iy, face_num;
    nest2xyf(hpx, pix, &ix, &iy, &face_num);
    return xyf2ring(hpx, ix, iy, face_num);
}

int64_t ring2nest(healpix_info *hpx, int64_t pix) {
    int ix, iy, face_num;
    ring2xyf(hpx, pix, &ix, &iy, &face_num);
    return xyf2nest(hpx, ix, iy, face_num);
}

int64_t loc2pix(healpix_info *hpx, double z, double phi, double sth, bool have_sth) {
    double za = fabs(z);
    double tt = fmodulo(phi * HPG_INV_HALFPI, 4.0);  // in [0,4)

    if (hpx->scheme == RING) {
        if (za <= HPG_TWOTHIRD)  // Equatorial region
        {
            int64_t nl4 = 4 * hpx->nside;
            double temp1 = hpx->nside * (0.5 + tt);
            double temp2 = hpx->nside * z * 0.75;
            int64_t jp = (int64_t)(temp1 - temp2);  // index of  ascending edge line
            int64_t jm = (int64_t)(temp1 + temp2);  // index of descending edge line

            // ring number counted from z=2/3
            int64_t ir = hpx->nside + 1 + jp - jm;  // in {1,2n+1}
            int64_t kshift = 1 - (ir & 1);          // kshift=1 if ir even, 0 otherwise

            int64_t t1 = jp + jm - hpx->nside + kshift + 1 + nl4 + nl4;
            int64_t ip =
                (hpx->order > 0) ? (t1 >> 1) & (nl4 - 1) : ((t1 >> 1) % nl4);  // in {0,4n-1}

            return hpx->ncap + (ir - 1) * nl4 + ip;
        } else  // North & South polar caps
        {
            double tp = tt - (int64_t)(tt);
            double tmp = ((za < 0.99) || (!have_sth))
                             ? hpx->nside * sqrt(3 * (1 - za))
                             : hpx->nside * sth / sqrt((1. + za) / 3.);

            int64_t jp = (int64_t)(tp * tmp);          // increasing edge line index
            int64_t jm = (int64_t)((1.0 - tp) * tmp);  // decreasing edge line index

            int64_t ir = jp + jm + 1;         // ring number counted from the closest pole
            int64_t ip = (int64_t)(tt * ir);  // in {0,4*ir-1}

            if (z > 0.) {
                return 2 * ir * (ir - 1) + ip;
            } else {
                return hpx->npix - 2 * ir * (ir + 1) + ip;
            }
        }
    } else  // is_nest
    {
        if (za <= HPG_TWOTHIRD)  // Equatorial region
        {
            double temp1 = hpx->nside * (0.5 + tt);
            double temp2 = hpx->nside * (z * 0.75);
            int64_t jp = (int64_t)(temp1 - temp2);  // index of  ascending edge line
            int64_t jm = (int64_t)(temp1 + temp2);  // index of descending edge line
            int64_t ifp = jp >> hpx->order;         // in {0,4}
            int64_t ifm = jm >> hpx->order;

            int face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));

            int ix = jm & (hpx->nside - 1), iy = hpx->nside - (jp & (hpx->nside - 1)) - 1;
            return xyf2nest(hpx, ix, iy, face_num);
        } else  // polar region, za > 2/3
        {
            int ntt = (int)tt;
            if (ntt >= 4) ntt = 3;
            double tp = tt - ntt;
            double tmp = ((za < 0.99) || (!have_sth))
                             ? hpx->nside * sqrt(3 * (1 - za))
                             : hpx->nside * sth / sqrt((1. + za) / 3.);

            int64_t jp = (int64_t)(tp * tmp);           // increasing edge line index
            int64_t jm = (int64_t)((1.0 - tp) * tmp);   // decreasing edge line index
            if (jp >= hpx->nside) jp = hpx->nside - 1;  // for points too close to the boundary
            if (jm >= hpx->nside) jm = hpx->nside - 1;
            return (z >= 0) ? xyf2nest(hpx, hpx->nside - jm - 1, hpx->nside - jp - 1, ntt)
                            : xyf2nest(hpx, jp, jm, ntt + 8);
        }
    }
}

void pix2loc(healpix_info *hpx, int64_t pix, double *z, double *phi, double *sth,
             bool *have_sth) {
    *have_sth = false;
    if (hpx->scheme == RING) {
        if (pix < hpx->ncap)  // North Polar cap
        {
            int64_t iring =
                (1 + (int64_t)(isqrt(1 + 2 * pix))) >> 1;  // counted from North pole
            int64_t iphi = (pix + 1) - 2 * iring * (iring - 1);

            double tmp = (iring * iring) * hpx->fact2;
            *z = 1.0 - tmp;
            if (*z > 0.99) {
                *sth = sqrt(tmp * (2. - tmp));
                *have_sth = true;
            }
            *phi = (iphi - 0.5) * HPG_HALFPI / iring;
        } else if (pix < (hpx->npix - hpx->ncap))  // Equatorial region
        {
            int64_t nl4 = 4 * hpx->nside;
            int64_t ip = pix - hpx->ncap;
            int64_t tmp = (hpx->order >= 0) ? ip >> (hpx->order + 2) : ip / nl4;
            int64_t iring = tmp + hpx->nside, iphi = ip - nl4 * tmp + 1;
            // 1 if iring+nside is odd, 1/2 otherwise
            double fodd = ((iring + hpx->nside) & 1) ? 1 : 0.5;

            *z = (2 * hpx->nside - iring) * hpx->fact1;
            *phi = (iphi - fodd) * HPG_PI * 0.75 * hpx->fact1;
        } else  // South Polar cap
        {
            int64_t ip = hpx->npix - pix;
            int64_t iring =
                (1 + (int64_t)(isqrt(2 * ip - 1))) >> 1;  // counted from South pole
            int64_t iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));

            double tmp = (iring * iring) * hpx->fact2;
            *z = tmp - 1.0;
            if (*z < -0.99) {
                *sth = sqrt(tmp * (2. - tmp));
                *have_sth = true;
            }
            *phi = (iphi - 0.5) * HPG_HALFPI / iring;
        }
    } else {
        int face_num, ix, iy;
        nest2xyf(hpx, pix, &ix, &iy, &face_num);

        int64_t jr = ((int64_t)(jrll[face_num]) << hpx->order) - ix - iy - 1;

        int64_t nr;
        if (jr < hpx->nside) {
            nr = jr;
            double tmp = (nr * nr) * hpx->fact2;
            *z = 1 - tmp;
            if (*z > 0.99) {
                *sth = sqrt(tmp * (2. - tmp));
                *have_sth = true;
            }
        } else if (jr > 3 * hpx->nside) {
            nr = hpx->nside * 4 - jr;
            double tmp = (nr * nr) * hpx->fact2;
            *z = tmp - 1;
            if (*z < -0.99) {
                *sth = sqrt(tmp * (2. - tmp));
                *have_sth = true;
            }
        } else {
            nr = hpx->nside;
            *z = (2 * hpx->nside - jr) * hpx->fact1;
        }

        int64_t tmp = (int64_t)(jpll[face_num]) * nr + ix - iy;
        if (tmp < 0) tmp += 8 * nr;
        *phi = (nr == hpx->nside) ? 0.75 * HPG_HALFPI * tmp * hpx->fact1
                                  : (0.5 * HPG_HALFPI * tmp) / nr;
    }
}

int64_t xyf2nest(healpix_info *hpx, int ix, int iy, int face_num) {
    return ((int64_t)face_num << (2 * hpx->order)) + spread_bits64(ix) +
           (spread_bits64(iy) << 1);
}

void nest2xyf(healpix_info *hpx, int64_t pix, int *ix, int *iy, int *face_num) {
    *face_num = pix >> (2 * hpx->order);
    pix &= (hpx->npface - 1);
    *ix = compress_bits64(pix);
    *iy = compress_bits64(pix >> 1);
}

int64_t xyf2ring(healpix_info *hpx, int ix, int iy, int face_num) {
    int64_t nl4 = 4 * hpx->nside;
    int64_t jr = (jrll[face_num] * hpx->nside) - ix - iy - 1;

    int64_t nr, kshift, n_before;

    bool shifted;
    get_ring_info_small(hpx, jr, &n_before, &nr, &shifted);
    nr >>= 2;
    kshift = 1 - shifted;
    int64_t jp = (jpll[face_num] * nr + ix - iy + 1 + kshift) / 2;
    if (jp < 1) jp += nl4;

    return n_before + jp - 1;
}

void ring2xyf(healpix_info *hpx, int64_t pix, int *ix, int *iy, int *face_num) {
    int64_t iring, iphi, kshift, nr;
    int64_t nl2 = 2 * hpx->nside;

    if (pix < hpx->ncap) {                      // North Polar cap
        iring = (1 + isqrt(1 + 2 * pix)) >> 1;  // counted from North pole
        iphi = (pix + 1) - 2 * iring * (iring - 1);
        kshift = 0;
        nr = iring;
        *face_num = special_div(iphi - 1, nr);
    } else if (pix < (hpx->npix - hpx->ncap)) {  // Equatorial region
        int64_t ip = pix - hpx->ncap;
        int64_t tmp = (hpx->order >= 0) ? ip >> (hpx->order + 2) : ip / (4 * hpx->nside);
        iring = tmp + hpx->nside;
        iphi = ip - tmp * 4 * hpx->nside + 1;
        kshift = (iring + hpx->nside) & 1;
        nr = hpx->nside;
        int64_t ire = tmp + 1, irm = nl2 + 1 - tmp;
        int64_t ifm = iphi - (ire >> 1) + hpx->nside - 1,
                ifp = iphi - (irm >> 1) + hpx->nside - 1;
        if (hpx->order >= 0) {
            ifm >>= hpx->order;
            ifp >>= hpx->order;
        } else {
            ifm /= hpx->nside;
            ifp /= hpx->nside;
        }
        *face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));
    } else {  // South Polar cap
        int64_t ip = hpx->npix - pix;
        iring = (1 + isqrt(2 * ip - 1)) >> 1;  // counted from South pole
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        kshift = 0;
        nr = iring;
        iring = 2 * nl2 - iring;
        *face_num = special_div(iphi - 1, nr) + 8;
    }

    int64_t irt = iring - ((2 + (*face_num >> 2)) * hpx->nside) + 1;
    int64_t ipt = 2 * iphi - jpll[*face_num] * nr - kshift - 1;
    if (ipt >= nl2) ipt -= 8 * hpx->nside;

    *ix = (ipt - irt) >> 1;
    *iy = (-ipt - irt) >> 1;
}

double ring2z(healpix_info *hpx, int64_t ring) {
    if (ring < hpx->nside) {
        return 1 - ring * ring * hpx->fact2;
    }
    if (ring <= 3 * hpx->nside) {
        return (2 * hpx->nside - ring) * hpx->fact1;
    }
    ring = 4 * hpx->nside - ring;
    return ring * ring * hpx->fact2 - 1;
}

int64_t spread_bits64(int v) {
    return (int64_t)(utab[v & 0xff]) | ((int64_t)(utab[(v >> 8) & 0xff]) << 16) |
           ((int64_t)(utab[(v >> 16) & 0xff]) << 32) |
           ((int64_t)(utab[(v >> 24) & 0xff]) << 48);
}

int compress_bits64(int64_t v) {
    int64_t raw = v & 0x5555555555555555ull;
    raw |= raw >> 15;
    return ctab[raw & 0xff] | (ctab[(raw >> 8) & 0xff] << 4) |
           (ctab[(raw >> 32) & 0xff] << 16) | (ctab[(raw >> 40) & 0xff] << 20);
}

double max_pixrad(healpix_info *hpx) {
    vec3 va, vb;

    va.z = 2. / 3.;
    double phi_a = HPG_PI / (4. * hpx->nside);
    double sintheta = sqrt((1.0 - va.z) * (1.0 + va.z));
    va.x = sintheta * cos(phi_a);
    va.y = sintheta * sin(phi_a);

    double t1 = 1. - 1. / hpx->nside;
    t1 *= t1;

    vb.z = 1. - t1 / 3.;
    sintheta = sqrt((1.0 - vb.z) * (1.0 + vb.z));
    // we set phi_b = 0.0, so cos(phi_b) = 1.0 and sin(phi_b) = 0.0
    vb.x = sintheta;
    vb.y = 0.0;

    vec3 vcross;
    vec3_crossprod(&va, &vb, &vcross);
    double vdot = vec3_dotprod(&va, &vb);
    return atan2(vec3_length(&vcross), vdot);
}

int64_t ring_above(healpix_info *hpx, double z) {
    double az = fabs(z);
    if (az <= HPG_TWOTHIRD)  // equatorial region
        return (int64_t)(hpx->nside * (2 - 1.5 * z));
    int64_t iring = (int64_t)(hpx->nside * sqrt(3 * (1 - az)));
    return (z > 0) ? iring : 4 * hpx->nside - iring - 1;
}

void get_ring_info_small(healpix_info *hpx, int64_t ring, int64_t *startpix, int64_t *ringpix,
                         bool *shifted) {
    if (ring < hpx->nside) {
        *shifted = true;
        *ringpix = 4 * ring;
        *startpix = 2 * ring * (ring - 1);
    } else if (ring < 3 * hpx->nside) {
        *shifted = ((ring - hpx->nside) & 1) == 0;
        *ringpix = 4 * hpx->nside;
        *startpix = hpx->ncap + (ring - hpx->nside) * (*ringpix);
    } else {
        *shifted = true;
        int64_t nr = 4 * hpx->nside - ring;
        *ringpix = 4 * nr;
        *startpix = hpx->npix - 2 * nr * (nr + 1);
    }
}

inline double cosdist_zphi(double z1, double phi1, double z2, double phi2) {
    return z1 * z2 + cos(phi1 - phi2) * sqrt((1. - z1 * z1) * (1. - z2 * z2));
}

/* Short note on the "zone":
   zone = 0: pixel lies completely outside the queried shape
          1: pixel may overlap with the shape, pixel center is outside
          2: pixel center is inside the shape, but maybe not the complete pixel
          3: pixel lies completely inside the shape */

void check_pixel_nest(int o, int order_, int omax, int zone, i64rangeset *pixset, int64_t pix,
                      i64stack *stk, bool inclusive, int *stacktop, int *status, char *err) {
    *status = 1;
    if (zone == 0) return;

    if (o < order_) {
        if (zone >= 3) {
            int sdist = 2 * (order_ - o);  // the "bit-shift distance" between map orders
            i64rangeset_append(pixset, pix << sdist, (pix + 1) << sdist, status,
                               err);  // output all subpixels
            if (!*status) return;
        } else {  // (1<=zone<=2)
            for (int i = 0; i < 4; i++) {
                // output all subpixels, a pair of pixel and order
                i64stack_push(stk, 4 * pix + 3 - i, status, err);
                if (!*status) return;
                i64stack_push(stk, o + 1, status, err);
                if (!*status) return;
            }
        }
    } else if (o > order_) {  // this implies that inclusive=true
        if (zone >= 2) {      // pixel center in shape
            i64rangeset_append_single(pixset, pix >> (2 * (o - order_)), status,
                                      err);  // output the parent pixel at order_
            if (!*status) return;
            i64stack_resize(stk, *stacktop, status, err);  // unwind the stack
            if (!*status) return;
        } else {                               // (zone==1): pixel center in safety range
            if (o < omax) {                    // check sublevels
                for (int i = 0; i < 4; i++) {  // add children in reverse order
                    i64stack_push(stk, 4 * pix + 3 - i, status, err);
                    if (!*status) return;
                    i64stack_push(stk, o + 1, status, err);
                    if (!*status) return;
                }
            } else {  // at resolution limit
                i64rangeset_append_single(pixset, pix >> (2 * (o - order_)), status,
                                          err);  // output the parent pixel at order_
                if (!*status) return;
                i64stack_resize(stk, *stacktop, status, err);  // unwind the stack
                if (!*status) return;
            }
        }
    } else {  // o==order_
        if (zone >= 2) {
            i64rangeset_append_single(pixset, pix, status, err);
            if (!*status) return;
        } else if (inclusive) {                // and (zone>=1)
            if (order_ < omax) {               // check sublevels
                *stacktop = stk->size;         // remember current stack position
                for (int i = 0; i < 4; i++) {  // add children in reverse order
                    i64stack_push(stk, 4 * pix + 3 - i, status, err);
                    if (!*status) return;
                    i64stack_push(stk, o + 1, status, err);
                    if (!*status) return;
                }
            } else {                                                  // at resolution limit
                i64rangeset_append_single(pixset, pix, status, err);  // output the pixel
                if (!*status) return;
            }
        }
    }
}

bool check_pixel_ring(healpix_info *hpx1, healpix_info *hpx2, int64_t pix, int64_t nr,
                      int64_t ipix1, int fct, double cz, double cphi, double cosrp2,
                      int64_t cpix) {
    if (pix >= nr) pix -= nr;
    if (pix < 0) pix += nr;
    pix += ipix1;
    if (pix == cpix) return false;  // disk center in pixel => overlap
    int px, py, pf;
    pix2xyf(hpx1, pix, &px, &py, &pf);
    for (int i = 0; i < fct - 1; i++) {  // go along the 4 edges
        int64_t ox = fct * px, oy = fct * py;
        double pz, pphi;
        pix2zphi(hpx2, xyf2pix(hpx2, ox + i, oy, pf), &pz, &pphi);
        if (cosdist_zphi(pz, pphi, cz, cphi) > cosrp2)  // overlap
            return false;
        pix2zphi(hpx2, xyf2pix(hpx2, ox + fct - 1, oy + i, pf), &pz, &pphi);
        if (cosdist_zphi(pz, pphi, cz, cphi) > cosrp2)  // overlap
            return false;
        pix2zphi(hpx2, xyf2pix(hpx2, ox + fct - 1 - i, oy + fct - 1, pf), &pz, &pphi);
        if (cosdist_zphi(pz, pphi, cz, cphi) > cosrp2)  // overlap
            return false;
        pix2zphi(hpx2, xyf2pix(hpx2, ox, oy + fct - 1 - i, pf), &pz, &pphi);
        if (cosdist_zphi(pz, pphi, cz, cphi) > cosrp2)  // overlap
            return false;
    }
    return true;
}

void query_disc(healpix_info *hpx, double ptg_theta, double ptg_phi, double radius, int fact,
                i64rangeset *pixset, int *status, char *err) {
    bool inclusive = (fact != 0);
    // this does not alter the storage
    pixset->stack->size = 0;

    if (hpx->scheme == RING) {
        int64_t fct = 1;
        if (inclusive) {
            fct = fact;
        }
        healpix_info hpx2;
        double rsmall, rbig;
        if (fct > 1) {
            hpx2 = healpix_info_from_nside(fct * hpx->nside, RING);
            rsmall = radius + max_pixrad(&hpx2);
            rbig = radius + max_pixrad(hpx);
        } else {
            rsmall = rbig = inclusive ? radius + max_pixrad(hpx) : radius;
        }

        if (rsmall >= HPG_PI) {
            i64rangeset_append(pixset, 0, hpx->npix, status, err);
            goto cleanup_ring;
        }

        if (rbig > HPG_PI) {
            rbig = HPG_PI;
        }

        double cosrsmall = cos(rsmall);
        double cosrbig = cos(rbig);

        double z0 = cos(ptg_theta);
        double xa = 1. / sqrt((1 - z0) * (1 + z0));

        int64_t cpix = loc2pix(hpx, z0, ptg_phi, 0., false);

        double rlat1 = ptg_theta - rsmall;
        double zmax = cos(rlat1);
        int64_t irmin = ring_above(hpx, zmax) + 1;

        if ((rlat1 <= 0) && (irmin > 1)) {  // north pole in the disk
            int64_t sp, rp;
            bool dummy;
            get_ring_info_small(hpx, irmin - 1, &sp, &rp, &dummy);
            i64rangeset_append(pixset, 0, sp + rp, status, err);
            if (!*status) goto cleanup_ring;
        }
        if ((fct > 1) && (rlat1 > 0)) irmin = i64max((int64_t)1, irmin - 1);

        double rlat2 = ptg_theta + rsmall;
        double zmin = cos(rlat2);
        int64_t irmax = ring_above(hpx, zmin);

        if ((fct > 1) && (rlat2 < HPG_PI)) irmax = i64min(4 * hpx->nside - 1, irmax + 1);

        for (int64_t iz = irmin; iz <= irmax; ++iz) {
            double z = ring2z(hpx, iz);
            double x = (cosrbig - z * z0) * xa;
            double ysq = 1 - z * z - x * x;
            double dphi = -1;
            if (ysq <= 0) {  // no intersection, ring completely inside or outside
                dphi = (fct == 1) ? 0 : HPG_PI - 1e-15;
            } else {
                dphi = atan2(sqrt(ysq), x);
            }
            if (dphi > 0) {
                int64_t nr, ipix1;
                bool shifted;
                get_ring_info_small(hpx, iz, &ipix1, &nr, &shifted);
                double shift = shifted ? 0.5 : 0.;

                int64_t ipix2 = ipix1 + nr - 1;  // highest pixel number in the ring

                int64_t ip_lo =
                    (int64_t)floor((nr / HPG_TWO_PI) * (ptg_phi - dphi) - shift) + 1;
                int64_t ip_hi = (int64_t)floor((nr / HPG_TWO_PI) * (ptg_phi + dphi) - shift);

                if (fct > 1) {
                    while ((ip_lo <= ip_hi) &&
                           check_pixel_ring(hpx, &hpx2, ip_lo, nr, ipix1, fct, z0, ptg_phi,
                                            cosrsmall, cpix))
                        ++ip_lo;
                    while ((ip_hi > ip_lo) &&
                           check_pixel_ring(hpx, &hpx2, ip_hi, nr, ipix1, fct, z0, ptg_phi,
                                            cosrsmall, cpix))
                        --ip_hi;
                }

                if (ip_lo <= ip_hi) {
                    if (ip_hi >= nr) {
                        ip_lo -= nr;
                        ip_hi -= nr;
                    }
                    if (ip_lo < 0) {
                        i64rangeset_append(pixset, ipix1, ipix1 + ip_hi + 1, status, err);
                        if (!*status) goto cleanup_ring;
                        i64rangeset_append(pixset, ipix1 + ip_lo + nr, ipix2 + 1, status, err);
                        if (!*status) goto cleanup_ring;
                    } else {
                        i64rangeset_append(pixset, ipix1 + ip_lo, ipix1 + ip_hi + 1, status,
                                           err);
                        if (!*status) goto cleanup_ring;
                    }
                }
            }
        }
        if ((rlat2 >= HPG_PI) && (irmax + 1 < 4 * hpx->nside)) {  // south pole in the disk
            int64_t sp, rp;
            bool dummy;
            get_ring_info_small(hpx, irmax + 1, &sp, &rp, &dummy);
            i64rangeset_append(pixset, sp, hpx->npix, status, err);
            if (!*status) goto cleanup_ring;
        }
    cleanup_ring:
        ;
    } else {                     // schema == NEST
        i64stack *stk = NULL;
        if (radius >= HPG_PI) {  // disk covers the whole sphere
            i64rangeset_append(pixset, 0, hpx->npix, status, err);
            return;
        }

        int oplus = 0;
        if (inclusive) {
            oplus = ilog2(fact);
        }
        int omax = hpx->order + oplus;  // the order up to which we test

        // Statically define the array of bases because it's not large.
        double ptg_z = cos(ptg_theta);
        struct healpix_info base[MAX_ORDER + 1];
        double crpdr[MAX_ORDER + 1], crmdr[MAX_ORDER + 1];
        double cosrad = cos(radius);
        for (int o = 0; o <= omax; o++) {
            base[o] = healpix_info_from_order(o, NEST);
            double dr = max_pixrad(&base[o]);  // safety distance
            crpdr[o] = ((radius + dr) > HPG_PI) ? -1. : cos(radius + dr);
            crmdr[o] = ((radius - dr) < 0.) ? 1. : cos(radius - dr);
        }

        stk = i64stack_new(2 * (12 + 3 * omax), status, err);
        if (!*status) goto cleanup_nest;
        for (int i = 0; i < 12; i++) {
            i64stack_push(stk, (int64_t)(11 - i), status, err);
            if (!*status) goto cleanup_nest;
            i64stack_push(stk, 0, status, err);
            if (!*status) goto cleanup_nest;
        }

        int stacktop = 0;  // a place to save a stack position
        while (stk->size > 0) {
            // pop current pixel number and order from the stack
            int64_t pix, temp;
            i64stack_pop_pair(stk, &pix, &temp, status, err);
            if (!*status) goto cleanup_nest;
            int o = (int)temp;

            double pix_z, pix_phi;
            pix2zphi(&base[o], pix, &pix_z, &pix_phi);
            // cosine of angular distance between pixel center and disk center
            double cangdist = cosdist_zphi(ptg_z, ptg_phi, pix_z, pix_phi);

            if (cangdist > crpdr[o]) {
                int zone = (cangdist < cosrad) ? 1 : ((cangdist <= crmdr[o]) ? 2 : 3);
                check_pixel_nest(o, hpx->order, omax, zone, pixset, pix, stk, inclusive,
                                 &stacktop, status, err);
                if (!*status) goto cleanup_nest;
            }
        }
    cleanup_nest:
        if (stk != NULL) i64stack_delete(stk);
    }
}

void xyf2loc(double x, double y, int face, double *z, double *phi, double *sth,
             bool *have_sth) {
    *have_sth = false;
    double jr = jrll[face] - x - y;
    double nr;
    if (jr < 1) {
        nr = jr;
        double tmp = nr * nr / 3.;
        *z = 1 - tmp;
        if (*z > 0.99) {
            *sth = sqrt(tmp * (2.0 - tmp));
            *have_sth = true;
        }
    } else if (jr > 3) {
        nr = 4 - jr;
        double tmp = nr * nr / 3.;
        *z = tmp - 1;
        if (*z < -0.99) {
            *sth = sqrt(tmp * (2. - tmp));
            *have_sth = true;
        }
    } else {
        nr = 1;
        *z = (2 - jr) * 2. / 3.;
    }

    double tmp = jpll[face] * nr + x - y;
    if (tmp < 0) tmp += 8;
    if (tmp >= 8) tmp -= 8;
    *phi = (nr < 1e-15) ? 0 : (0.5 * HPG_HALFPI * tmp) / nr;
}

void locToVec3(double z, double phi, double sth, bool have_sth, vec3 *vec) {
    if (have_sth) {
        vec->x = sth * cos(phi);
        vec->y = sth * sin(phi);
        vec->z = z;
    } else {
        double sintheta = sqrt((1.0 - z) * (1.0 + z));
        vec->x = sintheta * cos(phi);
        vec->y = sintheta * sin(phi);
        vec->z = z;
    }
}

void locToPtg(double z, double phi, double sth, bool have_sth, pointing *p) {
    p->phi = phi;

    if (have_sth) {
        p->theta = atan2(sth, z);
    } else {
        p->theta = acos(z);
    }
}

void boundaries(healpix_info *hpx, int64_t pix, size_t step, pointingarr *out, int *status) {
    *status = 1;

    if (out->size < 4 * step) {
        *status = 0;
        return;
    }

    int ix, iy, face;
    pix2xyf(hpx, pix, &ix, &iy, &face);
    double dc = 0.5 / hpx->nside;
    double xc = (ix + 0.5) / hpx->nside;
    double yc = (iy + 0.5) / hpx->nside;
    double d = 1.0 / (step * hpx->nside);
    for (size_t i = 0; i < step; i++) {
        double z, phi, sth;
        bool have_sth;
        xyf2loc(xc + dc - i * d, yc + dc, face, &z, &phi, &sth, &have_sth);
        locToPtg(z, phi, sth, have_sth, &out->data[i]);
        xyf2loc(xc - dc, yc + dc - i * d, face, &z, &phi, &sth, &have_sth);
        locToPtg(z, phi, sth, have_sth, &out->data[i + step]);
        xyf2loc(xc - dc + i * d, yc - dc, face, &z, &phi, &sth, &have_sth);
        locToPtg(z, phi, sth, have_sth, &out->data[i + 2 * step]);
        xyf2loc(xc + dc, yc - dc + i * d, face, &z, &phi, &sth, &have_sth);
        locToPtg(z, phi, sth, have_sth, &out->data[i + 3 * step]);
    }
}

const int nb_xoffset[] = {-1, -1, 0, 1, 1, 1, 0, -1};
const int nb_yoffset[] = {0, 1, 1, 1, 0, -1, -1, -1};
const int nb_facearray[][12] = {{8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9},  // S
                                {5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8},      // SE
                                {-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1},  // E
                                {4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10},      // SW
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},        // center
                                {1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4},          // NE
                                {-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1},  // W
                                {3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7},          // NW
                                {2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3}};     // N
const int nb_swaparray[][3] = {{0, 0, 3},                                      // S
                               {0, 0, 6},                                      // SE
                               {0, 0, 0},                                      // E
                               {0, 0, 5},                                      // SW
                               {0, 0, 0},                                      // center
                               {5, 0, 0},                                      // NE
                               {0, 0, 0},                                      // W
                               {6, 0, 0},                                      // NW
                               {3, 0, 0}};                                     // N

void neighbors(healpix_info *hpx, int64_t pix, i64stack *result, int *status, char *err) {
    *status = 1;
    if (result->size < 8) {
        snprintf(err, ERR_SIZE, "result stack of insufficient size.");
        *status = 0;
        return;
    }

    int ix, iy, face_num;
    pix2xyf(hpx, pix, &ix, &iy, &face_num);

    const int64_t nsm1 = hpx->nside - 1;
    if ((ix > 0) && (ix < nsm1) && (iy > 0) && (iy < nsm1)) {
        if (hpx->scheme == RING) {
            for (int m = 0; m < 8; m++) {
                result->data[m] =
                    xyf2ring(hpx, ix + nb_xoffset[m], iy + nb_yoffset[m], face_num);
            }
        } else {
            int64_t fpix = (int64_t)face_num << (2 * hpx->order);
            int64_t px0 = spread_bits64(ix), py0 = spread_bits64(iy) << 1;
            int64_t pxp = spread_bits64(ix + 1), pyp = spread_bits64(iy + 1) << 1;
            int64_t pxm = spread_bits64(ix - 1), pym = spread_bits64(iy - 1) << 1;

            result->data[0] = fpix + pxm + py0;
            result->data[1] = fpix + pxm + pyp;
            result->data[2] = fpix + px0 + pyp;
            result->data[3] = fpix + pxp + pyp;
            result->data[4] = fpix + pxp + py0;
            result->data[5] = fpix + pxp + pym;
            result->data[6] = fpix + px0 + pym;
            result->data[7] = fpix + pxm + pym;
        }
    } else {
        for (int i = 0; i < 8; i++) {
            int x = ix + nb_xoffset[i], y = iy + nb_yoffset[i];
            int nbnum = 4;
            if (x < 0) {
                x += hpx->nside;
                nbnum--;
            } else if (x >= hpx->nside) {
                x -= hpx->nside;
                nbnum++;
            }
            if (y < 0) {
                y += hpx->nside;
                nbnum -= 3;
            } else if (y >= hpx->nside) {
                y -= hpx->nside;
                nbnum += 3;
            }

            int f = nb_facearray[nbnum][face_num];
            if (f >= 0) {
                int bits = nb_swaparray[nbnum][face_num >> 2];
                if (bits & 1) x = hpx->nside - x - 1;
                if (bits & 2) y = hpx->nside - y - 1;
                if (bits & 4) {
                    int64_t temp = x;
                    x = y;
                    y = temp;
                }
                result->data[i] = xyf2pix(hpx, x, y, f);
            } else {
                result->data[i] = -1;
            }
        }
    }
}

static void get_circle_q12(vec3arr *point, size_t q1, size_t q2, vec3 *center,
                           double *cosrad) {
    vec3_add(&point->data[q1], &point->data[q2], center);
    vec3_normalize(center);
    *cosrad = vec3_dotprod(&point->data[q1], center);
    for (size_t i = 0; i < q1; i++) {
        if (vec3_dotprod(&point->data[i], center) <
            *cosrad) {  // point outside the current circle
            vec3 v1, v2;
            vec3_subtract(&point->data[q1], &point->data[i], &v1);
            vec3_subtract(&point->data[q2], &point->data[i], &v2);
            vec3_crossprod(&v1, &v2, center);
            vec3_normalize(center);
            *cosrad = vec3_dotprod(&point->data[i], center);
            if (*cosrad < 0) {
                vec3_flip(center);
                *cosrad = -*cosrad;
            }
        }
    }
}

static void get_circle_q(vec3arr *point, size_t q, vec3 *center, double *cosrad) {
    vec3_add(&point->data[0], &point->data[q], center);
    vec3_normalize(center);
    *cosrad = vec3_dotprod(&point->data[0], center);
    for (size_t i = 1; i < q; i++) {
        if (vec3_dotprod(&point->data[i], center) <
            *cosrad) {  // point outside the current circle
            get_circle_q12(point, i, q, center, cosrad);
        }
    }
}

static void find_enclosing_circle(vec3arr *point, vec3 *center, double *cosrad) {
    size_t np = point->size;
    vec3_add(&point->data[0], &point->data[1], center);
    vec3_normalize(center);
    *cosrad = vec3_dotprod(&point->data[0], center);
    for (size_t i = 2; i < np; i++) {
        if (vec3_dotprod(&point->data[i], center) <
            *cosrad) {  // point outside the current circle
            get_circle_q(point, i, center, cosrad);
        }
    }
}

void query_multidisc(healpix_info *hpx, vec3arr *norm, double *rad, int fact,
                     i64rangeset *pixset, int *status, char *err) {
    *status = 1;
    bool inclusive = (fact != 0);
    size_t nv = norm->size;
    // this does not alter the storage
    pixset->stack->size = 0;

    if (hpx->scheme == RING) {
        dblarr *z0 = NULL, *xa = NULL, *cosrsmall = NULL, *cosrbig = NULL;
        pointingarr *ptg = NULL;
        i64stack *cpix = NULL;
        i64rangeset *tr = NULL;
        int64_t fct = 1;
        if (inclusive) {
            fct = fact;
        }
        healpix_info hpx2;
        double rpsmall, rpbig;
        if (fct > 1) {
            hpx2 = healpix_info_from_nside(fct * hpx->nside, RING);
            rpsmall = max_pixrad(&hpx2);
            rpbig = max_pixrad(hpx);
        } else {
            rpsmall = rpbig = inclusive ? max_pixrad(hpx) : 0;
        }

        tr = i64rangeset_new(status, err);
        if (!*status) goto cleanup_ring;

        int64_t irmin = 1, irmax = 4 * hpx->nside - 1;
        z0 = dblarr_new(nv, status, err);
        if (!*status) goto cleanup_ring;
        xa = dblarr_new(nv, status, err);
        if (!*status) goto cleanup_ring;
        cosrsmall = dblarr_new(nv, status, err);
        if (!*status) goto cleanup_ring;
        cosrbig = dblarr_new(nv, status, err);
        if (!*status) goto cleanup_ring;
        ptg = pointingarr_new(nv, status, err);
        if (!*status) goto cleanup_ring;
        cpix = i64stack_new(nv, status, err);
        if (!*status) goto cleanup_ring;

        size_t counter = 0;
        for (size_t i = 0; i < nv; i++) {
            double rsmall = rad[i] + rpsmall;
            if (rsmall < HPG_PI) {
                double rbig = dblmin(HPG_PI, rad[i] + rpbig);
                pointing pnt;
                pointing_from_vec3(&norm->data[i], &pnt);
                cosrsmall->data[counter] = cos(rsmall);
                cosrbig->data[counter] = cos(rbig);
                double cth = cos(pnt.theta);
                z0->data[counter] = cth;
                if (fct > 1) {
                    cpix->data[counter] = loc2pix(hpx, cth, pnt.phi, 0, false);
                    if (!*status) goto cleanup_ring;
                }
                xa->data[counter] = 1. / sqrt((1 - cth) * (1 + cth));
                ptg->data[counter].theta = pnt.theta;
                ptg->data[counter].phi = pnt.phi;
                counter++;

                double rlat1 = pnt.theta - rsmall;
                double zmax = cos(rlat1);
                int64_t irmin_t = (rlat1 <= 0) ? 1 : ring_above(hpx, zmax) + 1;

                if ((fct > 1) && (rlat1 > 0)) irmin_t = i64max((int64_t)1, irmin_t - 1);

                double rlat2 = pnt.theta + rsmall;
                double zmin = cos(rlat2);
                int64_t irmax_t =
                    (rlat2 >= HPG_PI) ? 4 * hpx->nside - 1 : ring_above(hpx, zmin);

                if ((fct > 1) && (rlat2 < HPG_PI))
                    irmax_t = i64min(4 * hpx->nside - 1, irmax_t + 1);

                if (irmax_t < irmax) irmax = irmax_t;
                if (irmin_t > irmin) irmin = irmin_t;
            }
        }

        for (int64_t iz = irmin; iz <= irmax; iz++) {
            double z = ring2z(hpx, iz);
            int64_t ipix1, nr;
            bool shifted;
            get_ring_info_small(hpx, iz, &ipix1, &nr, &shifted);
            double shift = shifted ? 0.5 : 0.;
            tr->stack->size = 0;
            i64rangeset_append(tr, ipix1, ipix1 + nr, status, err);
            if (!*status) goto cleanup_ring;
            for (size_t j = 0; j < counter; j++) {
                double x = (cosrbig->data[j] - z * z0->data[j]) * xa->data[j];
                double ysq = 1. - z * z - x * x;
                double dphi = (ysq <= 0) ? HPG_PI - 1e-15 : atan2(sqrt(ysq), x);
                int64_t ip_lo =
                    (int64_t)floor((nr * HPG_INV_TWOPI) * (ptg->data[j].phi - dphi) - shift) +
                    1;
                int64_t ip_hi =
                    (int64_t)floor((nr * HPG_INV_TWOPI) * (ptg->data[j].phi + dphi) - shift);
                if (fct > 1) {
                    while ((ip_lo <= ip_hi) &&
                           check_pixel_ring(hpx, &hpx2, ip_lo, nr, ipix1, fct, z0->data[j],
                                            ptg->data[j].phi, cosrsmall->data[j],
                                            cpix->data[j]))
                        ++ip_lo;
                    while ((ip_hi > ip_lo) &&
                           check_pixel_ring(hpx, &hpx2, ip_hi, nr, ipix1, fct, z0->data[j],
                                            ptg->data[j].phi, cosrsmall->data[j],
                                            cpix->data[j]))
                        --ip_hi;
                }
                if (ip_hi >= nr) {
                    ip_lo -= nr;
                    ip_hi -= nr;
                }
                if (ip_lo < 0) {
                    i64rangeset_remove(tr, ipix1 + ip_hi + 1, ipix1 + ip_lo + nr, status, err);
                } else {
                    i64rangeset_intersect(tr, ipix1 + ip_lo, ipix1 + ip_hi + 1, status, err);
                }
                if (!*status) goto cleanup_ring;
            }
            i64rangeset_append_i64rangeset(pixset, tr, status, err);
            if (!*status) goto cleanup_ring;
        }

    cleanup_ring:
        if (z0 != NULL) dblarr_delete(z0);
        if (xa != NULL) dblarr_delete(xa);
        if (cosrsmall != NULL) dblarr_delete(cosrsmall);
        if (cosrbig != NULL) dblarr_delete(cosrbig);
        if (ptg != NULL) pointingarr_delete(ptg);
        if (tr != NULL) i64rangeset_delete(tr);

    } else {  // scheme == NEST
        dblarr *crlimit0[MAX_ORDER + 1];
        dblarr *crlimit1[MAX_ORDER + 1];
        dblarr *crlimit2[MAX_ORDER + 1];
        i64stack *stk = NULL;

        for (int o = 0; o <= MAX_ORDER; o++) {
            crlimit0[o] = NULL;
            crlimit1[o] = NULL;
            crlimit2[o] = NULL;
        }

        int oplus = 0;
        if (inclusive) {
            oplus = ilog2(fact);
        }
        int omax = hpx->order + oplus;  // the order up to which we test

        // TODO: ignore all disks with radius>=pi

        struct healpix_info base[MAX_ORDER + 1];
        for (int o = 0; o <= MAX_ORDER; o++) {  // prepare data at the required orders
            base[o] = healpix_info_from_order(o, NEST);
            crlimit0[o] = dblarr_new(nv, status, err);
            if (!*status) goto cleanup_nest;
            crlimit1[o] = dblarr_new(nv, status, err);
            if (!*status) goto cleanup_nest;
            crlimit2[o] = dblarr_new(nv, status, err);
            if (!*status) goto cleanup_nest;

            double dr = max_pixrad(&base[o]);  // safety distance

            for (size_t i = 0; i < nv; i++) {
                crlimit0[o]->data[i] = (rad[i] + dr > HPG_PI) ? -1. : cos(rad[i] + dr);
                crlimit1[o]->data[i] = (o == 0) ? cos(rad[i]) : crlimit1[0]->data[i];
                crlimit2[o]->data[i] = (rad[i] - dr < 0.) ? 1. : cos(rad[i] - dr);
            }
        }

        stk = i64stack_new(2 * (12 + 3 * omax), status, err);
        if (!*status) goto cleanup_nest;
        for (int i = 0; i < 12; i++) {
            i64stack_push(stk, (int64_t)(11 - i), status, err);
            if (!*status) goto cleanup_nest;
            i64stack_push(stk, 0, status, err);
            if (!*status) goto cleanup_nest;
        }

        int stacktop = 0;  // a place to save a stack position
        while (stk->size > 0) {
            // pop current pixel number and order from the stack
            int64_t pix, temp;
            i64stack_pop_pair(stk, &pix, &temp, status, err);
            if (!*status) goto cleanup_nest;
            int o = (int)temp;
            vec3 pv = pix2vec(&base[o], pix);

            size_t zone = 3;
            for (size_t i = 0; i < nv; i++) {
                double crad = vec3_dotprod(&pv, &norm->data[i]);
                double crlim;
                for (size_t iz = 0; iz < zone; iz++) {
                    if (iz == 0) {
                        crlim = crlimit0[o]->data[i];
                    } else if (iz == 1) {
                        crlim = crlimit1[o]->data[i];
                    } else {
                        crlim = crlimit2[o]->data[i];
                    }
                    if (crad < crlim)
                        if ((zone = iz) == 0) goto bailout;
                }
            }
            check_pixel_nest(o, hpx->order, omax, zone, pixset, pix, stk, inclusive, &stacktop,
                             status, err);
            if (!*status) goto cleanup_nest;
        bailout:;
        }
    cleanup_nest:
        for (int o = 0; o <= MAX_ORDER; o++) {
            if (crlimit0[o] != NULL) dblarr_delete(crlimit0[o]);
            if (crlimit1[o] != NULL) dblarr_delete(crlimit1[o]);
            if (crlimit2[o] != NULL) dblarr_delete(crlimit2[o]);
        }
        if (stk != NULL) i64stack_delete(stk);
    }
}

void query_polygon(healpix_info *hpx, pointingarr *vertex, int fact, i64rangeset *pixset,
                   int *status, char *err) {
    *status = 1;

    bool inclusive = (fact != 0);
    size_t nv = vertex->size;
    size_t ncirc = inclusive ? nv + 1 : nv;
    vec3arr *vv = NULL, *normal = NULL;
    double *rad = NULL;

    if (nv < 3) {
        snprintf(err, ERR_SIZE, "Polygon does not have enough vertices.");
        *status = 0;
        return;
    }

    vv = vec3arr_new(nv, status, err);
    if (!*status) goto cleanup;

    for (size_t i = 0; i < nv; i++) {
        vec3_from_pointing(&vertex->data[i], &vv->data[i]);
    }

    normal = vec3arr_new(ncirc, status, err);
    if (!*status) goto cleanup;
    int flip = 0;
    for (size_t i = 0; i < nv; i++) {
        vec3_crossprod(&vv->data[i], &vv->data[(i + 1) % nv], &normal->data[i]);
        vec3_normalize(&normal->data[i]);
        double hnd = vec3_dotprod(&normal->data[i], &vv->data[(i + 2) % nv]);
        if (fabs(hnd) < 1e-10) {
            snprintf(err, ERR_SIZE, "Polygon has degenerate corner.");
            *status = 0;
            goto cleanup;
        }
        if (i == 0)
            flip = (hnd < 0.) ? -1 : 1;
        else if (flip * hnd < 0) {
            snprintf(err, ERR_SIZE, "Polygon is not convex.");
            *status = 0;
            goto cleanup;
        }
        normal->data[i].x *= flip;
        normal->data[i].y *= flip;
        normal->data[i].z *= flip;
    }
    rad = (double *)calloc(ncirc, sizeof(double));
    if (rad == NULL) {
        snprintf(err, ERR_SIZE, "Could not allocate array memory.");
        *status = 0;
        goto cleanup;
    }
    for (size_t i = 0; i < ncirc; i++) rad[i] = HPG_HALFPI;
    if (inclusive) {
        double cosrad;
        find_enclosing_circle(vv, &normal->data[nv], &cosrad);
        rad[nv] = acos(cosrad);
    }
    query_multidisc(hpx, normal, rad, fact, pixset, status, err);

cleanup:
    if (vv != NULL) vec3arr_delete(vv);
    if (normal != NULL) vec3arr_delete(normal);
    if (rad != NULL) free(rad);
}

void query_ellipse(healpix_info *hpx, double ptg_theta, double ptg_phi, double semi_major,
                   double semi_minor, double alpha, int fact, struct i64rangeset *pixset,
                   int *status, char *err) {
    if (hpx->scheme == RING) {
        snprintf(err, ERR_SIZE, "query_ellipse only supports nest ordering.");
        *status = 0;
        return;
    }

    i64stack *stk = NULL;

    bool inclusive = (fact != 0);
    // this does not alter the storage
    pixset->stack->size = 0;

    /*
      The following math is adapted from
      https://math.stackexchange.com/questions/3747965/points-within-an-ellipse-on-the-globe

      The sign of alpha has been reversed from the equations posted there so that it is
      defined as the angle East (clockwise) of North.

      This foci of the ellipse are pre-computed from the center, semi-major and semi-minor
      axes, and the rotation angle (alpha).  This is a lot of trig, but it only has to
      be done once per query and not per pixel.

      The criterion is then that the sum of the distances from a pixel to each of the foci
      add up to less than 2*semi_major.
    */
    vec3 f1vec, f2vec;
    pointing f1ptg, f2ptg;

    double cos_alpha = cos(alpha);
    double sin_alpha = sin(alpha);
    double gamma = sqrt(semi_major * semi_major - semi_minor * semi_minor);
    double sin_gamma = sin(gamma);
    double cos_gamma = cos(gamma);
    double cos_phi = cos(ptg_phi);
    double cos_theta = cos(ptg_theta);
    double sin_phi = sin(ptg_phi);
    double sin_theta = sin(ptg_theta);

    f1vec.x = cos_alpha * sin_gamma * cos_phi * cos_theta + sin_alpha * sin_gamma * sin_phi +
              cos_gamma * cos_phi * sin_theta;
    f2vec.x = -cos_alpha * sin_gamma * cos_phi * cos_theta - sin_alpha * sin_gamma * sin_phi +
              cos_gamma * cos_phi * sin_theta;
    f1vec.y = cos_alpha * sin_gamma * sin_phi * cos_theta - sin_alpha * sin_gamma * cos_phi +
              cos_gamma * sin_phi * sin_theta;
    f2vec.y = -cos_alpha * sin_gamma * sin_phi * cos_theta + sin_alpha * sin_gamma * cos_phi +
              cos_gamma * sin_phi * sin_theta;
    f1vec.z = cos_gamma * cos_theta - cos_alpha * sin_gamma * sin_theta;
    f2vec.z = cos_gamma * cos_theta + cos_alpha * sin_gamma * sin_theta;

    pointing_from_vec3(&f1vec, &f1ptg);
    pointing_from_vec3(&f2vec, &f2ptg);

    if (semi_minor >= HPG_PI) {  // disk covers the whole sphere
        i64rangeset_append(pixset, 0, hpx->npix, status, err);
        goto cleanup;
    }

    int oplus = 0;
    if (inclusive) {
        oplus = ilog2(fact);
    }
    int omax = hpx->order + oplus;  // the order up to which we test

    // Statically define the array of bases because it's not large.
    struct healpix_info base[MAX_ORDER + 1];

    double dr[MAX_ORDER + 1];
    double dmdr[MAX_ORDER + 1];
    double dpdr[MAX_ORDER + 1];
    for (int o = 0; o <= omax; o++) {
        base[o] = healpix_info_from_order(o, NEST);
        dr[o] = max_pixrad(&base[o]);  // safety distance
        dmdr[o] = 2 * semi_major - 2 * dr[o];
        dpdr[o] = 2 * semi_major + 2 * dr[o];
    }

    stk = i64stack_new(2 * (12 + 3 * omax), status, err);
    if (!*status) goto cleanup;
    for (int i = 0; i < 12; i++) {
        i64stack_push(stk, (int64_t)(11 - i), status, err);
        if (!*status) goto cleanup;
        i64stack_push(stk, 0, status, err);
        if (!*status) goto cleanup;
    }

    int stacktop = 0;  // a place to save a stack position
    while (stk->size > 0) {
        // pop current pixel number and order from the stack
        int64_t pix, temp;
        i64stack_pop_pair(stk, &pix, &temp, status, err);
        if (!*status) goto cleanup;
        int o = (int)temp;

        double pix_z, pix_phi;
        pix2zphi(&base[o], pix, &pix_z, &pix_phi);
        double d = acos(cosdist_zphi(f1vec.z, f1ptg.phi, pix_z, pix_phi)) +
                   acos(cosdist_zphi(f2vec.z, f2ptg.phi, pix_z, pix_phi));
        if (d <= dpdr[o]) {
            int zone = (d >= 2 * semi_major) ? 1 : ((d > dmdr[o]) ? 2 : 3);
            check_pixel_nest(o, hpx->order, omax, zone, pixset, pix, stk, inclusive, &stacktop,
                             status, err);
            if (!*status) goto cleanup;
        }
    }
 cleanup:
    if (stk != NULL) i64stack_delete(stk);
}

void query_box(healpix_info *hpx, double ptg_theta0, double ptg_theta1, double ptg_phi0,
               double ptg_phi1, bool full_lon, int fact, struct i64rangeset *pixset,
               int *status, char *err) {
    if (hpx->scheme == RING) {
        snprintf(err, ERR_SIZE, "query_box only supports nest ordering.");
        *status = 0;
        return;
    }

    i64stack *stk = NULL;

    bool inclusive = (fact != 0);
    // this does not alter the storage
    pixset->stack->size = 0;

    // First check if we have an empty box
    if (ptg_theta0 == ptg_theta1) goto cleanup;
    if (ptg_phi0 == ptg_phi1 && !full_lon) goto cleanup;

    if (inclusive && !full_lon) {
        // This ensures that pixels which wrap around the edge are included
        if (ptg_phi0 == 0.0) ptg_phi0 = HPG_TWO_PI;
    }

    int oplus = 0;
    if (inclusive) {
        oplus = ilog2(fact);
    }
    int omax = hpx->order + oplus;  // the order up to which we test

    // Statically define the array of bases because it's not large.
    struct healpix_info base[MAX_ORDER + 1];
    double dr[MAX_ORDER + 1];
    for (int o = 0; o <= omax; o++) {
        base[o] = healpix_info_from_order(o, NEST);
        dr[o] = max_pixrad(&base[o]);  // safety distance
    }

    stk = i64stack_new(2 * (12 + 3 * omax), status, err);
    if (!*status) goto cleanup;
    for (int i = 0; i < 12; i++) {
        i64stack_push(stk, (int64_t)(11 - i), status, err);
        if (!*status) goto cleanup;
        i64stack_push(stk, 0, status, err);
        if (!*status) goto cleanup;
    }

    int stacktop = 0;  // a place to save a stack position
    while (stk->size > 0) {
        // pop current pixel number and order from the stack
        int64_t pix, temp;
        i64stack_pop_pair(stk, &pix, &temp, status, err);
        if (!*status) goto cleanup;
        int o = (int)temp;

        /* Short note on the "zone":
           zone = 0: pixel lies completely outside the queried shape
           1: pixel may overlap with the shape, pixel center is outside
           2: pixel center is inside the shape, but maybe not the complete pixel
           3: pixel lies completely inside the shape */

        double pix_theta, pix_phi;
        pix2ang(&base[o], pix, &pix_theta, &pix_phi);

        int zone_theta = 0, zone_phi = 0;

        /* Check in the colatitude (theta) direction */
        double tmdr = pix_theta - dr[o], tpdr = pix_theta + dr[o];
        if (tpdr >= (ptg_theta0 - HPG_EPSILON) && tmdr < ptg_theta1) {
            // Check if completely inside
            if (tmdr >= (ptg_theta0 - HPG_EPSILON) && tpdr < ptg_theta1) {
                zone_theta = 3;
            } else if (pix_theta >= (ptg_theta0 - HPG_EPSILON) && pix_theta < ptg_theta1) {
                zone_theta = 2;
            } else {
                zone_theta = 1;
            }
        }

        /* Check in the longitude (phi) direction */
        if (full_lon) {  // This has the full longitude range, always zone 3.
            zone_phi = 3;
        } else if (zone_theta > 0) {
            double stheta = sin(pix_theta);
            double pmdr = pix_phi - dr[o] / stheta;
            double ppdr = pix_phi + dr[o] / stheta;

            if (ptg_phi0 < ptg_phi1) {
                // Regular orientation
                if (ppdr >= (ptg_phi0 - HPG_EPSILON) && pmdr < ptg_phi1) {
                    // Check if completely inside
                    if (pmdr >= (ptg_phi0 - HPG_EPSILON) && ppdr < ptg_phi1) {
                        zone_phi = 3;
                    } else if (pix_phi >= (ptg_phi0 - HPG_EPSILON) && pix_phi < ptg_phi1) {
                        zone_phi = 2;
                    } else {
                        zone_phi = 1;
                    }
                }
            } else {
                // Reverse orientation
                if (pmdr < ptg_phi1 || ppdr >= (ptg_phi0 - HPG_EPSILON)) {
                    // Check if completely inside
                    if (ppdr < ptg_phi1 || pmdr >= (ptg_phi0 - HPG_EPSILON)) {
                        zone_phi = 3;
                    } else if (pix_phi < ptg_phi1 || pix_phi >= (ptg_phi0 - HPG_EPSILON)) {
                        zone_phi = 2;
                    } else {
                        zone_phi = 1;
                    }
                }
            }
        }

        int zone = intmin(zone_theta, zone_phi);
        if (zone > 0) {
            check_pixel_nest(o, hpx->order, omax, zone, pixset, pix, stk, inclusive, &stacktop,
                             status, err);
            if (!*status) goto cleanup;
        }
    }
 cleanup:
    if (stk != NULL) i64stack_delete(stk);
}

void get_ring_info2(healpix_info *hpx, int64_t ring, int64_t *startpix, int64_t *ringpix,
                    double *theta, bool *shifted) {
    int64_t northring = (ring > 2 * hpx->nside) ? 4 * hpx->nside - ring : ring;
    if (northring < hpx->nside) {
        double tmp = northring * northring * hpx->fact2;
        double costheta = 1 - tmp;
        double sintheta = sqrt(tmp * (2 - tmp));
        *theta = atan2(sintheta, costheta);
        *ringpix = 4 * northring;
        *shifted = true;
        *startpix = 2 * northring * (northring - 1);
    } else {
        *theta = acos((2 * hpx->nside - northring) * hpx->fact1);
        *ringpix = 4 * hpx->nside;
        *shifted = ((northring - hpx->nside) & 1) == 0;
        *startpix = hpx->ncap + (northring - hpx->nside) * *ringpix;
    }
    if (northring != ring) {  // southern hemisphere
        *theta = HPG_PI - *theta;
        *startpix = hpx->npix - *startpix - *ringpix;
    }
}

void get_interpol(healpix_info *hpx, double ptg_theta, double ptg_phi, int64_t *pixels,
                  double *weights) {
    double z = cos(ptg_theta);
    int64_t ir1 = ring_above(hpx, z);
    int64_t ir2 = ir1 + 1;

    double theta1, theta2, w1, tmp, dphi;
    int64_t sp, nr;
    bool shift;
    int64_t i1, i2;

    if (ir1 > 0) {
        get_ring_info2(hpx, ir1, &sp, &nr, &theta1, &shift);
        dphi = HPG_TWO_PI / nr;
        tmp = (ptg_phi / dphi - 0.5 * shift);
        i1 = (tmp < 0) ? (int64_t)tmp - 1 : (int64_t)tmp;
        w1 = (ptg_phi - (i1 + 0.5 * shift) * dphi) / dphi;
        i2 = i1 + 1;
        if (i1 < 0) {
            i1 += nr;
        }
        if (i2 >= nr) {
            i2 -= nr;
        }
        pixels[0] = sp + i1;
        pixels[1] = sp + i2;
        weights[0] = 1 - w1;
        weights[1] = w1;
    }
    if (ir2 < (4 * hpx->nside)) {
        get_ring_info2(hpx, ir2, &sp, &nr, &theta2, &shift);
        dphi = HPG_TWO_PI / nr;
        tmp = (ptg_phi / dphi - 0.5 * shift);
        i1 = (tmp < 0) ? (int64_t)tmp - 1 : (int64_t)tmp;
        w1 = (ptg_phi - (i1 + 0.5 * shift) * dphi) / dphi;
        i2 = i1 + 1;
        if (i1 < 0) {
            i1 += nr;
        }
        if (i2 >= nr) {
            i2 -= nr;
        }
        pixels[2] = sp + i1;
        pixels[3] = sp + i2;
        weights[2] = 1 - w1;
        weights[3] = w1;
    }
    if (ir1 == 0) {
        double wtheta = ptg_theta / theta2;  // ??
        weights[2] *= wtheta;
        weights[3] *= wtheta;
        double fac = (1 - wtheta) * 0.25;
        weights[0] = fac;
        weights[1] = fac;
        weights[2] += fac;
        weights[3] += fac;
        pixels[0] = (pixels[2] + 2) & 3;
        pixels[1] = (pixels[3] + 2) & 3;
    } else if (ir2 == 4 * hpx->nside) {
        double wtheta = (ptg_theta - theta1) / (HPG_PI - theta1);
        weights[0] *= (1 - wtheta);
        weights[1] *= (1 - wtheta);
        double fac = wtheta * 0.25;
        weights[0] += fac;
        weights[1] += fac;
        weights[2] = fac;
        weights[3] = fac;
        pixels[2] = ((pixels[0] + 2) & 3) + hpx->npix - 4;
        pixels[3] = ((pixels[1] + 2) & 3) + hpx->npix - 4;
    } else {
        double wtheta = (ptg_theta - theta1) / (theta2 - theta1);
        weights[0] *= (1 - wtheta);
        weights[1] *= (1 - wtheta);
        weights[2] *= wtheta;
        weights[3] *= wtheta;
    }

    if (hpx->scheme == NEST) {
        for (size_t m = 0; m < 4; m++) {
            pixels[m] = ring2nest(hpx, pixels[m]);
        }
    }
}
