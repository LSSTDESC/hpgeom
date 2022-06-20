/*
 *  This file is modified from Healpix_cxx/healpix_base.cc by
 *  Eli Rykoff, Matt Becker, Erin Sheldon.
 *  Copyright (C) 2022 LSST DESC
 *
 *  Healpix_cxx is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Healpix_cxx is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Healpix_cxx; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix, see http://healpix.sourceforge.net
 */

/*
 *  Healpix_cxx is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*
 *  Copyright (C) 2003-2016 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <stdio.h>

#include "healpix_geom.h"
#include "hpgeom_stack.h"

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

static inline int64_t i64max(int64_t v1, int64_t v2) {
  return v1 > v2 ? v1 : v2;
}
static inline int64_t i64min(int64_t v1, int64_t v2) {
  return v1 < v2 ? v1 : v2;
}

static inline int64_t special_div(int64_t a, int64_t b) {
  int64_t t = (a >= (b << 1));
  a -= t * (b << 1);
  return (t << 1) + (a >= b);
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

int64_t ang2pix(healpix_info hpx, double theta, double phi) {
  if ((theta < 0.01) || (theta > 3.14159 - 0.01)) {
    return loc2pix(hpx, cos(theta), phi, 0.0, false);
  } else {
    return loc2pix(hpx, cos(theta), phi, sin(theta), true);
  }
}

/*
int64_t vec2pix(int64_t nside_, int is_nest, double x, double y, double z) {
    double xl = 1./sqrt(x*x + y*y + z*z);
    double phi = atan2(y, x);
    double nz = z*xl;
    if (fabs(nz) > 0.99) {
        return loc2pix(nside_, is_nest, nz, phi, sqrt(x*x+y*y)*xl, true);
    } else {
        return loc2pix(nside_, is_nest, nz, phi, 0, false);
    }
}
*/

void pix2ang(healpix_info hpx, int64_t pix, double *theta, double *phi) {
  double z, sth;
  bool have_sth;
  pix2loc(hpx, pix, &z, phi, &sth, &have_sth);
  if (have_sth) {
    *theta = atan2(sth, z);
  } else {
    *theta = acos(z);
  }
}

void pix2zphi(healpix_info hpx, int64_t pix, double *z, double *phi) {
  bool dum_b;
  double dum_d;
  pix2loc(hpx, pix, z, phi, &dum_d, &dum_b);
}

void pix2xyf(healpix_info hpx, int64_t pix, int *ix, int *iy, int *face_num) {
  (hpx.scheme == RING) ? ring2xyf(hpx, pix, ix, iy, face_num)
                       : nest2xyf(hpx, pix, ix, iy, face_num);
}

int64_t xyf2pix(healpix_info hpx, int ix, int iy, int face_num) {
  return (hpx.scheme == RING) ? xyf2ring(hpx, ix, iy, face_num)
                              : xyf2nest(hpx, ix, iy, face_num);
}

int64_t nest2ring(healpix_info hpx, int64_t pix) {
  int ix, iy, face_num;
  nest2xyf(hpx, pix, &ix, &iy, &face_num);
  return xyf2ring(hpx, ix, iy, face_num);
}

int64_t ring2nest(healpix_info hpx, int64_t pix) {
  int ix, iy, face_num;
  ring2xyf(hpx, pix, &ix, &iy, &face_num);
  return xyf2nest(hpx, ix, iy, face_num);
}

/*
void pix2vec(int64_t nside_, int is_nest, int64_t pix, double &x, double &y,
double &z) { double phi, sth; bool have_sth; pix2loc(nside_, is_nest, pix, z,
&phi, &sth, &have_sth); if (have_sth) { *x = sth*cos(phi); *y = sth*sin(phi); }
else { sth = sqrt((1.0 - z)*(1.0 + z)); *x = sth*cos(phi); *y = sth*sin(phi);
    }
}
*/

int64_t loc2pix(healpix_info hpx, double z, double phi, double sth,
                bool have_sth) {
  double za = fabs(z);
  double tt = fmod(phi * M_2_PI, 4.0); // in [0,4)

  if (hpx.scheme == RING) {
    if (za <= M_TWOTHIRD) // Equatorial region
    {
      int64_t nl4 = 4 * hpx.nside;
      double temp1 = hpx.nside * (0.5 + tt);
      double temp2 = hpx.nside * z * 0.75;
      int64_t jp = (int64_t)(temp1 - temp2); // index of  ascending edge line
      int64_t jm = (int64_t)(temp1 + temp2); // index of descending edge line

      // ring number counted from z=2/3
      int64_t ir = hpx.nside + 1 + jp - jm; // in {1,2n+1}
      int64_t kshift = 1 - (ir & 1);        // kshift=1 if ir even, 0 otherwise

      int64_t t1 = jp + jm - hpx.nside + kshift + 1 + nl4 + nl4;
      int64_t ip = (hpx.order > 0) ? (t1 >> 1) & (nl4 - 1)
                                   : ((t1 >> 1) % nl4); // in {0,4n-1}

      return hpx.ncap + (ir - 1) * nl4 + ip;
    } else // North & South polar caps
    {
      double tp = tt - (int64_t)(tt);
      double tmp = ((za < 0.99) || (!have_sth))
                       ? hpx.nside * sqrt(3 * (1 - za))
                       : hpx.nside * sth / sqrt((1. + za) / 3.);

      int64_t jp = (int64_t)(tp * tmp);         // increasing edge line index
      int64_t jm = (int64_t)((1.0 - tp) * tmp); // decreasing edge line index

      int64_t ir = jp + jm + 1; // ring number counted from the closest pole
      int64_t ip = (int64_t)(tt * ir); // in {0,4*ir-1}

      if (z > 0.) {
        return 2 * ir * (ir - 1) + ip;
      } else {
        return hpx.npix - 2 * ir * (ir + 1) + ip;
      }
    }
  } else // is_nest
  {
    if (za <= M_TWOTHIRD) // Equatorial region
    {
      double temp1 = hpx.nside * (0.5 + tt);
      double temp2 = hpx.nside * (z * 0.75);
      int64_t jp = (int64_t)(temp1 - temp2); // index of  ascending edge line
      int64_t jm = (int64_t)(temp1 + temp2); // index of descending edge line
      int64_t ifp = jp >> hpx.order;         // in {0,4}
      int64_t ifm = jm >> hpx.order;

      int face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));

      int ix = jm & (hpx.nside - 1),
          iy = hpx.nside - (jp & (hpx.nside - 1)) - 1;
      return xyf2nest(hpx, ix, iy, face_num);
    } else // polar region, za > 2/3
    {
      int ntt = (int)tt;
      if (ntt >= 4)
        ntt = 3;
      double tp = tt - ntt;
      double tmp = ((za < 0.99) || (!have_sth))
                       ? hpx.nside * sqrt(3 * (1 - za))
                       : hpx.nside * sth / sqrt((1. + za) / 3.);

      int64_t jp = (int64_t)(tp * tmp);         // increasing edge line index
      int64_t jm = (int64_t)((1.0 - tp) * tmp); // decreasing edge line index
      if (jp >= hpx.nside)
        jp = hpx.nside - 1; // for points too close to the boundary
      if (jm >= hpx.nside)
        jm = hpx.nside - 1;
      return (z >= 0)
                 ? xyf2nest(hpx, hpx.nside - jm - 1, hpx.nside - jp - 1, ntt)
                 : xyf2nest(hpx, jp, jm, ntt + 8);
    }
  }
}

void pix2loc(healpix_info hpx, int64_t pix, double *z, double *phi, double *sth,
             bool *have_sth) {
  *have_sth = false;
  if (hpx.scheme == RING) {
    if (pix < hpx.ncap) // North Polar cap
    {
      int64_t iring =
          (1 + (int64_t)(isqrt(1 + 2 * pix))) >> 1; // counted from North pole
      int64_t iphi = (pix + 1) - 2 * iring * (iring - 1);

      double tmp = (iring * iring) * hpx.fact2;
      *z = 1.0 - tmp;
      if (*z > 0.99) {
        *sth = sqrt(tmp * (2. - tmp));
        *have_sth = true;
      }
      *phi = (iphi - 0.5) * M_PI_2 / iring;
    } else if (pix < (hpx.npix - hpx.ncap)) // Equatorial region
    {
      int64_t nl4 = 4 * hpx.nside;
      int64_t ip = pix - hpx.ncap;
      int64_t tmp = (hpx.order >= 0) ? ip >> (hpx.order + 2) : ip / nl4;
      int64_t iring = tmp + hpx.nside, iphi = ip - nl4 * tmp + 1;
      // 1 if iring+nside is odd, 1/2 otherwise
      double fodd = ((iring + hpx.nside) & 1) ? 1 : 0.5;

      *z = (2 * hpx.nside - iring) * hpx.fact1;
      *phi = (iphi - fodd) * M_PI * 0.75 * hpx.fact1;
    } else // South Polar cap
    {
      int64_t ip = hpx.npix - pix;
      int64_t iring =
          (1 + (int64_t)(isqrt(2 * ip - 1))) >> 1; // counted from South pole
      int64_t iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));

      double tmp = (iring * iring) * hpx.fact2;
      *z = tmp - 1.0;
      if (*z < -0.99) {
        *sth = sqrt(tmp * (2. - tmp));
        *have_sth = true;
      }
      *phi = (iphi - 0.5) * M_PI_2 / iring;
    }
  } else {
    int face_num, ix, iy;
    nest2xyf(hpx, pix, &ix, &iy, &face_num);

    int64_t jr = ((int64_t)(jrll[face_num]) << hpx.order) - ix - iy - 1;

    int64_t nr;
    if (jr < hpx.nside) {
      nr = jr;
      double tmp = (nr * nr) * hpx.fact2;
      *z = 1 - tmp;
      if (*z > 0.99) {
        *sth = sqrt(tmp * (2. - tmp));
        *have_sth = true;
      }
    } else if (jr > 3 * hpx.nside) {
      nr = hpx.nside * 4 - jr;
      double tmp = (nr * nr) * hpx.fact2;
      *z = tmp - 1;
      if (*z < -0.99) {
        *sth = sqrt(tmp * (2. - tmp));
        *have_sth = true;
      }
    } else {
      nr = hpx.nside;
      *z = (2 * hpx.nside - jr) * hpx.fact1;
    }

    int64_t tmp = (int64_t)(jpll[face_num]) * nr + ix - iy;
    if (tmp < 0)
      tmp += 8 * nr;
    *phi = (nr == hpx.nside) ? 0.75 * M_PI_2 * tmp * hpx.fact1
                             : (0.5 * M_PI_2 * tmp) / nr;
  }
}

int64_t xyf2nest(healpix_info hpx, int ix, int iy, int face_num) {
  return ((int64_t)face_num << (2 * hpx.order)) + spread_bits64(ix) +
         (spread_bits64(iy) << 1);
}

void nest2xyf(healpix_info hpx, int64_t pix, int *ix, int *iy, int *face_num) {
  *face_num = pix >> (2 * hpx.order);
  pix &= (hpx.npface - 1);
  *ix = compress_bits64(pix);
  *iy = compress_bits64(pix >> 1);
}

int64_t xyf2ring(healpix_info hpx, int ix, int iy, int face_num) {
  int64_t nl4 = 4 * hpx.nside;
  int64_t jr = (jrll[face_num] * hpx.nside) - ix - iy - 1;

  int64_t nr, kshift, n_before;

  bool shifted;
  get_ring_info_small(hpx, jr, &n_before, &nr, &shifted);
  nr >>= 2;
  kshift = 1 - shifted;
  int64_t jp = (jpll[face_num] * nr + ix - iy + 1 + kshift) / 2;
  if (jp < 1)
    jp += nl4;

  return n_before + jp - 1;
}

void ring2xyf(healpix_info hpx, int64_t pix, int *ix, int *iy, int *face_num) {
  int64_t iring, iphi, kshift, nr;
  int64_t nl2 = 2 * hpx.nside;

  if (pix < hpx.ncap) {                    // North Polar cap
    iring = (1 + isqrt(1 + 2 * pix)) >> 1; // counted from North pole
    iphi = (pix + 1) - 2 * iring * (iring - 1);
    kshift = 0;
    nr = iring;
    *face_num = special_div(iphi - 1, nr);
  } else if (pix < (hpx.npix - hpx.ncap)) { // Equatorial region
    int64_t ip = pix - hpx.ncap;
    int64_t tmp =
        (hpx.order >= 0) ? ip >> (hpx.order + 2) : ip / (4 * hpx.nside);
    iring = tmp + hpx.nside;
    iphi = ip - tmp * 4 * hpx.nside + 1;
    kshift = (iring + hpx.nside) & 1;
    nr = hpx.nside;
    int64_t ire = tmp + 1, irm = nl2 + 1 - tmp;
    int64_t ifm = iphi - (ire >> 1) + hpx.nside - 1,
            ifp = iphi - (irm >> 1) + hpx.nside - 1;
    if (hpx.order >= 0) {
      ifm >>= hpx.order;
      ifp >>= hpx.order;
    } else {
      ifm /= hpx.nside;
      ifp /= hpx.nside;
    }
    *face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));
  } else { // South Polar cap
    int64_t ip = hpx.npix - pix;
    iring = (1 + isqrt(2 * ip - 1)) >> 1; // counted from South pole
    iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
    kshift = 0;
    nr = iring;
    iring = 2 * nl2 - iring;
    *face_num = special_div(iphi - 1, nr) + 8;
  }

  int64_t irt = iring - ((2 + (*face_num >> 2)) * hpx.nside) + 1;
  int64_t ipt = 2 * iphi - jpll[*face_num] * nr - kshift - 1;
  if (ipt >= nl2)
    ipt -= 8 * hpx.nside;

  *ix = (ipt - irt) >> 1;
  *iy = (-ipt - irt) >> 1;
}

double ring2z(healpix_info hpx, int64_t ring) {
  if (ring < hpx.nside) {
    return 1 - ring * ring * hpx.fact2;
  }
  if (ring <= 3 * hpx.nside) {
    return (2 * hpx.nside - ring) * hpx.fact1;
  }
  ring = 4 * hpx.nside - ring;
  return ring * ring * hpx.fact2 - 1;
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

double max_pixrad(healpix_info hpx) {
  double z_a = 2. / 3.;
  double phi_a = M_PI / (4. * hpx.nside);
  double sintheta = sqrt((1 - z_a) * (1 + z_a));
  double x_a = sintheta * cos(phi_a);
  double y_a = sintheta * sin(phi_a);

  double t1 = 1. - 1. / hpx.nside;
  t1 *= t1;

  double z_b = 1. - t1 / 3.;
  double phi_b = 0.0;
  sintheta = sqrt((1 - z_b) * (1 + z_b));
  double x_b = sintheta * cos(phi_b);
  double y_b = sintheta * sin(phi_b);

  double angle = acos((x_a * x_b + y_a * y_b + z_a * z_b) /
                      (sqrt(x_a * x_a + y_a * y_a + z_a * z_a) *
                       sqrt(x_b * x_b + y_b * y_b + z_b * z_b)));
  return angle;
}

int64_t ring_above(healpix_info hpx, double z) {
  double az = fabs(z);
  if (az <= M_TWOTHIRD) // equatorial region
    return (int64_t)(hpx.nside * (2 - 1.5 * z));
  int64_t iring = (int64_t)(hpx.nside * sqrt(3 * (1 - az)));
  return (z > 0) ? iring : 4 * hpx.nside - iring - 1;
}

void get_ring_info_small(healpix_info hpx, int64_t ring, int64_t *startpix,
                         int64_t *ringpix, bool *shifted) {
  if (ring < hpx.nside) {
    *shifted = true;
    *ringpix = 4 * ring;
    *startpix = 2 * ring * (ring - 1);
  } else if (ring < 3 * hpx.nside) {
    *shifted = ((ring - hpx.nside) & 1) == 0;
    *ringpix = 4 * hpx.nside;
    *startpix = hpx.ncap + (ring - hpx.nside) * (*ringpix);
  } else {
    *shifted = true;
    int64_t nr = 4 * hpx.nside - ring;
    *ringpix = 4 * nr;
    *startpix = hpx.npix - 2 * nr * (nr + 1);
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

void check_pixel_nest(int o, int order_, int omax, int zone,
                      i64rangeset *pixset, int64_t pix,
                      i64stack *stk, bool inclusive, int *stacktop,
                      int *status, char *err) {
  *status = 1;
  if (zone == 0)
    return;

  if (o < order_) {
    if (zone >= 3) {
      int sdist =
          2 * (order_ - o); // the "bit-shift distance" between map orders
      i64rangeset_append(pixset, pix << sdist, (pix + 1) << sdist, status,
                         err); // output all subpixels
      if (!status)
        return;
    } else { // (1<=zone<=2)
      for (int i = 0; i < 4; i++) {
        // output all subpixels, a pair of pixel and order
        i64stack_push(stk, 4 * pix + 3 - i, status, err);
        if (!status)
          return;
        i64stack_push(stk, o + 1, status, err);
        if (!status)
          return;
      }
    }
  } else if (o > order_) { // this implies that inclusive=true
    if (zone >= 2) {       // pixel center in shape
      i64rangeset_append_single(pixset, pix >> (2 * (o - order_)), status,
                                err); // output the parent pixel at order_
      if (!status)
        return;
      i64stack_resize(stk, *stacktop, status, err); // unwind the stack
      if (!status)
        return;
    } else {                          // (zone==1): pixel center in safety range
      if (o < omax) {                 // check sublevels
        for (int i = 0; i < 4; i++) { // add children in reverse order
          i64stack_push(stk, 4 * pix + 3 - i, status, err);
          if (!status)
            return;
          i64stack_push(stk, o + 1, status, err);
          if (!status)
            return;
        }
      } else { // at resolution limit
        i64rangeset_append_single(pixset, pix >> (2 * (o - order_)), status,
                                  err); // output the parent pixel at order_
        if (!status)
          return;
        i64stack_resize(stk, *stacktop, status, err); // unwind the stack
        if (!status)
          return;
      }
    }
  } else { // o==order_
    if (zone >= 2) {
      i64rangeset_append_single(pixset, pix, status, err);
      if (!status)
        return;
    } else if (inclusive) {           // and (zone>=1)
      if (order_ < omax) {            // check sublevels
        *stacktop = stk->size;        // remember current stack position
        for (int i = 0; i < 4; i++) { // add children in reverse order
          i64stack_push(stk, 4 * pix + 3 - i, status, err);
          if (!status)
            return;
          i64stack_push(stk, o + 1, status, err);
          if (!status)
            return;
        }
      } else { // at resolution limit
        i64rangeset_append_single(pixset, pix, status, err); // output the pixel
        if (!status)
          return;
      }
    }
  }
}

bool check_pixel_ring(healpix_info hpx1, healpix_info hpx2, int64_t pix,
                      int64_t nr, int64_t ipix1, int fct, double cz,
                      double cphi, double cosrp2, int64_t cpix) {
  if (pix >= nr)
    pix -= nr;
  if (pix < 0)
    pix += nr;
  pix += ipix1;
  if (pix == cpix)
    return false; // disk center in pixel => overlap
  int px, py, pf;
  pix2xyf(hpx1, pix, &px, &py, &pf);
  for (int i = 0; i < fct - 1; i++) { // go along the 4 edges
    int64_t ox = fct * px, oy = fct * py;
    double pz, pphi;
    pix2zphi(hpx2, xyf2pix(hpx2, ox + i, oy, pf), &pz, &pphi);
    if (cosdist_zphi(pz, pphi, cz, cphi) > cosrp2) // overlap
      return false;
    pix2zphi(hpx2, xyf2pix(hpx2, ox + fct - 1, oy + i, pf), &pz, &pphi);
    if (cosdist_zphi(pz, pphi, cz, cphi) > cosrp2) // overlap
      return false;
    pix2zphi(hpx2, xyf2pix(hpx2, ox + fct - 1 - i, oy + fct - 1, pf), &pz,
             &pphi);
    if (cosdist_zphi(pz, pphi, cz, cphi) > cosrp2) // overlap
      return false;
    pix2zphi(hpx2, xyf2pix(hpx2, ox, oy + fct - 1 - i, pf), &pz, &pphi);
    if (cosdist_zphi(pz, pphi, cz, cphi) > cosrp2) // overlap
      return false;
  }
  return true;
}

void query_disc(healpix_info hpx, double ptg_theta, double ptg_phi,
                double radius, int fact, i64rangeset *pixset,
                int *status, char *err) {
  bool inclusive = (fact != 0);
  // this does not alter the storage
  pixset->stack->size = 0;

  if (hpx.scheme == RING) {
    int64_t fct = 1;
    if (inclusive) {
      fct = fact;
    }
    healpix_info hpx2;
    double rsmall, rbig;
    if (fct > 1) {
      hpx2 = healpix_info_from_nside(fct * hpx.nside, RING);
      rsmall = radius + max_pixrad(hpx2);
      rbig = radius + max_pixrad(hpx);
    } else {
      rsmall = rbig = inclusive ? radius + max_pixrad(hpx) : radius;
    }

    if (rsmall >= M_PI) {
      i64rangeset_append(pixset, 0, hpx.npix, status, err);
      return;
    }

    if (rbig > M_PI) {
      rbig = M_PI;
    }

    double cosrsmall = cos(rsmall);
    double cosrbig = cos(rbig);

    double z0 = cos(ptg_theta);
    double xa = 1. / sqrt((1 - z0) * (1 + z0));

    int64_t cpix = loc2pix(hpx, z0, ptg_phi, 0., false);

    double rlat1 = ptg_theta - rsmall;
    double zmax = cos(rlat1);
    int64_t irmin = ring_above(hpx, zmax) + 1;

    if ((rlat1 <= 0) && (irmin > 1)) { // north pole in the disk
      int64_t sp, rp;
      bool dummy;
      get_ring_info_small(hpx, irmin - 1, &sp, &rp, &dummy);
      i64rangeset_append(pixset, 0, sp + rp, status, err);
      if (!status)
        return;
    }
    if ((fct > 1) && (rlat1 > 0))
      irmin = i64max((int64_t)1, irmin - 1);

    double rlat2 = ptg_theta + rsmall;
    double zmin = cos(rlat2);
    int64_t irmax = ring_above(hpx, zmin);

    if ((fct > 1) && (rlat2 < M_PI))
      irmax = i64min(4 * hpx.nside - 1, irmax + 1);

    for (int64_t iz = irmin; iz <= irmax; ++iz) {
      double z = ring2z(hpx, iz);
      double x = (cosrbig - z * z0) * xa;
      double ysq = 1 - z * z - x * x;
      double dphi = -1;
      if (ysq <= 0) { // no intersection, ring completely inside or outside
        dphi = (fct == 1) ? 0 : M_PI - 1e-15;
      } else {
        dphi = atan2(sqrt(ysq), x);
      }
      if (dphi > 0) {
        int64_t nr, ipix1;
        bool shifted;
        get_ring_info_small(hpx, iz, &ipix1, &nr, &shifted);
        double shift = shifted ? 0.5 : 0.;

        int64_t ipix2 = ipix1 + nr - 1; // highest pixel number in the ring

        int64_t ip_lo =
            (int64_t)floor((nr / M_TWO_PI) * (ptg_phi - dphi) - shift) + 1;
        int64_t ip_hi =
            (int64_t)floor((nr / M_TWO_PI) * (ptg_phi + dphi) - shift);

        if (fct > 1) {
          while ((ip_lo <= ip_hi) &&
                 check_pixel_ring(hpx, hpx2, ip_lo, nr, ipix1, fct, z0, ptg_phi,
                                  cosrsmall, cpix))
            ++ip_lo;
          while ((ip_hi > ip_lo) &&
                 check_pixel_ring(hpx, hpx2, ip_hi, nr, ipix1, fct, z0, ptg_phi,
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
            if (!status)
              return;
            i64rangeset_append(pixset, ipix1 + ip_lo + nr, ipix2 + 1, status,
                               err);
            if (!status)
              return;
          } else {
            i64rangeset_append(pixset, ipix1 + ip_lo, ipix1 + ip_hi + 1, status,
                               err);
            if (!status)
              return;
          }
        }
      }
    }
    if ((rlat2 >= M_PI) &&
        (irmax + 1 < 4 * hpx.nside)) { // south pole in the disk
      int64_t sp, rp;
      bool dummy;
      get_ring_info_small(hpx, irmax + 1, &sp, &rp, &dummy);
      i64rangeset_append(pixset, sp, hpx.npix, status, err);
      if (!status)
        return;
    }
  } else {                // schema == NEST
    if (radius >= M_PI) { // disk covers the whole sphere
      i64rangeset_append(pixset, 0, hpx.npix, status, err);
      return;
    }

    int oplus = 0;
    if (inclusive) {
      oplus = ilog2(fact);
    }
    int omax = hpx.order + oplus; // the order up to which we test

    // Statically define the array of bases because it's not large.
    double ptg_z = cos(ptg_theta);
    struct healpix_info base[MAX_ORDER + 1];
    double crpdr[MAX_ORDER + 1], crmdr[MAX_ORDER + 1];
    double cosrad = cos(radius);
    for (int o = 0; o <= omax; o++) {
      base[o] = healpix_info_from_order(o, NEST);
      double dr = max_pixrad(base[o]); // safety distance
      crpdr[o] = ((radius + dr) > M_PI) ? -1. : cos(radius + dr);
      crmdr[o] = ((radius - dr) < 0.) ? 1. : cos(radius - dr);
    }

    i64stack *stk = i64stack_new(2 * (12 + 3 * omax), status, err);
    if (!status)
      return;
    for (int i = 0; i < 12; i++) {
      i64stack_push(stk, (int64_t)(11 - i), status, err);
      if (!status)
        return;
      i64stack_push(stk, 0, status, err);
      if (!status)
        return;
    }

    int stacktop = 0; // a place to save a stack position
    while (stk->size > 0) {
      // pop current pixel number and order from the stack
      int64_t pix = stk->data[stk->size - 2];
      int o = (int)stk->data[stk->size - 1];
      i64stack_resize(stk, stk->size - 2, status, err);
      if (!status)
        return;

      double pix_z, pix_phi;
      pix2zphi(base[o], pix, &pix_z, &pix_phi);
      // cosine of angular distance between pixel center and disk center
      double cangdist = cosdist_zphi(ptg_z, ptg_phi, pix_z, pix_phi);

      if (cangdist > crpdr[o]) {
        int zone = (cangdist < cosrad) ? 1 : ((cangdist <= crmdr[o]) ? 2 : 3);
        check_pixel_nest(o, hpx.order, omax, zone, pixset, pix, stk, inclusive,
                         &stacktop, status, err);
        if (!status)
          return;
      }
    }
  }
}

void xyf2loc(double x, double y, int face, double *z, double *phi, double *sth, bool *have_sth) {
    *have_sth = false;
    double jr = jrll[face] - x - y;
    double nr;
    if (jr<1) {
        nr = jr;
        double tmp = nr*nr/3.;
        *z = 1 - tmp;
        if (*z > 0.99) {
            *sth = sqrt(tmp*(2.0-tmp));
            *have_sth = true;
        }
    } else if (jr>3) {
        nr = 4-jr;
        double tmp = nr*nr/3.;
        *z = tmp - 1;
        if (*z<-0.99) {
            *sth = sqrt(tmp*(2.-tmp));
            *have_sth = true;
        }
    } else {
        nr = 1;
        *z = (2-jr)*2./3.;
    }

    double tmp=jpll[face]*nr+x-y;
    if (tmp<0) tmp+=8;
    if (tmp>=8) tmp-=8;
    *phi = (nr<1e-15) ? 0 : (0.5*M_PI_2*tmp)/nr;
}

void locToVec3(double z, double phi, double sth, bool have_sth, vec3 *vec) {
    if (have_sth) {
        vec->x = sth*cos(phi);
        vec->y = sth*sin(phi);
        vec->z = z;
    } else {
        double sintheta = sqrt((1.0 - z)*(1.0 + z));
        vec->x = sintheta*cos(phi);
        vec->y = sintheta*sin(phi);
        vec->z = z;
    }
}

void locToPtg(double z, double phi, double sth, bool have_sth, ptg *p) {
    p->phi = phi;

    if (have_sth) {
        p->theta = atan2(sth, z);
    } else {
        p->theta = acos(z);
    }
}

/*
void boundaries(healpix_info hpx, int64_t pix, size_t step, vec3arr *out, int *status, char *err) {
    *status = 1;

    if (out->size < 4*step) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Output vector of insufficient size.");
    }

    int ix, iy, face;
    pix2xyf(hpx, pix, &ix, &iy, &face);
    double dc = 0.5/hpx.nside;
    double xc = (ix + 0.5)/hpx.nside;
    double yc = (iy + 0.5)/hpx.nside;
    double d = 1.0/(step*hpx.nside);
    for (size_t i=0; i<step; i++) {
        double z, phi, sth;
        bool have_sth;
        xyf2loc(xc+dc-i*d, yc+dc, face, &z, &phi, &sth, &have_sth);
        locToVec3(z, phi, sth, have_sth, &out[i]);
        xyf2loc(xc-dc, yc+dc-i*d, face, &z, &phi, &sth, &have_sth);
        locToVec3(z, phi, sth, have_sth, &out[i+step]);
        xyf2loc(xc-dc+i*d, yc-dc, face, &z, &phi, &sth, &have_sth);
        locToVec3(z, phi, sth, have_sth, &out[i+2*step]);
        xyf2loc(xc+dc, yc-dc+i*d, face, &z, &phi, &sth, &have_sth);
        locToVec3(z, phi, sth, have_sth, &out[i+3*step]);
    }
}
*/

void boundaries(healpix_info hpx, int64_t pix, size_t step, ptgarr *out, int *status) {
    *status = 1;

    if (out->size < 4*step) {
        *status = 0;
        return;
    }

    int ix, iy, face;
    pix2xyf(hpx, pix, &ix, &iy, &face);
    double dc = 0.5/hpx.nside;
    double xc = (ix + 0.5)/hpx.nside;
    double yc = (iy + 0.5)/hpx.nside;
    double d = 1.0/(step*hpx.nside);
    for (size_t i=0; i<step; i++) {
        double z, phi, sth;
        bool have_sth;
        xyf2loc(xc+dc-i*d, yc+dc, face, &z, &phi, &sth, &have_sth);
        locToPtg(z, phi, sth, have_sth, &out->data[i]);
        xyf2loc(xc-dc, yc+dc-i*d, face, &z, &phi, &sth, &have_sth);
        locToPtg(z, phi, sth, have_sth, &out->data[i+step]);
        xyf2loc(xc-dc+i*d, yc-dc, face, &z, &phi, &sth, &have_sth);
        locToPtg(z, phi, sth, have_sth, &out->data[i+2*step]);
        xyf2loc(xc+dc, yc-dc+i*d, face, &z, &phi, &sth, &have_sth);
        locToPtg(z, phi, sth, have_sth, &out->data[i+3*step]);
    }
}
