/*
 *  This file is modified from Healpix_cxx/healpix_base.cc
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

// add check_nside code

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
  // need isqrt
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

    // what is jrll?
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
