#ifndef _HPGEOM_UTILS_H
#define _HPGEOM_UTILS_H

#include "healpix_geom.h"
#include <stdint.h>

#define ERR_SIZE 256

int check_nside(int64_t nside, Scheme scheme, char *err);
int check_theta_phi(double theta, double phi, char *err);
int check_pixel(healpix_info hpx, int64_t pix, char *err);
int lonlat_to_thetaphi(double lon, double lat, double *theta, double *phi,
                       bool degrees, char *err);
int thetaphi_to_lonlat(double theta, double phi, double *lon, double *lat,
                       bool degrees, char *err);

#endif
