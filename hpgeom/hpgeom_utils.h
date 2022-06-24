#ifndef _HPGEOM_UTILS_H
#define _HPGEOM_UTILS_H

#include "healpix_geom.h"
#include <stdint.h>

#define ERR_SIZE 256

int hpgeom_check_nside(int64_t nside, Scheme scheme, char *err);
int hpgeom_check_theta_phi(double theta, double phi, char *err);
int hpgeom_check_pixel(healpix_info *hpx, int64_t pix, char *err);
int hpgeom_lonlat_to_thetaphi(double lon, double lat, double *theta, double *phi, bool degrees,
                              char *err);
int hpgeom_thetaphi_to_lonlat(double theta, double phi, double *lon, double *lat, bool degrees,
                              char *err);
int hpgeom_check_fact(healpix_info *hpx, long fact, char *err);
int hpgeom_check_radius(double radius, char *err);

#endif
