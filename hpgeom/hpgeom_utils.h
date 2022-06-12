#ifndef _HPGEOM_UTILS_H
#define _HPGEOM_UTILS_H

#include <stdint.h>
#include "healpix_geom.h"

#define ERR_SIZE 256


int check_nside(int64_t nside, Scheme scheme, char *err);

int lonlat_to_thetaphi(double lon, double lat, double *theta, double *phi, bool degrees, char *err);

#endif
