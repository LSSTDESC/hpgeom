#include <stdio.h>

#include "healpix_geom.h"
#include "hpgeom_utils.h"

int hpgeom_check_nside(int64_t nside, Scheme scheme, char *err) {
  if (nside <= 0) {
    snprintf(err, ERR_SIZE, "nside %lld must be positive.", nside);
    return 0;
  }

  if (scheme == NEST && (nside & (nside - 1))) {
    snprintf(err, ERR_SIZE, "nside %lld must be power of 2 for NEST pixels",
             nside);
    return 0;
  }

  if (nside > MAX_NSIDE) {
    snprintf(err, ERR_SIZE, "nside %lld must not be greater than 2**%d", nside,
             MAX_ORDER);
    return 0;
  }
  return 1;
}

int hpgeom_check_theta_phi(double theta, double phi, char *err) {
  err[0] = '\0';

  if (theta < 0.0 || theta > M_PI) {
    snprintf(err, ERR_SIZE, "colatitude (theta) = %g out of range [0, pi]",
             theta);
    return 0;
  }
  if (phi < 0 || phi > M_TWO_PI) {
    snprintf(err, ERR_SIZE, "longitude (phi) = %g out of range [0, 2*pi]", phi);
    return 0;
  }
  return 1;
}

int hpgeom_check_pixel(healpix_info *hpx, int64_t pix, char *err) {
  err[0] = '\0';

  if (pix < 0 || pix >= hpx->npix) {
    snprintf(err, ERR_SIZE, "Pixel value %lld out of range for nside %lld", pix,
             hpx->nside);
    return 0;
  }
  return 1;
}

int hpgeom_check_fact(healpix_info *hpx, long fact, char *err) {
  err[0] = '\0';

  if (fact <= 0) {
    snprintf(err, ERR_SIZE, "Inclusive factor %ld must be >= 0.", fact);
    return 0;
  } else if (fact * hpx->nside > MAX_NSIDE) {
    snprintf(err, ERR_SIZE, "Inclusive factor * nside must be <= %lld",
             MAX_NSIDE);
    return 0;
  } else if ((hpx->scheme == NEST) & ((fact & (fact - 1)) > 0)) {
    snprintf(err, ERR_SIZE, "Inclusive factor %ld must be power of 2 for nest.",
             fact);
    return 0;
  }
  return 1;
}

int hpgeom_check_radius(double radius, char *err) {
  err[0] = '\0';

  if (radius <= 0) {
    snprintf(err, ERR_SIZE, "Radius must be positive.");
    return 0;
  }

  return 1;
}

int hpgeom_lonlat_to_thetaphi(double lon, double lat, double *theta,
                              double *phi, bool degrees, char *err) {
  err[0] = '\0';

  if (degrees) {
    lon = fmod(lon * D2R, M_TWO_PI);
    lat = lat * D2R;
  } else {
    lon = fmod(lon, M_TWO_PI);
  }

  if (lat < -M_PI_2 || lat > M_PI_2) {
    /* maybe improve this message depending on degrees...*/
    snprintf(err, ERR_SIZE, "lat = %g out of range [-90, 90]", lat);
    return 0;
  }

  *phi = lon;
  *theta = -lat + M_PI_2;

  return 1;
}

int hpgeom_thetaphi_to_lonlat(double theta, double phi, double *lon,
                              double *lat, bool degrees, char *err) {
  err[0] = '\0';

  *lon = phi;
  *lat = -(theta - M_PI_2);

  if (degrees) {
    *lon *= R2D;
    *lat *= R2D;
  }

  return 1;
}
