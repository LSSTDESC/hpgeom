#include <stdio.h>

#include "hpgeom_utils.h"

int check_nside(int64_t nside, Scheme scheme, char *err) {
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

int check_theta_phi(double theta, double phi, char *err) {
  err[0] = '\0';

  if (theta < 0.0 || theta > M_TWO_PI) {
    snprintf(err, ERR_SIZE, "longitude = %g out of range [0, 2*pi]", theta);
    return 0;
  }
  if (phi < 0 || phi > M_TWO_PI) {
    snprintf(err, ERR_SIZE, "colatitude = %g out of range [0, 2*pi]", phi);
    return 0;
  }
  return 1;
}

int check_pixel(healpix_info hpx, int64_t pix, char *err) {
  err[0] = '\0';

  if (pix < 0 || pix >= hpx.npix) {
    snprintf(err, ERR_SIZE, "Pixel value %lld out of range for nside %lld", pix,
             hpx.nside);
    return 0;
  }
  return 1;
}

int lonlat_to_thetaphi(double lon, double lat, double *theta, double *phi,
                       bool degrees, char *err) {
  err[0] = '\0';

  if (degrees) {
    lon = fmod(lon * D2R, M_TWO_PI);
    lat = lat * D2R;
  } else {
    lon = fmod(lon, M_TWO_PI);
  }

  if (lat < -M_PI || lat > M_PI) {
    /* maybe improve this message depending on degrees...*/
    snprintf(err, ERR_SIZE, "lat = %g out of range [-90, 90]", lat);
    return 0;
  }

  *phi = lon;
  *theta = -lat + M_PI_2;

  return 1;
}

int thetaphi_to_lonlat(double theta, double phi, double *lon, double *lat,
                       bool degrees, char *err) {
  err[0] = '\0';

  *lon = phi;
  *lat = -(theta - M_PI_2);

  if (degrees) {
    *lon *= R2D;
    *lat *= R2D;
  }

  return 1;
}
