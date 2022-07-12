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

#include <stdio.h>
#include <inttypes.h>

#include "healpix_geom.h"
#include "hpgeom_utils.h"

int hpgeom_check_nside(int64_t nside, Scheme scheme, char *err) {
    if (nside <= 0) {
        snprintf(err, ERR_SIZE, "nside %" PRId64 " must be positive.", nside);
        return 0;
    }

    if (scheme == NEST && (nside & (nside - 1))) {
        snprintf(err, ERR_SIZE, "nside %" PRId64 " must be power of 2 for NEST pixels", nside);
        return 0;
    }

    if (nside > MAX_NSIDE) {
        snprintf(err, ERR_SIZE, "nside %" PRId64 " must not be greater than 2**%d", nside,
                 MAX_ORDER);
        return 0;
    }
    return 1;
}

int hpgeom_check_theta_phi(double theta, double phi, char *err) {
    err[0] = '\0';

    if (theta < 0.0 || theta > HPG_PI) {
        snprintf(err, ERR_SIZE, "colatitude (theta) = %g out of range [0, pi]", theta);
        return 0;
    }
    if (phi < 0 || phi > HPG_TWO_PI) {
        snprintf(err, ERR_SIZE, "longitude (phi) = %g out of range [0, 2*pi]", phi);
        return 0;
    }
    return 1;
}

int hpgeom_check_pixel(healpix_info *hpx, int64_t pix, char *err) {
    err[0] = '\0';

    if (pix < 0 || pix >= hpx->npix) {
        snprintf(err, ERR_SIZE, "Pixel value %" PRId64 " out of range for nside %" PRId64, pix,
                 hpx->nside);
        return 0;
    }
    return 1;
}

int hpgeom_check_fact(healpix_info *hpx, long fact, char *err) {
    err[0] = '\0';

    if (fact <= 0) {
        snprintf(err, ERR_SIZE, "Inclusive factor %ld must be positive.", fact);
        return 0;
    } else if (fact * hpx->nside > MAX_NSIDE) {
        snprintf(err, ERR_SIZE, "Inclusive factor * nside must be <= %" PRId64, MAX_NSIDE);
        return 0;
    } else if ((hpx->scheme == NEST) & ((fact & (fact - 1)) > 0)) {
        snprintf(err, ERR_SIZE, "Inclusive factor %ld must be power of 2 for nest.", fact);
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

int hpgeom_check_semi(double semi_major, double semi_minor, char *err) {
    err[0] = '\0';

    if (semi_major <= 0) {
        snprintf(err, ERR_SIZE, "Semi-major axis must be positive.");
        return 0;
    }
    if (semi_minor <= 0) {
        snprintf(err, ERR_SIZE, "Semi-minor axis must be positive.");
        return 0;
    }
    if (semi_major < semi_minor) {
        snprintf(err, ERR_SIZE, "Semi-major axis must be >= semi-minor axis.");
        return 0;
    }

    return 1;
}

int hpgeom_lonlat_to_thetaphi(double lon, double lat, double *theta, double *phi, bool degrees,
                              char *err) {
    err[0] = '\0';

    if (degrees) {
        lon = fmodulo(lon * HPG_D2R, HPG_TWO_PI);
        lat = lat * HPG_D2R;
    } else {
        lon = fmodulo(lon, HPG_TWO_PI);
    }

    if (lat < -HPG_HALFPI || lat > HPG_HALFPI) {
        if (degrees) {
            snprintf(err, ERR_SIZE, "lat = %g out of range [-90, 90]", lat * HPG_R2D);
        } else {
            snprintf(err, ERR_SIZE, "lat = %g out of range [-pi/2, pi/2]", lat);
        }
        return 0;
    }

    *phi = lon;
    *theta = -lat + HPG_HALFPI;

    return 1;
}

int hpgeom_thetaphi_to_lonlat(double theta, double phi, double *lon, double *lat, bool degrees,
                              bool check_range, char *err) {
    int status = 1;
    err[0] = '\0';

    if (check_range) {
        status = hpgeom_check_theta_phi(theta, phi, err);
    }

    if (!status) return status;

    *lon = phi;
    *lat = -(theta - HPG_HALFPI);

    if (degrees) {
        *lon *= HPG_R2D;
        *lat *= HPG_R2D;
    }

    return status;
}
