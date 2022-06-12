#include <stdio.h>

#include "hpgeom_utils.h"


int check_nside(int64_t nside, Scheme scheme, char *err) {
    if (nside <= 0) {
        snprintf(err, ERR_SIZE, "nside %lld must be positive.", nside);
        return 0;
    }

    if (scheme == NEST && (nside&(nside-1))) {
        snprintf(err, ERR_SIZE, "nside %lld must be power of 2 for NEST pixels", nside);
        return 0;
    }

    if (nside > MAX_NSIDE) {
        snprintf(err, ERR_SIZE, "nside %lld must not be greater than 2**%d", nside, MAX_ORDER);
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


int lonlat_to_thetaphi(double lon, double lat, double *theta, double *phi, bool degrees, char *err) {
    int status = 0;

    err[0] = '\0';

    if (degrees) {
        lon = fmod(lon*D2R, M_TWO_PI);
        lat = lat*D2R;
    } else {
        lon = fmod(lon, M_TWO_PI);
    }

    if (lat < -M_PI || lat > M_PI) {
        /* maybe improve this message depending on degrees...*/
        snprintf(err, ERR_SIZE, "lat = %g out of range [-90, 90]", lat);
        status = 0;
        return status;
    }

    *phi = lon;
    *theta = -lat + M_PI_2;

    status = 1;

    return status;
}
