/*
 *  Copyright (C) 2022 LSST DESC
 *  Author: Eli Rykoff
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

// Generated with assistance from claude.

#include <math.h>
#include <ctype.h>
#include <string.h>

#include "frames.h"
#include "healpix_geom.h"


/* M[f] rotates ICRS -> frame f:  v_f = M[f] . v_icrs */
static const double M_ICRS_TO[HPG_FRAME_NFRAMES][3][3] = {
    /* equatorial (ICRS) */
    {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
    /* galactic */
    {{-0.054875657712591702, -0.87343705195561616, -0.48383507361671552},
     {0.49410943719272743, -0.44482972122329534, 0.74698218398666771},
     {-0.86766613755965871, -0.19807633727300067, 0.45598381368730168}},
    /* mean ecliptic & equinox of J2000 */
    {{0.99999999999999423, -7.0783689548483258e-08, 8.0562139837364217e-08},
     {3.2897004077419666e-08, 0.91748212991495848, 0.39777699944404804},
     {-1.0207044725484358e-07, -0.3977769994440431, 0.91748212991495559}},
};

static bool streq_nocase(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return false;
        a++;
        b++;
    }
    return *a == *b;
}

bool hpg_frame_from_string(const char *name, hpg_frame_t *frame) {
    if (name == NULL) return false;
    if (streq_nocase(name, "equatorial") || streq_nocase(name, "icrs") ||
        streq_nocase(name, "c")) {
        *frame = HPG_FRAME_EQUATORIAL;
    } else if (streq_nocase(name, "galactic") || streq_nocase(name, "g")) {
        *frame = HPG_FRAME_GALACTIC;
    } else if (streq_nocase(name, "ecliptic") || streq_nocase(name, "barycentricmeanecliptic") ||
               streq_nocase(name, "e")) {
        *frame = HPG_FRAME_ECLIPTIC;
    } else {
        return false;
    }
    return true;
}

bool hpg_frame_is_identity(hpg_frame_t from, hpg_frame_t to) { return from == to; }

void hpg_frame_rotation(hpg_frame_t from, hpg_frame_t to, double R[3][3]) {
    /* R = M[to] . M[from]^T */
    const double(*A)[3] = M_ICRS_TO[to];
    const double(*B)[3] = M_ICRS_TO[from];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j] = A[i][0] * B[j][0] + A[i][1] * B[j][1] + A[i][2] * B[j][2];
        }
    }
}

void hpg_rotate_vec(const double R[3][3], const double in[3], double out[3]) {
    double x = in[0], y = in[1], z = in[2];
    out[0] = R[0][0] * x + R[0][1] * y + R[0][2] * z;
    out[1] = R[1][0] * x + R[1][1] * y + R[1][2] * z;
    out[2] = R[2][0] * x + R[2][1] * y + R[2][2] * z;
}

void hpg_thetaphi_to_vec(double theta, double phi, double v[3]) {
    double st = sin(theta);
    v[0] = st * cos(phi);
    v[1] = st * sin(phi);
    v[2] = cos(theta);
}

void hpg_vec_to_thetaphi(const double v[3], double *theta, double *phi) {
    *theta = atan2(sqrt(v[0] * v[0] + v[1] * v[1]), v[2]);
    double p = atan2(v[1], v[0]);
    if (p < 0.0) p += HPG_TWO_PI;
    if (p >= HPG_TWO_PI) p -= HPG_TWO_PI;
    *phi = p;
}

void hpg_rotate_thetaphi(const double R[3][3], double theta, double phi, double *theta_out,
                         double *phi_out) {
    double v[3];
    hpg_thetaphi_to_vec(theta, phi, v);
    hpg_rotate_vec(R, v, v);
    hpg_vec_to_thetaphi(v, theta_out, phi_out);
}

/* Local north (decreasing theta) and east (increasing phi) unit tangents. */
static void local_basis(double theta, double phi, double north[3], double east[3]) {
    double st = sin(theta), ct = cos(theta), sp = sin(phi), cp = cos(phi);
    north[0] = -ct * cp;
    north[1] = -ct * sp;
    north[2] = st;
    east[0] = -sp;
    east[1] = cp;
    east[2] = 0.0;
}

double hpg_rotate_position_angle(const double R[3][3], double theta, double phi, double pa) {
    const double tiny = 1e-12;
    double theta_out, phi_out;
    hpg_rotate_thetaphi(R, theta, phi, &theta_out, &phi_out);
    if (sin(theta) < tiny || sin(theta_out) < tiny) return pa;

    double n_in[3], e_in[3], n_out[3], e_out[3], d[3];
    local_basis(theta, phi, n_in, e_in);
    local_basis(theta_out, phi_out, n_out, e_out);

    /* Direction with position angle pa in the input frame, rotated. */
    double cpa = cos(pa), spa = sin(pa);
    for (int i = 0; i < 3; i++) d[i] = cpa * n_in[i] + spa * e_in[i];
    hpg_rotate_vec(R, d, d);

    double pa_out = atan2(d[0] * e_out[0] + d[1] * e_out[1] + d[2] * e_out[2],
                          d[0] * n_out[0] + d[1] * n_out[1] + d[2] * n_out[2]);
    if (pa_out < 0.0) pa_out += HPG_TWO_PI;
    if (pa_out >= HPG_TWO_PI) pa_out -= HPG_TWO_PI;
    return pa_out;
}
