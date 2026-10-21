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

#ifndef HPG_FRAMES_H
#define HPG_FRAMES_H

#include <stdbool.h>

/*
 * Fixed-rotation celestial frames.
 *
 * Every frame here is related to ICRS by a constant 3x3 rotation matrix, so
 * great circles, small circles and angular distances are preserved exactly.
 *
 *   HPG_FRAME_EQUATORIAL  ICRS.
 *   HPG_FRAME_GALACTIC    astropy Galactic (IAU 1958 via FK5 J2000).
 *   HPG_FRAME_ECLIPTIC    mean ecliptic and equinox of J2000.0
 *                         (astropy BarycentricMeanEcliptic(equinox="J2000")).
 *                         The "true" ecliptic frames add nutation (~14 arcsec), and
 *                         the geocentric ones also add aberration (~20 arcsec), which
 *                         is not a rotation; neither is wanted for region selection.
 *
 * Matrices were generated with astropy 6.1.7 and agree with it to < 1e-9 arcsec.
 */

typedef enum {
    HPG_FRAME_EQUATORIAL = 0,
    HPG_FRAME_GALACTIC = 1,
    HPG_FRAME_ECLIPTIC = 2,
    HPG_FRAME_NFRAMES
} hpg_frame_t;

/* Parse a frame name (case-insensitive). Returns false if unknown.
   Accepts: "equatorial", "icrs", "c"; "galactic", "g";
            "ecliptic", "barycentricmeanecliptic", "e". */
bool hpg_frame_from_string(const char *name, hpg_frame_t *frame);

/* True if the rotation between the two frames is the identity. */
bool hpg_frame_is_identity(hpg_frame_t from, hpg_frame_t to);

/* R such that v_to = R . v_from. */
void hpg_frame_rotation(hpg_frame_t from, hpg_frame_t to, double R[3][3]);

/* out = R . in (in and out may alias). */
void hpg_rotate_vec(const double R[3][3], const double in[3], double out[3]);

/* Unit vector <-> (theta, phi); theta is colatitude in [0, pi], phi in [0, 2pi). */
void hpg_thetaphi_to_vec(double theta, double phi, double v[3]);
void hpg_vec_to_thetaphi(const double v[3], double *theta, double *phi);

/* Rotate a point given as (theta, phi). */
void hpg_rotate_thetaphi(const double R[3][3], double theta, double phi, double *theta_out,
                         double *phi_out);

/* Rotate a position angle attached to the point (theta, phi).
   pa is measured from local north (decreasing theta) toward local east
   (increasing phi), in radians.  Returns the position angle of the same
   direction at the rotated point, in the output frame, in [0, 2pi).
   Ill-defined exactly at either frame's poles; there the input pa is
   returned unchanged. */
double hpg_rotate_position_angle(const double R[3][3], double theta, double phi, double pa);

#endif
