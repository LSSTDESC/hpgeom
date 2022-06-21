#ifndef _HEALPIX_GEOM_H
#define _HEALPIX_GEOM_H

#include "hpgeom_stack.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef M_PI
#define M_E 2.7182818284590452354         /* e */
#define M_LOG2E 1.4426950408889634074     /* log_2 e */
#define M_LOG10E 0.43429448190325182765   /* log_10 e */
#define M_LN2 0.69314718055994530942      /* log_e 2 */
#define M_LN10 2.30258509299404568402     /* log_e 10 */
#define M_PI 3.14159265358979323846       /* pi */
#define M_PI_2 1.57079632679489661923     /* pi/2 */
#define M_PI_4 0.78539816339744830962     /* pi/4 */
#define M_1_PI 0.31830988618379067154     /* 1/pi */
#define M_2_PI 0.63661977236758134308     /* 2/pi */
#define M_2_SQRTPI 1.12837916709551257390 /* 2/sqrt(pi) */
#define M_SQRT2 1.41421356237309504880    /* sqrt(2) */
#define M_SQRT1_2 0.70710678118654752440  /* 1/sqrt(2) */

#endif

#define M_TWO_PI 6.28318530717958647693   /* 2*pi */
#define M_TWOTHIRD 0.66666666666666666666 /* 2/3 */

#define D2R 0.017453292519943295
#define R2D 57.295779513082323

#define MAX_ORDER 29
#define MAX_NSIDE (int64_t)(1) << MAX_ORDER

typedef enum Scheme { RING, NEST } Scheme;

typedef struct healpix_info {
  int order;
  int64_t nside;
  int64_t npface;
  int64_t ncap;
  int64_t npix;
  double fact2;
  double fact1;
  Scheme scheme;
} healpix_info;

healpix_info healpix_info_from_order(int order, enum Scheme scheme);
healpix_info healpix_info_from_nside(int64_t nside, enum Scheme scheme);

int64_t isqrt(int64_t i);
int ilog2(int64_t arg);
int64_t imodulo(int64_t v1, int64_t v2);
static inline int64_t i64max(int64_t v1, int64_t v2);
static inline int64_t i64min(int64_t v1, int64_t v2);
static inline int64_t special_div(int64_t a, int64_t b);

int64_t ang2pix(healpix_info hpx, double theta, double phi);
int64_t loc2pix(healpix_info hpx, double z, double phi, double sth,
                bool hav_sth);
int64_t xyf2nest(healpix_info hpx, int ix, int iy, int face_num);
int64_t xyf2ring(healpix_info hpx, int ix, int iy, int face_num);

double ring2z(healpix_info hpx, int64_t ring);
void pix2zphi(healpix_info hpx, int64_t pix, double *z, double *phi);
void pix2xyf(healpix_info hpx, int64_t pix, int *ix, int *iy, int *face_num);
int64_t xyf2pix(healpix_info hpx, int ix, int iy, int face_num);

void pix2ang(healpix_info hpx, int64_t pix, double *theta, double *phi);
void pix2loc(healpix_info hpx, int64_t pix, double *z, double *phi, double *sth,
             bool *have_sth);
void nest2xyf(healpix_info hpx, int64_t pix, int *ix, int *iy, int *face_num);
void ring2xyf(healpix_info hpx, int64_t pix, int *ix, int *iy, int *face_num);
int64_t nest2ring(healpix_info hpx, int64_t pix);
int64_t ring2nest(healpix_info hpx, int64_t pix);

int64_t spread_bits64(int v);
int compress_bits64(int64_t v);

int64_t ring_above(healpix_info hpx, double z);
void get_ring_info_small(healpix_info hpx, int64_t ring, int64_t *startpix,
                         int64_t *ringpix, bool *shifted);

double max_pixrad(healpix_info hpx);
bool check_pixel_ring(healpix_info hpx1, healpix_info hpx2, int64_t pix,
                      int64_t nr, int64_t ipix1, int fct, double cz,
                      double cphi, double cosrp2, int64_t cpix);
void check_pixel_nest(int o, int order_, int omax, int zone,
                      struct i64rangeset *pixset, int64_t pix,
                      struct i64stack *stk, bool inclusive, int *stacktop,
                      int *status, char *err);

void query_disc(healpix_info hpx, double theta, double phi, double radius,
                int fact, struct i64rangeset *pixset, int *status, char *err);

void xyf2loc(double x, double y, int face, double *z, double *phi, double *sth,
             bool *have_sth);
void locToVec3(double z, double phi, double sth, bool have_sth, vec3 *vec);
// void boundaries(healpix_info hpx, int64_t pix, size_t step, vec3 *out);
void boundaries(healpix_info hpx, int64_t pix, size_t step, ptgarr *out,
                int *status);

#endif
