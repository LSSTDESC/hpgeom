/*
 * Copyright 2022 LSST DESC
 * Author: Erin Sheldon, Eli Rykoff
 *
 * This code is adapted from https://github.com/esheldon/healpix_util
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
#ifndef _HPGEOM_STACK
#define _HPGEOM_STACK

#include <stdint.h>

#define STACK_PUSH_REALLOC_MULT 1
#define STACK_PUSH_REALLOC_MULTVAL 2
#define STACK_PUSH_INITSIZE 50

typedef struct i64stack {
    size_t size;                // number of elements that are visible to the user
    size_t allocated_size;      // number of allocated elements in data vector
    size_t push_realloc_style;  // Currently always STACK_PUSH_REALLOC_MULT,
                                // which is reallocate to
                                // allocated_size*realloc_multval
    size_t push_initsize;       // default size on first push, default STACK_PUSH_INITSIZE
    double realloc_multval;     // when allocated size is exceeded while pushing,
                                // reallocate to allocated_size*realloc_multval, default
                                // STACK_PUSH_REALLOC_MULTVAL
                                // if allocated_size was zero, we allocate to
                                // push_initsize
    int64_t *data;
} i64stack;

typedef struct i64rangeset {
    i64stack *stack;
} i64rangeset;

typedef struct pointing {
    double theta, phi;
} pointing;

typedef struct pointingarr {
    size_t size;
    pointing *data;
} pointingarr;

typedef struct vec3 {
    double x, y, z;
} vec3;

typedef struct vec3arr {
    size_t size;
    vec3 *data;
} vec3arr;

typedef struct dblarr {
    size_t size;
    double *data;
} dblarr;

i64stack *i64stack_new(size_t num, int *status, char *err);
void i64stack_realloc(i64stack *stack, size_t newsize, int *status, char *err);
void i64stack_resize(i64stack *stack, size_t newsize, int *status, char *err);
void i64stack_clear(i64stack *stack);
i64stack *i64stack_delete(i64stack *stack);
void i64stack_push(i64stack *stack, int64_t val, int *status, char *err);
int64_t i64stack_pop(i64stack *stack, int *status, char *err);
void i64stack_pop_pair(i64stack *stack, int64_t *first, int64_t *second, int *status,
                       char *err);
void i64stack_insert(struct i64stack *stack, size_t pos, size_t count, int64_t value,
                     int *status, char *err);
void i64stack_erase(struct i64stack *stack, size_t pos1, size_t pos2, int *status, char *err);

i64rangeset *i64rangeset_new(int *status, char *err);
void i64rangeset_append(i64rangeset *rangeset, int64_t v1, int64_t v2, int *status, char *err);
void i64rangeset_append_single(i64rangeset *rangeset, int64_t v1, int *status, char *err);
void i64rangeset_clear(i64rangeset *rangeset, int *status, char *err);
void i64rangeset_append_i64rangeset(i64rangeset *rangeset, i64rangeset *other, int *status,
                                    char *err);
i64rangeset *i64rangeset_delete(i64rangeset *rangeset);
size_t i64rangeset_npix(i64rangeset *rangeset);
void i64rangeset_fill_buffer(i64rangeset *rangeset, size_t npix, int64_t *buf);
void i64rangeset_remove(i64rangeset *rangeset, int64_t v1, int64_t v2, int *status, char *err);
void i64rangeset_intersect(i64rangeset *rangeset, int64_t v1, int64_t v2, int *status,
                           char *err);

void vec3_crossprod(vec3 *v1, vec3 *v2, vec3 *prod);
double vec3_dotprod(vec3 *v1, vec3 *v2);
double vec3_length(vec3 *v);
void vec3_add(vec3 *v1, vec3 *v2, vec3 *sum);
void vec3_subtract(vec3 *v1, vec3 *v2, vec3 *sum);
void vec3_normalize(vec3 *v);
void vec3_flip(vec3 *v);
void vec3_from_pointing(pointing *p, vec3 *v);
void pointing_from_vec3(vec3 *v, pointing *p);

vec3arr *vec3arr_new(size_t num, int *status, char *err);
vec3arr *vec3arr_delete(vec3arr *arr);

pointingarr *pointingarr_new(size_t num, int *status, char *err);
pointingarr *pointingarr_delete(pointingarr *arr);

dblarr *dblarr_new(size_t num, int *status, char *err);
dblarr *dblarr_delete(dblarr *arr);

#endif
