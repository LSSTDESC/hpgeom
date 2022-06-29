/*
 * This code is adapted from https://github.com/esheldon/healpix_util
 * Copyright 2022 LSST DESC
 * Author: Erin Sheldon, Eli Rykoff
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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hpgeom_stack.h"
#include "hpgeom_utils.h"

struct i64stack *i64stack_new(size_t num, int *status, char *err) {
    *status = 1;
    i64stack *stack = malloc(sizeof(i64stack));
    if (stack == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate i64stack");
        return NULL;
    }

    stack->size = 0;
    stack->allocated_size = num;
    stack->push_realloc_style = STACK_PUSH_REALLOC_MULT;
    stack->push_initsize = STACK_PUSH_INITSIZE;
    stack->realloc_multval = STACK_PUSH_REALLOC_MULTVAL;

    if (num == 0) {
        stack->data = NULL;
    } else {
        stack->data = calloc(num, sizeof(int64_t));
        if (stack->data == NULL) {
            *status = 0;
            snprintf(err, ERR_SIZE, "Could not allocate data in i64stack");
            return NULL;
        }
    }

    return stack;
}

void i64stack_realloc(struct i64stack *stack, size_t newsize, int *status, char *err) {
    *status = 1;

    size_t oldsize = stack->allocated_size;
    if (newsize != oldsize) {
        size_t elsize = sizeof(int64_t);

        int64_t *newdata = realloc(stack->data, newsize * elsize);
        if (newdata == NULL) {
            *status = 0;
            snprintf(err, ERR_SIZE, "Failed to reallocate i64stack");
            return;
        }

        if (newsize > stack->allocated_size) {
            // the allocated size is larger.  make sure to initialize the new
            // memory region.  This is the area starting from index [oldsize]
            size_t num_new_bytes = (newsize - oldsize) * elsize;
            memset(&newdata[oldsize], 0, num_new_bytes);
        } else if (stack->size > newsize) {
            // The viewed size is larger than the allocated size in this case,
            // we must set the size to the maximum it can be, which is the
            // allocated size
            stack->size = newsize;
        }

        stack->data = newdata;
        stack->allocated_size = newsize;
    }
}

void i64stack_resize(struct i64stack *stack, size_t newsize, int *status, char *err) {
    *status = 1;
    if (newsize > stack->allocated_size) {
        i64stack_realloc(stack, newsize, status, err);
        if (!status) {
            return;
        }
    }

    stack->size = newsize;
}

void i64stack_insert(struct i64stack *stack, size_t pos, size_t count, int64_t value,
                     int *status, char *err) {
    // insert count copies of value at position pos
    *status = 1;
    size_t nbytes_move = (stack->size - pos) * sizeof(int64_t);

    i64stack_resize(stack, stack->size + count, status, err);
    if (!*status) return;

    memmove(&stack->data[pos + count], &stack->data[pos], nbytes_move);

    for (size_t i = 0; i < count; i++) {
        stack->data[pos + i] = value;
    }
}

void i64stack_erase(struct i64stack *stack, size_t pos1, size_t pos2, int *status, char *err) {
    // erase elements from [pos1, pos2)
    *status = 1;
    size_t nbytes_move = (stack->size - pos2) * sizeof(int64_t);

    memmove(&stack->data[pos1], &stack->data[pos2], nbytes_move);

    i64stack_resize(stack, stack->size - (pos2 - pos1), status, err);
    if (!*status) return;
}

int64_t i64stack_pop(i64stack *stack, int *status, char *err) {
    *status = 1;

    if (stack->size < 1) {
        snprintf(err, ERR_SIZE, "Cannot pop from empty stack.");
        *status = 0;
        return -1;
    }

    int64_t retval = stack->data[stack->size - 1];
    i64stack_resize(stack, stack->size - 1, status, err);
    if (!*status) return -1;
    return retval;
}

void i64stack_pop_pair(i64stack *stack, int64_t *first, int64_t *second, int *status,
                       char *err) {
    *status = 1;

    if (stack->size < 2) {
        snprintf(err, ERR_SIZE, "Cannot pop pair from stack with <2 elements.");
        *status = 0;
        return;
    }

    *first = stack->data[stack->size - 2];
    *second = stack->data[stack->size - 1];

    i64stack_resize(stack, stack->size - 2, status, err);
}

void i64stack_clear(struct i64stack *stack) {
    stack->size = 0;
    stack->allocated_size = 0;
    if (stack->data != NULL) {
        free(stack->data);
        stack->data = NULL;
    }
}

struct i64stack *i64stack_delete(struct i64stack *stack) {
    if (stack != NULL) {
        i64stack_clear(stack);
        free(stack);
    }
    return NULL;
}

void i64stack_push(struct i64stack *stack, int64_t val, int *status, char *err) {
    *status = 1;
    // see if we have already filled the available data vector
    // if so, reallocate to larger storage
    if (stack->size == stack->allocated_size) {
        size_t newsize;
        if (stack->allocated_size == 0) {
            newsize = stack->push_initsize;
        } else {
            // currenly we always use the multiplier reallocation  method.
            if (stack->push_realloc_style != STACK_PUSH_REALLOC_MULT) {
                *status = 0;
                snprintf(err, ERR_SIZE,
                         "Currently only support push realloc style "
                         "STACK_PUSH_REALLOC_MULT");
                return;
            }
            // this will "floor" the size
            newsize = (size_t)(stack->allocated_size * stack->realloc_multval);
            // we want ceiling
            newsize++;
        }

        i64stack_realloc(stack, newsize, status, err);
        if (!status) {
            return;
        }
    }

    stack->size++;
    stack->data[stack->size - 1] = val;
}

struct i64rangeset *i64rangeset_new(int *status, char *err) {
    *status = 1;
    struct i64rangeset *rangeset = malloc(sizeof(struct i64rangeset));
    if (rangeset == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate struct i64rangeset");
        return NULL;
    }

    rangeset->stack = i64stack_new(0, status, err);
    if (!status) {
        free(rangeset);
        return NULL;
    }

    return rangeset;
}

void i64rangeset_append(struct i64rangeset *rangeset, int64_t v1, int64_t v2, int *status,
                        char *err) {
    *status = 1;

    if (v2 <= v1) {
        return;
    }

    if ((rangeset->stack->size > 0) &&
        (v1 <= rangeset->stack->data[rangeset->stack->size - 1])) {
        if (v2 > rangeset->stack->data[rangeset->stack->size - 1]) {
            rangeset->stack->data[rangeset->stack->size - 1] = v2;
        }
    } else {
        i64stack_push(rangeset->stack, v1, status, err);
        if (!status) return;
        i64stack_push(rangeset->stack, v2, status, err);
        if (!status) return;
    }
}

void i64rangeset_append_single(struct i64rangeset *rangeset, int64_t v1, int *status,
                               char *err) {
    i64rangeset_append(rangeset, v1, v1 + 1, status, err);
}

static ptrdiff_t iiv(i64rangeset *rangeset, int64_t val) {
    size_t mid;
    size_t low = 0;
    size_t high = rangeset->stack->size;

    while (low < high) {
        mid = low + (high - low) / 2;
        if (val >= rangeset->stack->data[mid]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return (ptrdiff_t)(low - 0) - 1;
}

void i64rangeset_remove(i64rangeset *rangeset, int64_t v1, int64_t v2, int *status,
                        char *err) {
    if (v2 <= v1) return;
    if (rangeset->stack->size == 0) return;
    if ((v2 <= rangeset->stack->data[0]) ||
        (v1 >= rangeset->stack->data[rangeset->stack->size - 1]))
        return;
    if ((v1 <= rangeset->stack->data[0]) &&
        (v2 >= rangeset->stack->data[rangeset->stack->size - 1])) {
        rangeset->stack->size = 0;
    }
    // addRemove(v1, v2, 0);
    ptrdiff_t v = 0;
    ptrdiff_t pos1 = iiv(rangeset, v1);
    ptrdiff_t pos2 = iiv(rangeset, v2);
    if ((pos1 >= 0) && (rangeset->stack->data[pos1] == v1)) --pos1;
    // first to delete is at pos1+1; last is at pos2
    bool insert_v1 = (pos1 & 1) == v;
    bool insert_v2 = (pos2 & 1) == v;
    ptrdiff_t rmstart = pos1 + 1 + (insert_v1 ? 1 : 0);
    ptrdiff_t rmend = pos2 - (insert_v2 ? 1 : 0);

    if (insert_v1 && insert_v2 && (pos1 + 1 > pos2)) {  // insert
        i64stack_insert(rangeset->stack, pos1 + 1, 2, v1, status, err);
        if (!*status) return;
        rangeset->stack->data[pos1 + 2] = v2;

    } else {  // erase
        if (insert_v1) rangeset->stack->data[pos1 + 1] = v1;
        if (insert_v2) rangeset->stack->data[pos2] = v2;
        i64stack_erase(rangeset->stack, rmstart, rmend + 1, status, err);
        if (!*status) return;
    }
}

void i64rangeset_intersect(i64rangeset *rangeset, int64_t a, int64_t b, int *status,
                           char *err) {
    // Remove all values not within [a, b) from the rangeset.
    *status = 1;
    if (rangeset->stack->size == 0) return;  // nothing to remove
    if ((b <= rangeset->stack->data[0]) ||
        (a >= rangeset->stack->data[rangeset->stack->size - 1])) {
        // no overlap
        rangeset->stack->size = 0;
        return;
    }
    if ((a <= rangeset->stack->data[0]) &&
        (b >= rangeset->stack->data[rangeset->stack->size - 1])) {
        // full rangeset in interval
        return;
    }

    ptrdiff_t pos2 = iiv(rangeset, b);
    if ((pos2 >= 0) && (rangeset->stack->data[pos2] == b)) --pos2;
    bool insert_b = (pos2 & 1) == 0;
    i64stack_erase(rangeset->stack, pos2 + 1, rangeset->stack->size, status, err);
    if (insert_b) {
        i64stack_push(rangeset->stack, b, status, err);
        if (!*status) return;
    }

    ptrdiff_t pos1 = iiv(rangeset, a);
    bool insert_a = (pos1 & 1) == 0;
    if (insert_a) rangeset->stack->data[pos1--] = a;
    if (pos1 >= 0) {
        i64stack_erase(rangeset->stack, 0, pos1 + 1, status, err);
        if (!*status) return;
    }
}

void i64rangeset_clear(struct i64rangeset *rangeset, int *status, char *err) {
    i64stack_clear(rangeset->stack);
}

void i64rangeset_append_i64rangeset(struct i64rangeset *rangeset, struct i64rangeset *other,
                                    int *status, char *err) {
    for (size_t j = 0; j < other->stack->size; j += 2) {
        i64rangeset_append(rangeset, other->stack->data[j], other->stack->data[j + 1], status,
                           err);
        if (!status) return;
    }
}

struct i64rangeset *i64rangeset_delete(struct i64rangeset *rangeset) {
    if (rangeset != NULL) {
        rangeset->stack = i64stack_delete(rangeset->stack);
        free(rangeset);
    }
    return NULL;
}

size_t i64rangeset_npix(struct i64rangeset *rangeset) {
    int64_t npix = 0;

    for (size_t j = 0; j < rangeset->stack->size; j += 2) {
        npix += (rangeset->stack->data[j + 1] - rangeset->stack->data[j]);
    }

    return npix;
}

void i64rangeset_fill_buffer(struct i64rangeset *rangeset, size_t npix, int64_t *buf) {
    size_t counter = 0;

    for (size_t j = 0; j < rangeset->stack->size; j += 2) {
        for (int64_t pix = rangeset->stack->data[j]; pix < rangeset->stack->data[j + 1];
             pix++) {
            buf[counter++] = pix;
        }
    }
}

void vec3_crossprod(vec3 *v1, vec3 *v2, vec3 *prod) {
    prod->x = v1->y * v2->z - v1->z * v2->y;
    prod->y = v1->z * v2->x - v1->x * v2->z;
    prod->z = v1->x * v2->y - v1->y * v2->x;
}

double vec3_dotprod(vec3 *v1, vec3 *v2) {
    return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

double vec3_length(vec3 *v) { return sqrt(v->x * v->x + v->y * v->y + v->z * v->z); }

void vec3_add(vec3 *v1, vec3 *v2, vec3 *sum) {
    sum->x = v1->x + v2->x;
    sum->y = v1->y + v2->y;
    sum->z = v1->z + v2->z;
}

void vec3_subtract(vec3 *v1, vec3 *v2, vec3 *sub) {
    sub->x = v1->x - v2->x;
    sub->y = v1->y - v2->y;
    sub->z = v1->z - v2->z;
}

void vec3_normalize(vec3 *v) {
    double norm = 1. / vec3_length(v);
    v->x *= norm;
    v->y *= norm;
    v->z *= norm;
}

void vec3_flip(vec3 *v) {
    v->x *= -1;
    v->y *= -1;
    v->z *= -1;
}

void vec3_from_pointing(pointing *p, vec3 *v) {
    double st = sin(p->theta);
    v->x = st * cos(p->phi);
    v->y = st * sin(p->phi);
    v->z = cos(p->theta);
}

static inline double safe_atan2(double y, double x) {
    return ((x == 0.) && (y == 0.)) ? 0.0 : atan2(y, x);
}

void pointing_from_vec3(vec3 *v, pointing *p) {
    p->theta = atan2(sqrt(v->x * v->x + v->y * v->y), v->z);
    p->phi = safe_atan2(v->y, v->x);
    if (p->phi < 0.) p->phi += HPG_TWO_PI;
}

vec3arr *vec3arr_new(size_t num, int *status, char *err) {
    *status = 1;
    vec3arr *arr = malloc(sizeof(vec3arr));
    if (arr == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate vec3arr");
        return NULL;
    }

    arr->size = num;
    arr->data = calloc(num, sizeof(vec3));
    if (arr->data == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate data in vec3arr");
        return NULL;
    }

    return arr;
}

vec3arr *vec3arr_delete(vec3arr *arr) {
    if (arr != NULL) {
        arr->size = 0;
        if (arr->data != NULL) {
            free(arr->data);
            arr->data = NULL;
        }
        free(arr);
    }
    return NULL;
}

pointingarr *pointingarr_new(size_t num, int *status, char *err) {
    *status = 1;
    pointingarr *arr = malloc(sizeof(pointingarr));
    if (arr == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate pointingarr");
        return NULL;
    }

    arr->size = num;
    arr->data = calloc(num, sizeof(vec3));
    if (arr->data == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate data in pointingarr");
        return NULL;
    }

    return arr;
}

pointingarr *pointingarr_delete(pointingarr *arr) {
    if (arr != NULL) {
        arr->size = 0;
        if (arr->data != NULL) {
            free(arr->data);
            arr->data = NULL;
        }
        free(arr);
    }
    return NULL;
}

dblarr *dblarr_new(size_t num, int *status, char *err) {
    *status = 1;
    dblarr *arr = malloc(sizeof(dblarr));
    if (arr == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate dblarr");
        return NULL;
    }

    arr->size = num;
    arr->data = calloc(num, sizeof(double));
    if (arr->data == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate data in dblarr");
        return NULL;
    }

    return arr;
}

dblarr *dblarr_delete(dblarr *arr) {
    if (arr != NULL) {
        arr->size = 0;
        if (arr->data != NULL) {
            free(arr->data);
            arr->data = NULL;
        }
        free(arr);
    }
    return NULL;
}
