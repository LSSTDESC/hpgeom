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

void i64stack_realloc(struct i64stack *stack, size_t newsize, int *status,
                      char *err) {
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

void i64stack_resize(struct i64stack *stack, size_t newsize, int *status,
                     char *err) {
  *status = 1;
  if (newsize > stack->allocated_size) {
    i64stack_realloc(stack, newsize, status, err);
    if (!status) {
      return;
    }
  }

  stack->size = newsize;
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

void i64stack_push(struct i64stack *stack, int64_t val, int *status,
                   char *err) {
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

void i64rangeset_append(struct i64rangeset *rangeset, int64_t v1, int64_t v2,
                        int *status, char *err) {
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
    if (!status)
      return;
    i64stack_push(rangeset->stack, v2, status, err);
    if (!status)
      return;
  }
}

void i64rangeset_append_single(struct i64rangeset *rangeset, int64_t v1,
                               int *status, char *err) {
  i64rangeset_append(rangeset, v1, v1 + 1, status, err);
}

void i64rangeset_clear(struct i64rangeset *rangeset, int *status, char *err) {
  i64stack_clear(rangeset->stack);
}

void i64rangeset_append_i64rangeset(struct i64rangeset *rangeset,
                                    struct i64rangeset *other, int *status,
                                    char *err) {
  for (size_t j = 0; j < other->stack->size; j += 2) {
    i64rangeset_append(rangeset, other->stack->data[j],
                       other->stack->data[j + 1], status, err);
    if (!status)
      return;
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

void i64rangeset_fill_buffer(struct i64rangeset *rangeset, size_t npix,
                             int64_t *buf) {
  size_t counter = 0;

  for (size_t j = 0; j < rangeset->stack->size; j += 2) {
    for (int64_t pix = rangeset->stack->data[j];
         pix < rangeset->stack->data[j + 1]; pix++) {
      buf[counter++] = pix;
    }
  }
}

void vec3_crossprod(vec3 *v1, vec3 *v2, vec3 *prod) {
    prod->x = v1->y*v2->z - v1->z*v2->y;
    prod->y = v1->z*v2->x - v1->x*v2->z;
    prod->z = v1->x*v2->y - v1->y*v2->x;
}

double vec3_dotprod(vec3 *v1, vec3 *v2) {
    return v1->x*v2->x + v1->y*v2->y + v1->z*v2->z;
}

double vec3_length(vec3 *v) {
    return sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
}

void vec3_add(vec3 *v1, vec3 *v2, vec3 *sum) {
    sum->x = v1->x + v2->x;
    sum->y = v1->y + v2->y;
    sum->z = v1->z + v2->z;
}

void vec3_normalize(vec3 *v) {
    double norm = 1./vec3_length(v);
    v->x *= norm;
    v->y *= norm;
    v->z *= norm;
}

void vec3_flip(vec3 *v) {
    v->x *= -1;
    v->y *= -1;
    v->z *= -1;
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

ptgarr *ptgarr_new(size_t num, int *status, char *err) {
    *status = 1;
    ptgarr *arr = malloc(sizeof(ptgarr));
    if (arr == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate ptgarr");
        return NULL;
    }

    arr->size = num;
    arr->data = calloc(num, sizeof(vec3));
    if (arr->data == NULL) {
        *status = 0;
        snprintf(err, ERR_SIZE, "Could not allocate data in ptgarr");
        return NULL;
    }

    return arr;
}

ptgarr *ptgarr_delete(ptgarr *arr) {
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

/*
struct i64stack* i64rangeset_full_stack(struct i64rangeset *rangeset, int
*status, char *err) { long npix = 0;

    for (size_t j = 0; j < rangeset->stack->size; j+=2) {
        npix += (rangeset->stack->data[j + 1] - rangeset->stack->data[j]);
    }

    struct i64stack *full_stack = i64stack_new(npix, status, err);
    if (!status) {
        return NULL;
    }

    for (size_t j = 0; j < rangeset->stack->size; j+=2) {
        for (int64_t pix=rangeset->stack->data[j]; pix <
rangeset->stack->data[j+1]; pix++) { i64stack_push(full_stack, pix, status,
err);
        }
    }

    return full_stack;
}
*/
