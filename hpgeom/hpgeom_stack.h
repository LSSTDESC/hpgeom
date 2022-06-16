/*
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

#define STACK_PUSH_REALLOC_MULT 1
#define STACK_PUSH_REALLOC_MULTVAL 2
#define STACK_PUSH_INITSIZE 50

struct i64stack {
  size_t size;               // number of elements that are visible to the user
  size_t allocated_size;     // number of allocated elements in data vector
  size_t push_realloc_style; // Currently always STACK_PUSH_REALLOC_MULT,
                             // which is reallocate to
                             // allocated_size*realloc_multval
  size_t
      push_initsize; // default size on first push, default STACK_PUSH_INITSIZE
  double
      realloc_multval; // when allocated size is exceeded while pushing,
                       // reallocate to allocated_size*realloc_multval, default
                       // STACK_PUSH_REALLOC_MULTVAL
                       // if allocated_size was zero, we allocate to
                       // push_initsize
  int64_t *data;
};

struct i64rangeset {
  struct i64stack *stack;
};

struct i64stack *i64stack_new(size_t num, int *status, char *err);
void i64stack_realloc(struct i64stack *stack, size_t newsize, int *status,
                      char *err);
void i64stack_resize(struct i64stack *stack, size_t newsize, int *status,
                     char *err);
void i64stack_clear(struct i64stack *stack);
struct i64stack *i64stack_delete(struct i64stack *stack);
void i64stack_push(struct i64stack *stack, int64_t val, int *status, char *err);

struct i64rangeset *i64rangeset_new(int *status, char *err);
void i64rangeset_append(struct i64rangeset *rangeset, int64_t v1, int64_t v2,
                        int *status, char *err);
void i64rangeset_append_single(struct i64rangeset *rangeset, int64_t v1,
                               int *status, char *err);
void i64rangeset_clear(struct i64rangeset *rangeset, int *status, char *err);
void i64rangeset_append_i64rangeset(struct i64rangeset *rangeset,
                                    struct i64rangeset *other, int *status,
                                    char *err);
struct i64rangeset *i64rangeset_delete(struct i64rangeset *rangeset);
size_t i64rangeset_npix(struct i64rangeset *rangeset);
void i64rangeset_fill_buffer(struct i64rangeset *rangeset, size_t npix,
                             int64_t *buf);

#endif
