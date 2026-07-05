#ifndef MATRIX_H
#define MATRIX_H
#include "arena.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct matrix_s matrix_t;

#ifndef USE_DOUBLE
typedef float mat_data_t;
#define MAT_POW(x, y) powf((x), (y))
#define MAT_EXP(x) expf(x)
#define MAT_SQRT(x) sqrtf((x))
#define MAT_LOG(x) logf((x))
#define MAT_FMT "%.6f"
#else
typedef double mat_data_t;
#define MAT_POW(x, y) pow((x), (y))
#define MAT_EXP(x) exp(x)
#define MAT_SQRT(x) sqrt((x))
#define MAT_LOG(x) log((x))
#define MAT_FMT "%.8lf"
#endif

typedef enum {
    MATRIX_OK = 0,
    MATRIX_ERR_DIM_MISMATCH = -1,
    MATRIX_ERR_INVALID_ARG = -2,
    MATRIX_ERR_MATH = -3,
    MATRIX_ERR_ARENA = -4,
    MATRIX_ERR_NULL_PTR = -5
} mat_err_t;

#ifndef NDEBUG
#define MAT_REQUIRE(cond, err_code)                                                 \
    do {                                                                            \
        if (!(cond))                                                                \
            return err_code;                                                        \
    } while (0)
#define MAT_CHECK(expr, label)                                                      \
    do {                                                                            \
        if ((expr) != MATRIX_OK)                                                    \
            goto label;                                                             \
    } while (0)
#else
#define MAT_REQUIRE(cond, err_code) (void)(cond)
#define MAT_CHECK(expr, label) (void)(expr)
#endif

#define MAT_VALID_INTERNAL(mat)                                                     \
    ((mat).data != NULL && (mat).rows > 0 && (mat).cols > 0)
#define MAT_SAME_DIM2_INTERNAL(mat1, mat2)                                          \
    ((mat1).rows == (mat2).rows && (mat1).cols == (mat2).cols)

#define MAT_PTR_NO_NULL(mat, err_code) MAT_REQUIRE(PTR_NO_NULL(mat), err_code)
#define MAT_PTR_IS_NULL(mat, err_code) MAT_REQUIRE(PTR_IS_NULL(mat), err_code)
#define MAT_VALID(mat) MAT_REQUIRE(MAT_VALID_INTERNAL(mat), MATRIX_ERR_INVALID_ARG)

#define MAT_PTR_VALID(mat)                                                          \
    MAT_REQUIRE(PTR_NO_NULL(mat) && MAT_VALID_INTERNAL(*(mat)),                     \
                MATRIX_ERR_INVALID_ARG)

#define MAT_SAME_DIM2(mat1, mat2)                                                   \
    MAT_REQUIRE(MAT_SAME_DIM2_INTERNAL(mat1, mat2), MATRIX_ERR_INVALID_ARG)

#define MAT_PTR_SAME_DIM2(mat1, mat2)                                               \
    MAT_REQUIRE(MAT_SAME_DIM2_INTERNAL(*(mat1), *(mat2)), MATRIX_ERR_INVALID_ARG)

#define MAT_PTR_SAME_DIM3(mat1, mat2, mat3)                                         \
    MAT_REQUIRE(MAT_SAME_DIM2_INTERNAL(*(mat1), *(mat2)) &&                         \
                    MAT_SAME_DIM2_INTERNAL(*(mat2), *(mat3)) &&                     \
                    MAT_SAME_DIM2_INTERNAL(*(mat1), *(mat3)),                       \
                MATRIX_ERR_INVALID_ARG)

struct matrix_s {
    mat_data_t *data;
    uint64_t rows;
    uint64_t cols;
};

#define MAT_AT(mat, r, c) ((mat).data[(r) * ((mat).cols) + (c)])
#define MAT_PTR_AT(mat, r, c) (MAT_AT(*(mat), r, c))

mat_err_t mat_init(arena_t *arena, const uint64_t, const uint64_t, matrix_t *);
void mat_free(matrix_t *);
mat_err_t mat_rand(matrix_t *);
mat_err_t mat_add(matrix_t *, const matrix_t *, const matrix_t *);
mat_err_t mat_scale_add(matrix_t *, const matrix_t *, const matrix_t *, mat_data_t);
mat_err_t mat_sub(matrix_t *, const matrix_t *, const matrix_t *);
mat_err_t mat_mul(matrix_t *, const matrix_t *, const matrix_t *);
mat_err_t mat_transpose(matrix_t *, const matrix_t *);
mat_err_t mat_copy(matrix_t *, const matrix_t *);
mat_err_t mat_slice(matrix_t *, const matrix_t *, const uint64_t, const uint64_t);
mat_err_t mat_row_view(matrix_t *, const matrix_t *, const uint64_t, const uint64_t);
mat_err_t mat_swap_rows(matrix_t *, const uint64_t, const uint64_t);
void mat_print(matrix_t *);
#endif
