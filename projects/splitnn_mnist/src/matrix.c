#include "../include/matrix.h"

mat_err_t mat_init(arena_t *arena, const uint64_t rows, const uint64_t cols,
                   matrix_t *dst)
{
    ARENA_CHECK(PTR_IS_NULL(arena), alloc_err);
    MAT_PTR_NO_NULL(dst, MATRIX_ERR_INVALID_ARG);
    ARENA_CHECK(arena_push(arena, sizeof(mat_data_t) * rows * cols, true,
                           (void **)&dst->data),
                alloc_err);
    dst->cols = cols;
    dst->rows = rows;
    return MATRIX_OK;
alloc_err:
    dst->cols = 0;
    dst->rows = 0;
    return MATRIX_ERR_ARENA;
}

mat_err_t mat_rand(matrix_t *mat)
{
    MAT_PTR_VALID(mat);
    for (uint64_t i = 0; i < mat->cols; i++) {
        for (uint64_t j = 0; j < mat->rows; j++) {
            MAT_PTR_AT(mat, j, i) = ((mat_data_t)rand() / RAND_MAX * 2.0f - 1.0f);
        }
    }
    return MATRIX_OK;
}

mat_err_t mat_transpose(matrix_t *dst, const matrix_t *mat)
{
    MAT_PTR_VALID(dst);
    MAT_PTR_VALID(mat);
    MAT_PTR_SAME_DIM2(dst, mat);
    for (uint64_t i = 0; i < mat->cols; i++) {
        for (uint64_t j = 0; j < mat->rows; j++) {
            MAT_PTR_AT(dst, i, j) = MAT_PTR_AT(mat, j, i);
        }
    }
    return MATRIX_OK;
}

mat_err_t mat_copy(matrix_t *dst, const matrix_t *mat)
{
    MAT_PTR_VALID(dst);
    MAT_PTR_VALID(mat);
    MAT_PTR_SAME_DIM2(dst, mat);
    for (uint64_t i = 0; i < mat->rows; i++) {
        for (uint64_t j = 0; j < mat->cols; j++) {
            MAT_PTR_AT(dst, i, j) = MAT_PTR_AT(mat, i, j);
        }
    }
    return MATRIX_OK;
}

mat_err_t mat_add(matrix_t *dst, const matrix_t *src1, const matrix_t *src2)
{
    MAT_PTR_VALID(dst);
    MAT_PTR_VALID(src1);
    MAT_PTR_VALID(src2);
    MAT_PTR_SAME_DIM3(dst, src1, src2);
    for (uint64_t i = 0; i < src1->rows; i++) {
        for (uint64_t j = 0; j < src1->cols; j++) {
            MAT_PTR_AT(dst, i, j) = MAT_PTR_AT(src1, i, j) + MAT_PTR_AT(src2, i, j);
        }
    }
    return MATRIX_OK;
}

mat_err_t mat_scale_add(matrix_t *dst, const matrix_t *src1, const matrix_t *src2,
                        mat_data_t scale)
{
    MAT_PTR_VALID(dst);
    MAT_PTR_VALID(src1);
    MAT_PTR_VALID(src2);
    MAT_PTR_SAME_DIM3(dst, src1, src2);
    for (uint64_t i = 0; i < src1->rows; i++) {
        for (uint64_t j = 0; j < src1->cols; j++) {
            MAT_PTR_AT(dst, i, j) =
                MAT_PTR_AT(src1, i, j) + MAT_PTR_AT(src2, i, j) * scale;
        }
    }
    return MATRIX_OK;
}

mat_err_t mat_sub(matrix_t *dst, const matrix_t *src1, const matrix_t *src2)
{
    MAT_PTR_VALID(dst);
    MAT_PTR_VALID(src1);
    MAT_PTR_VALID(src2);
    MAT_PTR_SAME_DIM3(dst, src1, src2);
    for (uint64_t i = 0; i < src1->rows; i++) {
        for (uint64_t j = 0; j < src1->cols; j++) {
            MAT_PTR_AT(dst, i, j) = MAT_PTR_AT(src1, i, j) - MAT_PTR_AT(src2, i, j);
        }
    }
    return MATRIX_OK;
}

mat_err_t mat_mul(matrix_t *dst, const matrix_t *src1, const matrix_t *src2)
{
    MAT_PTR_VALID(dst);
    MAT_PTR_VALID(src1);
    MAT_PTR_VALID(src2);
#ifndef NDEBUG
    if (src1->cols != src2->rows || dst->rows != src1->rows ||
        dst->cols != src2->cols)
        return MATRIX_ERR_INVALID_ARG;
#endif
    memset(dst->data, 0, sizeof(mat_data_t) * dst->cols * dst->rows);
#pragma omp parallel for if (src1->rows * src2->cols > 10000)
    for (uint64_t i = 0; i < dst->rows; i++) {
        for (uint64_t k = 0; k < src1->cols; k++) {
            mat_data_t r = MAT_PTR_AT(src1, i, k);
            for (uint64_t j = 0; j < dst->cols; j++) {
                MAT_PTR_AT(dst, i, j) += r * MAT_PTR_AT(src2, k, j);
            }
        }
    }
    return MATRIX_OK;
}

mat_err_t mat_row_view(matrix_t *dst, const matrix_t *src, const uint64_t stride,
                       const uint64_t cols)
{
    MAT_PTR_VALID(dst);
    MAT_PTR_VALID(src);
#ifndef NDEBUG
    if (src->cols < cols || src->rows < stride)
        return MATRIX_ERR_INVALID_ARG;
#endif
    dst->data = src->data + (stride * src->cols);
    dst->cols = cols;
    dst->rows = 1;
    return MATRIX_OK;
}

mat_err_t mat_swap_rows(matrix_t *mat, const uint64_t r1, const uint64_t r2)
{
    MAT_REQUIRE((r1 >= mat->rows || r2 >= mat->rows), MATRIX_ERR_INVALID_ARG);
    MAT_REQUIRE((r1 == r2), MATRIX_OK);

    mat_data_t *row1 = mat->data + (r1 * mat->cols);
    mat_data_t *row2 = mat->data + (r2 * mat->cols);

    for (uint64_t c = 0; c < mat->cols; c++) {
        mat_data_t temp = row1[c];
        row1[c] = row2[c];
        row2[c] = temp;
    }
    return MATRIX_OK;
}

void mat_print(matrix_t *mat)
{
    printf("[\n");
    for (uint64_t i = 0; i < mat->rows; i++) {
        for (uint64_t j = 0; j < mat->cols; j++) {
            printf(MAT_FMT, MAT_PTR_AT(mat, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}
