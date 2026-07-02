#include "../include/nn.h"
#include <math.h>

mat_err_t act_softmax_forward(matrix_t *dst, const matrix_t *src)
{
    for (uint64_t r = 0; r < src->rows; r++) {
        mat_data_t max_val = MAT_PTR_AT(src, r, 0);
        for (uint64_t c = 1; c < src->cols; c++) {
            if (MAT_PTR_AT(src, r, c) > max_val)
                max_val = MAT_PTR_AT(src, r, c);
        }

        mat_data_t sum = 0.0f;
        for (uint64_t c = 0; c < src->cols; c++) {
            mat_data_t e = MAT_EXP(MAT_PTR_AT(src, r, c) - max_val);
            MAT_PTR_AT(dst, r, c) = e;
            sum += e;
        }

        for (uint64_t c = 0; c < src->cols; c++) {
            MAT_PTR_AT(dst, r, c) /= sum;
        }
    }
    return MATRIX_OK;
}

mat_err_t act_softmax_backprop(matrix_t *dst, const matrix_t *das,
                               const matrix_t *a_out)
{
    (void)a_out;
    for (uint64_t i = 0; i < dst->rows * dst->cols; i++) {
        dst->data[i] = das->data[i];
    }
    return MATRIX_OK;
}

const act_api_t ACT_SOFTMAX = {.forward = act_softmax_forward,
                               .backprop = act_softmax_backprop};
static mat_err_t sigmoid_forward(matrix_t *dst, const matrix_t *src)
{
    NN_REQUIRE(dst->rows == src->rows && dst->cols == src->cols,
               MATRIX_ERR_DIM_MISMATCH);
    for (uint64_t i = 0; i < src->rows * src->cols; i++) {
        mat_data_t x = src->data[i];
        dst->data[i] = 1.0 / (1.0 + MAT_EXP(-x));
    }
    return MATRIX_OK;
}

static mat_err_t sigmoid_backprop(matrix_t *dst, const matrix_t *das,
                                  const matrix_t *a_out)
{
    NN_REQUIRE(dst->rows == das->rows && dst->cols == das->cols,
               MATRIX_ERR_DIM_MISMATCH);

    for (uint64_t i = 0; i < dst->rows * dst->cols; i++) {
        mat_data_t a = a_out->data[i];
        dst->data[i] = das->data[i] * (a * (1.0 - a));
    }
    return MATRIX_OK;
}

const act_api_t ACT_SIGMOID = {.forward = sigmoid_forward,
                               .backprop = sigmoid_backprop};

static mat_err_t relu_forward(matrix_t *dst, const matrix_t *src)
{
    for (uint64_t i = 0; i < src->rows * src->cols; i++) {
        mat_data_t x = src->data[i];
        dst->data[i] = (x > 0) ? x : 0;
    }
    return MATRIX_OK;
}

static mat_err_t relu_backprop(matrix_t *dst, const matrix_t *das,
                               const matrix_t *a_out)
{
    for (uint64_t i = 0; i < dst->rows * dst->cols; i++) {
        mat_data_t grad = (a_out->data[i] > 0) ? 1.0 : 0.0;
        dst->data[i] = das->data[i] * grad;
    }
    return MATRIX_OK;
}

const act_api_t ACT_RELU = {.forward = relu_forward, .backprop = relu_backprop};

mat_err_t linear_forward(matrix_t *dst, const matrix_t *src)
{
    if (src != dst) {
        memcpy(dst->data, src->data, sizeof(mat_data_t) * src->rows * src->cols);
    }
    return MATRIX_OK;
}

mat_err_t linear_backprop(matrix_t *dst, const matrix_t *das, const matrix_t *a_out)
{
    (void)(a_out);
    if (dst != das) {
        memcpy(dst->data, das->data, sizeof(mat_data_t) * dst->rows * dst->cols);
    }

    return MATRIX_OK;
}

const act_api_t ACT_LINEAR = {.forward = linear_forward,
                              .backprop = linear_backprop};
