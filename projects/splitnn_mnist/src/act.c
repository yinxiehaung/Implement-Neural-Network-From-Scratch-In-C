#include "../include/nn.h"
#include "../include/act.h"

static nn_err_t math_softmax_forward(matrix_t *out, const matrix_t *in)
{
    for (uint64_t r = 0; r < in->rows; r++) {
        mat_data_t max_val = MAT_PTR_AT(in, r, 0);
        for (uint64_t c = 1; c < in->cols; c++) {
            if (MAT_PTR_AT(in, r, c) > max_val)
                max_val = MAT_PTR_AT(in, r, c);
        }

        mat_data_t sum = 0.0f;
        for (uint64_t c = 0; c < in->cols; c++) {
            mat_data_t e = MAT_EXP(MAT_PTR_AT(in, r, c) - max_val);
            MAT_PTR_AT(out, r, c) = e;
            sum += e;
        }

        for (uint64_t c = 0; c < in->cols; c++) {
            MAT_PTR_AT(out, r, c) /= sum;
        }
    }
    return NN_OK;
}

static nn_err_t math_softmax_backprop(matrix_t *grad_out, const matrix_t *grad_in,
                                      const matrix_t *cache_in)
{
    (void)(cache_in);
    for (uint64_t i = 0; i < grad_out->rows * grad_out->cols; i++) {
        grad_out->data[i] = grad_in->data[i];
    }
    return NN_OK;
}

static nn_err_t math_sigmoid_forward(matrix_t *out, const matrix_t *in)
{
    NN_REQUIRE(in->rows == out->rows && out->cols == in->cols, NN_ERR_MAT_SHAPE);
    for (uint64_t i = 0; i < in->rows * in->cols; i++) {
        mat_data_t x = in->data[i];
        out->data[i] = 1.0 / (1.0 + MAT_EXP(-x));
    }
    return NN_OK;
}

static nn_err_t math_sigmoid_backprop(matrix_t *grad_out, const matrix_t *grad_in,
                                      const matrix_t *cache_in)
{
    NN_REQUIRE(grad_out->rows == grad_in->rows && grad_out->cols == grad_in->cols,
               NN_ERR_MAT_SHAPE);
    for (uint64_t i = 0; i < grad_in->rows * grad_in->cols; i++) {
        mat_data_t x = cache_in->data[i];
        mat_data_t a = 1.0 / (1.0 + MAT_EXP(-x));
        grad_out->data[i] = grad_in->data[i] * (a * (1.0 - a));
    }
    return NN_OK;
}

static nn_err_t math_relu_forward(matrix_t *out, const matrix_t *in)
{
    for (uint64_t i = 0; i < in->rows * in->cols; i++) {
        mat_data_t x = in->data[i];
        out->data[i] = (x > 0) ? x : 0;
    }
    return NN_OK;
}

static nn_err_t math_relu_backprop(matrix_t *grad_out, const matrix_t *grad_in,
                                   const matrix_t *cache_in)
{
    for (uint64_t i = 0; i < grad_out->rows * grad_out->cols; i++) {
        mat_data_t grad = (cache_in->data[i] > 0) ? 1.0 : 0.0;
        grad_out->data[i] = grad_in->data[i] * grad;
    }
    return NN_OK;
}

static nn_err_t math_linear_forward(matrix_t *out, const matrix_t *in)
{
    if (out != in) {
        memcpy(out->data, in->data, sizeof(mat_data_t) * out->rows * in->cols);
    }
    return NN_OK;
}

static nn_err_t math_linear_backprop(matrix_t *grad_out, const matrix_t *grad_in,
                                     const matrix_t *cache_in)
{
    (void)(cache_in);
    if (grad_in != grad_out) {
        memcpy(grad_out->data, grad_in->data,
               sizeof(mat_data_t) * grad_in->rows * grad_out->cols);
    }
    return NN_OK;
}

const act_api_t ACT_SOFTMAX = {.forward = math_softmax_forward,
                               .backprop = math_softmax_backprop};
const act_api_t ACT_SIGMOID = {.forward = math_sigmoid_forward,
                               .backprop = math_sigmoid_backprop};
const act_api_t ACT_RELU = {.forward = math_relu_forward,
                            .backprop = math_relu_backprop};
const act_api_t ACT_LINEAR = {.forward = math_linear_forward,
                              .backprop = math_linear_backprop};
