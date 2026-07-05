#include "../include/init.h"
#include <stdint.h>

static inline mat_data_t rand_uniform(void)
{
    return ((mat_data_t)rand() / (mat_data_t)RAND_MAX) * 2.0f - 1.0f;
}

static mat_err_t xavier_init_w(matrix_t *w, uint64_t fan_in, uint64_t fan_out)
{
    mat_data_t limit = sqrtf(6.0f / (mat_data_t)(fan_in + fan_out));
    for (uint64_t i = 0; i < w->rows * w->cols; i++) {
        w->data[i] = rand_uniform() * limit;
    }
    return MATRIX_OK;
}

static mat_err_t kaiming_init_w(matrix_t *w, uint64_t fan_in, uint64_t fan_out)
{
    (void)fan_out;
    mat_data_t limit = sqrtf(6.0f / (mat_data_t)fan_in);
    for (uint64_t i = 0; i < w->rows * w->cols; i++) {
        w->data[i] = rand_uniform() * limit;
    }
    return MATRIX_OK;
}

static mat_err_t zeros_init_b(matrix_t *b)
{
    for (uint64_t i = 0; i < b->rows * b->cols; i++) {
        b->data[i] = 0.0f;
    }
    return MATRIX_OK;
}

static mat_err_t standard_init_b(matrix_t *b)
{
    for (uint64_t i = 0; i < b->rows * b->cols; i++) {
        b->data[i] = rand_uniform() * 0.01f;
    }
    return MATRIX_OK;
}

const init_api_t INIT_XAVIER = {.init_weight = xavier_init_w,
                                .init_bias = zeros_init_b};
const init_api_t INIT_KAIMING = {.init_weight = kaiming_init_w,
                                 .init_bias = zeros_init_b};
const init_api_t INIT_ZEROS = {.init_weight = NULL, .init_bias = zeros_init_b};
const init_api_t INIT_STANDER = {.init_weight = NULL, .init_bias = standard_init_b};
