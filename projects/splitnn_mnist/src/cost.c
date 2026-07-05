#include "../include/nn.h"
#include "../include/cost.h"
#include <math.h>

static mat_data_t mse_cost(const matrix_t *a, const matrix_t *y)
{
    NN_REQUIRE(a->rows == y->rows && a->cols == y->cols, MATRIX_ERR_DIM_MISMATCH);

    mat_data_t sum = 0;
    uint64_t n = a->rows * a->cols;

    for (uint64_t i = 0; i < n; i++) {
        mat_data_t diff = a->data[i] - y->data[i];
        sum += diff * diff;
    }

    return sum / (mat_data_t)n;
}

static mat_err_t mse_grad(matrix_t *dst, const matrix_t *a, const matrix_t *y)
{
    MAT_REQUIRE(dst->rows == a->rows && dst->cols == a->cols,
                MATRIX_ERR_DIM_MISMATCH);
    MAT_REQUIRE(a->rows == y->rows && a->cols == y->cols, MATRIX_ERR_DIM_MISMATCH);

    uint64_t n = a->rows * a->cols;
    mat_data_t scale = 2.0f / n;

    for (uint64_t i = 0; i < n; i++) {
        dst->data[i] = (a->data[i] - y->data[i]) * scale;
    }

    return MATRIX_OK;
}

const cost_api_t COST_MSE = {.cost = mse_cost, .grad = mse_grad};

#define EPSILON 1e-7f

static mat_data_t bce_cost(const matrix_t *a, const matrix_t *y)
{
    MAT_REQUIRE(a->rows == y->rows && a->cols == y->cols, MATRIX_ERR_DIM_MISMATCH);

    mat_data_t sum = 0;
    uint64_t n = a->rows * a->cols;

    for (uint64_t i = 0; i < n; i++) {
        mat_data_t val_a = a->data[i];
        mat_data_t val_y = y->data[i];
        if (val_a < EPSILON)
            val_a = EPSILON;
        if (val_a > 1.0 - EPSILON)
            val_a = 1.0 - EPSILON;

        sum += val_y * MAT_LOG(val_a) + (1.0 - val_y) * MAT_LOG(1.0 - val_a);
    }

    return -sum / (mat_data_t)n;
}

static mat_err_t bce_grad(matrix_t *dst, const matrix_t *a, const matrix_t *y)
{
    NN_REQUIRE(dst->rows == a->rows && dst->cols == a->cols,
               MATRIX_ERR_DIM_MISMATCH);
    NN_REQUIRE(a->rows == y->rows && a->cols == y->cols, MATRIX_ERR_DIM_MISMATCH);

    uint64_t n = a->rows * a->cols;

    for (uint64_t i = 0; i < n; i++) {
        mat_data_t val_a = a->data[i];
        mat_data_t val_y = y->data[i];

        if (val_a < EPSILON)
            val_a = EPSILON;
        if (val_a > 1.0 - EPSILON)
            val_a = 1.0 - EPSILON;

        mat_data_t denominator = val_a * (1.0 - val_a);
        dst->data[i] = (val_a - val_y) / (n * denominator);
    }

    return MATRIX_OK;
}

const cost_api_t COST_BCE = {.cost = bce_cost, .grad = bce_grad};

mat_data_t cost_ce_calc(const matrix_t *pred, const matrix_t *target)
{
    mat_data_t loss = 0.0f;
    for (uint64_t i = 0; i < pred->rows * pred->cols; i++) {
        if (target->data[i] == 1.0f) {
            mat_data_t p = pred->data[i];
            if (p < 1e-7f)
                p = 1e-7f;
            loss += -MAT_LOG(p);
        }
    }
    return loss / pred->rows;
}

mat_err_t cost_ce_grad(matrix_t *dst, const matrix_t *pred, const matrix_t *target)
{
    for (uint64_t i = 0; i < pred->rows * pred->cols; i++) {
        dst->data[i] = pred->data[i] - target->data[i];
        dst->data[i] /= (mat_data_t)pred->rows;
    }
    return MATRIX_OK;
}

const cost_api_t COST_CROSS_ENTROPY = {.cost = cost_ce_calc, .grad = cost_ce_grad};
