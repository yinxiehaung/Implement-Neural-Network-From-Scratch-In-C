#include "../include/nn.h"
#include <stdint.h>

static nn_err_t nn_model_alloc(arena_t *arena, const nn_model_cfg_t *cfg,
                               nn_model_t *out_nn)
{
    ARENA_CHECK(arena_push(arena, sizeof(matrix_t) * cfg->arch_count, true,
                           (void **)&out_nn->ws),
                alloc_err);
    ARENA_CHECK(arena_push(arena, sizeof(matrix_t) * cfg->arch_count, true,
                           (void **)&out_nn->bs),
                alloc_err);
    for (uint64_t i = 0; i < cfg->arch_count - 1; i++) {
        uint64_t rows = cfg->arch[i];
        uint64_t cols = cfg->arch[i + 1];
        MAT_CHECK(mat_init(arena, rows, cols, &out_nn->ws[i]), alloc_err);
        MAT_CHECK(mat_init(arena, 1, cols, &out_nn->bs[i]), alloc_err);
    }
    return NN_OK;
alloc_err:
    return NN_ERR_ARENA;
}

nn_err_t nn_model_init(arena_t *arena, const nn_model_cfg_t *cfg,
                       nn_model_t *out_model)
{
    NN_PTR_NO_NULL(arena, NN_ERR_INVALID_ARG);
    NN_PTR_MODEL_CFG_VALID(cfg);
    for (uint64_t i = 0; i < cfg->arch_count; i++) {
        NN_REQUIRE(cfg->arch[i] != 0, NN_ERR_INVALID_ARCH);
    }
    tmp_arena_t check_point;
    begin_tmp_arena(&check_point, arena);
    NN_CHECK(nn_model_alloc(arena, cfg, out_model), alloc_err);
    out_model->cfg = *cfg;
    for (uint64_t i = 0; i < cfg->arch_count - 1; i++) {
        uint64_t fan_in = cfg->arch[i], fan_out = cfg->arch[i + 1];
        if (PTR_NO_NULL(cfg->initializers) && cfg->initializers[i]->init_weight)
            MAT_CHECK(cfg->initializers[i]->init_weight(&out_model->ws[i], fan_in,
                                                        fan_out),
                      init_err);
        if (PTR_NO_NULL(cfg->initializers) && cfg->initializers[i]->init_bias)
            MAT_CHECK(cfg->initializers[i]->init_bias(&out_model->bs[i]), init_err);
    }
    return NN_OK;
alloc_err:
    end_tmp_arena(&check_point);
    return NN_ERR_ARENA;
init_err:
    end_tmp_arena(&check_point);
    return NN_ERR_INIT;
}

nn_err_t nn_trainer_alloc(arena_t *arena, const nn_train_cfg_t *cfg,
                          const nn_model_t *model, nn_trainer_t *trainer)
{
    uint64_t layer = model->cfg.arch_count;
    uint64_t batch_size = cfg->batch_size;

    ARENA_CHECK(
        arena_push(arena, sizeof(matrix_t) * layer, true, (void **)&trainer->dws),
        alloc_err);
    ARENA_CHECK(
        arena_push(arena, sizeof(matrix_t) * layer, true, (void **)&trainer->dbs),
        alloc_err);
    ARENA_CHECK(arena_push(arena, sizeof(matrix_t) * layer, true,
                           (void **)&trainer->transposed_w_scratch),
                alloc_err);
    ARENA_CHECK(arena_push(arena, sizeof(matrix_t) * (layer + 1), true,
                           (void **)&trainer->das),
                alloc_err);

    ARENA_CHECK(arena_push(arena, sizeof(matrix_t) * (layer + 1), true,
                           (void **)&trainer->as),
                alloc_err);
    ARENA_CHECK(arena_push(arena, sizeof(matrix_t) * (layer + 1), true,
                           (void **)&trainer->delta_act_scratch),
                alloc_err);
    for (uint64_t i = 0; i < layer - 1; i++) {
        uint64_t rows = model->cfg.arch[i];
        uint64_t cols = model->cfg.arch[i + 1];

        MAT_CHECK(mat_init(arena, rows, cols, &trainer->dws[i]), alloc_err);
        MAT_CHECK(mat_init(arena, rows, cols, &trainer->transposed_w_scratch[i]),
                  alloc_err);
        MAT_CHECK(mat_init(arena, 1, cols, &trainer->dbs[i]), alloc_err);
    }

    for (uint64_t i = 0; i < layer; i++) {
        uint64_t cols = model->cfg.arch[i];
        MAT_CHECK(mat_init(arena, batch_size, cols, &trainer->das[i]), alloc_err);
        MAT_CHECK(mat_init(arena, batch_size, cols, &trainer->as[i]), alloc_err);
        MAT_CHECK(mat_init(arena, batch_size, cols, &trainer->delta_act_scratch[i]),
                  alloc_err);
    }
    trainer->model = (nn_model_t *)model;
    trainer->cfg = *cfg;
    return NN_OK;
alloc_err:
    return NN_ERR_ARENA;
}

nn_err_t nn_trainer_init(arena_t *arena, const nn_train_cfg_t *cfg,
                         const nn_model_t *model, nn_trainer_t *trainer)
{
    NN_PTR_NO_NULL(arena, NN_ERR_INVALID_ARG);
    NN_PTR_MODEL_VALID(model);
    NN_PTR_TRAIN_CFG_VALID(cfg);
    return nn_trainer_alloc(arena, cfg, model, trainer);
}

nn_err_t nn_forward_train(nn_trainer_t *trainer, matrix_t *input)
{
    NN_PTR_TRAINER_VALID(trainer);
    NN_PTR_NO_NULL(input, NN_ERR_INVALID_ARG);
    matrix_t curr;
    mat_copy(&trainer->as[0], input);
    curr = trainer->as[0];
    nn_model_t *model = trainer->model;
    for (uint64_t i = 0; i < trainer->model->cfg.arch_count - 1; i++) {
        mat_mul(&trainer->as[i + 1], &curr, &model->ws[i]);
        mat_add(&trainer->as[i + 1], &trainer->as[i + 1], &model->bs[i]);
        MAT_CHECK(model->cfg.activations[i]->forward(&trainer->as[i + 1],
                                                     &trainer->as[i + 1]),
                  act_err);
        curr = trainer->as[i + 1];
    }
    return NN_OK;
act_err:
    return NN_ERR_INVALID_ACT;
}

nn_err_t nn_forward_predict(arena_t *arena, const nn_model_t *model,
                            const matrix_t *x, matrix_t *out_mat)
{
    NN_PTR_MODEL_VALID(model);
    NN_PTR_NO_NULL(x, NN_ERR_INVALID_ARG);
    NN_PTR_NO_NULL(out_mat, NN_ERR_INVALID_ARG);
    NN_REQUIRE(model->cfg.arch[0] == x->cols * x->rows, NN_ERR_MAT_SHAPE);
    tmp_arena_t check_point;
    begin_tmp_arena(&check_point, arena);
    matrix_t ping = *x, pong;
    ping = *x;
    for (uint64_t i = 0; i < model->cfg.arch_count - 1; i++) {
        if (i != model->cfg.arch_count - 2) {
            MAT_CHECK(mat_init(arena, ping.rows, model->cfg.arch[i + 1], &pong),
                      alloc_err);
        } else {
            pong = *out_mat;
        }
        MAT_CHECK(mat_mul(&pong, &ping, &model->ws[i]), op_err);
        MAT_CHECK(mat_add(&pong, &pong, &model->bs[i]), op_err);
        MAT_CHECK(model->cfg.activations[i]->forward(&pong, &pong), act_err);
        ping = pong;
    }
    *out_mat = pong;
    end_tmp_arena(&check_point);
    return NN_OK;
act_err:
    end_tmp_arena(&check_point);
    return NN_ERR_INVALID_ACT;
op_err:
    end_tmp_arena(&check_point);
    return NN_ERR_MAT_OP;
alloc_err:
    end_tmp_arena(&check_point);
    return NN_ERR_ARENA;
}

static nn_err_t nn_cost(nn_trainer_t *trainer, matrix_t *x, matrix_t *y,
                        cost_func_ptr cost, mat_data_t *result)
{
    nn_err_t err_code;
    *result = 0.0f;
    for (uint64_t i = 0; i < x->rows; i++) {
        NN_CHECK(err_code = nn_forward_train(trainer, x), forward_err);
        for (uint64_t j = 0; j < y->cols; j++) {
            *result += cost(&trainer->as[trainer->model->cfg.arch_count], y);
        }
    }
    *result = (*result) / x->rows;
    return NN_OK;
forward_err:
    return err_code;
}

nn_err_t nn_finte_diff(nn_trainer_t *trainer, matrix_t *x, matrix_t *y,
                       const mat_data_t esp, const cost_api_t *cost_func)
{
    NN_PTR_TRAINER_VALID(trainer);
    NN_PTR_NO_NULL(cost_func, NN_ERR_INVALID_ARG);
    NN_PTR_NO_NULL(x, NN_ERR_INVALID_ARG);
    NN_PTR_NO_NULL(y, NN_ERR_INVALID_ARG);
    NN_REQUIRE(esp > 0, NN_ERR_INVALID_ARG);
    mat_data_t c;
    mat_data_t dc;
    NN_CHECK(nn_cost(trainer, x, y, cost_func->cost, &c), finte_err);
    nn_model_t *model = trainer->model;
    for (uint64_t i = 0; i < model->cfg.arch_count - 1; i++) {
        for (uint64_t j = 0; j < model->ws[i].rows; j++) {
            for (uint64_t k = 0; k < model->ws[i].cols; k++) {
                mat_data_t saved = MAT_PTR_AT(&model->ws[i], j, k);
                MAT_PTR_AT(&model->ws[i], j, k) += esp;
                NN_CHECK(nn_cost(trainer, x, y, cost_func->cost, &dc), finte_err);
                MAT_PTR_AT(&trainer->dws[i], j, k) = ((dc - c) / esp);
                MAT_PTR_AT(&model->ws[i], j, k) = saved;
            }
        }
        for (uint64_t j = 0; j < model->bs[i].cols; j++) {
            mat_data_t saved = MAT_PTR_AT(&model->bs[i], 0, j);
            MAT_PTR_AT(&model->bs[i], 0, j) += esp;
            NN_CHECK(nn_cost(trainer, x, y, cost_func->cost, &dc), finte_err);
            MAT_PTR_AT(&trainer->dbs[i], 0, j) = ((dc - c) / esp);
            MAT_PTR_AT(&model->bs[i], 0, j) = saved;
        }
    }
    return NN_OK;
finte_err:
    return NN_ERR_FINTED_DIFF;
}
static mat_err_t mat_sum_rows(matrix_t *dst, const matrix_t *src)
{
    MAT_PTR_VALID(dst);
    MAT_PTR_VALID(src);
    MAT_REQUIRE(dst->rows == 1 && dst->cols == src->cols, MATRIX_ERR_DIM_MISMATCH);

    for (uint64_t c = 0; c < dst->cols; c++) {
        MAT_PTR_AT(dst, 0, c) = 0.0f;
    }
    for (uint64_t r = 0; r < src->rows; r++) {
        for (uint64_t c = 0; c < src->cols; c++) {
            MAT_PTR_AT(dst, 0, c) += MAT_PTR_AT(src, r, c);
        }
    }

    return MATRIX_OK;
}
static inline mat_err_t mat_mul_left_transposed(matrix_t *dst, const matrix_t *src1,
                                                const matrix_t *src2)
{
    MAT_REQUIRE(src1->rows == src2->rows, MATRIX_ERR_DIM_MISMATCH);
    MAT_REQUIRE(dst->rows == src1->cols && dst->cols == src2->cols,
                MATRIX_ERR_DIM_MISMATCH);
    memset(dst->data, 0, sizeof(mat_data_t) * dst->cols * dst->rows);
#pragma omp parallel for if (src1->rows * src2->cols > 10000)
    for (uint64_t i = 0; i < src1->rows; i++) {
        mat_data_t *src1_row = &src1->data[i * src1->cols];
        mat_data_t *src2_row = &src2->data[i * src2->cols];
        for (uint64_t k = 0; k < src1->cols; k++) {
            mat_data_t src1_val = src1_row[k];
            mat_data_t *dst_row = &dst->data[k * dst->cols];
            for (uint64_t j = 0; j < src2->cols; j++) {
                dst_row[j] += src1_val * src2_row[j];
            }
        }
    }
    return MATRIX_OK;
}

static inline mat_err_t mat_mul_right_transposed(matrix_t *dst, const matrix_t *src1,
                                                 const matrix_t *src2)
{
    MAT_REQUIRE(src1->cols == src2->cols, MATRIX_ERR_DIM_MISMATCH);
    MAT_REQUIRE(dst->rows == src1->rows && dst->cols == src2->rows,
                MATRIX_ERR_DIM_MISMATCH);
#pragma omp parallel for if (src1->rows * src2->cols > 10000)
    for (uint64_t i = 0; i < src1->rows; i++) {
        mat_data_t *src1_row = &src1->data[i * src1->cols];
        mat_data_t *dst_row = &dst->data[i * dst->cols];
        for (uint64_t j = 0; j < src2->rows; j++) {
            mat_data_t *src2_row = &src2->data[j * src2->cols];
            mat_data_t sum = 0;
            for (uint64_t k = 0; k < src1->cols; k++) {
                sum += src1_row[k] * src2_row[k];
            }
            dst_row[j] = sum;
        }
    }
    return MATRIX_OK;
}
nn_err_t nn_backprop(nn_trainer_t *trainer, matrix_t *target,
                     const cost_api_t *cost_func)
{
    nn_model_t *model = trainer->model;
    uint64_t last_layer = model->cfg.arch_count - 1;

    MAT_CHECK(
        cost_func->grad(&trainer->das[last_layer], &trainer->as[last_layer], target),
        backprop_err);

    for (uint64_t layer = last_layer - 1; layer != UINT64_MAX; layer--) {
        matrix_t *a_in = &trainer->as[layer];
        matrix_t *a_out = &trainer->as[layer + 1];
        matrix_t *w = &model->ws[layer];
        matrix_t *das_curr = &trainer->das[layer + 1];
        matrix_t *delta_act = das_curr;

        MAT_CHECK(
            model->cfg.activations[layer]->backprop(delta_act, das_curr, a_out),
            backprop_err);

        matrix_t *dws = &trainer->dws[layer];
        MAT_CHECK(mat_mul_left_transposed(dws, a_in, delta_act), backprop_err);
        MAT_CHECK(mat_sum_rows(&trainer->dbs[layer], delta_act), backprop_err);
        if (layer > 0) {
            matrix_t *das_prev = &trainer->das[layer];
            MAT_CHECK(mat_mul_right_transposed(das_prev, delta_act, w),
                      backprop_err);
        }
    }
    return NN_OK;
backprop_err:
    return NN_ERR_BACKPROP;
}

void nn_model_print(nn_model_t *model) { return; }
void nn_trainer_print(nn_trainer_t *trainer) { return; }
