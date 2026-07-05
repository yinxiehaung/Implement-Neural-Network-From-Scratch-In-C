#include "../include/nn.h"
#include <stdint.h>
#ifndef offsetof
#define offsetof(TYPE, MEMBER) ((size_t) & ((TYPE *)0)->MEMBER)
#endif

#ifndef container_of
#define container_of(ptr, type, member)                                             \
    ((type *)((char *)(ptr) - offsetof(type, member)))
#endif

static mat_err_t mat_add_bias(matrix_t *out, const matrix_t *in,
                              const matrix_t *bias)
{
    MAT_PTR_VALID(out);
    MAT_PTR_VALID(in);
    MAT_PTR_VALID(bias);

    MAT_REQUIRE(bias->rows == 1 && bias->cols == in->cols, MATRIX_ERR_DIM_MISMATCH);
    MAT_REQUIRE(out->rows == in->rows && out->cols == in->cols,
                MATRIX_ERR_DIM_MISMATCH);

    for (uint64_t r = 0; r < in->rows; r++) {
        for (uint64_t c = 0; c < in->cols; c++) {
            MAT_PTR_AT(out, r, c) = MAT_PTR_AT(in, r, c) + MAT_PTR_AT(bias, 0, c);
        }
    }
    return MATRIX_OK;
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

static nn_err_t dense_layer_forward(matrix_t *out, const matrix_t *in,
                                    nn_layer_t *self)
{
    nn_dense_layer_t *layer = container_of(self, nn_dense_layer_t, base);
    out->rows = in->rows;
    out->cols = layer->w.cols;
    if (mat_copy(&layer->cache_in, in) != MATRIX_OK) {
        return NN_ERR_MAT_OP;
    }
    if (mat_mul(out, in, &layer->w) != MATRIX_OK) {
        return NN_ERR_MAT_OP;
    }

    if (mat_add_bias(out, out, &layer->b) != MATRIX_OK) {
        return NN_ERR_MAT_OP;
    }

    return NN_OK;
}

static nn_err_t dense_layer_backprop(matrix_t *grad_out, const matrix_t *grad_in,
                                     nn_layer_t *self)
{
    nn_dense_layer_t *layer = container_of(self, nn_dense_layer_t, base);

    if (mat_mul_left_transposed(&layer->dw, &layer->cache_in, grad_in) != MATRIX_OK)
        return NN_ERR_BACKPROP;

    if (mat_sum_rows(&layer->db, grad_in) != MATRIX_OK)
        return NN_ERR_BACKPROP;

    if (grad_out != NULL) {
        grad_out->rows = grad_in->rows;
        grad_out->cols = layer->w.rows;
        if (mat_mul_right_transposed(grad_out, grad_in, &layer->w) != MATRIX_OK)
            return NN_ERR_BACKPROP;
    }

    return NN_OK;
}

static nn_err_t dense_layer_init_params(nn_layer_t *layer,
                                        const init_api_t *initializer)
{

    nn_dense_layer_t *dense = (nn_dense_layer_t *)layer;
    if (initializer->init_weight != NULL) {
        initializer->init_weight(&dense->w, dense->w.rows, dense->w.cols);
    }
    if (initializer->init_bias != NULL) {
        initializer->init_bias(&dense->b);
    }
    return NN_OK;
}

static nn_err_t dense_layer_compile(nn_layer_t *layer, arena_t *arena,
                                    uint64_t batch_size, uint64_t in_dim,
                                    uint64_t *out_dim, uint64_t *buffer_size)
{
    nn_dense_layer_t *dense = (nn_dense_layer_t *)layer;
    if (in_dim != dense->w.rows) {
        return NN_ERR_INVALID_ARG;
    }
    *out_dim = dense->w.cols;
    *buffer_size = (in_dim > dense->w.cols) ? in_dim : dense->w.cols;
    MAT_REQUIRE(mat_init(arena, batch_size, in_dim, &dense->cache_in) == MATRIX_OK,
                NN_ERR_COMPILE);
    return NN_OK;
}

nn_err_t dense_layer_serialize(const nn_layer_t *layer, FILE *fp)
{
    nn_dense_layer_t *dense = (nn_dense_layer_t *)layer;
    fwrite(&dense->w.rows, sizeof(uint64_t), 1, fp);
    fwrite(&dense->w.cols, sizeof(uint64_t), 1, fp);
    fwrite(dense->w.data, sizeof(mat_data_t), dense->w.rows * dense->w.cols, fp);
    fwrite(&dense->b.rows, sizeof(uint64_t), 1, fp);
    fwrite(&dense->b.cols, sizeof(uint64_t), 1, fp);
    fwrite(dense->b.data, sizeof(mat_data_t), dense->b.rows * dense->b.cols, fp);
    return NN_OK;
}

nn_err_t dense_layer_deserialize(nn_model_t *model, FILE *fp, bool should_load)
{
    uint64_t w_rows, w_cols, b_rows, b_cols;

    if (fread(&w_rows, sizeof(uint64_t), 1, fp) != 1)
        return NN_ERR_IO;
    if (fread(&w_cols, sizeof(uint64_t), 1, fp) != 1)
        return NN_ERR_IO;
    size_t w_size = w_rows * w_cols * sizeof(mat_data_t);

    if (should_load) {
        nn_model_add_dense(model, w_rows, w_cols);
        nn_dense_layer_t *d =
            (nn_dense_layer_t *)model->layers[model->layer_count - 1];
        if (fread(d->w.data, 1, w_size, fp) != w_size)
            return NN_ERR_IO;
    } else {
        fseek(fp, w_size, SEEK_CUR);
    }

    if (fread(&b_rows, sizeof(uint64_t), 1, fp) != 1)
        return NN_ERR_IO;
    if (fread(&b_cols, sizeof(uint64_t), 1, fp) != 1)
        return NN_ERR_IO;
    size_t b_size = b_rows * b_cols * sizeof(mat_data_t);

    if (should_load) {
        nn_dense_layer_t *d =
            (nn_dense_layer_t *)model->layers[model->layer_count - 1];
        if (fread(d->b.data, 1, b_size, fp) != b_size)
            return NN_ERR_IO;
    } else {
        fseek(fp, b_size, SEEK_CUR);
    }

    return NN_OK;
}

nn_err_t nn_model_add_dense(nn_model_t *model, uint64_t in_features,
                            uint64_t out_features)
{
    NN_PTR_MODEL_VALID(model);
    NN_REQUIRE(model->layer_count < model->layer_capacity, NN_ERR_INVALID_MODEL);
    nn_dense_layer_t *dense = NULL;
    arena_err_t a_err = arena_push(model->model_arena, sizeof(nn_dense_layer_t),
                                   true, (void **)&dense);
    NN_REQUIRE(a_err == ARENA_OK, NN_ERR_INVALID_MODEL);

    dense->base.type_id = LAYER_TYPE_DENSE;
    dense->base.forward = dense_layer_forward;
    dense->base.backprop = dense_layer_backprop;
    dense->base.deserialize = dense_layer_deserialize;
    dense->base.serialize = dense_layer_serialize;
    dense->base.compile = dense_layer_compile;
    dense->base.init_params = dense_layer_init_params;
    dense->in_dim = in_features;
    dense->out_dim = out_features;

    if (mat_init(model->model_arena, in_features, out_features, &dense->w) !=
        MATRIX_OK)
        return NN_ERR_MAT_INIT;
    if (mat_init(model->model_arena, in_features, out_features, &dense->dw) !=
        MATRIX_OK)
        return NN_ERR_MAT_INIT;

    if (mat_init(model->model_arena, 1, out_features, &dense->b) != MATRIX_OK)
        return NN_ERR_MAT_INIT;
    if (mat_init(model->model_arena, 1, out_features, &dense->db) != MATRIX_OK)
        return NN_ERR_MAT_INIT;

    dense->cache_in.data = NULL;
    dense->cache_in.rows = 0;
    dense->cache_in.cols = 0;
    model->layers[model->layer_count++] = (nn_layer_t *)dense;
    return NN_OK;
}
