#ifndef NN_DENSE_LAYER_H
#define NN_DENSE_LAYER_H
#include "nn_layer.h"
#include <stdint.h>

typedef struct nn_dense_layer_s nn_dense_layer_t;
typedef struct layer_desc_s layer_desc_t;
typedef struct nn_model_s nn_model_t;

struct nn_dense_layer_s {
    nn_layer_t base;
    matrix_t w;
    matrix_t b;
    matrix_t dw;
    matrix_t db;
    matrix_t cache_in;
    uint64_t in_dim;
    uint64_t out_dim;
};

nn_err_t nn_model_add_dense(nn_model_t *model, uint64_t rows, uint64_t cols);
nn_err_t dense_layer_serialize(const nn_layer_t *layer, FILE *fp);
nn_err_t dense_layer_deserialize(nn_model_t *model, FILE *fp, bool should_load);
#endif
