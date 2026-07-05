#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "matrix.h"
#include "nn_err_types.h"
#include "nn_layer_type.h"

typedef struct nn_layer_s nn_layer_t;
typedef struct init_api_s init_api_t;
typedef struct nn_model_s nn_model_t;

typedef nn_err_t (*layer_forward_fn)(matrix_t *out, const matrix_t *in,
                                     nn_layer_t *self);
typedef nn_err_t (*layer_backprop_fn)(matrix_t *grad_out, const matrix_t *grad_in,
                                      nn_layer_t *self);
typedef nn_err_t (*layer_compile_fn)(nn_layer_t *layer, arena_t *arena,
                                     uint64_t batch_size, uint64_t in_dim,
                                     uint64_t *out_dim, uint64_t *req_buffer_size);
typedef nn_err_t (*layer_init_params_fn)(nn_layer_t *layer,
                                         const init_api_t *initializer);
typedef nn_err_t (*layer_serialize_fn)(const nn_layer_t *layer, FILE *fp);

typedef nn_err_t (*layer_deserialize_fn)(nn_model_t *model, FILE *fp,
                                         bool should_load);

struct nn_layer_s {
    layer_type_t type_id;
    layer_forward_fn forward;
    layer_backprop_fn backprop;
    layer_compile_fn compile;
    layer_init_params_fn init_params;
    layer_serialize_fn serialize;
    layer_deserialize_fn deserialize;
};
#endif
