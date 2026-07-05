#ifndef NN_ACTIVATION_LAYER_H
#define NN_ACTIVATION_LAYER_H

#include "nn_layer.h"
#include "act.h"

typedef struct nn_model_s nn_model_t;

typedef struct {
    nn_layer_t base;
    const act_api_t *api;
    matrix_t cache_in;
} nn_activation_layer_t;

nn_err_t nn_model_add_activation(nn_model_t *model, const act_api_t *api);
nn_err_t act_layer_serialize(const nn_layer_t *layer, FILE *fp);
nn_err_t act_layer_deserialize(nn_model_t *model, FILE *fp, bool should_load);
#endif
