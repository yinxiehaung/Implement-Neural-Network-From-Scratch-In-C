#ifndef ACT_H
#define ACT_H

#include "nn_layer.h"

typedef struct act_api_s act_api_t;
typedef nn_err_t (*act_forward_fn)(matrix_t *out, const matrix_t *in);
typedef nn_err_t (*act_backprop_fn)(matrix_t *grad_out, const matrix_t *grad_in,
                                    const matrix_t *cache_in);

struct act_api_s {
    act_forward_fn forward;
    act_backprop_fn backprop;
};

extern const act_api_t ACT_SIGMOID;
extern const act_api_t ACT_RELU;
extern const act_api_t ACT_TANH;
extern const act_api_t ACT_LINEAR;
extern const act_api_t ACT_SOFTMAX;
extern const act_api_t ACT_DENSE;
#endif
