#ifndef INIT_H
#define INIT_H

#include "matrix.h"

typedef mat_err_t (*nn_init_weight_fn)(matrix_t *w, uint64_t fan_in,
                                       uint64_t fan_out);
typedef mat_err_t (*nn_init_bias_fn)(matrix_t *b);

typedef struct init_api_s init_api_t;
struct init_api_s {
    nn_init_weight_fn init_weight;
    nn_init_bias_fn init_bias;
};

extern const init_api_t INIT_XAVIER;
extern const init_api_t INIT_KAIMING;
extern const init_api_t INIT_ZEROS;
extern const init_api_t INIT_STANDER;
#endif
