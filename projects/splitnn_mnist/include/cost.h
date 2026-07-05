#ifndef COST_H
#define COST_H

#include "nn_layer.h"

typedef struct cost_api_s cost_api_t;
typedef mat_data_t (*nn_cost_fn)(const matrix_t *, const matrix_t *);
typedef mat_err_t (*cost_grad_fn)(matrix_t *loss_grad, const matrix_t *y_pred,
                                  const matrix_t *y_true);
struct cost_api_s {
    nn_cost_fn cost;
    cost_grad_fn grad;
};

extern const cost_api_t COST_MSE;
extern const cost_api_t COST_CROSS_ENTROPY;
extern const cost_api_t COST_BCE;
#endif
