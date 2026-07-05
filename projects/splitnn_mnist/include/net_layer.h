#ifndef NET_LAYER_H
#define NET_LAYER_H

#include "nn.h"

typedef struct {
    nn_layer_t base;
    int sockfd;
} nn_net_emitter_layer_t;

typedef struct {
    nn_layer_t base;
    int sockfd;
} nn_net_receiver_layer_t;

nn_err_t net_send_matrix(int sockfd, const matrix_t *mat);
nn_err_t net_recv_matrix(int sockfd, matrix_t *mat, arena_t *arena);
nn_err_t nn_model_add_emitter(nn_model_t *model, int sockfd);
nn_err_t nn_model_add_receiver(nn_model_t *model, int sockfd);
#endif
