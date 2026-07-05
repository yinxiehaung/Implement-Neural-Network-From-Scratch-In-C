#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include "../include/net_layer.h"

static int send_all(int sockfd, const void *buf, size_t len)
{
    size_t total_sent = 0;
    const char *p = (const char *)buf;

    while (total_sent < len) {
        ssize_t n = send(sockfd, p + total_sent, len - total_sent, 0);

        if (n <= 0) {
            perror("[NetLayer Error] send_all failed or connection closed");
            return -1;
        }
        total_sent += n;
    }
    return 0;
}

static int recv_all(int sockfd, void *buf, size_t len)
{
    size_t total_received = 0;
    char *p = (char *)buf;

    while (total_received < len) {
        ssize_t n =
            recv(sockfd, p + total_received, len - total_received, MSG_WAITALL);

        if (n <= 0) {
            if (n == 0) {
                printf("[NetLayer Error] recv_all: Connection closed by peer.\n");
            } else {
                perror("[NetLayer Error] recv_all failed");
            }
            return -1;
        }
        total_received += n;
    }
    return 0;
}

nn_err_t net_send_matrix(int sockfd, const matrix_t *mat)
{
    if (send_all(sockfd, &mat->rows, sizeof(uint64_t)) < 0)
        return NN_ERR_IO;

    if (send_all(sockfd, &mat->cols, sizeof(uint64_t)) < 0)
        return NN_ERR_IO;

    size_t payload_bytes = mat->rows * mat->cols * sizeof(mat_data_t);
    if (send_all(sockfd, mat->data, payload_bytes) < 0)
        return NN_ERR_IO;

    return NN_OK;
}

nn_err_t net_recv_matrix(int sockfd, matrix_t *mat, arena_t *arena)
{
    if (recv_all(sockfd, &mat->rows, sizeof(uint64_t)) < 0)
        return NN_ERR_IO;
    if (recv_all(sockfd, &mat->cols, sizeof(uint64_t)) < 0)
        return NN_ERR_IO;

    if (mat->data == NULL) {
        if (mat_init(arena, mat->rows, mat->cols, mat) != MATRIX_OK) {
            return NN_ERR_MATH;
        }
    }

    size_t payload_bytes = mat->rows * mat->cols * sizeof(mat_data_t);
    if (recv_all(sockfd, mat->data, payload_bytes) < 0)
        return NN_ERR_IO;

    return NN_OK;
}

static nn_err_t emitter_compile(nn_layer_t *layer, arena_t *arena,
                                uint64_t batch_size, uint64_t in_dim,
                                uint64_t *out_dim, uint64_t *buffer_size)
{
    *out_dim = in_dim;
    *buffer_size = in_dim;
    return NN_OK;
}

static nn_err_t emitter_forward(matrix_t *out, const matrix_t *in, nn_layer_t *layer)
{
    nn_net_emitter_layer_t *emitter = (nn_net_emitter_layer_t *)layer;

    nn_err_t err = net_send_matrix(emitter->sockfd, in);
    if (err != NN_OK)
        return err;

    for (uint64_t i = 0; i < in->rows * in->cols; i++) {
        out->data[i] = in->data[i];
    }
    return NN_OK;
}

nn_err_t nn_model_add_emitter(nn_model_t *model, int sockfd)
{
    NN_REQUIRE(model->layer_count < model->layer_capacity, NN_ERR_INVALID_MODEL);

    nn_net_emitter_layer_t *emitter;
    arena_push(model->model_arena, sizeof(nn_net_emitter_layer_t), true,
               (void **)&emitter);

    emitter->sockfd = sockfd;
    emitter->base.type_id = LAYER_TYPE_NET_EMITTER;
    emitter->base.forward = emitter_forward;
    emitter->base.compile = emitter_compile;

    emitter->base.init_params = NULL;
    emitter->base.serialize = NULL;

    model->layers[model->layer_count++] = (nn_layer_t *)emitter;
    return NN_OK;
}

static nn_err_t receiver_compile(nn_layer_t *layer, arena_t *arena,
                                 uint64_t batch_size, uint64_t in_dim,
                                 uint64_t *out_dim, uint64_t *buffer_size)
{
    *out_dim = in_dim;
    *buffer_size = in_dim;
    return NN_OK;
}

static nn_err_t receiver_forward(matrix_t *out, const matrix_t *in,
                                 nn_layer_t *layer)
{
    nn_net_receiver_layer_t *receiver = (nn_net_receiver_layer_t *)layer;

    uint64_t rows, cols;
    if (recv_all(receiver->sockfd, &rows, sizeof(uint64_t)) < 0)
        return NN_ERR_IO;
    if (recv_all(receiver->sockfd, &cols, sizeof(uint64_t)) < 0)
        return NN_ERR_IO;

    size_t payload_bytes = rows * cols * sizeof(mat_data_t);
    if (recv_all(receiver->sockfd, out->data, payload_bytes) < 0)
        return NN_ERR_IO;

    out->rows = rows;
    out->cols = cols;

    return NN_OK;
}

nn_err_t nn_model_add_receiver(nn_model_t *model, int sockfd)
{
    nn_net_receiver_layer_t *receiver;
    arena_push(model->model_arena, sizeof(nn_net_receiver_layer_t), true,
               (void **)&receiver);

    receiver->sockfd = sockfd;
    receiver->base.type_id = LAYER_TYPE_NET_RECEIVER;
    receiver->base.forward = receiver_forward;
    receiver->base.compile = receiver_compile;
    receiver->base.init_params = NULL;
    receiver->base.serialize = NULL;

    model->layers[model->layer_count++] = (nn_layer_t *)receiver;
    return NN_OK;
}
