#include "../include/nn.h"
#include <stdint.h>

nn_err_t nn_model_init(nn_model_t *model, arena_t *arena, uint64_t max_layer)
{
    NN_PTR_NO_NULL(model, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(arena, NN_ERR_NULL_PTR);

    model->layer_count = 0;
    model->layer_capacity = max_layer;
    model->model_arena = arena;

    arena_err_t a_err = arena_push(arena, sizeof(nn_layer_t *) * max_layer, true,
                                   (void **)&model->layers);
    NN_REQUIRE(a_err == ARENA_OK, NN_ERR_ARENA);

    model->buffer_ping.data = NULL;
    model->buffer_pong.data = NULL;
    model->buffer_ping.rows = 0;
    model->buffer_ping.cols = 0;
    model->buffer_pong.rows = 0;
    model->buffer_pong.cols = 0;

    return NN_OK;
}

nn_err_t nn_forward_train(nn_model_t *model, matrix_t *in)
{
    NN_PTR_MODEL_VALID(model);
    NN_REQUIRE(in && MAT_VALID_INTERNAL(*in), NN_ERR_INVALID_ARG);

    matrix_t *current_in = in;
    matrix_t *current_out = &model->buffer_ping;

    for (uint64_t i = 0; i < model->layer_count; i++) {
        nn_layer_t *layer = model->layers[i];
        nn_err_t err = layer->forward(current_out, current_in, layer);
        NN_REQUIRE(err == NN_OK, err);
        current_in = current_out;
        current_out = (current_in == &model->buffer_ping) ? &model->buffer_pong
                                                          : &model->buffer_ping;
    }
    model->last_output = current_in;
    return NN_OK;
}

nn_err_t nn_forward_predict(const nn_model_t *model, const matrix_t *in,
                            matrix_t *out)
{
    NN_PTR_MODEL_VALID(model);
    NN_REQUIRE(in && MAT_VALID_INTERNAL(*in), NN_ERR_INVALID_ARG);
    NN_REQUIRE(out && MAT_VALID_INTERNAL(*out), NN_ERR_INVALID_ARG);

    matrix_t *current_in = (matrix_t *)in;
    matrix_t *current_out = (matrix_t *)&model->buffer_ping;

    for (uint64_t i = 0; i < model->layer_count; i++) {
        nn_layer_t *layer = model->layers[i];

        nn_err_t err = layer->forward(current_out, current_in, layer);
        NN_REQUIRE(err == NN_OK, err);

        current_in = current_out;
        current_out = (current_in == &model->buffer_ping)
                          ? (matrix_t *)&model->buffer_pong
                          : (matrix_t *)&model->buffer_ping;
    }
    mat_err_t mat_err = mat_copy(out, current_in);
    NN_REQUIRE(mat_err == MATRIX_OK, NN_ERR_MAT_OP);
    return NN_OK;
}

nn_err_t nn_backprop(nn_model_t *model, nn_trainer_t *trainer, matrix_t *y_true,
                     const cost_api_t *cost)
{
    NN_PTR_MODEL_VALID(model);
    NN_PTR_NO_NULL(trainer, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(cost, NN_ERR_NULL_PTR);
    NN_REQUIRE(y_true && MAT_VALID_INTERNAL(*y_true), NN_ERR_INVALID_ARG);
    matrix_t *y_pred = model->last_output;
    matrix_t *dLoss =
        (y_pred == &model->buffer_ping) ? &model->buffer_pong : &model->buffer_ping;

    mat_err_t mat_err = cost->grad(dLoss, y_pred, y_true);
    NN_REQUIRE(mat_err == MATRIX_OK, NN_ERR_MATH);

    const matrix_t *current_grad_in = dLoss;

    matrix_t *current_grad_out = y_pred;

    for (int64_t i = model->layer_count - 1; i >= 0; i--) {
        nn_layer_t *layer = model->layers[i];
        if (i == 0) {
            current_grad_out = NULL;
        }
        nn_err_t err = layer->backprop(current_grad_out, current_grad_in, layer);
        NN_REQUIRE(err == NN_OK, err);
        current_grad_in = current_grad_out;
        if (current_grad_out != NULL) {
            current_grad_out = (current_grad_out == &model->buffer_ping)
                                   ? &model->buffer_pong
                                   : &model->buffer_ping;
        }
    }
    return NN_OK;
}

nn_err_t nn_model_init_params(nn_model_t *model, init_api_t *initializer,
                              uint64_t init_size)
{
    NN_PTR_MODEL_VALID(model);
    NN_PTR_NO_NULL(initializer, NN_ERR_NULL_PTR);
    uint64_t params_idx = 0;
    for (uint64_t i = 0; i < model->layer_count && params_idx < init_size; i++) {
        nn_layer_t *layer = model->layers[i];
        if (layer->init_params != NULL) {
            layer->init_params(layer, &initializer[params_idx]);
            params_idx++;
        }
    }
    return NN_OK;
}

nn_err_t nn_model_compile(nn_model_t *model, uint64_t batch_size, uint64_t input_dim)
{
    NN_PTR_MODEL_VALID(model);
    uint64_t current_dim = input_dim;
    uint64_t max_buffer = 0;
    for (uint64_t i = 0; i < model->layer_count; i++) {
        nn_layer_t *layer = model->layers[i];
        uint64_t out_dim = 0;
        uint64_t buffer_size = 0;
        if (layer->compile != NULL) {
            nn_err_t err = layer->compile(layer, model->model_arena, batch_size,
                                          current_dim, &out_dim, &buffer_size);
            if (err != NN_OK)
                return err;
        }

        current_dim = out_dim;
        if (buffer_size > max_buffer)
            max_buffer = buffer_size;
    }
    if (mat_init(model->model_arena, batch_size, max_buffer, &model->buffer_ping) !=
        MATRIX_OK) {
        return NN_ERR_ARENA;
    }

    if (mat_init(model->model_arena, batch_size, max_buffer, &model->buffer_pong) !=
        MATRIX_OK) {
        return NN_ERR_ARENA;
    }

    return NN_OK;
}
nn_err_t nn_model_build_from_desc(nn_model_t *model, const layer_desc_t *arch,
                                  uint64_t arch_count)
{
    NN_PTR_MODEL_VALID(model);
    NN_PTR_NO_NULL(arch, NN_ERR_NULL_PTR);
    for (uint64_t i = 0; i < arch_count; i++) {
        NN_CHECK(nn_model_add_dense(model, arch[i].in, arch[i].out), build_err);
        NN_CHECK(nn_model_add_activation(model, arch[i].activation), build_err);
    }
    return NN_OK;
build_err:
    return NN_ERR_INVALID_MODEL;
}

void nn_model_layer_print(const nn_layer_t *layer) { return; }

void nn_model_print(const nn_model_t *model)
{
    printf("The Model Layer:%lu\nThe Model Capacity:%lu\n", model->layer_count,
           model->layer_capacity);
    for (uint64_t i = 0; i < model->layer_count; i++) {
        printf("The Layer info: %lu\n-----------------------\n", i);
        if (model->layers[i]->type_id == LAYER_TYPE_DENSE) {
            const nn_dense_layer_t *dense = (nn_dense_layer_t *)model->layers[i];
            printf(
                "The Input dim:%lu\nThe Output dim:%lu\n-----------------------\n",
                dense->in_dim, dense->out_dim);
        }
    }
    return;
}
void nn_trainer_print(const nn_trainer_t *trainer) { return; }
