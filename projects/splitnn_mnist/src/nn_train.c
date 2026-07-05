#include "../include/nn.h"

nn_err_t nn_trainer_step(nn_model_t *model, const nn_trainer_t *trainer)
{
    NN_PTR_MODEL_VALID(model);
    NN_PTR_NO_NULL(trainer, NN_ERR_NULL_PTR);

    for (uint64_t i = 0; i < model->layer_count; i++) {
        nn_layer_t *layer = model->layers[i];
        mat_data_t lr = trainer->learning_rate;
        if (layer->type_id == LAYER_TYPE_DENSE) {
            nn_dense_layer_t *dense = (nn_dense_layer_t *)layer;
            MAT_CHECK(mat_scale_add(&dense->w, &dense->w, &dense->dw, -lr), op_err);
            MAT_CHECK(mat_scale_add(&dense->b, &dense->b, &dense->db, -lr), op_err);
        }
    }
    return NN_OK;

op_err:
    return NN_ERR_MAT_OP;
}

nn_err_t nn_train(nn_model_t *model, nn_trainer_t *trainer, nn_dataset_t *dataset,
                  const cost_api_t *cost)
{
    NN_PTR_MODEL_VALID(model);
    NN_PTR_NO_NULL(trainer, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(dataset, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(cost, NN_ERR_NULL_PTR);

    NN_REQUIRE(dataset->X.rows == dataset->Y.rows, NN_ERR_MAT_SHAPE);

    matrix_t x_batch;
    matrix_t y_batch;

    nn_dataloader_t loader;

    for (uint64_t epoch = 0; epoch < trainer->max_epochs; epoch++) {
        mat_data_t epoch_loss = 0.0f;
        nn_dataset_shuffle(dataset);
        nn_dataloader_init(&loader, dataset, trainer->batch_size);
        while (nn_dataloader_next(&loader, &x_batch, &y_batch)) {
            NN_CHECK(nn_forward_train(model, &x_batch), train_err);

            if (cost->cost != NULL) {
                matrix_t *y_pred = model->last_output;
                epoch_loss += cost->cost(y_pred, &y_batch);
            }
            for (uint64_t i = 0; i < model->layer_count; i++) {
                if (model->layers[i]->type_id == LAYER_TYPE_DENSE) {
                    nn_dense_layer_t *dense = (nn_dense_layer_t *)model->layers[i];
                    memset(dense->dw.data, 0,
                           sizeof(mat_data_t) * dense->dw.rows * dense->dw.cols);
                    memset(dense->db.data, 0,
                           sizeof(mat_data_t) * dense->db.rows * dense->db.cols);
                }
            }
            NN_CHECK(nn_backprop(model, trainer, &y_batch, cost), train_err);
            NN_CHECK(nn_trainer_step(model, trainer), train_err);
        }

        if (cost->cost != NULL && loader.num_batches > 0) {
            printf("Epoch %lu/%lu - Loss: " MAT_FMT "\n", epoch + 1,
                   trainer->max_epochs, epoch_loss / loader.num_batches);
        }
    }

    return NN_OK;

train_err:
    return NN_ERR_TRAIN_FAILED;
}
