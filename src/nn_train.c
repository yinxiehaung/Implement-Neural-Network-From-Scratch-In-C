#include "../include/nn_train.h"
#include <stdint.h>

void nn_train(nn_trainer_t *trainer, nn_dataloader_t *train_loader,
              const cost_api_t *cost_func, const nn_epoch_callback_t on_epoch_end)
{
    nn_model_t *model = trainer->model;
    nn_model_cfg_t model_cfg = model->cfg;
    nn_dataset_t *train_set = (nn_dataset_t *)train_loader->dataset;
    const nn_train_cfg_t *train_cfg = &trainer->cfg;
    uint64_t epoch = 1;
    for (; epoch <= train_cfg->max_epochs; epoch++) {
        nn_dataset_shuffle(train_set);
        matrix_t batch_x, batch_y;
        mat_data_t total_loss = 0.0f;
        while (nn_dataloader_next(train_loader, &batch_x, &batch_y)) {
            nn_forward_train(trainer, &batch_x);
            nn_backprop(trainer, &batch_y, cost_func);
            mat_data_t scaled_lr = train_cfg->learning_rate / train_cfg->batch_size;
            for (uint64_t i = 0; i < model->cfg.arch_count - 1; i++) {
                mat_scale_add(&model->ws[i], &model->ws[i], &trainer->dws[i],
                              -scaled_lr);
                mat_scale_add(&model->bs[i], &model->bs[i], &trainer->dbs[i],
                              -scaled_lr);
            }
            matrix_t *pred = &trainer->as[model_cfg.arch_count - 1];
            total_loss += cost_func->cost(pred, &batch_y);
        }

        mat_data_t avg_train_loss = total_loss / train_loader->num_batches;
        if (on_epoch_end) {
            on_epoch_end(epoch, avg_train_loss);
        } else {
            printf("Epoch %03llu | Train Loss: " MAT_FMT "\n",
                   (unsigned long long)epoch, avg_train_loss);
        }
    }
}
