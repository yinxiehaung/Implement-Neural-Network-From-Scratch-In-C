#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include "nn.h"
#include "act.h"
#include "init.h"
#include "cost.h"
#include "nn_dataset.h"

typedef void (*nn_epoch_callback_t)(uint64_t epoch, mat_data_t train_loss);
void nn_train(nn_trainer_t *trainer, nn_dataloader_t *train_loader,
              const cost_api_t *cost_func, const nn_epoch_callback_t on_epoch_end);
#endif
