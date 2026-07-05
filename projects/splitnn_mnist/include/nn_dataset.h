#ifndef NN_DATASET_H
#define NN_DATASET_H

#include "matrix.h"

typedef struct {
    matrix_t X;
    matrix_t Y;
} nn_dataset_t;

typedef struct {
    const nn_dataset_t *dataset;
    uint64_t batch_size;
    uint64_t current_batch;
    uint64_t num_batches;
} nn_dataloader_t;

void nn_dataloader_init(nn_dataloader_t *loader, const nn_dataset_t *dataset,
                        uint64_t batch_size);

bool nn_dataloader_next(nn_dataloader_t *loader, matrix_t *out_px, matrix_t *out_py);

void nn_dataset_shuffle(nn_dataset_t *dataset);

#endif
