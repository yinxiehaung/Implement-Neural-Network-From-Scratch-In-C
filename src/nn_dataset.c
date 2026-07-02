#include "../include/nn_dataset.h"
#include <stdlib.h>

void nn_dataloader_init(nn_dataloader_t *loader, const nn_dataset_t *dataset,
                        uint64_t batch_size)
{
    loader->dataset = dataset;
    loader->batch_size = batch_size;
    loader->current_batch = 0;
    loader->num_batches = dataset->X.rows / batch_size;
}

bool nn_dataloader_next(nn_dataloader_t *loader, matrix_t *out_px, matrix_t *out_py)
{

    if (loader->current_batch >= loader->num_batches) {
        loader->current_batch = 0;
        return false;
    }

    uint64_t b = loader->current_batch;
    uint64_t bs = loader->batch_size;
    uint64_t x_cols = loader->dataset->X.cols;
    uint64_t y_cols = loader->dataset->Y.cols;

    out_px->data = loader->dataset->X.data + (b * bs * x_cols);
    out_px->rows = bs;
    out_px->cols = x_cols;

    out_py->data = loader->dataset->Y.data + (b * bs * y_cols);
    out_py->rows = bs;
    out_py->cols = y_cols;

    loader->current_batch++;
    return true;
}

void nn_dataset_shuffle(nn_dataset_t *dataset)
{
    for (uint64_t i = dataset->X.rows - 1; i > 0; i--) {
        uint64_t j = rand() % (i + 1);
        mat_swap_rows(&dataset->X, i, j);
        mat_swap_rows(&dataset->Y, i, j);
    }
}
