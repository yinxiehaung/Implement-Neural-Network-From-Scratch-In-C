#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../../include/nn.h"
// #define NDEBUG

static uint32_t reverse_bytes(uint32_t val)
{
    return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) | ((val >> 8) & 0xff00) |
           ((val << 24) & 0xff000000);
}

void shuffle_data(matrix_t *x, matrix_t *y)
{
    for (uint64_t i = x->rows - 1; i > 0; i--) {
        uint64_t j = rand() % (i + 1);
        mat_swap_rows(x, i, j);
        mat_swap_rows(y, i, j);
    }
}

mat_err_t mnist_load_images(arena_t *arena, const char *filepath, matrix_t *out_x)
{
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filepath);
        return MATRIX_ERR_INVALID_ARG;
    }

    uint32_t magic = 0, count = 0, rows = 0, cols = 0;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1)
        goto file_err;
    if (fread(&count, sizeof(uint32_t), 1, file) != 1)
        goto file_err;
    if (fread(&rows, sizeof(uint32_t), 1, file) != 1)
        goto file_err;
    if (fread(&cols, sizeof(uint32_t), 1, file) != 1)
        goto file_err;

    magic = reverse_bytes(magic);
    count = reverse_bytes(count);
    rows = reverse_bytes(rows);
    cols = reverse_bytes(cols);
    if (magic != 2051) {
        printf("Error: Invalid MNIST image magic number (%u)\n", magic);
        fclose(file);
        return MATRIX_ERR_INVALID_ARG;
    }
    uint64_t pixels_per_image = rows * cols;
    if (mat_init(arena, count, pixels_per_image, out_x) != MATRIX_OK) {
        fclose(file);
        return MATRIX_ERR_ARENA;
    }
    uint8_t *buffer = (uint8_t *)malloc(pixels_per_image);
    if (!buffer)
        goto file_err;

    for (uint32_t i = 0; i < count; i++) {
        if (fread(buffer, sizeof(uint8_t), pixels_per_image, file) !=
            pixels_per_image) {
            free(buffer);
            goto file_err;
        }
        for (uint64_t j = 0; j < pixels_per_image; j++) {
            MAT_PTR_AT(out_x, i, j) = (mat_data_t)buffer[j] / 255.0f;
        }
    }

    free(buffer);
    fclose(file);
    printf("Successfully loaded %u MNIST images from %s\n", count, filepath);
    return MATRIX_OK;

file_err:
    fclose(file);
    return MATRIX_ERR_INVALID_ARG;
}

mat_err_t mnist_load_labels(arena_t *arena, const char *filepath, matrix_t *out_y)
{
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filepath);
        return MATRIX_ERR_INVALID_ARG;
    }

    uint32_t magic = 0, count = 0;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1)
        goto file_err;
    if (fread(&count, sizeof(uint32_t), 1, file) != 1)
        goto file_err;

    magic = reverse_bytes(magic);
    count = reverse_bytes(count);
    if (magic != 2049) {
        printf("Error: Invalid MNIST label magic number (%u)\n", magic);
        fclose(file);
        return MATRIX_ERR_INVALID_ARG;
    }
    if (mat_init(arena, count, 10, out_y) != MATRIX_OK) {
        fclose(file);
        return MATRIX_ERR_ARENA;
    }
    for (uint64_t i = 0; i < count * 10; i++) {
        out_y->data[i] = 0.0f;
    }

    for (uint32_t i = 0; i < count; i++) {
        uint8_t label;
        if (fread(&label, sizeof(uint8_t), 1, file) != 1)
            goto file_err;
        if (label < 10) {
            MAT_PTR_AT(out_y, i, label) = 1.0f;
        }
    }
    fclose(file);
    printf("Successfully loaded %u MNIST labels from %s\n", count, filepath);
    return MATRIX_OK;
file_err:
    fclose(file);
    return MATRIX_ERR_INVALID_ARG;
}

int infer(void)
{
    srand(time(NULL));
    arena_t arena = {0};
    if (arena_init(&arena, MiB(100)) != ARENA_OK) {
        printf("Error: Arena init failed\n");
        return 1;
    }

    nn_model_t model = {0};
    nn_model_init(&model, &arena, 20);
    if (nn_model_load(&model, "./mnist_model.bin") != NN_OK) {
        printf("Error: Model load failed\n");
        arena_free(&arena);
        return 1;
    }
    nn_model_compile(&model, 1, 784);
    nn_dataset_t test_dataset;
    mnist_load_images(&arena, "test-images-idx3-ubyte/t10k-images.idx3-ubyte",
                      &test_dataset.X);
    mnist_load_labels(&arena, "test-labels-idx1-ubyte/t10k-labels.idx1-ubyte",
                      &test_dataset.Y);
    matrix_t single_img;
    int r_idx = rand() % test_dataset.X.rows;
    single_img.rows = 1;
    single_img.cols = 784;
    single_img.data = test_dataset.X.data + (r_idx * 784);

    matrix_t out;
    if (mat_init(&arena, 1, 10, &out) != MATRIX_OK) {
        printf("Error: Output matrix init failed\n");
        arena_free(&arena);
        return 1;
    }

    if (nn_forward_predict(&model, &single_img, &out) != NN_OK) {
        printf("Error: Prediction failed\n");
        arena_free(&arena);
        return 1;
    }

    int best_digit = 0;
    mat_data_t max_prob = out.data[0];
    mat_data_t sum_prob = out.data[0];
    for (int i = 1; i < 10; i++) {

        printf("Lable %d prob:" MAT_FMT "\n", i, out.data[i]);
        if (out.data[i] > max_prob) {

            max_prob = out.data[i];

            best_digit = i;
        }
        sum_prob += out.data[i];
    }
    printf("sum_prob:" MAT_FMT "\n", sum_prob);
    int actual_digit = 0;
    mat_data_t *y_row = test_dataset.Y.data + (r_idx * 10);
    for (int i = 0; i < 10; i++) {
        if (y_row[i] == 1.0f) {
            actual_digit = i;
            break;
        }
    }

    printf("Predicted: %d\n", best_digit);
    printf("Actual: %d\n", actual_digit);
    printf("Confidence: %.2f%%\n", max_prob * 100.0f);

    arena_free(&arena);
    return 0;
}
int train(void)
{
    srand(67);
    arena_t arena = {0};
    if (arena_init(&arena, MiB(500)) != ARENA_OK) {
        printf("Error: Arena init failed\n");
        return 1;
    }

    nn_dataset_t train_dataset;
    mnist_load_images(&arena, "train-images-idx3-ubyte/train-images.idx3-ubyte",
                      &train_dataset.X);
    mnist_load_labels(&arena, "train-labels-idx1-ubyte/train-labels.idx1-ubyte",
                      &train_dataset.Y);

    nn_model_t model = {0};
    layer_desc_t arch[] = {{784, 128, &ACT_RELU}, {128, 10, &ACT_SOFTMAX}};
    init_api_t apis[] = {INIT_KAIMING, INIT_XAVIER};
    nn_model_init(&model, &arena, 20);
    if (nn_model_add_dense(&model, arch[0].in, arch[0].out) != NN_OK ||
        nn_model_add_activation(&model, &ACT_RELU) != NN_OK ||
        nn_model_add_dense(&model, arch[1].in, arch[1].out) != NN_OK ||
        nn_model_add_activation(&model, &ACT_SOFTMAX) != NN_OK) {
        printf("Error: Model build failed\n");
        arena_free(&arena);
        return 1;
    }
    nn_trainer_t trainer = {.weight_decay = 0.0f,
                            .learning_rate = 0.1f,
                            .batch_size = 128,
                            .max_epochs = 30};
    nn_model_compile(&model, trainer.batch_size, 784);
    nn_model_init_params(&model, apis, 2);
    nn_model_print(&model);
    printf("Starting training...\n");
    if (nn_train(&model, &trainer, &train_dataset, &COST_CROSS_ENTROPY) != NN_OK) {
        printf("Error: Training failed\n");
        arena_free(&arena);
        return 1;
    }

    if (nn_model_save(&model, "./mnist_model.bin") != NN_OK) {
        printf("Error: Model save failed\n");
        arena_free(&arena);
        return 1;
    }

    printf("Model saved to ./mnist_model.bin");

    arena_free(&arena);
    return 0;
}
int main(void)
{
    train();
    infer();
}
