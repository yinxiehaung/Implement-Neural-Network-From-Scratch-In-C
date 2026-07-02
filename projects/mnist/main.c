#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../../include/nn_train.h"
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

int main(void)
{
    srand(time(NULL));
    arena_t arena;
    arena_init(&arena, MiB(250));
    matrix_t train_x, train_y;
    mnist_load_images(&arena, "./train-images-idx3-ubyte/train-images.idx3-ubyte",
                      &train_x);
    mnist_load_labels(&arena, "./train-labels-idx1-ubyte/train-labels.idx1-ubyte",
                      &train_y);
    uint64_t total_samples = train_x.rows;
    nn_dataset_t train_set = {.X = train_x, .Y = train_y};
    nn_dataloader_t loader;
    nn_dataloader_init(&loader, &train_set, 100);
    uint64_t arch[] = {784, 128, 10};
    act_api_t *acts[] = {(act_api_t *)&ACT_RELU, (act_api_t *)&ACT_SOFTMAX};
    init_api_t *inits[] = {(init_api_t *)&INIT_KAIMING, (init_api_t *)&INIT_XAVIER};
    nn_model_cfg_t model_cfg = {
        .arch_count = 3, .arch = arch, .activations = acts, .initializers = inits};
    nn_model_t model;
    nn_model_init(&arena, &model_cfg, &model);
    nn_train_cfg_t train_cfg = {
        .learning_rate = 0.5f, .max_epochs = 20, .batch_size = loader.batch_size};
    nn_trainer_t trainer;
    nn_trainer_init(&arena, &train_cfg, &model, &trainer);
    printf("Starting MNIST training...\n");
    nn_train(&trainer, &loader, &COST_CROSS_ENTROPY, NULL);
    printf("\nSaving trained model weights...\n");
    if (nn_model_save(&model, "mnist_model.bin") == NN_OK) {
        printf("Model successfully persisted to disk!\n");
    }
    arena_free(&arena);
    return 0;
}

/*int main(void)
{
    srand(time(NULL));

    arena_t model_arena;
    arena_init(&model_arena, MiB(250));

    arena_t scratch_arena;
    arena_init(&scratch_arena, MiB(250));

    uint64_t arch[] = {784, 128, 10};
    act_api_t *acts[] = {(act_api_t *)&ACT_RELU, (act_api_t *)&ACT_SOFTMAX};

    nn_model_cfg_t model_cfg = {
        .arch_count = 3, .arch = arch, .activations = acts, .initializers = NULL};

    nn_model_t model;
    nn_model_init(&model_arena, &model_cfg, &model);

    printf("Load Model...\n");
    if (nn_model_load(&model, "mnist_model.bin") != NN_OK) {
        printf("Error: mnist_model.bin Can't load.\n");
        arena_free(&model_arena);
        arena_free(&scratch_arena);
        return -1;
    }

    matrix_t test_x, test_y;
    mnist_load_images(&model_arena,
                      "./train-images-idx3-ubyte/train-images.idx3-ubyte", &test_x);
    mnist_load_labels(&model_arena,
                      "./train-labels-idx1-ubyte/train-labels.idx1-ubyte", &test_y);

    matrix_t prediction;
    if (mat_init(&model_arena, 1, 10, &prediction) != MATRIX_OK) {
        return -1;
    }

    for (int i = 0; i < 5; i++) {
        uint64_t rand_idx = rand() % test_x.rows;

        matrix_t single_image = {
            .data = test_x.data + (rand_idx * 784), .rows = 1, .cols = 784};

        if (nn_forward_predict(&scratch_arena, &model, &single_image, &prediction) !=
            NN_OK) {
            printf("Error:Backprop\n");
            break;
        }

        int pred_label = 0;
        mat_data_t max_prob = MAT_PTR_AT(&prediction, 0, 0);
        for (int j = 1; j < 10; j++) {
            mat_data_t prob = MAT_PTR_AT(&prediction, 0, j);
            if (prob > max_prob) {
                max_prob = prob;
                pred_label = j;
            }
        }
        int true_label = 0;
        for (int j = 0; j < 10; j++) {
            if (MAT_PTR_AT(&test_y, rand_idx, j) == 1.0f) {
                true_label = j;
                break;
            }
        }
        printf("Test Sample #%05llu | AI Infer: [%d] (Confindence: %5.1f%%) | True
Label: "
               "[%d] %s\n",
               (unsigned long long)rand_idx, pred_label, max_prob * 100.0f,
               true_label, (pred_label == true_label) ? "PASS ✅" : "FAIL ❌");
    }
    arena_free(&model_arena);
    arena_free(&scratch_arena);
    return 0;
}*/
