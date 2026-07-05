#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "../include/net_layer.h"

static uint32_t reverse_bytes(uint32_t val)
{
    return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) | ((val >> 8) & 0xff00) |
           ((val << 24) & 0xff000000);
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

int main()
{
    srand(time(NULL));
    arena_t arena = {0};
    if (arena_init(&arena, MiB(100)) != ARENA_OK)
        return 1;

    nn_dataset_t test_dataset;
    mnist_load_images(&arena, "./test-images-idx3-ubyte/t10k-images.idx3-ubyte",
                      &test_dataset.X);
    mnist_load_labels(&arena, "./test-labels-idx1-ubyte/t10k-labels.idx1-ubyte",
                      &test_dataset.Y);

    int r_idx = rand() % test_dataset.X.rows;
    matrix_t single_img;
    single_img.rows = 1;
    single_img.cols = 784;
    single_img.data = test_dataset.X.data + (r_idx * 784);

    int actual_digit = 0;
    mat_data_t *y_row = test_dataset.Y.data + (r_idx * 10);
    for (int i = 0; i < 10; i++) {
        if (y_row[i] == 1.0f) {
            actual_digit = i;
            break;
        }
    }

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv_addr = {.sin_family = AF_INET, .sin_port = htons(8080)};
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("[Error] Connection failed");
        return -1;
    }

    nn_model_t model = {0};
    nn_model_init(&model, &arena, 10);

    if (nn_model_load_partial(&model, "mnist_model.bin", 0, 1) != NN_OK) {
        printf("[Error] Failed to load partial model.\n");
        return 1;
    }

    if (nn_model_add_emitter(&model, sock) != NN_OK) {
        printf("[Error] Failed to add emitter layer.\n");
        return 1;
    }

    if (nn_model_compile(&model, 1, 784) != NN_OK) {
        printf("[Error] Model compile failed.\n");
        return 1;
    }

    matrix_t dummy_out;
    mat_init(&arena, 1, 128, &dummy_out);

    printf("[Client] Running partial inference & sending features...\n");

    if (nn_forward_predict(&model, &single_img, &dummy_out) != NN_OK) {
        printf("[Error] Prediction failed\n");
        return 1;
    }
    matrix_t final_pred = {0};
    printf("[Client] Waiting for prediction from Server...\n");

    if (net_recv_matrix(sock, &final_pred, &arena) != NN_OK) {
        printf("[Error] Failed to receive prediction.\n");
        return 1;
    }
    int best_digit = 0;
    mat_data_t max_prob = final_pred.data[0];
    for (int i = 1; i < 10; i++) {
        if (final_pred.data[i] > max_prob) {
            max_prob = final_pred.data[i];
            best_digit = i;
        }
    }

    printf("\n========================================\n");
    printf("🎯 Predicted Digit: %d\n", best_digit);
    printf("📝 Actual Digit:    %d\n", actual_digit);
    printf("📊 Confidence:      %.2f%%\n", max_prob * 100.0f);
    printf("========================================\n");

    close(sock);
    arena_free(&arena);
    return 0;
}
