#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "../include/net_layer.h"

int main()
{
    arena_t arena = {0};
    if (arena_init(&arena, MiB(100)) != ARENA_OK) {
        printf("Error: Arena init failed\n");
        return 1;
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in address = {.sin_family = AF_INET,
                                  .sin_addr.s_addr = INADDR_ANY,
                                  .sin_port = htons(8080)};

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("[Error] Bind failed");
        return -1;
    }
    listen(server_fd, 3);

    printf("[Server] Waiting for Client on port 8080...\n");
    int client_socket = accept(server_fd, NULL, NULL);
    if (client_socket < 0) {
        perror("[Error] Accept failed");
        return -1;
    }
    printf("[Server] Client Connected!\n");

    nn_model_t model = {0};
    nn_model_init(&model, &arena, 10);

    if (nn_model_add_receiver(&model, client_socket) != NN_OK) {
        printf("[Error] Failed to Add Receiver.\n");
        return 1;
    }

    if (nn_model_load_partial(&model, "mnist_model.bin", 2, 3) != NN_OK) {
        printf("[Error] Failed to load partial model.\n");
        return 1;
    }

    if (nn_model_compile(&model, 1, 128) != NN_OK) {
        printf("[Error] Model compile failed.\n");
        return 1;
    }

    matrix_t dummy_in = {0};
    mat_init(&arena, 1, 128, &dummy_in);
    matrix_t final_out;
    mat_init(&arena, 1, 10, &final_out);

    printf("[Server] Waiting for features & running inference...\n");

    nn_err_t err_code = nn_forward_predict(&model, &dummy_in, &final_out);

    if (err_code != NN_OK) {
        printf("[Error] Prediction failed with internal error: %d\n", err_code);
        return 1;
    }
    printf("[Server] Inference complete. Sending 1x10 predictions back...\n");
    if (net_send_matrix(client_socket, &final_out) != NN_OK) {
        printf("[Error] Failed to send predictions back.\n");
        return 1;
    }

    close(client_socket);
    close(server_fd);
    arena_free(&arena);
    return 0;
}
