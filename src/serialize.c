#include "../include/nn.h"
#include <stdio.h>
#include <stdlib.h>

#define NN_MAGIC 0x59585348

nn_err_t nn_model_save(const nn_model_t *model, const char *filepath)
{
    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        printf("Error: Could not open file %s for saving.\n", filepath);
        return NN_ERR_FILE;
    }

    uint32_t magic = NN_MAGIC;
    fwrite(&magic, sizeof(uint32_t), 1, fp);

    uint64_t arch_count = model->cfg.arch_count;
    fwrite(&arch_count, sizeof(uint64_t), 1, fp);
    fwrite(model->cfg.arch, sizeof(uint64_t), arch_count, fp);

    for (uint64_t i = 0; i < arch_count - 1; i++) {
        uint64_t w_size = model->ws[i].rows * model->ws[i].cols;
        uint64_t b_size = model->bs[i].rows * model->bs[i].cols;

        fwrite(model->ws[i].data, sizeof(mat_data_t), w_size, fp);
        fwrite(model->bs[i].data, sizeof(mat_data_t), b_size, fp);
    }

    fclose(fp);
    printf("Model successfully saved to %s\n", filepath);
    return NN_OK;
}

nn_err_t nn_model_load(nn_model_t *model, const char *filepath)
{
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        printf("Error: Could not open file %s for loading.\n", filepath);
        return NN_ERR_FILE;
    }

    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != NN_MAGIC) {
        printf("Error: Invalid model file format (Magic Number mismatch).\n");
        fclose(fp);
        return NN_ERR_FILE;
    }

    uint64_t arch_count;
    if (fread(&arch_count, sizeof(uint64_t), 1, fp) != 1 ||
        arch_count != model->cfg.arch_count) {
        printf("Error: Architecture count mismatch (File: %llu, Model: %llu).\n",
               (unsigned long long)arch_count,
               (unsigned long long)model->cfg.arch_count);
        fclose(fp);
        return NN_ERR_FILE;
    }

    uint64_t *file_arch = (uint64_t *)malloc(sizeof(uint64_t) * arch_count);
    fread(file_arch, sizeof(uint64_t), arch_count, fp);
    for (uint64_t i = 0; i < arch_count; i++) {
        if (file_arch[i] != model->cfg.arch[i]) {
            printf("Error: Layer %llu size mismatch (File: %llu, Model: %llu).\n",
                   (unsigned long long)i, (unsigned long long)file_arch[i],
                   (unsigned long long)model->cfg.arch[i]);
            free(file_arch);
            fclose(fp);
            return NN_ERR_FILE;
        }
    }
    free(file_arch);

    for (uint64_t i = 0; i < arch_count - 1; i++) {
        uint64_t w_size = model->ws[i].rows * model->ws[i].cols;
        uint64_t b_size = model->bs[i].rows * model->bs[i].cols;

        fread(model->ws[i].data, sizeof(mat_data_t), w_size, fp);
        fread(model->bs[i].data, sizeof(mat_data_t), b_size, fp);
    }

    fclose(fp);
    printf("Model successfully loaded from %s\n", filepath);
    return NN_OK;
}
