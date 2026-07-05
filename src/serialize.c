#include <string.h>
#include "../include/nn.h"

static layer_deserialize_fn layer_registry[MAX_LAYER_TYPES] = {
    [LAYER_TYPE_DENSE] = dense_layer_deserialize,
    [LAYER_TYPE_ACTIVATION] = act_layer_deserialize,
};

nn_err_t nn_model_save(const nn_model_t *model, const char *filepath)
{
    NN_PTR_NO_NULL(model, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(filepath, NN_ERR_NULL_PTR);

    FILE *fp = fopen(filepath, "wb");
    NN_REQUIRE(fp != NULL, NN_ERR_IO);

    if (fwrite(&model->layer_count, sizeof(uint64_t), 1, fp) != 1) {
        fclose(fp);
        return NN_ERR_IO;
    }

    for (uint64_t i = 0; i < model->layer_count; i++) {
        nn_layer_t *layer = model->layers[i];
        if (fwrite(&layer->type_id, sizeof(uint32_t), 1, fp) != 1) {
            fclose(fp);
            return NN_ERR_IO;
        }
        if (layer->serialize != NULL) {
            nn_err_t err = layer->serialize(layer, fp);
            if (err != NN_OK) {
                fclose(fp);
                return err;
            }
        }
    }
    fclose(fp);
    return NN_OK;
}

nn_err_t nn_model_load(nn_model_t *model, const char *filepath)
{
    FILE *fp = fopen(filepath, "rb");
    NN_REQUIRE(fp != NULL, NN_ERR_IO);

    uint64_t total_layers;
    if (fread(&total_layers, sizeof(uint64_t), 1, fp) != 1) {
        fclose(fp);
        return NN_ERR_IO;
    }

    for (uint64_t i = 0; i < total_layers; i++) {
        layer_type_t type_id;
        if (fread(&type_id, sizeof(uint32_t), 1, fp) != 1) {
            fclose(fp);
            return NN_ERR_IO;
        }
        if (type_id < LAYER_TYPE_COUNT && layer_registry[type_id] != NULL) {
            nn_err_t err = layer_registry[type_id](model, fp, true);
            if (err != NN_OK) {
                fclose(fp);
                return err;
            }
        } else {
            fclose(fp);
            return NN_ERR_INVALID_MODEL;
        }
    }

    fclose(fp);
    return NN_OK;
}
nn_err_t nn_model_load_partial(nn_model_t *model, const char *filepath,
                               uint64_t start_layer, uint64_t end_layer)
{
    FILE *fp = fopen(filepath, "rb");
    NN_REQUIRE(fp != NULL, NN_ERR_IO);

    uint64_t total_layers;
    fread(&total_layers, sizeof(uint64_t), 1, fp);

    for (uint64_t i = 0; i < total_layers; i++) {
        uint32_t type_id;
        fread(&type_id, sizeof(uint32_t), 1, fp);
        bool should_load = (i >= start_layer && i <= end_layer);
        if (layer_registry[type_id] != NULL) {
            nn_err_t err = layer_registry[type_id](model, fp, should_load);
            if (err != NN_OK) {
                fclose(fp);
                return err;
            }
        } else {
            fclose(fp);
            return NN_ERR_INVALID_MODEL;
        }
    }
    fclose(fp);
    return NN_OK;
}

nn_err_t nn_model_split(nn_model_t *source, nn_model_t *part1, nn_model_t *part2,
                        uint64_t split_idx)
{
    NN_PTR_NO_NULL(part1, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(part2, NN_ERR_NULL_PTR);
    NN_REQUIRE(source->layer_count > split_idx, NN_ERR_INVALID_ARG);
    nn_model_init(part1, source->model_arena, split_idx);
    nn_model_init(part2, source->model_arena, source->layer_count - split_idx);
    for (uint32_t i = 0; i < split_idx; i++) {
        part1->layers[part1->layer_count++] = source->layers[i];
    }

    for (uint32_t i = split_idx; i < source->layer_count; i++) {
        part2->layers[part2->layer_count++] = source->layers[i];
    }
    return NN_OK;
}
