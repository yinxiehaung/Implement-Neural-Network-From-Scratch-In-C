#ifndef NN_H
#define NN_H
#include "matrix.h"
#include "nn_err_types.h"
#include "nn_layer_type.h"
#include "nn_layer.h"
#include "nn_dense_layer.h"
#include "nn_dataset.h"
#include "nn_activation_layer.h"
#include "act.h"
#include "init.h"
#include "cost.h"

typedef struct nn_model_s nn_model_t;
typedef struct nn_trainer_s nn_trainer_t;
typedef struct layer_desc_s layer_desc_t;

struct layer_desc_s {
    uint64_t in;
    uint64_t out;
    const act_api_t *activation;
};

struct nn_model_s {
    nn_layer_t **layers;
    uint64_t layer_count;
    uint64_t layer_capacity;
    matrix_t buffer_ping;
    matrix_t buffer_pong;
    matrix_t *last_output;
    arena_t *model_arena;
};

struct nn_trainer_s {
    mat_data_t learning_rate;
    mat_data_t weight_decay;
    uint64_t max_epochs;
    uint64_t batch_size;
};

#ifndef NDEBUG
#define NN_REQUIRE(cond, err_code)                                                  \
    do {                                                                            \
        if (!(cond)) {                                                              \
            return err_code;                                                        \
        }                                                                           \
    } while (0)
#define NN_CHECK(expr, label)                                                       \
    do {                                                                            \
        if ((expr) != NN_OK) {                                                      \
            goto label;                                                             \
        }                                                                           \
    } while (0)
#else
#define NN_REQUIRE(cond, err_code)                                                  \
    do {                                                                            \
        (void)(cond)                                                                \
    } while (0)
#define NN_CHECK(expr, label)                                                       \
    do {                                                                            \
        (void)(expr)                                                                \
    } while (0)
#endif
#define NN_PTR_NO_NULL(nn, err_code) NN_REQUIRE(PTR_NO_NULL(nn), err_code);
#define NN_PTR_IS_NULL(nn, err_code) NN_REQUIRE(PTR_IS_NULL(nn), err_code);

#define NN_LAYER_DESC_VAILD_INTERNAL(desc)                                          \
    ((desc).activation != NULL && (desc).initializer != NULL)
#define NN_LAYER_DESC_VAILD(desc) (NN_LAYER_DESC_VAILD_INTERNAL(desc))
#define NN_PTR_LAYER_DESC_VAILD(desc)                                               \
    (PTR_NO_NULL((desc)) && NN_LAYER_DESC_VAILD((*desc)))

#define NN_MODEL_VALID_INTERNAL(model)                                              \
    ((model).layers != NULL && (model).model_arena != NULL &&                       \
     (model).layer_capacity > 2)
#define NN_MODEL_VALID(model)                                                       \
    NN_REQUIRE(NN_MODEL_VALID_INTERNAL((model)), NN_ERR_INVALID_MODEL)
#define NN_PTR_MODEL_VALID(model)                                                   \
    NN_REQUIRE(PTR_NO_NULL((model)) && NN_MODEL_VALID_INTERNAL(*(model)),           \
               NN_ERR_INVALID_MODEL)

nn_err_t nn_model_init(nn_model_t *model, arena_t *arena, uint64_t max_layer);
nn_err_t nn_forward_train(nn_model_t *model, matrix_t *in);
nn_err_t nn_forward_predict(const nn_model_t *model, const matrix_t *in,
                            matrix_t *out);

nn_err_t nn_train(nn_model_t *model, nn_trainer_t *trainer, nn_dataset_t *dataset,
                  const cost_api_t *cost);
nn_err_t nn_model_compile(nn_model_t *model, uint64_t batch_size,
                          uint64_t input_dim);
nn_err_t nn_model_build_from_desc(nn_model_t *model, const layer_desc_t *arch,
                                  uint64_t arch_count);
nn_err_t nn_backprop(nn_model_t *model, nn_trainer_t *trainer, matrix_t *y_true,
                     const cost_api_t *cost);
nn_err_t nn_model_save(const nn_model_t *model, const char *filepath);
nn_err_t nn_model_load(nn_model_t *model, const char *filepath);
nn_err_t nn_model_load_partial(nn_model_t *model, const char *filepath,
                               uint64_t start_layer, uint64_t end_layer);
nn_err_t nn_model_split(nn_model_t *source, nn_model_t *part1, nn_model_t *part2,
                        uint64_t split_idx);
nn_err_t nn_model_init_params(nn_model_t *model, init_api_t *initializer,
                              uint64_t init_size);
void nn_model_layer_print(const nn_layer_t *);
void nn_model_print(const nn_model_t *);
void nn_trainer_print(const nn_trainer_t *);
#endif
