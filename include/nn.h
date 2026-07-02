#ifndef NN_H
#define NN_H
#include "matrix.h"

typedef struct nn_model_s nn_model_t;
typedef struct nn_trainer_s nn_trainer_t;
typedef struct nn_model_cfg_s nn_model_cfg_t;
typedef struct nn_train_cfg_s nn_train_cfg_t;

typedef mat_err_t (*act_func_ptr)(matrix_t *, const matrix_t *);
typedef mat_err_t (*grad_func_ptr)(matrix_t *, const matrix_t *, const matrix_t *);
typedef mat_data_t (*cost_func_ptr)(const matrix_t *, const matrix_t *);
typedef mat_err_t (*w_init_func_ptr)(matrix_t *w, uint64_t fan_in, uint64_t fan_out);
typedef mat_err_t (*b_init_func_ptr)(matrix_t *b);

typedef struct act_api_s act_api_t;
typedef struct cost_api_s cost_api_t;
typedef struct init_api_s init_api_t;

struct init_api_s {
    w_init_func_ptr init_weight;
    b_init_func_ptr init_bias;
};

struct act_api_s {
    act_func_ptr forward;
    grad_func_ptr backprop;
};

struct cost_api_s {
    cost_func_ptr cost;
    grad_func_ptr grad;
};

struct nn_model_cfg_s {
    uint64_t arch_count;
    uint64_t *arch;
    act_api_t **activations;
    init_api_t **initializers;
};

struct nn_train_cfg_s {
    mat_data_t learning_rate;
    mat_data_t weight_decay;
    uint64_t max_epochs;
    uint64_t batch_size;
};

struct nn_model_s {
    nn_model_cfg_t cfg;
    matrix_t *ws;
    matrix_t *bs;
};

struct nn_trainer_s {
    struct nn_model_s *model;
    struct nn_train_cfg_s cfg;
    matrix_t *as;
    matrix_t *dws;
    matrix_t *dbs;
    matrix_t *das;
    matrix_t *transposed_w_scratch;
    matrix_t *delta_act_scratch;
};

typedef enum {
    NN_OK = 0,
    NN_ERR_INVALID_ARCH = -1,
    NN_ERR_ARENA = -2,
    NN_ERR_MAT_SHAPE = -3,
    NN_ERR_MAT_OP = -4,
    NN_ERR_NULL_PTR = -5,
    NN_ERR_INVALID_ARG = -6,
    NN_ERR_INVALID_ACT = -7,
    NN_ERR_FINTED_DIFF = -8,
    NN_ERR_BACKPROP = -9,
    NN_ERR_INVALID_MODEL = -10,
    NN_ERR_INVALID_TRAINER = -11,
    NN_ERR_INIT = -12,
    NN_ERR_FILE = -13
} nn_err_t;

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

#define NN_MODEL_CFG_VALID_INTERNAL(cfg)                                            \
    ((cfg).arch_count >= 2 && (cfg).arch != NULL && (cfg).activations != NULL)
#define NN_MODEL_CFG_VALID(cfg)                                                     \
    NN_REQUIRE(NN_MODEL_CFG_VALID_INTERNAL((cfg)), NN_ERR_INVALID_ARCH)
#define NN_PTR_MODEL_CFG_VALID(cfg)                                                 \
    NN_REQUIRE(PTR_NO_NULL(cfg) && NN_MODEL_CFG_VALID_INTERNAL(*(cfg)),             \
               NN_ERR_INVALID_ARCH)

#define NN_TRAIN_CFG_VALID_INTERNAL(cfg)                                            \
    ((cfg).batch_size > 0 && (cfg).max_epochs > 0)
#define NN_TRAIN_CFG_VALID(cfg)                                                     \
    NN_REQUIRE(NN_TRAIN_CFG_VALID_INTERNAL((cfg)), NN_ERR_INVALID_ARG)
#define NN_PTR_TRAIN_CFG_VALID(cfg)                                                 \
    NN_REQUIRE(PTR_NO_NULL(cfg) && NN_TRAIN_CFG_VALID_INTERNAL(*(cfg)),             \
               NN_ERR_INVALID_ARG)

#define NN_MODEL_VALID_INTERNAL(model)                                              \
    (NN_MODEL_CFG_VALID_INTERNAL((model).cfg) && (model).ws != NULL &&              \
     (model).bs != NULL)
#define NN_MODEL_VALID(model)                                                       \
    NN_REQUIRE(NN_MODEL_VALID_INTERNAL((model)), NN_ERR_INVALID_MODEL)
#define NN_PTR_MODEL_VALID(model)                                                   \
    NN_REQUIRE(PTR_NO_NULL((model)) && NN_MODEL_VALID_INTERNAL(*(model)),           \
               NN_ERR_INVALID_MODEL)

#define NN_TRAINER_VALID_INTERNAL(trainer)                                          \
    (NN_MODEL_VALID_INTERNAL(*((trainer).model)) &&                                 \
     NN_TRAIN_CFG_VALID_INTERNAL((trainer).cfg) && (trainer).as != NULL &&          \
     (trainer).dws != NULL && (trainer).dbs != NULL && (trainer).das != NULL &&     \
     (trainer).transposed_w_scratch != NULL && (trainer).delta_act_scratch != NULL)
#define NN_TRAINER_VALID(trainer)                                                   \
    NN_REQUIRE(NN_TRAINER_VALID_INTERNAL((trainer)), NN_ERR_INVALID_TRAINER)
#define NN_PTR_TRAINER_VALID(trainer)                                               \
    NN_REQUIRE(PTR_NO_NULL(trainer) && NN_TRAINER_VALID_INTERNAL(*(trainer)),       \
               NN_ERR_INVALID_TRAINER)

nn_err_t nn_model_init(arena_t *, const nn_model_cfg_t *, nn_model_t *);
nn_err_t nn_trainer_init(arena_t *, const nn_train_cfg_t *, const nn_model_t *,
                         nn_trainer_t *);
nn_err_t nn_forward_train(nn_trainer_t *, matrix_t *);
nn_err_t nn_forward_predict(arena_t *, const nn_model_t *, const matrix_t *,
                            matrix_t *);
nn_err_t nn_finte_diff(nn_trainer_t *, matrix_t *, matrix_t *, const mat_data_t,
                       const cost_api_t *);
nn_err_t nn_backprop(nn_trainer_t *, matrix_t *, const cost_api_t *);
nn_err_t nn_model_save(const nn_model_t *, const char *);
nn_err_t nn_model_load(nn_model_t *model, const char *filepath);

void nn_model_print(nn_model_t *);
void nn_trainer_print(nn_trainer_t *);
#endif
