#ifndef NN_ERR_TYPES_H
#define NN_ERR_TYPES_H
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
    NN_ERR_IO = -13,
    NN_ERR_MATH = -14,
    NN_ERR_MODEL_FULL = -15,
    NN_ERR_TRAIN_FAILED = -16,
    NN_ERR_COMPILE = -17,
    NN_ERR_LAYER_ADD = -18,
    NN_ERR_MAT_INIT = -19
} nn_err_t;
#endif
