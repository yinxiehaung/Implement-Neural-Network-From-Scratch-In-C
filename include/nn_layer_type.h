#ifndef NN_LAYER_TYPE_H
#define NN_LAYER_TYPE_H
typedef enum {
    LAYER_TYPE_UNKNOWN = 0,
    LAYER_TYPE_DENSE = 1,
    LAYER_TYPE_ACTIVATION = 2,
    LAYER_TYPE_COUNT,
} layer_type_t;

#define MAX_LAYER_TYPES LAYER_TYPE_COUNT
#endif
