#ifndef ARENA_H
#define ARENA_H
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/mman.h>

typedef struct arena_s arena_t;
typedef struct tmp_arena_s tmp_arena_t;
typedef uint8_t byte_t;

typedef enum {
    ARENA_OK = 0,
    ARENA_ERR_NOMEM = -1,
    ARENA_ERR_INVALID = -2,
    ARENA_ERR_SYS = -3
} arena_err_t;

struct arena_s {
    size_t capacity;
    size_t at;
    byte_t *base;
};

struct tmp_arena_s {
    struct arena_s *arena;
    size_t old_at;
};

#define MiB(x) ((x) << 20)
#define KiB(x) ((x) << 10)

arena_err_t arena_init(arena_t *, size_t);
void arena_free(arena_t *);
arena_err_t arena_push(arena_t *, size_t, bool, void **);

arena_err_t begin_tmp_arena(tmp_arena_t *, arena_t *);
void end_tmp_arena(tmp_arena_t *);

#define PTR_NO_NULL(ptr) ((ptr) != NULL)
#define PTR_IS_NULL(ptr) ((ptr) == NULL)
#define ARENA_MMAP_FLAGS MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE
#define PUSH_ALIGN_SIZE sizeof(void *)
#define ALIGN_UP_POW_2(x, align_size)                                               \
    (((size_t)(x) + ((size_t)(align_size) - 1)) & (~((size_t)(align_size) - 1)))

#define ARENA_CHECK(expr, label)                                                    \
    do {                                                                            \
        if ((expr) != ARENA_OK) {                                                   \
            goto label;                                                             \
        }                                                                           \
    } while (0)
#endif
