#include "../include/arena.h"
arena_err_t arena_init(arena_t *arena, size_t size)
{
    if (arena == NULL) {
        return ARENA_ERR_INVALID;
    }
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1) {
        return ARENA_ERR_SYS;
    }
    size_t align_size = ALIGN_UP_POW_2(size, page_size);
    byte_t *mem = mmap(NULL, sizeof(byte_t) * align_size, PROT_WRITE | PROT_READ,
                       ARENA_MMAP_FLAGS, -1, 0);
    if (mem == NULL) {
        return ARENA_ERR_SYS;
    }
    arena->base = mem;
    arena->at = 0;
    arena->capacity = align_size;
    return ARENA_OK;
}

void arena_free(arena_t *arena)
{
    if (arena == NULL) {
        return;
    }
    if (munmap(arena->base, arena->capacity) != 0) {
        return;
    }
    arena->at = 0;
    arena->capacity = 0;
    arena->base = NULL;
}

arena_err_t arena_push(arena_t *arena, size_t req_size, bool set_zero,
                       void **ptr_out)
{
    if (arena == NULL || ptr_out == NULL)
        return ARENA_ERR_INVALID;
    size_t align_at = ALIGN_UP_POW_2(arena->at, PUSH_ALIGN_SIZE);
    size_t new_at = align_at + req_size;

    if (new_at > arena->capacity) {
        return ARENA_ERR_NOMEM;
    }
    *ptr_out = (void *)(arena->base + align_at);
    arena->at = new_at;
    if (set_zero)
        memset(*ptr_out, 0, req_size);
    return ARENA_OK;
}

arena_err_t begin_tmp_arena(tmp_arena_t *tmp_arena, arena_t *arena)
{
    if (arena == NULL || tmp_arena == NULL) {
        return ARENA_ERR_INVALID;
    }
    tmp_arena->arena = arena;
    tmp_arena->old_at = arena->at;
    return ARENA_OK;
}

void end_tmp_arena(tmp_arena_t *tmp_arena)
{
    if (tmp_arena == NULL) {
        return;
    }
    tmp_arena->arena->at = tmp_arena->old_at;
    tmp_arena->arena = NULL;
}
