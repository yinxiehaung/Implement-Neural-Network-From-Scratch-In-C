CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp -g -march=native -I../../include
LDFLAGS = -lm

SRC_DIR = ../../src
OBJ_DIR = obj

TEST_SRC = main.c
ENGINE_SRCS = $(wildcard $(SRC_DIR)/*.c)

TEST_OBJ = $(OBJ_DIR)/main.o
ENGINE_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(ENGINE_SRCS))

OBJS = $(TEST_OBJ) $(ENGINE_OBJS)

TARGET = mnist

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
