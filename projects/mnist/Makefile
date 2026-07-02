CC = gcc
# -I../include 告訴編譯器：往上一層去找 include 資料夾
CFLAGS = -Wall -Wextra -O3 -fopenmp -g -march=native -I../../include
LDFLAGS = -lm

# 路徑設定 (相對於 tests/ 目錄)
SRC_DIR = ../../src
OBJ_DIR = obj

# 抓取檔案
TEST_SRC = main.c
ENGINE_SRCS = $(wildcard $(SRC_DIR)/*.c)

# 將 .c 轉換為 obj/ 下的 .o
TEST_OBJ = $(OBJ_DIR)/main.o
ENGINE_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(ENGINE_SRCS))

OBJS = $(TEST_OBJ) $(ENGINE_OBJS)

# 最終執行檔名稱 (直接產在 tests/ 目錄下)
TARGET = mnist

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 編譯 tests/ 裡面的 main.c
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# 編譯 ../src/ 裡面的引擎核心代碼
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
