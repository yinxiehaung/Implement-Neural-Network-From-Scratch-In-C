CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -Iinclude
LDLIBS = -lm

ALL_SRCS = $(wildcard src/*.c)

NN_SRCS = $(filter-out src/server.c src/client.c, $(ALL_SRCS))

SERVER_SRCS = src/server.c $(NN_SRCS)
CLIENT_SRCS = src/client.c $(NN_SRCS)

.PHONY: all clean

all: server client

server: $(SERVER_SRCS)
	$(CC) $(CFLAGS) -o server $(SERVER_SRCS) $(LDLIBS)

client: $(CLIENT_SRCS)
	$(CC) $(CFLAGS) -o client $(CLIENT_SRCS) $(LDLIBS)

clean:
	rm -f server client
