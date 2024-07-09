# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O2

# Raylib and math libraries
LIBS = -lraylib -lm

# Output executable name
TARGET = neural_network

# Source file
SRC = neural_network.c

# Include directory for Raylib (adjust if necessary)
INCLUDES = -I/usr/local/include/raylib

# Library directory for Raylib (adjust if necessary)
LIBDIRS = -L/usr/local/lib

# Default rule to build the target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(INCLUDES) $(LIBDIRS) $(LIBS) -o $(TARGET)

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)

.PHONY: all clean
