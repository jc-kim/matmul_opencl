TARGET=mat_mul
OBJS=mat_mul.o timers.o opencl_errors.o

CC=gcc
CFLAGS=-g -O4 -Wall
LDFLAGS=-lOpenCL

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add ./$(TARGET) -v

run_test: $(TARGET)
	./$(TARGET) -v -s 10

run_gpu: $(TARGET)
	thorq --add --device gpu ./$(TARGET) -v -d gpu -s 10000
