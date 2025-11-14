NVFLAGS := -std=c++11 -O3
TARGET := main

.PHONY: all
all: $(TARGET)

$(TARGET): main.cu
	nvcc $(NVFLAGS) -o main main.cu

clean:
	rm -rf main *.o




