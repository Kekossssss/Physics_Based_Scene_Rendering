GCCFLAGS := -std=c++17 -O3
NVFLAGS := -std=c++17 -O3 -Xcompiler -fopenmp
TARGET := main

.PHONY: all
all: $(TARGET)

$(TARGET): main.cu
	nvcc $(NVFLAGS) -o main main.cu

main_cpu:  main_cpu.cpp
	g++ $(GCCFLAGS) utils.cpp cpu_renderer.cpp main_cpu.cpp -o main_cpu 

.PHONY: gpu
all: gpu_exec

gpu_exec: gpu_part.cu
	nvcc $(NVFLAGS) -o gpu_exec gpu_part.cu utils.cpp

.PHONY: cpu
all: cpu_renderer

cpu_renderer: cpu_renderer.cpp
	g++ $(GCCFLAGS) -o cpu_renderer cpu_renderer.cpp utils.cpp

clean:
	rm -rf main *.o




