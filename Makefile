GCCFLAGS := -std=c++17 -O3 -fopenmp
NVFLAGS := -std=c++17 -O3 -Xcompiler -fopenmp
TARGET := main

.PHONY: all
all: $(TARGET)

$(TARGET): main.cu
	nvcc $(NVFLAGS) -o main main.cu

seq: sequential_code.cpp
	g++ $(GCCFLAGS) sequential_code.cpp cpu_renderer.cpp utils.cpp video_writer.cpp cpu_part.cpp cpu_converter.cpp -o sequential_code

main_cpu:  main_cpu.cpp
	g++ $(GCCFLAGS) main_cpu.cpp cpu_renderer.cpp utils.cpp video_writer.cpp cpu_part.cpp cpu_converter.cpp -o main_cpu

main_gpu : gpu_main.cu
	nvcc $(NVFLAGS) -o main_gpu gpu_main.cu gpu_part.cu utils.cpp video_writer.cpp cpu_part.cpp cpu_converter.cpp  

clean:
	rm -rf main *.o




