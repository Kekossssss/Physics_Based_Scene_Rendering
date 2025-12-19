#ifndef GPU_PART_HPP
#define GPU_PART_HPP

#define RENDERED_FRAMES 100

#include "utils.hpp"
#include "cuda.h"
#include <vector>

bool allocate_gpu_stream(cudaStream_t* gpu_stream);
void initiate_video_memory(id_array* gpu_id_array, image_array* gpu_image, gpu_object_pointers* gpu_obj_pointers, cudaStream_t* gpu_stream);
void clean_video_memory(id_array* gpu_id_array, image_array* gpu_image, gpu_object_pointers* gpu_obj_pointers, cudaStream_t* gpu_stream);

void copy_initial_data_to_video_memory_for_all_streams(gpu_object_pointers* gpu_obj_pointers, object_to_gpu obj, cudaStream_t* gpu_stream);
void copy_data_to_video_memory(gpu_object_pointers& gpu_obj_pointers, object_to_gpu& obj, cudaStream_t& gpu_stream);
void copy_data_from_video_memory(image_array& gpu_image, image_array& img, cudaStream_t& gpu_stream);

void synchronize_gpu_image(bool image_is_valid, gpu_object_pointers *gpu_obj_pointers, cudaStream_t *gpu_stream);

bool allocate_gpu_thread(dim3& numBlocks, dim3& threadsPerBlock);

bool draw_image(object_to_gpu& tab_pos, image_array& image,
              id_array* identifier_array, image_array* gpu_image, gpu_object_pointers* gpu_obj_pointers,
              cudaStream_t* gpu_stream,
              dim3 numBlocks, dim3 threadsPerBlock);

void benchmark_performance(int i, std::chrono::_V2::system_clock::time_point before_image_draw, time_benchmarking* time_table, gpu_object_pointers* gpu_obj_pointers, cudaStream_t* gpu_stream);
void reinit_terminal(int i);
void print_intermediate_bench_values(int i, time_benchmarking* time_table);
void compute_bench_values(int i, values_benchmarking& bench_values, time_benchmarking* time_table);

#endif