#include "utils.hpp"

//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//
#include "cuda.h"
#include "omp.h"
#include "cpu_part.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Library used for sleep function
#include <unistd.h>

#include "cpu_converter.hpp"
#include "cpu_renderer.hpp"
#include "cpu_part.hpp"
#include "gpu_part.hpp"

#include <stdlib.h>
#include <iostream>
#include <thread>
#include <pthread.h>

#include "video_writer.hpp"

extern double gravity;

//------------------------------------------------------------------------------------------//
// Main
//------------------------------------------------------------------------------------------//
int main(int argc, char **argv)
{
    printf("Program Starting\n");

    const char *output_file = "output.mp4";

    // Performance debug values
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::_V2::system_clock::time_point before_image_draw;
    time_benchmarking time_table[RENDERED_FRAMES];
    values_benchmarking bench_values;
    bench_values.min_time = std::numeric_limits<double>::infinity();
    bench_values.max_time = 0.0;
    bench_values.mean_time = 0.0;
    bench_values.mean_time_render = 0.0;
    bench_values.mean_time_copy_to = 0.0;
    bench_values.mean_time_compute = 0.0;
    bench_values.mean_time_copy_from = 0.0;

    //----------------------------------------------------------------------
    //      Exemple of objects
    //----------------------------------------------------------------------
    std::vector<Shape *> shapes;

    int k = 2.5;
    int c = 1000;

    shapes.push_back(new Sphere(Point3D(200 * k + c, 200 * k, 0), 90 * k, Point3D(200, 100, 0), 100, 0.95, 9.81));
    shapes.push_back(new Cube(Point3D(400 * k + c, 200 * k, 0), 90 * k, Point3D(-100, 200, 0), Point3D(0.2, 0.2, 0), Point3D(1, 1, 1), 100, 0.95, 9.81));
    shapes.push_back(new Sphere(Point3D(600 * k + c, 200 * k, 0), 90 * k, Point3D(50, 50, 0), 100, 0.95, 9.81));
    shapes.push_back(new Cube(Point3D(800 * k + c, 200 * k, 0), 90 * k, Point3D(-200, 100, 0), Point3D(0, 0, 0), Point3D(2, 0, 1), 100, 0.95, 9.81));

    shapes.push_back(new Cube(Point3D(250 * k + c, 400 * k, 0), 90 * k, Point3D(150, -50, 0), Point3D(0.5, 0, 0), Point3D(0, 2, 0), 100, 0.95, 9.81));
    shapes.push_back(new Sphere(Point3D(450 * k + c, 400 * k, 0), 90 * k, Point3D(-50, -150, 0), 100, 0.95, 9.81));
    shapes.push_back(new Cube(Point3D(650 * k + c, 400 * k, 0), 90 * k, Point3D(100, 100, 0), Point3D(0, 0, 0.5), Point3D(1, 1, 0), 100, 0.95, 9.81));
    shapes.push_back(new Sphere(Point3D(850 * k + c, 400 * k, 0), 90 * k, Point3D(-300, 0, 0), 100, 0.95, 9.81));

    shapes.push_back(new Sphere(Point3D(200 * k + c, 600 * k, 0), 90 * k, Point3D(0, -400, 0), 100, 0.95, 9.81));
    shapes.push_back(new Cube(Point3D(400 * k + c, 600 * k, 0), 90 * k, Point3D(50, -200, 0), Point3D(0, 0, 0), Point3D(3, 0, 0), 100, 0.95, 9.81));
    shapes.push_back(new Sphere(Point3D(600 * k + c, 600 * k, 0), 90 * k, Point3D(-50, -200, 0), 100, 0.95, 9.81));
    shapes.push_back(new Cube(Point3D(800 * k + c, 600 * k, 0), 90 * k, Point3D(200, -300, 0), Point3D(0.1, 0.1, 0.1), Point3D(0, 3, 0), 100, 0.95, 9.81));

    
    // Positions of objects on the screen
    object_to_gpu tab_pos;

    // GPU conversion
    int numObjects = convertSceneToGPU(shapes, tab_pos, true);
    std::cout << "Converted " << numObjects << " objects\n";

    double dt = 1.0 / 60.0;

    // Double buffering: two image arrays
    image_array image_current;
    int img_size = IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT;

    //----------------------------------------------------------------------
    //  Saving into MP4 File
    //----------------------------------------------------------------------

    //Current image
    image_current.red = new unsigned char[RESOLUTION];
    image_current.green = new unsigned char[RESOLUTION];
    image_current.blue = new unsigned char[RESOLUTION];
    image_current.alpha = new unsigned char[RESOLUTION];

    memset(image_current.red, 128, img_size);
    memset(image_current.green, 128, img_size);
    memset(image_current.blue, 128, img_size);
    memset(image_current.alpha, 255, img_size);
    
    // Double buffering for RGB data (for FFmpeg)
    unsigned char *rgb_buffer_1 = new unsigned char[img_size * 3];
    unsigned char *rgb_buffer_2 = new unsigned char[img_size * 3];

    // Create video encoder
    MP4VideoEncoder encoder(output_file, IMAGE_RESOLUTION_WIDTH, IMAGE_RESOLUTION_HEIGHT, DEBUG_PERF, 60);

    //----------------------------------------------------------------------
    //      Defining threads which will write MP4 File
    //----------------------------------------------------------------------

    // Setup async encoding pipeline
    MP4FrameData frame_data;
    frame_data.ready = false;
    frame_data.done = false;
    frame_data.rgb_buffer = rgb_buffer_1;
    pthread_mutex_init(&frame_data.mutex, nullptr);
    pthread_cond_init(&frame_data.cond, nullptr);

    VideoEncoderThreadData encoder_thread_data;
    encoder_thread_data.encoder = &encoder;
    encoder_thread_data.frame_data = &frame_data;

    pthread_t encoder_thread;
    pthread_create(&encoder_thread, nullptr, asyncVideoEncoderThread, &encoder_thread_data);

    // Track which RGB buffer to use (ping-pong)
    unsigned char *current_rgb = rgb_buffer_1;
    unsigned char *backup_rgb = rgb_buffer_2;

    //----------------------------------------------------------------------
    //      GPU global memory definition
    //----------------------------------------------------------------------

    // Video Memory initialisation
    bool image_validity;
    cudaStream_t *gpu_stream;
    gpu_stream = new cudaStream_t[NB_STREAM];
    if (allocate_gpu_stream(gpu_stream))
        return 1;
    dim3 numBlocks, threadsPerBlock;
    id_array *gpu_id_array;
    gpu_id_array = new id_array[NB_STREAM];
    image_array *gpu_image;
    gpu_image = new image_array[NB_STREAM];
    gpu_object_pointers *gpu_obj_pointers;
    gpu_obj_pointers = new gpu_object_pointers[NB_STREAM];
    if (allocate_gpu_thread(numBlocks, threadsPerBlock))
        return 1;
    initiate_video_memory(gpu_id_array, gpu_image, gpu_obj_pointers, gpu_stream);
    copy_initial_data_to_video_memory_for_all_streams(gpu_obj_pointers, tab_pos, gpu_stream);
    for (int i = 0; i < NB_STREAM; i++)
        gpu_obj_pointers[i].state = NONE;

    printf("Initialisation finished, Waiting to start\n");
    sleep(5);
    

    auto after_init = std::chrono::high_resolution_clock::now();
    if (DEBUG_PERF)
    {
        std::chrono::duration<double, std::milli> duration_after_init = after_init - start;
        printf("Execution time after initialisation: %f ms\n", duration_after_init.count());
    }

    double waiting_duration = 0;
    double physics_update = 0;

    long long totalCollisions = 0;

    printf("--------------Start Rendering---------------\n");
    for (int i = 0; i < RENDERED_FRAMES; i++)
    {
        //Progression
        if (i % (RENDERED_FRAMES / 4) == 0) {
            std::cout << i / (RENDERED_FRAMES / 4) * 25 << " % of frames completed\n";
        }

        //----------------------------------------------------------------------
        //      Update physics
        //----------------------------------------------------------------------
        auto start_update = std::chrono::high_resolution_clock::now();
        
        simulationStep(shapes, dt, totalCollisions);
        
        auto end_update = std::chrono::high_resolution_clock::now();
        double time_update = std::chrono::duration<double, std::milli>(end_update - start_update).count();

        physics_update += time_update;
        
        
        //----------------------------------------------------------------------
        //      Rendering
        //----------------------------------------------------------------------
        convertSceneToGPU(shapes, tab_pos, true);

        // Main function to use in order to draw an image from the set of objects
        if (DEBUG_PERF)
        {
            before_image_draw = std::chrono::high_resolution_clock::now();
        }
        
        //Update GPU state & Render
        updateGPUPhysicsState(shapes, tab_pos, numObjects);
        image_validity = draw_image(tab_pos, image_current, gpu_id_array, gpu_image, gpu_obj_pointers, gpu_stream, numBlocks, threadsPerBlock);
        if (DEBUG_PERF)
        {
            benchmark_performance(i, before_image_draw, time_table, gpu_obj_pointers, gpu_stream);
            reinit_terminal(i);
            print_intermediate_bench_values(i, time_table);
            compute_bench_values(i, bench_values, time_table);
        }

        
        //----------------------------------------------------------------------
        //      Saving in MP4 File asynchronously
        //----------------------------------------------------------------------
        //Copy to backup buffer
        synchronize_gpu_image(image_validity, gpu_obj_pointers, gpu_stream);
        if (image_validity) {
            // Convert to RGB
            convertToRGBParallel(image_current, current_rgb);

            // Wait for previous encode to complete
            auto start_wait = std::chrono::high_resolution_clock::now();
            pthread_mutex_lock(&frame_data.mutex);
            while (frame_data.ready)
            {
                pthread_cond_wait(&frame_data.cond, &frame_data.mutex);
            }
            pthread_mutex_unlock(&frame_data.mutex);
            auto end_wait = std::chrono::high_resolution_clock::now();
            double time_wait = std::chrono::duration<double, std::milli>(end_wait - start_wait).count();

            waiting_duration += time_wait;

            // Queue encode (async)
            pthread_mutex_lock(&frame_data.mutex);
            frame_data.rgb_buffer = current_rgb;
            frame_data.ready = true;
            pthread_cond_signal(&frame_data.cond);
            pthread_mutex_unlock(&frame_data.mutex);

            // Swap buffers for next frame
            std::swap(current_rgb, backup_rgb);
        }
    }
    printf("--------------End of Rendering--------------\n");

    // Output performance metrics
    printf("\n--------------Run Parameters Recap---------------\n");
    printf("Image resolution : %d * %d\n", IMAGE_RESOLUTION_WIDTH, IMAGE_RESOLUTION_HEIGHT);
    printf("Number of pixels to compute : %d\n", RESOLUTION);
    printf("Number of objects in simulation : %d\n", NB_OBJECT);
    printf("Number of rendered frames : %d\n", RENDERED_FRAMES);
    printf("-------------------------------------------------\n");

    auto end = std::chrono::high_resolution_clock::now();
    if (DEBUG_PERF)
    {
        std::chrono::duration<double, std::milli> duration_end = end - start;
        printf("\n--------------Performance Metrics---------------\n");
        printf("Total execution time: %f ms\n", duration_end.count());
        printf("Mean time between 2 frames : %f ms\n", bench_values.mean_time / ((float)RENDERED_FRAMES - (NB_STREAM - 1)));
        printf("Maximum time between 2 frames (%d -> %d): %f ms\n", bench_values.index_max_time - 1, bench_values.index_max_time, bench_values.max_time);
        printf("Minimum time between 2 frames (%d -> %d): %f ms\n", bench_values.index_min_time - 1, bench_values.index_min_time, bench_values.min_time);
        printf("Mean FPS : %f\n", 1000.0 * ((float)RENDERED_FRAMES - (NB_STREAM - 1)) / bench_values.mean_time);
        printf("Mean image rendering time : %f ms\n", bench_values.mean_time_render / ((float)RENDERED_FRAMES - (NB_STREAM - 1)));
        if (NB_STREAM == 2)
        {
            printf("Mean time spent copying/compute on GPU : %f ms\n", bench_values.mean_time_compute / ((float)RENDERED_FRAMES - (NB_STREAM - 1)));
            printf("Mean time spent copying from the GPU : %f ms\n", bench_values.mean_time_copy_from / ((float)RENDERED_FRAMES - (NB_STREAM - 1)));
        }
        else if (NB_STREAM == 3)
        {
            printf("Mean time spent copying to the GPU : %f ms\n", bench_values.mean_time_copy_to / ((float)RENDERED_FRAMES - (NB_STREAM - 1)));
            printf("Mean time spent computing on the GPU : %f ms\n", bench_values.mean_time_compute / ((float)RENDERED_FRAMES - (NB_STREAM - 1)));
            printf("Mean time spent copying from the GPU : %f ms\n", bench_values.mean_time_copy_from / ((float)RENDERED_FRAMES - (NB_STREAM - 1)));
        }
        printf("-------------------------------------------------\n");

        
        std::cout << "\n=== ENCODING COMPLETE ===\n";
        std::cout << "Total time:     " << duration_end.count() << " ms\n";
        std::cout << "Average frame:  " << duration_end.count() / RENDERED_FRAMES << " ms\n";
        std::cout << "Average FPS:    " << (RENDERED_FRAMES * 1000.0) / duration_end.count() << "\n";
        std::cout << "Average Physics Calculation Time:    " << physics_update / duration_end.count() << " ms\n";
        std::cout << "Average Waiting Time:    " << waiting_duration / duration_end.count() << " ms\n";
        std::cout << "Output file:    " << output_file << "\n";
    }

    // Wait for final encode
    pthread_mutex_lock(&frame_data.mutex);
    while (frame_data.ready)
    {
        pthread_cond_wait(&frame_data.cond, &frame_data.mutex);
    }
    pthread_mutex_unlock(&frame_data.mutex);

    // Shutdown encoder thread
    pthread_mutex_lock(&frame_data.mutex);
    frame_data.done = true;
    pthread_cond_signal(&frame_data.cond);
    pthread_mutex_unlock(&frame_data.mutex);
    pthread_join(encoder_thread, nullptr);

    auto end_total = std::chrono::high_resolution_clock::now();

    clean_video_memory(gpu_id_array, gpu_image, gpu_obj_pointers, gpu_stream);
}