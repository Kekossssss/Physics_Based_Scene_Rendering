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

extern double gravity;

// ============================================================================
// PTHREAD STRUCTURES
// ============================================================================

struct ThreadData
{
    std::vector<Shape *> *shapes;
    int start_idx;
    int end_idx;
    double dt;
};

struct MP4FrameData
{
    image_array *image;
    unsigned char *rgb_buffer;
    bool ready;
    bool done;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

// ============================================================================
// MP4 VIDEO ENCODER CLASS
// ============================================================================

class MP4VideoEncoder
{
private:
    FILE *ffmpeg_pipe;
    int width;
    int height;
    int fps;
    bool is_open;

public:
    MP4VideoEncoder(const char *filename, int w, int h, int framerate = 60)
        : width(w), height(h), fps(framerate), is_open(false)
    {
        // Build FFmpeg command
        char command[512];
        snprintf(command, sizeof(command),
                "singularity exec ffmpeg_latest.sif "
         "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d "
         "-framerate %d -i - -c:v libx264 -preset ultrafast -crf 18 "
         "-pix_fmt yuv420p -movflags +faststart %s 2>/dev/null",
                 width, height, fps, filename); 

        std::cout << "Opening FFmpeg pipe...\n";
        ffmpeg_pipe = popen(command, "w");

        if (ffmpeg_pipe)
        {
            is_open = true;
            std::cout << "FFmpeg encoder started: " << filename << "\n";
        }
        else
        {
            std::cerr << "ERROR: Failed to open FFmpeg pipe!\n";
            std::cerr << "Make sure FFmpeg is installed: sudo apt install ffmpeg\n";
        }
    }

    bool writeFrame(const image_array &image)
    {
        if (!is_open)
            return false;

        // Convert RGBA to RGB24 (FFmpeg format)
        unsigned char *rgb_buffer = new unsigned char[width * height * 3];

        for (int i = 0; i < width * height; i++)
        {
            rgb_buffer[i * 3 + 0] = image.red[i];
            rgb_buffer[i * 3 + 1] = image.green[i];
            rgb_buffer[i * 3 + 2] = image.blue[i];
        }

        // Write frame to FFmpeg
        size_t written = fwrite(rgb_buffer, 1, width * height * 3, ffmpeg_pipe);
        delete[] rgb_buffer;

        return (written == (size_t)(width * height * 3));
    }

    bool writeFrameRGB(const unsigned char *rgb_buffer)
    {
        if (!is_open)
            return false;

        size_t written = fwrite(rgb_buffer, 1, width * height * 3, ffmpeg_pipe);
        return (written == (size_t)(width * height * 3));
    }

// ============================================================================
// PTHREAD WORKER FUNCTIONS
// ============================================================================

    ~MP4VideoEncoder()
    {
        if (is_open)
        {
            std::cout << "Closing FFmpeg encoder...\n";
            pclose(ffmpeg_pipe);
        }
    }
};

// ============================================================================
// PTHREAD WORKER FUNCTIONS
// ============================================================================

void *updateShapesThread(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start_idx; i < data->end_idx; i++)
    {
        (*data->shapes)[i]->update(data->dt);
    }
    return nullptr;
}

// ============================================================================
// PARALLEL UPDATE
// ============================================================================

void updateShapesParallel(std::vector<Shape *> &shapes, double dt, int num_threads)
{
    int num_shapes = shapes.size();
    pthread_t *threads = new pthread_t[num_threads];
    ThreadData *thread_data = new ThreadData[num_threads];

    int shapes_per_thread = num_shapes / num_threads;
    int remainder = num_shapes % num_threads;
    int current_start = 0;

    for (int i = 0; i < num_threads; i++)
    {
        thread_data[i].shapes = &shapes;
        thread_data[i].start_idx = current_start;
        thread_data[i].end_idx = current_start + shapes_per_thread + (i < remainder ? 1 : 0);
        thread_data[i].dt = dt;

        pthread_create(&threads[i], nullptr, updateShapesThread, &thread_data[i]);
        current_start = thread_data[i].end_idx;
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    delete[] threads;
    delete[] thread_data;
}

// ============================================================================
// PARALLEL RGB CONVERSION
// ============================================================================

struct RGBConversionData
{
    const image_array *image;
    unsigned char *rgb_buffer;
    int start_idx;
    int end_idx;
};

void *convertToRGBThread(void *arg)
{
    RGBConversionData *data = (RGBConversionData *)arg;

    for (int i = data->start_idx; i < data->end_idx; i++)
    {
        data->rgb_buffer[i * 3 + 0] = data->image->red[i];
        data->rgb_buffer[i * 3 + 1] = data->image->green[i];
        data->rgb_buffer[i * 3 + 2] = data->image->blue[i];
    }

    return nullptr;
}

void convertToRGBParallel(const image_array &image, unsigned char *rgb_buffer, int num_threads = 4)
{
    int total_pixels = IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT;
    pthread_t *threads = new pthread_t[num_threads];
    RGBConversionData *thread_data = new RGBConversionData[num_threads];

    int pixels_per_thread = total_pixels / num_threads;
    int remainder = total_pixels % num_threads;
    int current_start = 0;

    for (int i = 0; i < num_threads; i++)
    {
        thread_data[i].image = &image;
        thread_data[i].rgb_buffer = rgb_buffer;
        thread_data[i].start_idx = current_start;
        thread_data[i].end_idx = current_start + pixels_per_thread + (i < remainder ? 1 : 0);

        pthread_create(&threads[i], nullptr, convertToRGBThread, &thread_data[i]);
        current_start = thread_data[i].end_idx;
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    delete[] threads;
    delete[] thread_data;
}

// ============================================================================
// ASYNC VIDEO ENCODING THREAD
// ============================================================================

struct VideoEncoderThreadData
{
    MP4VideoEncoder *encoder;
    MP4FrameData *frame_data;
};

void *asyncVideoEncoderThread(void *arg)
{
    VideoEncoderThreadData *data = (VideoEncoderThreadData *)arg;
    MP4VideoEncoder *encoder = data->encoder;
    MP4FrameData *frame_data = data->frame_data;

    while (true)
    {
        pthread_mutex_lock(&frame_data->mutex);

        // Wait for frame to be ready
        while (!frame_data->ready && !frame_data->done)
        {
            pthread_cond_wait(&frame_data->cond, &frame_data->mutex);
        }

        if (frame_data->done)
        {
            pthread_mutex_unlock(&frame_data->mutex);
            break;
        }

        // Take ownership of the buffer
        unsigned char *rgb_to_encode = frame_data->rgb_buffer;
        frame_data->ready = false;

        pthread_mutex_unlock(&frame_data->mutex);

        // Encode frame (unlocked - allows next frame to prepare)
        encoder->writeFrameRGB(rgb_to_encode);

        // Signal completion
        pthread_cond_signal(&frame_data->cond);
    }

    return nullptr;
}

// ============================================================================
// HELPER: Copy image data
// ============================================================================

void copyImageArray(const image_array &src, image_array &dst)
{
    int size = IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT;
    memcpy(dst.red, src.red, size);
    memcpy(dst.green, src.green, size);
    memcpy(dst.blue, src.blue, size);
    memcpy(dst.alpha, src.alpha, size);
}

//------------------------------------------------------------------------------------------//
// Main
//------------------------------------------------------------------------------------------//
int main(int argc, char **argv)
{
    printf("Program Starting\n");
    const char *output_file = (argc >= 4) ? argv[3] : "output.mp4";

    int num_threads = 5;

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

    // Test objects positions
    object_to_gpu tab_pos;

    // Create scene
    /*std::vector<Shape *> shapes;
    shapes.push_back(new Sphere(Point3D(50, 50, 10), 100, Point3D(0, 0, 0), 500, 0.8, 9.81));
    shapes.push_back(new Cube(Point3D(2000, 200, 20), 200, Point3D(-100, 0, 0),
                              Point3D(0, 0, 0), Point3D(0.1, 0.1, 0.1), 10, 0.8, 9.81));
    shapes.push_back(new RectangularPrism(Point3D(100, 200, 50), 400, 300, 200,
                                          Point3D(100, 5, 0), Point3D(0, 0, 0),
                                          Point3D(0, 0, 0), 10, 0.8, 9.81));
    shapes.push_back(new Sphere(Point3D(130, 50, 10), 100, Point3D(0, 0, 0), 10, 0.8, 9.81));*/

    std::vector<Shape *> shapes;

    shapes.push_back(new RectangularPrism(
        Point3D(500, 950, 0), 50, 1500, 500, 
        Point3D(0, 0, 0), Point3D(0, 0, 0), Point3D(0, 0, 0), 
        1e20, 0.2, 0.0
    ));

    shapes.push_back(new Cube(
        Point3D(600, 875, 0), 100, 
        Point3D(0, 0, 0), Point3D(0,0,0), Point3D(0,0,0), 
        100, 0.0, 9.81
    ));

    shapes.push_back(new Cube(
        Point3D(600, 775, 0), 100, 
        Point3D(0, 0, 0), Point3D(0,0,0), Point3D(0,0,0), 
        100, 0.0, 9.81
    ));

    shapes.push_back(new Cube(
        Point3D(620, 675, 0), 100, 
        Point3D(0, 0, 0), Point3D(0.1, 0.1, 0.0), Point3D(0,0,0), 
        80, 0.0, 9.81
    ));

    shapes.push_back(new Sphere(
        Point3D(50, 750, 0), 80, 
        Point3D(800, -50, 0),   
        400, 0.5, 9.81          
    ));

    shapes.push_back(new RectangularPrism(
        Point3D(500, 50, 0), 50, 1500, 500, 
        Point3D(0, 0, 0), Point3D(0, 0, 0), Point3D(0, 0, 0), 
        1e20, 0.5, 0.0
    ));
    
    // GPU conversion
    int numObjects = convertSceneToGPU(shapes, tab_pos, true);
    std::cout << "Converted " << numObjects << " objects\n";

    double dt = 1.0 / 60.0 * 10.0;

    // Double buffering: two image arrays
    image_array image_current;
    int img_size = IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT;

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
    MP4VideoEncoder encoder(output_file, IMAGE_RESOLUTION_WIDTH, IMAGE_RESOLUTION_HEIGHT, 60);

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

    long long collisionsDetected = 0;
    long long collisionsResolved = 0;
    bool resolveCollisions = true;

    int N = shapes.size();

    printf("--------------Start Rendering---------------\n");
    for (int i = 0; i < RENDERED_FRAMES; i++)
    {
        // Main function to use in order to draw an image from the set of objects
        if (DEBUG_PERF)
        {
            before_image_draw = std::chrono::high_resolution_clock::now();
        }
        
        
        //Update GPU state & Render
        auto start_render = std::chrono::high_resolution_clock::now();
        updateGPUPhysicsState(shapes, tab_pos, numObjects);
        image_validity = draw_image(tab_pos, image_current, gpu_id_array, gpu_image, gpu_obj_pointers, gpu_stream, numBlocks, threadsPerBlock);
        auto end_render = std::chrono::high_resolution_clock::now();
        if (DEBUG_PERF)
        {
            benchmark_performance(i, before_image_draw, time_table, gpu_obj_pointers, gpu_stream);
            reinit_terminal(i);
            print_intermediate_bench_values(i, time_table);
            compute_bench_values(i, bench_values, time_table);
        }
        
        if (i % 5 == 0) {
            std::cout << "1- S-S: " << shapes[0]->getCenter().y << " | ";
            std::cout << "C-C: " << shapes[2]->getCenter().y << " | ";
        }

        int N = (int)shapes.size();
        for (int k = 0;k < N;i++)
        {
            shapes[k]->update(dt);
        }

        if (i % 5 == 0) {
            std::cout << "2 - S-S: " << shapes[0]->getCenter().y << " | ";
            std::cout << "C-C: " << shapes[2]->getCenter().y << " | ";
        }

        #pragma omp parallel for schedule(dynamic) reduction(+:collisionsDetected,collisionsResolved)
        for(int k=0; k<N; k++) {
            for(int j=k+1; j<N; j++) {
                bool collision = false;
                
                Sphere* s1 = dynamic_cast<Sphere*>(shapes[k]);
                Sphere* s2 = dynamic_cast<Sphere*>(shapes[j]);
                RigidBody* r1 = dynamic_cast<RigidBody*>(shapes[k]);
                RigidBody* r2 = dynamic_cast<RigidBody*>(shapes[j]);

                if(s1 && s2) {
                    collision = checkSphereCollision(*s1, *s2);
                    if(collision) {
                        resolveSphereSphereCollision(s1, s2);
                        collisionsResolved++;
                    }
                }
                else if(r1 && r2) {
                    collision = checkOBBCollision(*r1, *r2);
                    if(collision) {
                        resolveRigidRigidCollision(r1, r2);
                        collisionsResolved++;
                    }
                }
                else if(s1 && r2) {
                    collision = checkSphereRigidCollision(*s1, *r2);
                    if(collision) {
                        resolveSphereRigidCollision(s1, r2);
                        collisionsResolved++;
                    }
                }
                else if(r1 && s2) {
                    collision = checkSphereRigidCollision(*s2, *r1);
                    if(collision) {
                        resolveSphereRigidCollision(s2, r1);
                        collisionsResolved++;
                    }
                }

                if(collision) collisionsDetected++;
            }
        }

        if (i % 5 == 0) {
            std::cout << "S-S: " << shapes[0]->getCenter().y << " | ";
            std::cout << "C-C: " << shapes[2]->getCenter().y << " | ";
        }
        
        convertSceneToGPU(shapes, tab_pos, true);

        double time_render = std::chrono::duration<double, std::milli>(end_render - start_render).count();

        //Copy to backup buffer
        synchronize_gpu_image(image_validity, gpu_obj_pointers, gpu_stream);
        if (image_validity) {
                // Convert to RGB
            auto start_convert = std::chrono::high_resolution_clock::now();
            convertToRGBParallel(image_current, current_rgb, num_threads);
            auto end_convert = std::chrono::high_resolution_clock::now();
            double time_convert = std::chrono::duration<double, std::milli>(end_convert - start_convert).count();

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

            // Queue encode (async)
            pthread_mutex_lock(&frame_data.mutex);
            frame_data.rgb_buffer = current_rgb;
            frame_data.ready = true;
            pthread_cond_signal(&frame_data.cond);
            pthread_mutex_unlock(&frame_data.mutex);

            // Swap buffers for next frame
            std::swap(current_rgb, backup_rgb);

            if (i % 30 == 0 and DEBUG_PERF)
            {
                //std::cout << "Update:   " << time_update << " ms\n";
                std::cout << "Render:   " << time_render << " ms\n";
                std::cout << "Convert:  " << time_convert << " ms\n";
                std::cout << "Wait:     " << time_wait << " ms\n";
                //std::cout << "Frame:    " << time_frame << " ms\n";
                //std::cout << "FPS:      " << 1000.0 / time_frame << "\n";
            }
        }

        // usleep(3000000);
    }
    printf("--------------End of Rendering--------------\n");

    // Output performance metrics
    printf("\n--------------Run Parameters Recap---------------\n");
    printf("Image resolution : %d * %d\n", IMAGE_RESOLUTION_WIDTH, IMAGE_RESOLUTION_HEIGHT);
    printf("Number of pixels to compute : %d\n", RESOLUTION);
    printf("Number of objects in simulation : %d\n", NB_OBJECT);
    printf("Number of rendered frames : %d\n", RENDERED_FRAMES);
    printf("-------------------------------------------------\n");

    std::cout << "Collisions       : " << collisionsDetected << std::endl;
    if(resolveCollisions) {
        std::cout << "Résolutions      : " << collisionsResolved << std::endl;
    }

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
    //double duration_total = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    std::cout << "\n=== ENCODING COMPLETE ===\n";
    //std::cout << "Total time:     " << duration_total << " ms\n";
    //std::cout << "Average frame:  " << duration_total / num_frames << " ms\n";
    //std::cout << "Average FPS:    " << (num_frames * 1000.0) / duration_total << "\n";
    std::cout << "Output file:    " << output_file << "\n";

    clean_video_memory(gpu_id_array, gpu_image, gpu_obj_pointers, gpu_stream);
}