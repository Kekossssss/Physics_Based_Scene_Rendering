#include "cpu_converter.hpp"
#include "cpu_renderer.hpp"
#include "cpu_part.hpp"
#include "video_writer.hpp"
#include "utils.hpp"

#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <omp.h>

double gravity = 9.81;

// ============================================================================
// MAIN WITH PIPELINED EXECUTION
// ============================================================================

int main(int argc, char *argv[])
{
    
    printf("Program Starting\n");
    //Output file
    const char *output_file = "output.mp4";

    auto start = std::chrono::high_resolution_clock::now();

    int k = 2.5;
    int c = 1000;

    //----------------------------------------------------------------------
    //      Exemple of objects
    //----------------------------------------------------------------------
    std::vector<Shape *> shapes;
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
    
    // Object Conversion
    int numObjects = convertSceneToGPU(shapes, tab_pos, true);
    std::cout << "Converted " << numObjects << " objects\n";

    //Timestep
    double dt = 1.0 / 60.0;

    // Double buffering: two image arrays
    image_array image_current;
    int img_size = IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT;

    //----------------------------------------------------------------------
    //  Saving into MP4 File
    //----------------------------------------------------------------------

    // Current image
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

    auto start_total = std::chrono::high_resolution_clock::now();

    long long totalCollisions = 0;

    double render_duration = 0;
    double waiting_duration = 0;
    double physics_update = 0;

    //----------------------------------------------------------------------
    //      Starting Simulation
    //----------------------------------------------------------------------

    for (int frame = 0; frame < RENDERED_FRAMES; frame++)
    {
        //Progression
        if (frame % (RENDERED_FRAMES / 4) == 0) {
            std::cout << frame / (RENDERED_FRAMES / 4) * 25 << " % of frames completed\n";
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
        auto start_render = std::chrono::high_resolution_clock::now();
        updateGPUPhysicsState(shapes, tab_pos, numObjects);
        draw_image(tab_pos, image_current);
        auto end_render = std::chrono::high_resolution_clock::now();
        double time_render = std::chrono::duration<double, std::milli>(end_render - start_render).count();

        render_duration += time_render;
        
        //----------------------------------------------------------------------
        //      Saving in MP4 File asynchronously
        //----------------------------------------------------------------------
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

        //Next frame will compute while this frame saves in background!
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
    double duration_total = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    if (DEBUG_PERF) {
        std::cout << "\n=== ENCODING COMPLETE ===\n";
        std::cout << "Total time:     " << duration_total << " ms\n";
        std::cout << "Average time to generate and store one frame:  " << duration_total / RENDERED_FRAMES << " ms\n";
        std::cout << "Average FPS:    " << (RENDERED_FRAMES * 1000.0) / duration_total << "\n";std::cout << "Average Physics Calculation Time:    " << physics_update / duration_total << " ms\n";
        std::cout << "Average Rendering Time:    " << render_duration / duration_total << " ms\n";
        std::cout << "Average Waiting Time:    " << waiting_duration / duration_total << " ms\n";
        
        std::cout << "Output file:    " << output_file << "\n";
    }

    // Cleanup
    delete[] image_current.red;
    delete[] image_current.green;
    delete[] image_current.blue;
    delete[] image_current.alpha;

    return 0;
}