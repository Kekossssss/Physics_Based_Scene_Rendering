#include "cpu_converter.hpp"
#include "cpu_renderer.hpp"

#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <omp.h>

double gravity = 9.81;

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

struct BMPThreadData
{
    image_array *image;
    int start_row;
    int end_row;
    FILE *file;
    unsigned char *row_buffer;
    pthread_mutex_t *file_mutex;
};

struct PipelineData
{
    image_array *image;
    char filename[256];
    int num_threads;
    bool ready;
    bool done;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
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

void *saveBMPThread(void *arg)
{
    BMPThreadData *data = (BMPThreadData *)arg;

    for (int y = data->start_row; y < data->end_row; y++)
    {
        int row_start = y * IMAGE_RESOLUTION_WIDTH;

        for (int x = 0; x < IMAGE_RESOLUTION_WIDTH; x++)
        {
            int idx = row_start + x;
            int buffer_idx = x * 3;
            data->row_buffer[buffer_idx] = data->image->blue[idx];
            data->row_buffer[buffer_idx + 1] = data->image->green[idx];
            data->row_buffer[buffer_idx + 2] = data->image->red[idx];
        }

        pthread_mutex_lock(data->file_mutex);
        fseek(data->file, 54 + y * (IMAGE_RESOLUTION_WIDTH * 3), SEEK_SET);
        fwrite(data->row_buffer, 1, IMAGE_RESOLUTION_WIDTH * 3, data->file);
        pthread_mutex_unlock(data->file_mutex);
    }

    return nullptr;
}

// ============================================================================
// PARALLEL UPDATE FUNCTIONS
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
// PARALLEL BMP SAVE
// ============================================================================

bool save_as_bmp_parallel(image_array &image, const char *filename, int num_threads = 4)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
        return false;

    // BMP Header
    unsigned char bmp_header[54] = {
        'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int width = IMAGE_RESOLUTION_WIDTH;
    int height = IMAGE_RESOLUTION_HEIGHT;
    int file_size = 54 + width * height * 3;

    *((int *)&bmp_header[2]) = file_size;
    *((int *)&bmp_header[18]) = width;
    *((int *)&bmp_header[22]) = height;

    fwrite(bmp_header, 1, 54, file);

    pthread_t *threads = new pthread_t[num_threads];
    BMPThreadData *thread_data = new BMPThreadData[num_threads];
    pthread_mutex_t file_mutex;
    pthread_mutex_init(&file_mutex, nullptr);

    int rows_per_thread = height / num_threads;
    int remainder = height % num_threads;
    int current_start = 0;

    for (int i = 0; i < num_threads; i++)
    {
        thread_data[i].image = &image;
        thread_data[i].start_row = current_start;
        thread_data[i].end_row = current_start + rows_per_thread + (i < remainder ? 1 : 0);
        thread_data[i].file = file;
        thread_data[i].file_mutex = &file_mutex;
        thread_data[i].row_buffer = new unsigned char[width * 3];

        pthread_create(&threads[i], nullptr, saveBMPThread, &thread_data[i]);
        current_start = thread_data[i].end_row;
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], nullptr);
        delete[] thread_data[i].row_buffer;
    }

    pthread_mutex_destroy(&file_mutex);
    delete[] threads;
    delete[] thread_data;

    fclose(file);
    return true;
}

// ============================================================================
// PIPELINE THREAD FOR ASYNC BMP SAVING
// ============================================================================

void *asyncBMPSaverThread(void *arg)
{
    PipelineData *pipeline = (PipelineData *)arg;

    while (true)
    {
        pthread_mutex_lock(&pipeline->mutex);

        // Wait for data to be ready
        while (!pipeline->ready && !pipeline->done)
        {
            pthread_cond_wait(&pipeline->cond, &pipeline->mutex);
        }

        if (pipeline->done)
        {
            pthread_mutex_unlock(&pipeline->mutex);
            break;
        }

        // Take ownership of the image to save
        image_array *img_to_save = pipeline->image;
        char filename[256];
        strcpy(filename, pipeline->filename);
        int num_threads = pipeline->num_threads;

        pipeline->ready = false;
        pthread_mutex_unlock(&pipeline->mutex);

        // Save the image (unlocked, allows next frame to compute)
        save_as_bmp_parallel(*img_to_save, filename, num_threads);

        // Signal that save is complete
        pthread_cond_signal(&pipeline->cond);
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

// ============================================================================
// MAIN WITH PIPELINED EXECUTION
// ============================================================================

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <dt_multiplier> <num_threads>\n";
        return 1;
    }

    int num_threads = atoi(argv[2]);

    // Create scene
    std::vector<Shape *> shapes;
    shapes.push_back(new Sphere(Point3D(50, 50, 10), 100, Point3D(0, 0, 0)));
    shapes.push_back(new Cube(Point3D(200, 200, 20), 200, Point3D(0, 0, 0),
                              Point3D(0, 0, 0), Point3D(0.1, 0.1, 0.1)));
    shapes.push_back(new RectangularPrism(Point3D(100, 100, 50), 400, 300, 200,
                                          Point3D(5, 5, 0), Point3D(0, 0, 0),
                                          Point3D(0, 0, 0)));

    // GPU conversion
    object_to_gpu gpuScene;
    int numObjects = convertSceneToGPU(shapes, gpuScene, true);
    std::cout << "Converted " << numObjects << " objects\n";

    double dt = 0.016 * atoi(argv[1]);

    // Double buffering: two image arrays
    image_array image_current, image_backup;
    int img_size = IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT;

    image_current.red = new unsigned char[img_size];
    image_current.green = new unsigned char[img_size];
    image_current.blue = new unsigned char[img_size];
    image_current.alpha = new unsigned char[img_size];

    image_backup.red = new unsigned char[img_size];
    image_backup.green = new unsigned char[img_size];
    image_backup.blue = new unsigned char[img_size];
    image_backup.alpha = new unsigned char[img_size];

    // Setup pipeline for async BMP saving
    PipelineData pipeline;
    pipeline.ready = false;
    pipeline.done = false;
    pipeline.num_threads = num_threads;
    pthread_mutex_init(&pipeline.mutex, nullptr);
    pthread_cond_init(&pipeline.cond, nullptr);

    pthread_t saver_thread;
    pthread_create(&saver_thread, nullptr, asyncBMPSaverThread, &pipeline);

    auto start_total = std::chrono::high_resolution_clock::now();

    // Simulation loop
    for (int frame = 0; frame < 100; frame++)
    {
        std::cout << "\n=== FRAME " << frame << " ===\n";

        auto frame_start = std::chrono::high_resolution_clock::now();

        // ---- PHASE 1: Update physics (parallel) ----
        auto start_update = std::chrono::high_resolution_clock::now();
        // updateShapesParallel(shapes, dt, num_threads);
        for (auto shape : shapes)
        {
            shape->update(dt);
        }
        auto end_update = std::chrono::high_resolution_clock::now();
        double time_update = std::chrono::duration<double, std::milli>(end_update - start_update).count();

        // ---- PHASE 2: Update GPU state & Render ----
        auto start_render = std::chrono::high_resolution_clock::now();
        updateGPUPhysicsState(shapes, gpuScene, numObjects);
        draw_image(gpuScene, image_current);
        auto end_render = std::chrono::high_resolution_clock::now();
        double time_render = std::chrono::duration<double, std::milli>(end_render - start_render).count();

        // ---- PHASE 3: Copy to backup buffer ----
        auto start_copy = std::chrono::high_resolution_clock::now();
        copyImageArray(image_current, image_backup);
        auto end_copy = std::chrono::high_resolution_clock::now();
        double time_copy = std::chrono::duration<double, std::milli>(end_copy - start_copy).count();

        // ---- PHASE 4: Wait for previous save to complete ----
        auto start_wait = std::chrono::high_resolution_clock::now();
        pthread_mutex_lock(&pipeline.mutex);
        while (pipeline.ready)
        {
            pthread_cond_wait(&pipeline.cond, &pipeline.mutex);
        }
        pthread_mutex_unlock(&pipeline.mutex);
        auto end_wait = std::chrono::high_resolution_clock::now();
        double time_wait = std::chrono::duration<double, std::milli>(end_wait - start_wait).count();

        // ---- PHASE 5: Queue save for backup buffer (async) ----
        pthread_mutex_lock(&pipeline.mutex);
        pipeline.image = &image_backup;
        snprintf(pipeline.filename, sizeof(pipeline.filename), "frame.bmp", frame);
        pipeline.ready = true;
        pthread_cond_signal(&pipeline.cond);
        pthread_mutex_unlock(&pipeline.mutex);

        auto frame_end = std::chrono::high_resolution_clock::now();
        double time_frame = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

        std::cout << "Update:  " << time_update << " ms\n";
        std::cout << "Render:  " << time_render << " ms\n";
        std::cout << "Copy:    " << time_copy << " ms\n";
        std::cout << "Wait:    " << time_wait << " ms\n";
        std::cout << "Frame:   " << time_frame << " ms\n";

        // Note: Next frame will compute while this frame saves in background!
    }

    // Wait for final save to complete
    pthread_mutex_lock(&pipeline.mutex);
    while (pipeline.ready)
    {
        pthread_cond_wait(&pipeline.cond, &pipeline.mutex);
    }
    pthread_mutex_unlock(&pipeline.mutex);

    // Shutdown pipeline
    pthread_mutex_lock(&pipeline.mutex);
    pipeline.done = true;
    pthread_cond_signal(&pipeline.cond);
    pthread_mutex_unlock(&pipeline.mutex);
    pthread_join(saver_thread, nullptr);

    auto end_total = std::chrono::high_resolution_clock::now();
    double duration_total = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    std::cout << "\n=== TOTAL SIMULATION TIME: " << duration_total << " ms ===\n";
    std::cout << "Average frame time: " << duration_total / 100.0 << " ms\n";

    // Cleanup
    delete[] image_current.red;
    delete[] image_current.green;
    delete[] image_current.blue;
    delete[] image_current.alpha;
    delete[] image_backup.red;
    delete[] image_backup.green;
    delete[] image_backup.blue;
    delete[] image_backup.alpha;

    pthread_mutex_destroy(&pipeline.mutex);
    pthread_cond_destroy(&pipeline.cond);

    deleteShapes(shapes);

    return 0;
}