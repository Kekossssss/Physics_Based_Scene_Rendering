#include "cpu_converter.hpp"
#include "cpu_renderer.hpp"

#include <stdlib.h> // Required for atoi()

#include <iostream>
#include <chrono>
#include <thread>

#include <pthread.h>
#include <omp.h>

double gravity = 9.81;

// ============================================================================
// PTHREAD STRUCTURE FOR SHAPE UPDATE
// ============================================================================

/**
 * Structure to pass data to each pthread worker
 */
struct ThreadData
{
    std::vector<Shape *> *shapes; // Pointer to the shapes vector
    int start_idx;                // Start index for this thread
    int end_idx;                  // End index (exclusive) for this thread
    double dt;                    // Time step
};

// ============================================================================
// PTHREAD WORKER FUNCTION
// ============================================================================

/**
 * Worker function executed by each thread
 * Updates shapes from start_idx to end_idx-1
 */
void *updateShapesThread(void *arg)
{
    ThreadData *data = (ThreadData *)arg;

    // Update shapes assigned to this thread
    for (int i = data->start_idx; i < data->end_idx; i++)
    {
        (*data->shapes)[i]->update(data->dt);
    }

    return nullptr;
}

// ============================================================================
// PARALLELIZED UPDATE FUNCTION
// ============================================================================

pthread_t *threads;
ThreadData *thread_data;

/**
 * Update all shapes using pthreads
 * @param shapes: Vector of shape pointers
 * @param dt: Time step for physics update
 * @param num_threads: Number of threads to use (default: 4)
 */
void updateShapesParallel(std::vector<Shape *> &shapes, double dt, int num_threads = 4)
{
    int num_shapes = shapes.size();

    // Calculate work distribution
    int shapes_per_thread = num_shapes / num_threads;
    int remainder = num_shapes % num_threads;

    int current_start = 0;

    // Create threads
    for (int i = 0; i < num_threads; i++)
    {
        thread_data[i].shapes = &shapes;
        thread_data[i].start_idx = current_start;
        thread_data[i].end_idx = current_start + shapes_per_thread + (i < remainder ? 1 : 0);
        thread_data[i].dt = dt;

        // Create thread
        int rc = pthread_create(&threads[i], nullptr, updateShapesThread, &thread_data[i]);
        if (rc != 0)
        {
            std::cerr << "Error: pthread_create failed with code " << rc << std::endl;
        }

        current_start = thread_data[i].end_idx;
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], nullptr);
    }
    /*
    // Cleanup
    delete[] threads;
    delete[] thread_data;*/
}

// ============================================================================
// ALTERNATIVE: FIXED THREAD POOL VERSION
// ============================================================================

/**
 * Thread pool for repeated updates (more efficient for simulation loops)
 */
class ShapeUpdateThreadPool
{
private:
    pthread_t *threads;
    ThreadData *thread_data;
    int num_threads;
    std::vector<Shape *> *shapes;

    // Synchronization
    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;
    bool running;
    bool should_exit;

    static void *workerLoop(void *arg)
    {
        ThreadData *data = (ThreadData *)arg;
        ShapeUpdateThreadPool *pool = (ShapeUpdateThreadPool *)data->shapes;

        while (true)
        {
            // Wait for start signal
            pthread_barrier_wait(&pool->start_barrier);

            if (pool->should_exit)
                break;

            // Do work
            for (int i = data->start_idx; i < data->end_idx; i++)
            {
                (*pool->shapes)[i]->update(data->dt);
            }

            // Signal completion
            pthread_barrier_wait(&pool->end_barrier);
        }

        return nullptr;
    }

public:
    /**
     * Constructor: Initialize thread pool
     */
    ShapeUpdateThreadPool(std::vector<Shape *> &shapes_ref, int num_threads = 4)
        : num_threads(num_threads), shapes(&shapes_ref), running(false), should_exit(false)
    {

        if (num_threads > (int)shapes_ref.size())
        {
            num_threads = shapes_ref.size();
        }

        threads = new pthread_t[num_threads];
        thread_data = new ThreadData[num_threads];

        // Initialize barriers
        pthread_barrier_init(&start_barrier, nullptr, num_threads + 1);
        pthread_barrier_init(&end_barrier, nullptr, num_threads + 1);

        // Calculate work distribution
        int num_shapes = shapes_ref.size();
        int shapes_per_thread = num_shapes / num_threads;
        int remainder = num_shapes % num_threads;

        int current_start = 0;

        // Create persistent threads
        for (int i = 0; i < num_threads; i++)
        {
            thread_data[i].shapes = (std::vector<Shape *> *)this; // Hack to pass pool pointer
            thread_data[i].start_idx = current_start;
            thread_data[i].end_idx = current_start + shapes_per_thread + (i < remainder ? 1 : 0);
            thread_data[i].dt = 0.0;

            pthread_create(&threads[i], nullptr, workerLoop, &thread_data[i]);

            current_start = thread_data[i].end_idx;
        }

        running = true;
    }

    /**
     * Update all shapes with given time step
     */
    void update(double dt)
    {
        if (!running)
            return;

        // Update dt for all threads
        for (int i = 0; i < num_threads; i++)
        {
            thread_data[i].dt = dt;
        }

        // Signal threads to start
        pthread_barrier_wait(&start_barrier);

        // Wait for completion
        pthread_barrier_wait(&end_barrier);
    }

    /**
     * Destructor: Clean up threads
     */
    ~ShapeUpdateThreadPool()
    {
        if (running)
        {
            should_exit = true;
            pthread_barrier_wait(&start_barrier);

            for (int i = 0; i < num_threads; i++)
            {
                pthread_join(threads[i], nullptr);
            }

            pthread_barrier_destroy(&start_barrier);
            pthread_barrier_destroy(&end_barrier);
        }

        delete[] threads;
        delete[] thread_data;
    }
};

int main(int argc, char *argv[])
{
    // Create CPU scene
    std::vector<Shape *> shapes;
    shapes.push_back(new Sphere(Point3D(50, 50, 10), 100, Point3D(0, 0, 0)));
    shapes.push_back(new Cube(Point3D(200, 200, 20), 200, Point3D(0, 0, 0),
                              Point3D(0, 0, 0), Point3D(0.1, 0.1, 0.1)));
    shapes.push_back(new RectangularPrism(Point3D(100, 100, 50), 400, 300, 200,
                                          Point3D(5, 5, 0), Point3D(0, 0, 0),
                                          Point3D(0, 0, 0)));

    threads = new pthread_t[shapes.size()];
    thread_data = new ThreadData[shapes.size()];

    // Create GPU object structure
    object_to_gpu gpuScene;

    // Initial conversion
    int numObjects = convertSceneToGPU(shapes, gpuScene, true);
    std::cout << "Converted " << numObjects << " objects\n";

    // Simulation loop
    double dt = 0.016 * atoi(argv[1]); // 60 FPS

    // Test image arrays
    image_array image;
    image.red = new unsigned char[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];
    image.green = new unsigned char[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];
    image.blue = new unsigned char[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];
    image.alpha = new unsigned char[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];

    auto start = std::chrono::high_resolution_clock::now();

    ShapeUpdateThreadPool pool(shapes, 1);

    for (int frame = 0; frame < 100; frame++)
    {

        // Update GPU data (only positions/rotations for efficiency)
        updateGPUPhysicsState(shapes, gpuScene, numObjects);

        // Pass gpuScene to your GPU rendering function
        // draw_image(gpuScene, image, ...);

        draw_image(gpuScene, image);

        // Update physics on CPU
        auto start_seq = std::chrono::high_resolution_clock::now();
        /*for (int i = 0; i < shapes.size(); i++)
        {
            shapes[i]->update(dt);
        }*/
        auto end_seq = std::chrono::high_resolution_clock::now();
        double time_seq = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

        auto start_par = std::chrono::high_resolution_clock::now();
        // updateShapesParallel(shapes, dt, 1);
        auto end_par = std::chrono::high_resolution_clock::now();

        double time_par = std::chrono::duration<double, std::milli>(end_par - start_par).count();

        auto start_pool = std::chrono::high_resolution_clock::now();
        pool.update(dt);
        auto end_pool = std::chrono::high_resolution_clock::now();
        double time_pool = std::chrono::duration<double, std::milli>(end_pool - start_pool).count();

        std::cout << "=== BENCHMARK RESULTS (" << frame << " iterations) ===\n";
        std::cout << "Sequential:     " << time_seq << " ms\n";
        std::cout << "Parallel:       " << time_par << " ms (speedup: " << time_seq / time_par << "x)\n";
        std::cout << "Thread Pool:    " << time_pool << " ms (speedup: " << time_seq / time_pool << "x)\n";
        if (save_as_bmp(image, "test_image_cpu.bmp") == false)
        {
            printf("Image saving error, leaving loop\n");
            break;
        }

        /*
        std::cout << "Waiting for 3 seconds..." << std::endl;
        // Pause the current thread for 3 seconds
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Done waiting." << std::endl;*/
    }

    auto end = std::chrono::high_resolution_clock::now();

    double duration_cool = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Sequential:     " << duration_cool << " ms\n";

    // Cleanup
    deleteShapes(shapes);

    return 0;
}