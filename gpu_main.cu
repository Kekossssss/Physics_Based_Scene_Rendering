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


#define RENDERED_FRAMES 1000

//------------------------------------------------------------------------------------------//
// CPU Functions (Video Memory Management)
//------------------------------------------------------------------------------------------//
bool allocate_gpu_stream(cudaStream_t *gpu_stream)
{
    for (int i = 0; i < NB_STREAM; i++)
    {
        if (cudaStreamCreate(&gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Stream Initialisation failed (stream %d)\n", i);
            return 1;
        }
    }
    return 0;
}

void initiate_video_memory(id_array *gpu_id_array, image_array *gpu_image, gpu_object_pointers *gpu_obj_pointers, cudaStream_t *gpu_stream)
{
    // Allocate references to variables in GPU memory
    for (int i = 0; i < NB_STREAM; i++)
    {
        //// GPU only id_array variable
        cudaStreamSynchronize(gpu_stream[i]);
        if (cudaMallocAsync(&gpu_id_array[i].id, sizeof(int) * RESOLUTION, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_id_array[i].side, sizeof(int) * RESOLUTION, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        //// GPU Image variable
        if (cudaMallocAsync(&gpu_image[i].red, sizeof(unsigned char) * RESOLUTION, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_image[i].green, sizeof(unsigned char) * RESOLUTION, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_image[i].blue, sizeof(unsigned char) * RESOLUTION, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_image[i].alpha, sizeof(unsigned char) * RESOLUTION, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        //// GPU objects variable allocation
        if (cudaMallocAsync(&gpu_obj_pointers[i].type, sizeof(unsigned char) * NB_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].pos_x, sizeof(float) * NB_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].pos_y, sizeof(float) * NB_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].pos_z, sizeof(float) * NB_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].rot_x, sizeof(float) * NB_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].rot_y, sizeof(float) * NB_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].rot_z, sizeof(float) * NB_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].dimension, sizeof(float) * NB_OBJECT * MAX_DIMENSIONS_OBJECTS, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].red, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].green, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].blue, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        if (cudaMallocAsync(&gpu_obj_pointers[i].is_single_color, sizeof(bool) * NB_OBJECT, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Malloc Failed\n");
        }
        cudaStreamSynchronize(gpu_stream[i]);
    }
}

void copy_initial_data_to_video_memory(gpu_object_pointers &gpu_obj_pointers, object_to_gpu &obj, cudaStream_t &gpu_stream)
{
    // Copy data that doesn't need to be refetch after initialisation
    unsigned char col_r[NB_OBJECT * MAX_FACES_OBJECT];
    unsigned char col_g[NB_OBJECT * MAX_FACES_OBJECT];
    unsigned char col_b[NB_OBJECT * MAX_FACES_OBJECT];
    float dimension[NB_OBJECT * MAX_DIMENSIONS_OBJECTS];
    for (int i = 0; i < NB_OBJECT; i++)
    {
        if (obj.is_single_color[i] == true)
        {
            col_r[i * MAX_FACES_OBJECT] = obj.col[i][0].red;
            col_g[i * MAX_FACES_OBJECT] = obj.col[i][0].green;
            col_b[i * MAX_FACES_OBJECT] = obj.col[i][0].blue;
            for (int j = 1; j < MAX_FACES_OBJECT; j++)
            {
                col_r[i * MAX_FACES_OBJECT + j] = 0;
                col_g[i * MAX_FACES_OBJECT + j] = 0;
                col_b[i * MAX_FACES_OBJECT + j] = 0;
            }
        }
        else
        {
            for (int j = 0; j < MAX_FACES_OBJECT; j++)
            {
                col_r[i * MAX_FACES_OBJECT + j] = obj.col[i][j].red;
                col_g[i * MAX_FACES_OBJECT + j] = obj.col[i][j].green;
                col_b[i * MAX_FACES_OBJECT + j] = obj.col[i][j].blue;
            }
        }
        for (int j = 0; j < MAX_DIMENSIONS_OBJECTS; j++)
        {
            if (obj.nb_dim[i] > j)
            {
                dimension[i * MAX_DIMENSIONS_OBJECTS + j] = obj.dimension[i][j];
            }
            else
            {
                dimension[i * MAX_DIMENSIONS_OBJECTS + j] = 0.0;
            }
        }
    }
    // Copy data that doesn't need to be refetch after initialisation
    if (cudaMemcpyAsync(gpu_obj_pointers.type, obj.type, sizeof(unsigned char) * NB_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.dimension, dimension, sizeof(float) * NB_OBJECT * MAX_DIMENSIONS_OBJECTS, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.red, col_r, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.green, col_g, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.blue, col_b, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.is_single_color, obj.is_single_color, sizeof(bool) * NB_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    cudaStreamSynchronize(gpu_stream);
}

void copy_initial_data_to_video_memory_for_all_streams(gpu_object_pointers *gpu_obj_pointers, object_to_gpu obj, cudaStream_t *gpu_stream)
{
    for (int i = 0; i < NB_STREAM; i++)
    {
        copy_initial_data_to_video_memory(gpu_obj_pointers[i], obj, gpu_stream[i]);
    }
}

void copy_data_to_video_memory(gpu_object_pointers &gpu_obj_pointers, object_to_gpu &obj, cudaStream_t &gpu_stream)
{
    // Puts positional and rotation data in adequate arrays for GPU
    float pos_x[NB_OBJECT];
    float pos_y[NB_OBJECT];
    float pos_z[NB_OBJECT];
    float rot_x[NB_OBJECT];
    float rot_y[NB_OBJECT];
    float rot_z[NB_OBJECT];
    for (int i = 0; i < NB_OBJECT; i++)
    {
        pos_x[i] = obj.pos[i].x;
        pos_y[i] = obj.pos[i].y;
        pos_z[i] = obj.pos[i].z;
        rot_x[i] = obj.rot[i].theta_x;
        rot_y[i] = obj.rot[i].theta_y;
        rot_z[i] = obj.rot[i].theta_z;
    }
    // Copy positional data which has been computed by the CPU
    if (cudaMemcpyAsync(gpu_obj_pointers.pos_x, pos_x, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.pos_y, pos_y, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.pos_z, pos_z, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.rot_x, rot_x, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.rot_y, rot_y, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpyAsync(gpu_obj_pointers.rot_z, rot_z, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to device)\n");
    }
}

void copy_data_from_video_memory(image_array &gpu_image, image_array &img, cudaStream_t &gpu_stream)
{
    // Copy image data from video memory
    if (cudaMemcpyAsync(img.red, gpu_image.red, sizeof(unsigned char) * RESOLUTION, ::cudaMemcpyDeviceToHost, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to host)\n");
    }
    if (cudaMemcpyAsync(img.green, gpu_image.green, sizeof(unsigned char) * RESOLUTION, ::cudaMemcpyDeviceToHost, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to host)\n");
    }
    if (cudaMemcpyAsync(img.blue, gpu_image.blue, sizeof(unsigned char) * RESOLUTION, ::cudaMemcpyDeviceToHost, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to host)\n");
    }
    if (cudaMemcpyAsync(img.alpha, gpu_image.alpha, sizeof(unsigned char) * RESOLUTION, ::cudaMemcpyDeviceToHost, gpu_stream) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to host)\n");
    }
}

void clean_video_memory(id_array *gpu_id_array, image_array *gpu_image, gpu_object_pointers *gpu_obj_pointers, cudaStream_t *gpu_stream)
{
    // Allocate references to variables in GPU memory
    for (int i = 0; i < NB_STREAM; i++)
    {
        //// GPU only id_array variable
        cudaStreamSynchronize(gpu_stream[i]);
        if (cudaFreeAsync(gpu_id_array[i].id, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_id_array[i].side, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        //// GPU Image variable
        if (cudaFreeAsync(gpu_image[i].red, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_image[i].green, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_image[i].blue, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_image[i].alpha, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        //// GPU objects variable allocation
        if (cudaFreeAsync(gpu_obj_pointers[i].type, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].pos_x, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].pos_y, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].pos_z, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].rot_x, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].rot_y, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].rot_z, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].dimension, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].red, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].green, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].blue, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        if (cudaFreeAsync(gpu_obj_pointers[i].is_single_color, gpu_stream[i]) != cudaSuccess)
        {
            printf("Cuda Free Failed\n");
        }
        cudaStreamSynchronize(gpu_stream[i]);
        cudaStreamDestroy(gpu_stream[i]);
    }
}

void synchronize_gpu_image(bool image_is_valid, gpu_object_pointers *gpu_obj_pointers, cudaStream_t *gpu_stream) {
    for (int i = 0; i < NB_STREAM; i++) {
        if (image_is_valid==true and (gpu_obj_pointers[i].state == COPY_TO_GPU or gpu_obj_pointers[i].state == COPY_AND_COMPUTE or gpu_obj_pointers[i].state == ALL_ACTIONS)) {
            if (cudaStreamSynchronize(gpu_stream[i]) != cudaSuccess) {
                printf("Cuda Synchronization for stream %d as failed\n", i);
            }
        }
    }
}


//------------------------------------------------------------------------------------------//
// CPU Functions (Block/Thread allocation management)
//------------------------------------------------------------------------------------------//
__global__ void get_warpSize(int *size)
{
    *size = warpSize;
}

bool allocate_gpu_thread(dim3 &numBlocks, dim3 &threadsPerBlock)
{
    // Getting Warp Size
    int *warp_size_device;
    int warp_size;
    if (cudaMalloc(&warp_size_device, sizeof(int)) != cudaSuccess)
    {
        printf("Cuda Malloc Failed\n");
    }
    get_warpSize<<<1, 1>>>(warp_size_device);
    if (cudaMemcpy(&warp_size, warp_size_device, sizeof(int), ::cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("Cuda Memcpy failed (to host)\n");
    }
    if (cudaFree(warp_size_device) != cudaSuccess)
    {
        printf("Cuda Free Failed\n");
    }
    // Compute thread allocation
    if (RESOLUTION <= 1024)
    {
        numBlocks.x = 1;
        numBlocks.y = 1;
        threadsPerBlock.x = IMAGE_RESOLUTION_HEIGHT;
        threadsPerBlock.y = IMAGE_RESOLUTION_WIDTH;
    }
    else
    {
        if ((IMAGE_RESOLUTION_HEIGHT % warp_size == 0) and (IMAGE_RESOLUTION_WIDTH % warp_size == 0) and (warp_size <= 32))
        {
            numBlocks.x = IMAGE_RESOLUTION_HEIGHT / warp_size;
            numBlocks.y = IMAGE_RESOLUTION_WIDTH / warp_size;
            threadsPerBlock.x = warp_size;
            threadsPerBlock.y = warp_size;
        }
        else if ((IMAGE_RESOLUTION_HEIGHT % (warp_size / 2) == 0) and (IMAGE_RESOLUTION_WIDTH % (warp_size / 2) == 0) and (warp_size <= 64))
        {
            numBlocks.x = IMAGE_RESOLUTION_HEIGHT / (warp_size / 2);
            numBlocks.y = IMAGE_RESOLUTION_WIDTH / (warp_size / 2);
            threadsPerBlock.x = (warp_size / 2);
            threadsPerBlock.y = (warp_size / 2);
        }
        else if ((IMAGE_RESOLUTION_HEIGHT % (warp_size / 4) == 0) and (IMAGE_RESOLUTION_WIDTH % (2 * warp_size) == 0) and (warp_size <= 64))
        {
            numBlocks.x = IMAGE_RESOLUTION_HEIGHT / (warp_size / 4);
            numBlocks.y = IMAGE_RESOLUTION_WIDTH / (2 * warp_size);
            threadsPerBlock.x = (warp_size / 4);
            threadsPerBlock.y = (2 * warp_size);
        }
        else
        {
            printf("Resolutions that are not a multiple of %d/%d are not supported yet\n", warp_size, 2 * warp_size);
            return 1;
        }
    }
    numBlocks.z = 1;
    threadsPerBlock.z = 1;
    return 0;
}

//------------------------------------------------------------------------------------------//
// GPU Functions (General Geometry functions)
//------------------------------------------------------------------------------------------//
__device__ float dist2D(float A_x, float A_y, float B_x, float B_y)
{
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return sqrt(dx * dx + dy * dy);
}

__device__ float dist3D(position A, position B)
{
    float dx = B.x - A.x;
    float dy = B.y - A.y;
    float dz = B.z - A.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__device__ float squareDist2D(float A_x, float A_y, float B_x, float B_y)
{
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return dx * dx + dy * dy;
}

__device__ float squareDist3D(position A, position B)
{
    float dx = B.x - A.x;
    float dy = B.y - A.y;
    float dz = B.z - A.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ void rotate2D_x(float &new_y, float &new_z, float old_y, float old_z, float theta)
{
    new_y = old_y * cos(theta) - old_z * sin(theta);
    new_z = old_y * sin(theta) + old_z * cos(theta);
}

__device__ void rotate2D_y(float &new_x, float &new_z, float old_x, float old_z, float theta)
{
    new_z = old_z * cos(theta) - old_x * sin(theta);
    new_x = old_z * sin(theta) + old_x * cos(theta);
}

__device__ void rotate2D_z(float &new_x, float &new_y, float old_x, float old_y, float theta)
{
    new_x = old_x * cos(theta) - old_y * sin(theta);
    new_y = old_x * sin(theta) + old_y * cos(theta);
}

__device__ void rotate3D(position &new_pos, position pos, float theta_x, float theta_y, float theta_z)
{
    rotate2D_x(new_pos.y, new_pos.z, pos.y, pos.z, theta_x);
    rotate2D_y(new_pos.x, new_pos.z, pos.x, new_pos.z, theta_y);
    rotate2D_z(new_pos.x, new_pos.y, new_pos.x, new_pos.y, theta_z);
}

__device__ position update_camera_perspective(position old_pos)
{
    position new_pos;
    float ratio;
    if (old_pos.z <= 0.0)
    {
        ratio = old_pos.z / (0.0 - CAMERA_Z);
    }
    else
    {
        ratio = old_pos.z / (old_pos.z - CAMERA_Z);
    }
    new_pos.x = (CAMERA_X - old_pos.x) * ratio + old_pos.x;
    new_pos.y = (CAMERA_Y - old_pos.y) * ratio + old_pos.y;
    new_pos.z = old_pos.z;
    return new_pos;
}

__device__ int sort_dist_list(float *list_dist, int *list_faces, int size)
{
    float temp_f;
    int temp_c;
    for (int i = 0; i < size - 1; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            if (list_dist[i] > list_dist[j])
            {
                temp_f = list_dist[i];
                temp_c = list_faces[i];
                list_dist[i] = list_dist[j];
                list_faces[i] = list_faces[j];
                list_dist[j] = temp_f;
                list_faces[j] = temp_c;
            }
        }
    }
    for (int i = 0; i < size; i++)
    {
        if (list_faces[i] != -1)
            return list_faces[i];
    }
    return -1;
}

__device__ bool belongs_2D_4side_convex_polygone(float x, float y, position A0, position A1, position A2, position A3, float D, int face)
{
    float det1 = (A1.x - A0.x) * (y - A0.y) - (A1.y - A0.y) * (x - A0.x);
    float det2 = (A2.x - A1.x) * (y - A1.y) - (A2.y - A1.y) * (x - A1.x);
    if (det1 * det2 <= 0)
        return -1;
    float det3 = (A3.x - A2.x) * (y - A2.y) - (A3.y - A2.y) * (x - A2.x);
    if (det2 * det3 <= 0)
        return -1;
    float det4 = (A0.x - A3.x) * (y - A3.y) - (A0.y - A3.y) * (x - A3.x);
    if (det3 * det4 <= 0)
        return -1;
    else
        return face;
}

__device__ int belongs_2D_4side_convex_polygone_with_sides(float x, float y, position A0, position A1, position A2, position A3, float D, int face)
{
    // Point is equal to one of the points of the polygon
    if (x == A0.x and y == A0.y)
        return face;
    if (x == A1.x and y == A1.y)
        return face;
    if (x == A2.x and y == A2.y)
        return face;
    if (x == A3.x and y == A3.y)
        return face;
    // Check if it is inside/on the sides
    float det1 = (A1.x - A0.x) * (y - A0.y) - (A1.y - A0.y) * (x - A0.x);
    float det2 = (A2.x - A1.x) * (y - A1.y) - (A2.y - A1.y) * (x - A1.x);
    float det3 = (A3.x - A2.x) * (y - A2.y) - (A3.y - A2.y) * (x - A2.x);
    float det4 = (A0.x - A3.x) * (y - A3.y) - (A0.y - A3.y) * (x - A3.x);
    // Point is on the sides
    if (det1 == 0.0)
    {
        if (det2 * det3 <= 0)
            return -1;
        if (det3 * det4 <= 0)
            return -1;
    }
    else if (det2 == 0.0)
    {
        if (det1 * det3 <= 0)
            return -1;
        if (det3 * det4 <= 0)
            return -1;
    }
    else if (det3 == 0.0)
    {
        if (det1 * det2 <= 0)
            return -1;
        if (det2 * det4 <= 0)
            return -1;
    }
    else if (det4 == 0.0)
    {
        if (det1 * det2 <= 0)
            return -1;
        if (det2 * det3 <= 0)
            return -1;
        // Point is inside
    }
    else
    {
        if (det1 * det2 < 0)
            return -1;
        if (det2 * det3 < 0)
            return -1;
        if (det3 * det4 < 0)
            return -1;
    }
    return face;
}

//------------------------------------------------------------------------------------------//
// GPU Functions (Object detection)
//------------------------------------------------------------------------------------------//
__device__ int is_in_cube(float x, float y, float pos_x, float pos_y, float pos_z, float rot_x, float rot_y, float rot_z, float *dimensions)
{
    // Compute positions of cube summits
    //    E----F
    //   /    /|
    //  A----B G
    //  |    |/
    //  D----C
    position A;
    A.x = -dimensions[0] / 2.0;
    A.y = -dimensions[0] / 2.0;
    A.z = -dimensions[0] / 2.0;
    rotate3D(A, A, rot_x, rot_y, rot_z);
    A.x += pos_x;
    A.y += pos_y;
    A.z += pos_z;
    position B;
    B.x = dimensions[0] / 2.0;
    B.y = -dimensions[0] / 2.0;
    B.z = -dimensions[0] / 2.0;
    rotate3D(B, B, rot_x, rot_y, rot_z);
    B.x += pos_x;
    B.y += pos_y;
    B.z += pos_z;
    position C;
    C.x = dimensions[0] / 2.0;
    C.y = dimensions[0] / 2.0;
    C.z = -dimensions[0] / 2.0;
    rotate3D(C, C, rot_x, rot_y, rot_z);
    C.x += pos_x;
    C.y += pos_y;
    C.z += pos_z;
    position D;
    D.x = -dimensions[0] / 2.0;
    D.y = dimensions[0] / 2.0;
    D.z = -dimensions[0] / 2.0;
    rotate3D(D, D, rot_x, rot_y, rot_z);
    D.x += pos_x;
    D.y += pos_y;
    D.z += pos_z;
    position E;
    E.x = -dimensions[0] / 2.0;
    E.y = -dimensions[0] / 2.0;
    E.z = dimensions[0] / 2.0;
    rotate3D(E, E, rot_x, rot_y, rot_z);
    E.x += pos_x;
    E.y += pos_y;
    E.z += pos_z;
    position F;
    F.x = dimensions[0] / 2.0;
    F.y = -dimensions[0] / 2.0;
    F.z = dimensions[0] / 2.0;
    rotate3D(F, F, rot_x, rot_y, rot_z);
    F.x += pos_x;
    F.y += pos_y;
    F.z += pos_z;
    position G;
    G.x = dimensions[0] / 2.0;
    G.y = dimensions[0] / 2.0;
    G.z = dimensions[0] / 2.0;
    rotate3D(G, G, rot_x, rot_y, rot_z);
    G.x += pos_x;
    G.y += pos_y;
    G.z += pos_z;
    position H;
    H.x = -dimensions[0] / 2.0;
    H.y = dimensions[0] / 2.0;
    H.z = dimensions[0] / 2.0;
    rotate3D(H, H, rot_x, rot_y, rot_z);
    H.x += pos_x;
    H.y += pos_y;
    H.z += pos_z;
    // Compute position of faces centers
    position camera;
    camera.x = CAMERA_X;
    camera.y = CAMERA_Y;
    camera.z = CAMERA_Z;
    position center_front;
    center_front.x = (A.x + B.x + C.x + D.x) / 4.0;
    center_front.y = (A.y + B.y + C.y + D.y) / 4.0;
    center_front.z = (A.z + B.z + C.z + D.z) / 4.0;
    position center_back;
    center_back.x = (E.x + F.x + G.x + H.x) / 4.0;
    center_back.y = (E.y + F.y + G.y + H.y) / 4.0;
    center_back.z = (E.z + F.z + G.z + H.z) / 4.0;
    position center_top;
    center_top.x = (A.x + B.x + F.x + E.x) / 4.0;
    center_top.y = (A.y + B.y + F.y + E.y) / 4.0;
    center_top.z = (A.z + B.z + F.z + E.z) / 4.0;
    position center_bottom;
    center_bottom.x = (D.x + C.x + G.x + H.x) / 4.0;
    center_bottom.y = (D.y + C.y + G.y + H.y) / 4.0;
    center_bottom.z = (D.z + C.z + G.z + H.z) / 4.0;
    position center_right;
    center_right.x = (B.x + F.x + G.x + C.x) / 4.0;
    center_right.y = (B.y + F.y + G.y + C.y) / 4.0;
    center_right.z = (B.z + F.z + G.z + C.z) / 4.0;
    position center_left;
    center_left.x = (A.x + E.x + H.x + D.x) / 4.0;
    center_left.y = (A.y + E.y + H.y + D.y) / 4.0;
    center_left.z = (A.z + E.z + H.z + D.z) / 4.0;
    // Compute distance to faces
    float dist_to_cam[6];
    dist_to_cam[0] = squareDist3D(center_front, camera);
    dist_to_cam[1] = squareDist3D(center_back, camera);
    dist_to_cam[2] = squareDist3D(center_top, camera);
    dist_to_cam[3] = squareDist3D(center_bottom, camera);
    dist_to_cam[4] = squareDist3D(center_right, camera);
    dist_to_cam[5] = squareDist3D(center_left, camera);
    // Update Points position on screen relative to camera viewpoint
    A = update_camera_perspective(A);
    B = update_camera_perspective(B);
    C = update_camera_perspective(C);
    D = update_camera_perspective(D);
    E = update_camera_perspective(E);
    F = update_camera_perspective(F);
    G = update_camera_perspective(G);
    H = update_camera_perspective(H);
    // Check hit faces
    int list_faces[6];
    // Front
    list_faces[0] = belongs_2D_4side_convex_polygone_with_sides(x, y, A, B, C, D, dimensions[0], 2);
    // Back
    list_faces[1] = belongs_2D_4side_convex_polygone_with_sides(x, y, E, F, G, H, dimensions[0], 3);
    // Top
    list_faces[2] = belongs_2D_4side_convex_polygone_with_sides(x, y, A, B, F, E, dimensions[0], 0);
    // Bottom
    list_faces[3] = belongs_2D_4side_convex_polygone_with_sides(x, y, D, C, G, H, dimensions[0], 5);
    // Right
    list_faces[4] = belongs_2D_4side_convex_polygone_with_sides(x, y, B, F, G, C, dimensions[0], 1);
    // Left
    list_faces[5] = belongs_2D_4side_convex_polygone_with_sides(x, y, A, E, H, D, dimensions[0], 4);
    // Return closest hit face
    return sort_dist_list(dist_to_cam, list_faces, 6);
}

__device__ bool is_in_sphere(float x, float y, float pos_x, float pos_y, float pos_z, float dimensions)
{
    float new_pos_x, new_pos_y;
    float ratio;
    float new_dim;
    if (pos_z < 0.0)
    {
        ratio = pos_z / (-CAMERA_Z);
    }
    else
    {
        ratio = pos_z / (pos_z - CAMERA_Z);
    }
    new_dim = (1.0 - ratio) * dimensions;
    new_pos_x = (CAMERA_X - pos_x) * ratio + pos_x;
    new_pos_y = (CAMERA_Y - pos_y) * ratio + pos_y;
    return (squareDist2D(x, y, new_pos_x, new_pos_y) <= new_dim * new_dim);
}

__device__ int is_in_object(float x, float y, unsigned char id, float pos_x, float pos_y, float pos_z, float rot_x, float rot_y, float rot_z, float *dimensions)
{
    bool side;
    if (id == CUBE)
    {
        return is_in_cube(x, y, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, dimensions);
    }
    else if (id == SPHERE)
    {
        side = is_in_sphere(x, y, pos_x, pos_y, pos_z, dimensions[0]);
        if (side == true)
            return 0;
        else
            return -1;
    }
    else
    {
        return -1;
    }
}

//------------------------------------------------------------------------------------------//
// GPU Functions (Color related)
//------------------------------------------------------------------------------------------//
__device__ colors get_colors(int id, int side, gpu_object_pointers &gpu_obj_pointers)
{
    colors col;
    int single = (gpu_obj_pointers.is_single_color[id] == false) ? side : 0;
    unsigned char is_valid = (id != -1) ? 1 : 0;
    col.red = is_valid * gpu_obj_pointers.red[id * MAX_FACES_OBJECT + single];
    col.green = is_valid * gpu_obj_pointers.green[id * MAX_FACES_OBJECT + single];
    col.blue = is_valid * gpu_obj_pointers.blue[id * MAX_FACES_OBJECT + single];
    return col;
}

//------------------------------------------------------------------------------------------//
// GPU Functions (Scene rendering)
//------------------------------------------------------------------------------------------//
__global__ void update_identifiers(gpu_object_pointers gpu_obj_pointers, id_array identifier_array)
{
    int height = blockIdx.x * blockDim.x + threadIdx.x;
    int width = blockIdx.y * blockDim.y + threadIdx.y;
    float x = IMAGE_OFFSET_WIDTH + PIXEL_WIDTH_SIZE / 2.0 + ((float)width) * PIXEL_WIDTH_SIZE;
    float y = IMAGE_OFFSET_HEIGHT + PIXEL_HEIGHT_SIZE / 2.0 + ((float)height) * PIXEL_HEIGHT_SIZE;
    identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = -1;
    identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = -1;
    for (int i = 0; i < NB_OBJECT; i++)
    {
        float dim[MAX_DIMENSIONS_OBJECTS];
#pragma unroll
        for (int j = 0; j < MAX_DIMENSIONS_OBJECTS; j++)
        {
            dim[j] = gpu_obj_pointers.dimension[i * MAX_DIMENSIONS_OBJECTS + j];
        }
        if (identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] == -1)
        {
            if (is_in_sphere(x, y, gpu_obj_pointers.pos_x[i], gpu_obj_pointers.pos_y[i], gpu_obj_pointers.pos_z[i], dim[0]))
            {
                int is_in = is_in_object(x, y, gpu_obj_pointers.type[i], gpu_obj_pointers.pos_x[i], gpu_obj_pointers.pos_y[i], gpu_obj_pointers.pos_z[i], gpu_obj_pointers.rot_x[i], gpu_obj_pointers.rot_y[i], gpu_obj_pointers.rot_z[i], dim);
                if (is_in != -1)
                {
                    identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = i;
                    identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = is_in;
                }
            }
        }
        else if (gpu_obj_pointers.pos_z[i] < gpu_obj_pointers.pos_z[identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width]])
        {
            if (is_in_sphere(x, y, gpu_obj_pointers.pos_x[i], gpu_obj_pointers.pos_y[i], gpu_obj_pointers.pos_z[i], dim[0]))
            {
                int is_in = is_in_object(x, y, gpu_obj_pointers.type[i], gpu_obj_pointers.pos_x[i], gpu_obj_pointers.pos_y[i], gpu_obj_pointers.pos_z[i], gpu_obj_pointers.rot_x[i], gpu_obj_pointers.rot_y[i], gpu_obj_pointers.rot_z[i], dim);
                if (is_in != -1)
                {
                    identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = i;
                    identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = is_in;
                }
            }
        }
    }
}

__global__ void update_image(id_array identifier_array, gpu_object_pointers gpu_obj_pointers, image_array image)
{
    int height = blockIdx.x * blockDim.x + threadIdx.x;
    int width = blockIdx.y * blockDim.y + threadIdx.y;
    colors col = get_colors(identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width], identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width], gpu_obj_pointers);
    image.red[height * IMAGE_RESOLUTION_WIDTH + width] = col.red;
    image.green[height * IMAGE_RESOLUTION_WIDTH + width] = col.green;
    image.blue[height * IMAGE_RESOLUTION_WIDTH + width] = col.blue;
    image.alpha[height * IMAGE_RESOLUTION_WIDTH + width] = 0;
}

__global__ void simple_anti_aliasing(image_array image)
{
    int height = blockIdx.x * blockDim.x + threadIdx.x;
    int width = blockIdx.y * blockDim.y + threadIdx.y;
    int AA_result_red = 0;
    int AA_result_green = 0;
    int AA_result_blue = 0;
    if ((height > AA_SIMPLE_SURROUNDING_PIXEL - 1) and (height < IMAGE_RESOLUTION_HEIGHT - AA_SIMPLE_SURROUNDING_PIXEL))
    {
        if ((width > AA_SIMPLE_SURROUNDING_PIXEL - 1) and (width < IMAGE_RESOLUTION_WIDTH - AA_SIMPLE_SURROUNDING_PIXEL))
        {
            for (int i = -AA_SIMPLE_SURROUNDING_PIXEL; i <= AA_SIMPLE_SURROUNDING_PIXEL; i++)
            {
                for (int j = -AA_SIMPLE_SURROUNDING_PIXEL; j <= AA_SIMPLE_SURROUNDING_PIXEL; j++)
                {
                    AA_result_red += image.red[(height + i) * IMAGE_RESOLUTION_WIDTH + (width + j)];
                    AA_result_green += image.green[(height + i) * IMAGE_RESOLUTION_WIDTH + (width + j)];
                    AA_result_blue += image.blue[(height + i) * IMAGE_RESOLUTION_WIDTH + (width + j)];
                }
            }
            AA_result_red /= (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1) * (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1);
            image.red[height * IMAGE_RESOLUTION_WIDTH + width] = AA_result_red;
            AA_result_green /= (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1) * (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1);
            image.green[height * IMAGE_RESOLUTION_WIDTH + width] = AA_result_green;
            AA_result_blue /= (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1) * (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1);
            image.blue[height * IMAGE_RESOLUTION_WIDTH + width] = AA_result_blue;
        }
    }
}

//------------------------------------------------------------------------------------------//
// GPU Functions (Debug)
//------------------------------------------------------------------------------------------//
__global__ void print_identifier_array(id_array identifier_array)
{
    for (int height = 0; height < IMAGE_RESOLUTION_HEIGHT; height++)
    {
        printf(" ");
        for (int width = 0; width < IMAGE_RESOLUTION_WIDTH; width++)
        {
            int carac = identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width];
            int side = identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width];
            if (carac == -1 or carac >= 10)
            {
                printf("%d", carac);
            }
            else
            {
                printf(" %d", carac);
            }
            if (side == -1 or side >= 10)
            {
                printf(",%d | ", side);
            }
            else
            {
                printf(", %d | ", side);
            }
        }
        printf("\n");
    }
}

//------------------------------------------------------------------------------------------//
// Draw image function
//------------------------------------------------------------------------------------------//
bool draw_image(object_to_gpu &tab_pos, image_array &image, id_array *identifier_array, image_array *gpu_image, gpu_object_pointers *gpu_obj_pointers, cudaStream_t *gpu_stream, dim3 numBlocks, dim3 threadsPerBlock)
{
    bool image_is_valid = false;
    if (USE_SYNCHRONOUS_GPU == true)
    {
        // Synchronize with GPU stream to be sure that last operations are finished
        if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
        {
            printf("Cuda Synchronization for stream 0 as failed\n");
        }

        // Copy data to video memory
        copy_data_to_video_memory(gpu_obj_pointers[0], tab_pos, gpu_stream[0]);
        if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
        {
            printf("Cuda Synchronization for stream 0 as failed\n");
        }

        // Compute which object is visible (and which face can we see) for each pixel
        update_identifiers<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(gpu_obj_pointers[0], identifier_array[0]);
        if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
        {
            printf("Cuda Synchronization for stream 0 as failed\n");
        }
        // print_identifier_array<<<1, 1>>>(identifier_array);

        // Assigns colors to each pixel, simply based on which object is visible (no light computations)
        update_image<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(identifier_array[0], gpu_obj_pointers[0], gpu_image[0]);
        if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
        {
            printf("Cuda Synchronization for stream 0 as failed\n");
        }

        // AntiAliasing
        if (AA == "simple")
        {
            simple_anti_aliasing<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(gpu_image[0]);
            if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
            {
                printf("Cuda Synchronization for stream 0 as failed\n");
            }
        }

        // Copy data from video memory (TODO: Needs improvement as this is the slowest point of the "compute")
        copy_data_from_video_memory(gpu_image[0], image, gpu_stream[0]);
        if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
        {
            printf("Cuda Synchronization for stream 0 as failed\n");
        }
        image_is_valid = true;
        gpu_obj_pointers[0].state = ALL_ACTIONS;
    }
    else if (ENABLE_MULTISTREAM == false)
    {
        // Synchronize with GPU stream to be sure that last operations are finished
        if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
        {
            printf("Cuda Synchronization for stream 0 as failed\n");
        }

        // Copy data to video memory
        copy_data_to_video_memory(gpu_obj_pointers[0], tab_pos, gpu_stream[0]);

        // Compute which object is visible (and which face can we see) for each pixel
        update_identifiers<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(gpu_obj_pointers[0], identifier_array[0]);
        // print_identifier_array<<<1, 1>>>(identifier_array);

        // Assigns colors to each pixel, simply based on which object is visible (no light computations)
        update_image<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(identifier_array[0], gpu_obj_pointers[0], gpu_image[0]);

        // AntiAliasing
        if (AA == "simple")
        {
            simple_anti_aliasing<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(gpu_image[0]);
        }

        // Copy data from video memory (TODO: Needs improvement as this is the slowest point of the "compute")
        copy_data_from_video_memory(gpu_image[0], image, gpu_stream[0]);
        if (gpu_obj_pointers[0].state == ALL_ACTIONS)
        {
            image_is_valid = true;
        }
        gpu_obj_pointers[0].state = ALL_ACTIONS;
    }
    else if (ENABLE_LOW_LATENCY_MULTISTREAM == true)
    {
        if (gpu_obj_pointers[0].state == NONE)
        {
            gpu_obj_pointers[0].state = COPY_FROM_GPU;
            gpu_obj_pointers[1].state = COPY_AND_COMPUTE;
            if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
            {
                printf("Cuda Synchronization for stream %d as failed\n", 0);
            }
            // Copy data to video memory
            copy_data_to_video_memory(gpu_obj_pointers[0], tab_pos, gpu_stream[0]);

            // Compute which object is visible (and which face can we see) for each pixel
            update_identifiers<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(gpu_obj_pointers[0], identifier_array[0]);
            // print_identifier_array<<<1, 1>>>(identifier_array);

            // Assigns colors to each pixel, simply based on which object is visible (no light computations)
            update_image<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(identifier_array[0], gpu_obj_pointers[0], gpu_image[0]);

            // AntiAliasing
            if (AA == "simple")
            {
                simple_anti_aliasing<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(gpu_image[0]);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(NB_STREAM) schedule(static)
            for (int i = 0; i < 2; i++)
            {
                // Synchronize with GPU stream to be sure that last operations are finished
                if (gpu_obj_pointers[i].state != NONE)
                {
                    if (cudaStreamSynchronize(gpu_stream[i]) != cudaSuccess)
                    {
                        printf("Cuda Synchronization for stream %d as failed\n", i);
                    }
                }
                if (gpu_obj_pointers[i].state == COPY_AND_COMPUTE)
                {
                    // Copy data to video memory
                    copy_data_to_video_memory(gpu_obj_pointers[i], tab_pos, gpu_stream[i]);

                    // Compute which object is visible (and which face can we see) for each pixel
                    update_identifiers<<<numBlocks, threadsPerBlock, 0, gpu_stream[i]>>>(gpu_obj_pointers[i], identifier_array[i]);
                    // print_identifier_array<<<1, 1>>>(identifier_array);

                    // Assigns colors to each pixel, simply based on which object is visible (no light computations)
                    update_image<<<numBlocks, threadsPerBlock, 0, gpu_stream[i]>>>(identifier_array[i], gpu_obj_pointers[i], gpu_image[i]);

                    // AntiAliasing
                    if (AA == "simple")
                    {
                        simple_anti_aliasing<<<numBlocks, threadsPerBlock, 0, gpu_stream[i]>>>(gpu_image[i]);
                    }
                    gpu_obj_pointers[i].state = COPY_FROM_GPU;
                }
                else if (gpu_obj_pointers[i].state == COPY_FROM_GPU)
                {
                    // Copy data from video memory (TODO: Needs improvement as this is the slowest point of the "compute")
                    copy_data_from_video_memory(gpu_image[i], image, gpu_stream[i]);
                    image_is_valid = true;
                    gpu_obj_pointers[i].state = COPY_AND_COMPUTE;
                }
            }
        }
    }
    else
    {
        if (gpu_obj_pointers[0].state == NONE)
        {
            gpu_obj_pointers[0].state = COMPUTE;
            gpu_obj_pointers[1].state = COPY_TO_GPU;
            gpu_obj_pointers[2].state = NONE;
            if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
            {
                printf("Cuda Synchronization for stream %d as failed\n", 0);
            }
            // Copy data to video memory
            copy_data_to_video_memory(gpu_obj_pointers[0], tab_pos, gpu_stream[0]);
        }
        else if (gpu_obj_pointers[2].state == NONE)
        {
            gpu_obj_pointers[0].state = COPY_FROM_GPU;
            gpu_obj_pointers[1].state = COMPUTE;
            gpu_obj_pointers[2].state = COPY_TO_GPU;
            if (cudaStreamSynchronize(gpu_stream[0]) != cudaSuccess)
            {
                printf("Cuda Synchronization for stream %d as failed\n", 0);
            }
            // Compute which object is visible (and which face can we see) for each pixel
            update_identifiers<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(gpu_obj_pointers[0], identifier_array[0]);
            // print_identifier_array<<<1, 1>>>(identifier_array);

            // Assigns colors to each pixel, simply based on which object is visible (no light computations)
            update_image<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(identifier_array[0], gpu_obj_pointers[0], gpu_image[0]);

            // AntiAliasing
            if (AA == "simple")
            {
                simple_anti_aliasing<<<numBlocks, threadsPerBlock, 0, gpu_stream[0]>>>(gpu_image[0]);
            }
            if (cudaStreamSynchronize(gpu_stream[1]) != cudaSuccess)
            {
                printf("Cuda Synchronization for stream %d as failed\n", 1);
            }
            // Copy data to video memory
            copy_data_to_video_memory(gpu_obj_pointers[1], tab_pos, gpu_stream[1]);
        }
        else
        {
            #pragma omp parallel for num_threads(NB_STREAM) schedule(static)
            for (int i = 0; i < 3; i++)
            {
                // Synchronize with GPU stream to be sure that last operations are finished
                if (cudaStreamSynchronize(gpu_stream[i]) != cudaSuccess)
                {
                    printf("Cuda Synchronization for stream %d as failed\n", i);
                }
                if (gpu_obj_pointers[i].state == COPY_TO_GPU)
                {
                    // Copy data to video memory
                    copy_data_to_video_memory(gpu_obj_pointers[i], tab_pos, gpu_stream[i]);
                    gpu_obj_pointers[i].state = COMPUTE;
                }
                else if (gpu_obj_pointers[i].state == COMPUTE)
                {
                    // Compute which object is visible (and which face can we see) for each pixel
                    update_identifiers<<<numBlocks, threadsPerBlock, 0, gpu_stream[i]>>>(gpu_obj_pointers[i], identifier_array[i]);
                    // print_identifier_array<<<1, 1>>>(identifier_array);

                    // Assigns colors to each pixel, simply based on which object is visible (no light computations)
                    update_image<<<numBlocks, threadsPerBlock, 0, gpu_stream[i]>>>(identifier_array[i], gpu_obj_pointers[i], gpu_image[i]);

                    // AntiAliasing
                    if (AA == "simple")
                    {
                        simple_anti_aliasing<<<numBlocks, threadsPerBlock, 0, gpu_stream[i]>>>(gpu_image[i]);
                    }
                    gpu_obj_pointers[i].state = COPY_FROM_GPU;
                }
                else if (gpu_obj_pointers[i].state == COPY_FROM_GPU)
                {
                    // Copy data from video memory (TODO: Needs improvement as this is the slowest point of the "compute")
                    copy_data_from_video_memory(gpu_image[i], image, gpu_stream[i]);
                    image_is_valid = true;
                    gpu_obj_pointers[i].state = COPY_TO_GPU;
                }
            }
        }
    }
    return image_is_valid;
}

//------------------------------------------------------------------------------------------//
// Benchmark debug functions
//------------------------------------------------------------------------------------------//
void benchmark_performance(int i, std::chrono::_V2::system_clock::time_point before_image_draw, time_benchmarking *time_table, gpu_object_pointers *gpu_obj_pointers, cudaStream_t *gpu_stream)
{
    #pragma omp parallel for num_threads(NB_STREAM) schedule(static)
    for (int st = 0; st < NB_STREAM; st++)
    {
        if (gpu_obj_pointers[st].state != NONE)
        {
            if (cudaStreamSynchronize(gpu_stream[st]) != cudaSuccess)
            {
                printf("Cuda Synchronization for stream %d as failed (Performance Debug)\n", st);
            }
        }
        auto after_image_draw = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> temp = after_image_draw - before_image_draw;
        if ((i >= NB_STREAM - 1) and gpu_obj_pointers[st].state == COPY_TO_GPU)
        {
            int image_id = i - 2;
            if (image_id == 0)
            {
                time_table[image_id].copy_from_time = temp.count();
                time_table[image_id].time_since_start = temp.count() + time_table[image_id].copy_to_time + time_table[image_id].compute_time;
            }
            else
            {
                time_table[image_id].copy_from_time = temp.count();
                time_table[image_id].time_since_start = temp.count() + time_table[image_id - 1].time_since_start;
            }
            time_table[image_id].time_frame_rendering = temp.count() + time_table[image_id].copy_to_time + time_table[image_id].compute_time;
        }
        else if (gpu_obj_pointers[st].state == COMPUTE)
        {
            time_table[i].copy_to_time = temp.count();
        }
        else if (gpu_obj_pointers[st].state == COPY_FROM_GPU)
        {
            if (NB_STREAM == 2)
            {
                time_table[i].copy_to_and_compute_time = temp.count();
            }
            else
            {
                time_table[i - 1].compute_time = temp.count();
            }
        }
        else if (i < NB_STREAM - 1 and gpu_obj_pointers[st].state == COMPUTE)
        {
            time_table[i].copy_to_time = temp.count();
        }
        else if ((i >= NB_STREAM - 1) and gpu_obj_pointers[st].state == COPY_AND_COMPUTE)
        {
            int image_id = i - 1;
            if (image_id == 0)
            {
                time_table[image_id].copy_from_time = temp.count();
                time_table[image_id].time_since_start = temp.count() + time_table[image_id].copy_to_and_compute_time;
            }
            else
            {
                time_table[image_id].copy_from_time = temp.count();
                time_table[image_id].time_since_start = temp.count() + time_table[image_id - 1].time_since_start;
            }
            time_table[image_id].time_frame_rendering = temp.count() + time_table[image_id].copy_to_and_compute_time;
        }
        else if (gpu_obj_pointers[st].state == ALL_ACTIONS)
        {
            if (i > 0)
            {
                time_table[i].time_since_start = temp.count() + time_table[i - 1].time_since_start;
            }
            else
            {
                time_table[i].time_since_start = temp.count();
            }
            time_table[i].time_since_last_frame = temp.count();
            time_table[i].time_frame_rendering = temp.count();
        }
        else
        {
        }
    }
    if (NB_STREAM == 3)
    {
        int image_id = i - 2;
        double time_max = std::max(std::max(time_table[image_id + 2].copy_to_time, time_table[image_id + 1].compute_time), time_table[image_id].copy_from_time);
        if (image_id == 0)
        {
            time_table[image_id].time_since_last_frame = time_max + time_table[image_id].copy_to_time + time_table[image_id].compute_time;
        }
        else
        {
            time_table[image_id].time_since_last_frame = time_max;
        }
    }
    else if (NB_STREAM == 2)
    {
        int image_id = i - 1;
        double time_max = std::max(time_table[image_id + 1].copy_to_and_compute_time, time_table[image_id].copy_from_time);
        if (image_id == 0)
        {
            time_table[image_id].time_since_last_frame = time_max + time_table[image_id].copy_to_and_compute_time;
        }
        else
        {
            time_table[image_id].time_since_last_frame = time_max;
        }
    }
}

void reinit_terminal(int i)
{
    if (i > 0)
    {
        printf("\x1b[1F"); // Move to beginning of previous line
        printf("\x1b[2K"); // Clear entire line
        if (NB_STREAM == 2)
        {
            printf("\x1b[1F"); // Move to beginning of previous line
            printf("\x1b[2K"); // Clear entire line
            if (i >= 2)
            {
                printf("\x1b[1F"); // Move to beginning of previous line
                printf("\x1b[2K"); // Clear entire line
            }
        }
        else if (NB_STREAM == 3)
        {
            printf("\x1b[1F"); // Move to beginning of previous line
            printf("\x1b[2K"); // Clear entire line
            if (i >= 2)
            {
                printf("\x1b[1F"); // Move to beginning of previous line
                printf("\x1b[2K"); // Clear entire line
                if (i >= 3)
                {
                    printf("\x1b[1F"); // Move to beginning of previous line
                    printf("\x1b[2K"); // Clear entire line
                }
            }
        }
    }
}

void print_intermediate_bench_values(int i, time_benchmarking *time_table)
{
    if (i >= (NB_STREAM - 1))
    {
        printf("Step: %d | Image %d / %d took %f ms to render\n", i + 1, i - (NB_STREAM - 1) + 1, RENDERED_FRAMES, time_table[i - (NB_STREAM - 1)].time_since_last_frame);
        if (NB_STREAM == 2)
        {
            printf("\tCopy and Compute step for image %d took %f ms\n", i - (NB_STREAM - 1) + 2, time_table[i - (NB_STREAM - 1) + 1].copy_to_and_compute_time);
            printf("\tCopy from GPU memory step for image %d took %f ms\n", i - (NB_STREAM - 1) + 1, time_table[i - (NB_STREAM - 1)].copy_from_time);
        }
        else if (NB_STREAM == 3)
        {
            printf("\tCopy to GPU memory step for image %d took %f ms\n", i - (NB_STREAM - 1) + 3, time_table[i - (NB_STREAM - 1) + 2].copy_to_time);
            printf("\tCompute step for image %d took %f ms\n", i - (NB_STREAM - 1) + 2, time_table[i - (NB_STREAM - 1) + 1].compute_time);
            printf("\tCopy from GPU memory step for image %d took %f ms\n", i - (NB_STREAM - 1) + 1, time_table[i - (NB_STREAM - 1)].copy_from_time);
        }
    }
    else
    {
        printf("Step: %d | Image %d / %d is not yet valid\n", i + 1, 1, RENDERED_FRAMES);
        if (NB_STREAM == 2)
        {
            printf("\tCopy and Compute step for image %d took %f ms\n", i + 1, time_table[i].copy_to_and_compute_time);
        }
        else if (NB_STREAM == 3)
        {
            printf("\tCopy to GPU memory step for image %d took %f ms\n", i + 1, time_table[i].copy_to_time);
            if (i == 1)
            {
                printf("\tCompute step for image %d took %f ms\n", i, time_table[i - 1].compute_time);
            }
        }
    }
}

void compute_bench_values(int i, values_benchmarking &bench_values, time_benchmarking *time_table)
{
    if (i >= (NB_STREAM - 1))
    {
        bench_values.mean_time += time_table[i - (NB_STREAM - 1)].time_since_last_frame;
        bench_values.mean_time_render += time_table[i - (NB_STREAM - 1)].time_frame_rendering;
        bench_values.mean_time_copy_to += time_table[i - (NB_STREAM - 1)].copy_to_time;
        if (NB_STREAM == 2)
        {
            bench_values.mean_time_compute += time_table[i - (NB_STREAM - 1)].copy_to_and_compute_time;
            bench_values.mean_time_copy_from += time_table[i - (NB_STREAM - 1)].copy_from_time;
        }
        else if (NB_STREAM == 3)
        {
            bench_values.mean_time_compute += time_table[i - (NB_STREAM - 1)].compute_time;
            bench_values.mean_time_copy_from += time_table[i - (NB_STREAM - 1)].copy_from_time;
        }
        if (bench_values.min_time > time_table[i - (NB_STREAM - 1)].time_since_last_frame)
        {
            bench_values.min_time = time_table[i - (NB_STREAM - 1)].time_since_last_frame;
            bench_values.index_min_time = i;
        }
        if (bench_values.max_time < time_table[i - (NB_STREAM - 1)].time_since_last_frame)
        {
            bench_values.max_time = time_table[i - (NB_STREAM - 1)].time_since_last_frame;
            bench_values.index_max_time = i;
        }
    }
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
    std::vector<Shape *> shapes;
    shapes.push_back(new Sphere(Point3D(50, 50, 10), 100, Point3D(0, 0, 0), 10, 0.5, 9.81));
    shapes.push_back(new Cube(Point3D(500, 200, 20), 200, Point3D(0, 0, 0),
                              Point3D(0, 0, 0), Point3D(0.1, 0.1, 0.1), 10, 0.5, 9.81));
    shapes.push_back(new Cube(Point3D(100, 100, 50), 400,
                                          Point3D(5, 5, 0), Point3D(0, 0, 0),
                                          Point3D(0, 0, 0), 10, 0.5, 9.81));
    shapes.push_back(new Sphere(Point3D(130, 50, 10), 100, Point3D(0, 0, 0), 10, 0.5, 9.81));
    shapes.push_back(new Sphere(Point3D(700, 50, 10), 100, Point3D(0, 0, 0), 10, 0.5, 9.81));

    // GPU conversion
    int numObjects = convertSceneToGPU(shapes, tab_pos, true);
    std::cout << "Converted " << numObjects << " objects\n";

    double dt = 1.0 / 60.0;

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

        for(int i = 0; i < shapes.size(); i++) {
            Point3D pos = shapes[i]->getCenter();
            Point3D vel = shapes[i]->getVelocity();
            printf("  Shape %d: pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f)\n", 
                i, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z);
        }

        int N = (int)shapes.size();
        for (int i = 0;i < N;i++)
        {
            shapes[i]->update(dt);
        }

        #pragma omp parallel for schedule(dynamic) reduction(+:collisionsDetected,collisionsResolved)
        for(int i=0; i<N; i++) {
            for(int j=i+1; j<N; j++) {
                bool collision = false;
                
                Sphere* s1 = dynamic_cast<Sphere*>(shapes[i]);
                Sphere* s2 = dynamic_cast<Sphere*>(shapes[j]);
                RigidBody* r1 = dynamic_cast<RigidBody*>(shapes[i]);
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
        
        convertSceneToGPU(shapes, tab_pos, true);
        
        //Update GPU state & Render
        auto start_render = std::chrono::high_resolution_clock::now();
        image_validity = draw_image(tab_pos, image_current, gpu_id_array, gpu_image, gpu_obj_pointers, gpu_stream, numBlocks, threadsPerBlock);
        auto end_render = std::chrono::high_resolution_clock::now();
        if (DEBUG_PERF)
        {
            benchmark_performance(i, before_image_draw, time_table, gpu_obj_pointers, gpu_stream);
            reinit_terminal(i);
            print_intermediate_bench_values(i, time_table);
            compute_bench_values(i, bench_values, time_table);
        }

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