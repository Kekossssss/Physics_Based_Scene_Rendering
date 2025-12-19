#include "utils.hpp"
#include "cpu_part.hpp"

//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Library used for sleep function
#include <unistd.h>

#include <stdlib.h>
#include <iostream>
#include <thread>
#include <pthread.h>

#include "video_writer.hpp"

// ============================================================================
// MP4 VIDEO ENCODER CLASS
// ============================================================================

MP4VideoEncoder::MP4VideoEncoder(const char *filename, int w, int h, bool debug, int framerate = 60)
    : width(w), height(h), fps(framerate), is_open(false)
{
    // Build FFmpeg command
    char command[512];
    if (not debug) {
        snprintf(command, sizeof(command),
            "singularity exec ffmpeg_latest.sif "
        "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d "
        "-framerate %d -i - -c:v libx264 -preset ultrafast -crf 18 "
        "-pix_fmt yuv420p -movflags +faststart %s 2>/dev/null",
                width, height, fps, filename); 
    } else {
        snprintf(command, sizeof(command),
            "singularity exec ffmpeg_latest.sif "
        "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d "
        "-framerate %d -i - -c:v libx264 -preset ultrafast -crf 18 "
        "-pix_fmt yuv420p -movflags +faststart %s",
                width, height, fps, filename); 
    }

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

bool MP4VideoEncoder::writeFrame(const image_array &image)
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

bool MP4VideoEncoder::writeFrameRGB(const unsigned char *rgb_buffer)
{
    if (!is_open)
        return false;

    size_t written = fwrite(rgb_buffer, 1, width * height * 3, ffmpeg_pipe);
    return (written == (size_t)(width * height * 3));
}

// ============================================================================
// PTHREAD WORKER FUNCTIONS
// ============================================================================

MP4VideoEncoder::~MP4VideoEncoder()
{
    if (is_open)
    {
        std::cout << "Closing FFmpeg encoder...\n";
        pclose(ffmpeg_pipe);
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

void convertToRGBParallel(const image_array &image, unsigned char *rgb_buffer, int num_threads)
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
