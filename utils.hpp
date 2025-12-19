//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//

#ifndef UTILS_HPP
#define UTILS_HPP

#include "stddef.h"
#include <fstream>
#include <iostream>
#include <cstring>

//------------------------------------------------------------------------------------------//
// GPU Rendering Quality Parameters
//------------------------------------------------------------------------------------------//
// Anti-Aliasing parameters : off, simple, ...
#define AA "simple"
#define AA_SIMPLE_SURROUNDING_PIXEL 1


//------------------------------------------------------------------------------------------//
// GPU Related parameters
//------------------------------------------------------------------------------------------//
#define USE_SYNCHRONOUS_GPU false
#define ENABLE_MULTISTREAM false
#define ENABLE_LOW_LATENCY_MULTISTREAM true
#if ENABLE_MULTISTREAM == true and USE_SYNCHRONOUS_GPU == false
#   if ENABLE_LOW_LATENCY_MULTISTREAM == true
#       define NB_STREAM 2
#   else
#       define NB_STREAM 3
#   endif
#else
#   define NB_STREAM 1
#endif

//------------------------------------------------------------------------------------------//
// Global parameters definition
//------------------------------------------------------------------------------------------//
// Configurable DEBUG parameters
#define DEBUG_PERF true
#define DEBUG_VALUES false
#if DEBUG_PERF == true
#   define KEEP_VALUES_HISTORIC false
#endif
#define ONLY_FINAL_FRAME false

// Images resolution for the GPU to render
#define IMAGE_RESOLUTION_WIDTH 640
#define IMAGE_RESOLUTION_WIDTH_FLOAT 640.0
#define IMAGE_RESOLUTION_HEIGHT 320
#define IMAGE_RESOLUTION_HEIGHT_FLOAT 320.0
#define RESOLUTION IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT

// Image real size in the space
#define IMAGE_TO_BOX_RATIO 100.0
#if IMAGE_RESOLUTION_WIDTH>IMAGE_RESOLUTION_HEIGHT
    #define IMAGE_WIDTH BOX_WIDTH*(IMAGE_TO_BOX_RATIO/100.0)*(IMAGE_RESOLUTION_WIDTH_FLOAT/IMAGE_RESOLUTION_HEIGHT_FLOAT)
    #define IMAGE_HEIGHT BOX_HEIGHT*(IMAGE_TO_BOX_RATIO/100.0)
#else
    #define IMAGE_WIDTH BOX_WIDTH*(IMAGE_TO_BOX_RATIO/100.0)
    #define IMAGE_HEIGHT BOX_HEIGHT*(IMAGE_TO_BOX_RATIO/100.0)*(IMAGE_RESOLUTION_HEIGHT_FLOAT/IMAGE_RESOLUTION_WIDTH_FLOAT)
#endif

// Top Coordinates (0,0) of the image in the space (position of the top left corner of the image in the space)
#define IMAGE_OFFSET_WIDTH 0.0
#define IMAGE_OFFSET_HEIGHT 0.0

// Defines the number of objects that are going to be simulated
#define NB_OBJECT 5

// Defines the maximum number of dimensions that an object can have (radius for a sphere, lenght/height/width for a rectangle, side for a square...)
#define MAX_DIMENSIONS_OBJECTS 3

// Defines the maximum number of faces that an object can have (used to define the colors of each face)
#define MAX_FACES_OBJECT 6

// Space in which the simulation will occur, and so, in which object can move
#define BOX_WIDTH 2000.0
#define BOX_HEIGHT 2000.0
#define BOX_DEPTH 2000.0

// Pixel sizes
#define PIXEL_WIDTH_SIZE IMAGE_WIDTH/IMAGE_RESOLUTION_WIDTH_FLOAT
#define PIXEL_HEIGHT_SIZE IMAGE_HEIGHT/IMAGE_RESOLUTION_HEIGHT_FLOAT

// Maximum simulation time before end of program (ms)
#define MAX_SIMU_TIME 120.0

// Maximum number of rendered images before end of program
#define MAX_RENDERED_FRAMES 12

// Fixed time between 2 rendered frame/New position of objects (ms)
#define INTERVAL_TIME 10.0

// Correspondance between ID of objects and their types
#define CUBE 0
#define SPHERE 1

// Camera viewing position
#define CAMERA_X IMAGE_WIDTH/2.0
#define CAMERA_Y IMAGE_HEIGHT/2.0
#define CAMERA_Z -BOX_DEPTH*2.0

//------------------------------------------------------------------------------------------//
// GPU benchmarking structure
//------------------------------------------------------------------------------------------//
struct time_benchmarking {
    // Enables tracking image output time
    double time_since_start;
    // Enables FPS, mean, max and min render time computation
    double time_since_last_frame;
    // Enables ananlysis of whole frame rendering time
    double time_frame_rendering;
    // Enables analysis of GPU computing times for different parts of the process (only available with 2 or 3 streams)
    double copy_to_time;
    double compute_time;
    double copy_from_time;
    double copy_to_and_compute_time;
};

struct values_benchmarking {
    int index_min_time;
    double min_time;
    int index_max_time;
    double max_time;
    double mean_time;
    double mean_time_render;
    double mean_time_copy_to;
    double mean_time_compute;
    double mean_time_copy_from;
};

//------------------------------------------------------------------------------------------//
// GPU memory pointers structure
//------------------------------------------------------------------------------------------//
enum STREAM_STATE {
    NONE,
    ALL_ACTIONS,
    COPY_AND_COMPUTE,
    COPY_TO_GPU,
    COMPUTE,
    COPY_FROM_GPU
};

struct gpu_object_pointers {
    STREAM_STATE state;
    unsigned char* type;
    float* pos_x;
    float* pos_y;
    float* pos_z;
    float* rot_x;
    float* rot_y;
    float* rot_z;
    float* dimension;
    bool* is_single_color;
    unsigned char* red;
    unsigned char* green;
    unsigned char* blue;
};

//------------------------------------------------------------------------------------------//
// Structures definition
//------------------------------------------------------------------------------------------//
// Structure to simplify the access to the position of an object
struct position {
    float x;
    float y;
    float z;
};

// Structure to simplify the access to the rotation of an object
struct rotation {
    float theta_x;
    float theta_y;
    float theta_z;
};

// Structure to simplify the access to the colors of an object
struct colors {
    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

struct id_array {
    int* id;
    int* side;
};


struct object_to_gpu {
    unsigned char type[NB_OBJECT]; 
    position pos[NB_OBJECT];
    rotation rot[NB_OBJECT];
    int nb_dim[NB_OBJECT];
    float dimension[NB_OBJECT][MAX_DIMENSIONS_OBJECTS]; 
    bool is_single_color[NB_OBJECT];
    colors col[NB_OBJECT][MAX_FACES_OBJECT]; 
};


struct object {
    unsigned char id[NB_OBJECT][2];
    position pos[NB_OBJECT];
    rotation rot[NB_OBJECT];
    float dimension[NB_OBJECT][MAX_DIMENSIONS_OBJECTS]; 
};

struct image_array {
    unsigned char* red;
    unsigned char* green;
    unsigned char* blue;
    unsigned char* alpha;  //Opacity of the color
};

bool save_as_bmp(const image_array& img, const char* filename);

#endif