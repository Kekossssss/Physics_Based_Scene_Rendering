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
#define AA "off"
#define AA_SIMPLE_SURROUNDING_PIXEL 1


//------------------------------------------------------------------------------------------//
// GPU Related parameters
//------------------------------------------------------------------------------------------//
#define ENABLE_MULTISTREAM true
#define ENABLE_LOW_LATENCY_MULTISTREAM true
#if ENABLE_MULTISTREAM==true
    #if ENABLE_LOW_LATENCY_MULTISTREAM==true
        #define NB_STREAM 2
    #else
        #define NB_STREAM 3
    #endif
#else
    #define NB_STREAM 1
#endif

//------------------------------------------------------------------------------------------//
// Global parameters definition
//------------------------------------------------------------------------------------------//
// Configurable DEBUG parameters
#define DEBUG_PERF true
#define DEBUG_VALUES false

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
#define CAMERA_Z -BOX_DEPTH/16.0

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


// Structure to gather all of the objects at a defined time with all of their characteristics (with only GPU related values)
struct object_to_gpu {
    unsigned char type[NB_OBJECT]; // Object type
    position pos[NB_OBJECT];
    rotation rot[NB_OBJECT];
    int nb_dim[NB_OBJECT];
    float dimension[NB_OBJECT][MAX_DIMENSIONS_OBJECTS]; //Table of the dimensions for each kind of shape, use a function to assign each index to it's corresponding dimension according to the kind of shape
    bool is_single_color[NB_OBJECT];
    colors col[NB_OBJECT][MAX_FACES_OBJECT]; // Follows the following logic for assigning colors : Top -> Bottom of the shape, Right -> Left of the shape, Front -> Back of the shape
};

// IMPORTANT: CPU dev only need to provide this structure, GPU dev will do the translation towards the previous struct
// Structure to gather all of the objects at a defined time with all of their characteristics
struct object {
    unsigned char id[NB_OBJECT][2]; // [0] = Object type, [1] = Object identifier
    position pos[NB_OBJECT];
    rotation rot[NB_OBJECT];
    float dimension[NB_OBJECT][MAX_DIMENSIONS_OBJECTS]; //Table of the dimensions for each kind of shape, use a function to assign each index to it's corresponding dimension according to the kind of shape
    // CPU dev can add more object related values here
};

// Image array outputed by the GPU in order to be visualised
struct image_array {
    unsigned char* red;
    unsigned char* green;
    unsigned char* blue;
    unsigned char* alpha;  //Opacity of the color
};

// Small AI generated function to output an image from the image array (does not currently support alpha values)
bool save_as_bmp(const image_array& img, const char* filename);

#endif