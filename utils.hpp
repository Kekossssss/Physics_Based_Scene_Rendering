//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//
#include "stddef.h"

//------------------------------------------------------------------------------------------//
// Global parameters definition
//------------------------------------------------------------------------------------------//
// Configurable DEBUG parameters
#define DEBUG_PERF true
#define DEBUG_VALUES false

// Images resolution for the GPU to render
#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024

// Defines the number of objects that are going to be simulated
#define NB_OBJECT 10

// Defines the maximum number of dimensions that an object can have (radius for a sphere, lenght/height/width for a rectangle, side for a square...)
#define MAX_DIMENSIONS_OBJECTS 3

// Space in which the simulation will occur, and so, in which object can move
#define BOX_WIDTH 2000.0
#define BOX_HEIGHT 2000.0
#define BOX_DEPTH 2000.0

// Maximum simulation time before end of program (ms)
#define MAX_SIMU_TIME 120.0

// Maximum number of rendered images before end of program
#define MAX_RENDERED_FRAMES 12

// Fixed time between 2 rendered frame/New position of objects (ms)
#define INTERVAL_TIME 10.0

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

// Structure to gather all of the objects at a defined time with all of their characteristics (with only GPU related values)
struct object_to_gpu {
    unsigned char id[NB_OBJECT][2]; // [0] = Object type, [1] = Object identifier
    position pos[NB_OBJECT];
    rotation rot[NB_OBJECT];
    float dimension[NB_OBJECT][MAX_DIMENSIONS_OBJECTS];
};

// IMPORTANT: CPU dev only need to provide this structure, GPU dev will do the translation towards the previous struct
// Structure to gather all of the objects at a defined time with all of their characteristics
struct object {
    unsigned char id[NB_OBJECT][2]; // [0] = Object type, [1] = Object identifier
    position pos[NB_OBJECT];
    rotation rot[NB_OBJECT];
    float dimension[NB_OBJECT][MAX_DIMENSIONS_OBJECTS];
    // CPU dev can add more object related values here
};
//------------------------------------------------------------------------------------------//