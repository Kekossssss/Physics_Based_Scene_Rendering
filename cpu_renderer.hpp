#ifndef CPU_RENDERER_HPP // include guard
#define CPU_RENDERER_HPP

#include "utils.hpp"

struct GPUPoint3D
{
    float x, y, z;
};

struct GPUObjectData
{
    // Object type (0=Sphere, 1=Cube, 2=RectangularPrism, etc.)
    unsigned char type;

    // Position
    float pos_x, pos_y, pos_z;

    // Rotation angles
    float rot_x, rot_y, rot_z;

    // Dimensions (up to MAX_DIMENSIONS_OBJECTS)
    float dimensions[8]; // Adjust size as needed

    // Color data (for rendering)
    unsigned char red[6]; // MAX_FACES_OBJECT
    unsigned char green[6];
    unsigned char blue[6];
    bool is_single_color;

    // Velocity (for physics simulation)
    float vel_x, vel_y, vel_z;

    // Angular velocity (for rigid bodies)
    float ang_vel_x, ang_vel_y, ang_vel_z;
};

struct GPUSceneData
{
    int num_objects;
    GPUObjectData *objects; // Array on GPU
};

// Draw an image
void draw_image(object_to_gpu &tab_pos, image_array &image);

#endif
