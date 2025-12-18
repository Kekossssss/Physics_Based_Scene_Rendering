#ifndef CPU_GPU_CONVERTER_HPP
#define CPU_GPU_CONVERTER_HPP

#include "cpu_part.hpp"
#include "utils.hpp"
#include <vector>
#include <array>
#include <cstring>
#include <algorithm>

// ============================================================================
// CONVERSION FUNCTIONS: CPU Objects → GPU Format
// ============================================================================

/**
 * Convert a single CPU Shape object to GPU object_to_gpu structure
 * @param shape: CPU shape object (Sphere, Cube, RectangularPrism, etc.)
 * @param gpuObj: Output GPU object_to_gpu structure
 * @param index: Index in the arrays (0 to NB_OBJECT-1)
 * @param colorR: Red color component (0-255)
 * @param colorG: Green color component (0-255)
 * @param colorB: Blue color component (0-255)
 * @param multicolor: If true, assign different colors to each face (for cubes/prisms)
 */
inline void convertShapeToGPU(const Shape *shape, object_to_gpu &gpuObj, int index,
                              unsigned char colorR = 255,
                              unsigned char colorG = 255,
                              unsigned char colorB = 255,
                              bool multicolor = false)
{

    // Get basic properties
    Point3D center = shape->getCenter();
    Point3D rotation, angVel;
    shape->getRotation(rotation);

    // Set type based on shape type
    int shapeType = shape->getType();
    if (shapeType == 0)
    {
        gpuObj.type[index] = SPHERE;
    }
    else if (shapeType == 1 || shapeType == 2)
    {
        gpuObj.type[index] = CUBE; // Both Cube and RectangularPrism use CUBE type
    }

    // Set position
    gpuObj.pos[index].x = static_cast<float>(center.x);
    gpuObj.pos[index].y = static_cast<float>(center.y);
    gpuObj.pos[index].z = static_cast<float>(center.z);

    // Set rotation
    gpuObj.rot[index].theta_x = static_cast<float>(rotation.x);
    gpuObj.rot[index].theta_y = static_cast<float>(rotation.y);
    gpuObj.rot[index].theta_z = static_cast<float>(rotation.z);

    // Set dimensions based on type
    if (shapeType == 0)
    { // Sphere
        const Sphere *sphere = dynamic_cast<const Sphere *>(shape);
        if (sphere)
        {
            gpuObj.nb_dim[index] = 1;
            gpuObj.dimension[index][0] = static_cast<float>(sphere->getDiameter());
            gpuObj.dimension[index][1] = 0.0f;
            gpuObj.dimension[index][2] = 0.0f;
        }
    }
    else if (shapeType == 1)
    { // Cube
        gpuObj.nb_dim[index] = 1;
        gpuObj.dimension[index][0] = static_cast<float>(shape->getLength());
        gpuObj.dimension[index][1] = 0.0f;
        gpuObj.dimension[index][2] = 0.0f;
    }
    else if (shapeType == 2)
    { // RectangularPrism
        gpuObj.nb_dim[index] = 3;
        gpuObj.dimension[index][0] = static_cast<float>(shape->getLength());
        gpuObj.dimension[index][1] = static_cast<float>(shape->getHeight());
        gpuObj.dimension[index][2] = static_cast<float>(shape->getWidth());
    }

    // Set color data
    gpuObj.is_single_color[index] = !multicolor;

    if (multicolor && (shapeType == 1 || shapeType == 2))
    {
        // Different colors for each face
        // Order: Top, Right, Front, Back, Left, Bottom
        unsigned char faceColors[6][3] = {
            {255, 255, 255}, // Top - White
            {0, 0, 255},     // Right - Blue
            {255, 0, 0},     // Front - Red
            {255, 0, 255},   // Back - Magenta
            {0, 255, 0},     // Left - Green
            {255, 255, 0}    // Bottom - Yellow
        };
        for (int i = 0; i < MAX_FACES_OBJECT; i++)
        {
            gpuObj.col[index][i].red = faceColors[i][0];
            gpuObj.col[index][i].green = faceColors[i][1];
            gpuObj.col[index][i].blue = faceColors[i][2];
        }
    }
    else
    {
        // Single color for all faces
        gpuObj.col[index][0].red = colorR;
        gpuObj.col[index][0].green = colorG;
        gpuObj.col[index][0].blue = colorB;
        for (int i = 1; i < MAX_FACES_OBJECT; i++)
        {
            gpuObj.col[index][i].red = 0;
            gpuObj.col[index][i].green = 0;
            gpuObj.col[index][i].blue = 0;
        }
    }
}

/**
 * Convert entire scene from CPU vector to GPU object_to_gpu structure
 * @param shapes: Vector of CPU shape pointers
 * @param gpuObj: Output GPU object_to_gpu structure
 * @param randomColors: If true, assign random colors to objects
 * @return: Number of objects converted (limited by NB_OBJECT)
 */
inline int convertSceneToGPU(const std::vector<Shape *> &shapes, object_to_gpu &gpuObj,
                             bool randomColors = true)
{

    int numObjects = std::min((int)shapes.size(), NB_OBJECT);

    for (int i = 0; i < numObjects; i++)
    {
        // Generate colors
        unsigned char r, g, b;
        if (randomColors)
        {
            r = 100 + (i * 50) % 156;
            g = 100 + (i * 80) % 156;
            b = 100 + (i * 110) % 156;
        }
        else
        {
            r = 255;
            g = 255;
            b = 255;
        }

        // Check if object is a cube/prism for multicolor option
        bool multicolor = (shapes[i]->getType() == 1 || shapes[i]->getType() == 2);

        // Convert shape
        convertShapeToGPU(shapes[i], gpuObj, i, r, g, b, multicolor);
    }

    // Fill remaining slots with default values if needed
    for (int i = numObjects; i < NB_OBJECT; i++)
    {
        gpuObj.type[i] = SPHERE;
        gpuObj.pos[i] = {0.0f, 0.0f, 0.0f};
        gpuObj.rot[i] = {0.0f, 0.0f, 0.0f};
        gpuObj.nb_dim[i] = 1;
        gpuObj.dimension[i][0] = 0.0f;
        gpuObj.dimension[i][1] = 0.0f;
        gpuObj.dimension[i][2] = 0.0f;
        gpuObj.is_single_color[i] = true;
        gpuObj.col[i][0] = {0, 0, 0};
    }

    return numObjects;
}

/**
 * Update only position and rotation data (efficient for physics updates)
 * @param shapes: Vector of CPU shape pointers
 * @param gpuObj: GPU object_to_gpu structure to update
 * @param numObjects: Number of objects to update (must be <= shapes.size())
 */
inline void updateGPUPhysicsState(const std::vector<Shape *> &shapes,
                                  object_to_gpu &gpuObj,
                                  int numObjects = -1)
{

    if (numObjects < 0)
    {
        numObjects = std::min((int)shapes.size(), NB_OBJECT);
    }

#pragma omp parallel for
    for (int i = 0; i < numObjects; i++)
    {
        Point3D center = shapes[i]->getCenter();
        Point3D rotation;
        shapes[i]->getRotation(rotation);

        // Update position
        gpuObj.pos[i].x = static_cast<float>(center.x);
        gpuObj.pos[i].y = static_cast<float>(center.y);
        gpuObj.pos[i].z = static_cast<float>(center.z);

        // Update rotation
        gpuObj.rot[i].theta_x = static_cast<float>(rotation.x);
        gpuObj.rot[i].theta_y = static_cast<float>(rotation.y);
        gpuObj.rot[i].theta_z = static_cast<float>(rotation.z);
    }
}

/**
 * Update full object data (including dimensions and colors)
 * Use this if objects change shape or color during simulation
 */
inline void updateGPUFullState(const std::vector<Shape *> &shapes,
                               object_to_gpu &gpuObj,
                               int numObjects = -1)
{

    if (numObjects < 0)
    {
        numObjects = std::min((int)shapes.size(), NB_OBJECT);
    }

    for (int i = 0; i < numObjects; i++)
    {
        // Keep existing colors
        unsigned char r = gpuObj.col[i][0].red;
        unsigned char g = gpuObj.col[i][0].green;
        unsigned char b = gpuObj.col[i][0].blue;
        bool multicolor = !gpuObj.is_single_color[i];

        // Reconvert the entire object
        convertShapeToGPU(shapes[i], gpuObj, i, r, g, b, multicolor);
    }
}

/**
 * Create object_to_gpu structure with custom colors per object
 * @param shapes: Vector of CPU shape pointers
 * @param colors: Vector of {r, g, b} colors for each object
 * @param gpuObj: Output GPU object_to_gpu structure
 */
inline int convertSceneWithColors(const std::vector<Shape *> &shapes,
                                  const std::vector<std::array<unsigned char, 3>> &colors,
                                  object_to_gpu &gpuObj)
{

    int numObjects = std::min({(int)shapes.size(), (int)colors.size(), NB_OBJECT});

    for (int i = 0; i < numObjects; i++)
    {
        bool multicolor = (shapes[i]->getType() == 1 || shapes[i]->getType() == 2);
        convertShapeToGPU(shapes[i], gpuObj, i,
                          colors[i].at(0), colors[i].at(1), colors[i].at(2),
                          multicolor);
    }

    // Fill remaining slots
    for (int i = numObjects; i < NB_OBJECT; i++)
    {
        gpuObj.type[i] = SPHERE;
        gpuObj.pos[i] = {0.0f, 0.0f, 0.0f};
        gpuObj.rot[i] = {0.0f, 0.0f, 0.0f};
        gpuObj.nb_dim[i] = 1;
        for (int j = 0; j < MAX_DIMENSIONS_OBJECTS; j++)
        {
            gpuObj.dimension[i][j] = 0.0f;
        }
        gpuObj.is_single_color[i] = true;
        gpuObj.col[i][0] = {0, 0, 0};
    }

    return numObjects;
}

/**
 * Print GPU object data for debugging
 */
inline void printGPUObject(const object_to_gpu &gpuObj, int index)
{
    std::cout << "=== GPU Object " << index << " ===\n";
    std::cout << "Type: " << (int)gpuObj.type[index]
              << (gpuObj.type[index] == SPHERE ? " (SPHERE)" : " (CUBE)") << "\n";
    std::cout << "Position: (" << gpuObj.pos[index].x << ", "
              << gpuObj.pos[index].y << ", "
              << gpuObj.pos[index].z << ")\n";
    std::cout << "Rotation: (" << gpuObj.rot[index].theta_x << ", "
              << gpuObj.rot[index].theta_y << ", "
              << gpuObj.rot[index].theta_z << ")\n";
    std::cout << "Dimensions (" << gpuObj.nb_dim[index] << "): ";
    for (int i = 0; i < gpuObj.nb_dim[index]; i++)
    {
        std::cout << gpuObj.dimension[index][i] << " ";
    }
    std::cout << "\n";
    std::cout << "Single Color: " << (gpuObj.is_single_color[index] ? "Yes" : "No") << "\n";
    std::cout << "Color: RGB(" << (int)gpuObj.col[index][0].red << ", "
              << (int)gpuObj.col[index][0].green << ", "
              << (int)gpuObj.col[index][0].blue << ")\n";
}

/**
 * Print all GPU objects
 */
inline void printAllGPUObjects(const object_to_gpu &gpuObj, int numObjects = NB_OBJECT)
{
    std::cout << "\n=== GPU Scene: " << numObjects << " objects ===\n";
    for (int i = 0; i < numObjects; i++)
    {
        printGPUObject(gpuObj, i);
        std::cout << "\n";
    }
}

// ============================================================================
// USAGE EXAMPLE
// ============================================================================

/*
#include "cpu_gpu_converter.hpp"

double gravity = -9.81;

int main() {
    // Create CPU scene
    std::vector<Shape*> shapes;
    shapes.push_back(new Sphere(Point3D(50, 50, 10), 100, Point3D(0, 0, 0)));
    shapes.push_back(new Cube(Point3D(200, 200, 20), 200, Point3D(0, 0, 0),
                              Point3D(0, 0, 0), Point3D(0.1, 0.1, 0.1)));
    shapes.push_back(new RectangularPrism(Point3D(100, 100, 50), 400, 300, 200,
                                          Point3D(5, 5, 0), Point3D(0, 0, 0),
                                          Point3D(0, 0, 0)));

    // Create GPU object structure
    object_to_gpu gpuScene;

    // Initial conversion
    int numObjects = convertSceneToGPU(shapes, gpuScene, true);
    std::cout << "Converted " << numObjects << " objects\n";

    // Simulation loop
    double dt = 0.016; // 60 FPS
    for (int frame = 0; frame < 100; frame++) {
        // Update physics on CPU
        for (auto shape : shapes) {
            shape->update(dt);
        }

        // Update GPU data (only positions/rotations for efficiency)
        updateGPUPhysicsState(shapes, gpuScene, numObjects);

        // Pass gpuScene to your GPU rendering function
        // draw_image(gpuScene, image, ...);
    }

    // Cleanup
    deleteShapes(shapes);

    return 0;
}
*/

#endif // CPU_GPU_CONVERTER_HPP