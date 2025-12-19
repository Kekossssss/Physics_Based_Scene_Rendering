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
void convertShapeToGPU(const Shape *shape, object_to_gpu &gpuObj, int index,
                              unsigned char colorR = 255,
                              unsigned char colorG = 255,
                              unsigned char colorB = 255,
                              bool multicolor = false);

/**
 * Convert entire scene from CPU vector to GPU object_to_gpu structure
 * @param shapes: Vector of CPU shape pointers
 * @param gpuObj: Output GPU object_to_gpu structure
 * @param randomColors: If true, assign random colors to objects
 * @return: Number of objects converted (limited by NB_OBJECT)
 */
int convertSceneToGPU(const std::vector<Shape *> &shapes, object_to_gpu &gpuObj,
                             bool randomColors = true);

/**
 * Update only position and rotation data (efficient for physics updates)
 * @param shapes: Vector of CPU shape pointers
 * @param gpuObj: GPU object_to_gpu structure to update
 * @param numObjects: Number of objects to update (must be <= shapes.size())
 */
void updateGPUPhysicsState(const std::vector<Shape *> &shapes,
                                  object_to_gpu &gpuObj,
                                  int numObjects = -1);

/**
 * Update full object data (including dimensions and colors)
 * Use this if objects change shape or color during simulation
 */
void updateGPUFullState(const std::vector<Shape *> &shapes,
                               object_to_gpu &gpuObj,
                               int numObjects = -1);

/**
 * Create object_to_gpu structure with custom colors per object
 * @param shapes: Vector of CPU shape pointers
 * @param colors: Vector of {r, g, b} colors for each object
 * @param gpuObj: Output GPU object_to_gpu structure
 */
int convertSceneWithColors(const std::vector<Shape *> &shapes,
                                  const std::vector<std::array<unsigned char, 3>> &colors,
                                  object_to_gpu &gpuObj);

/**
 * Print GPU object data for debugging
 */
void printGPUObject(const object_to_gpu &gpuObj, int index);

/**
 * Print all GPU objects
 */
void printAllGPUObjects(const object_to_gpu &gpuObj, int numObjects = NB_OBJECT);

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