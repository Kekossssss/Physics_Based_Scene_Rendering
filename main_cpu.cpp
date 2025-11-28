//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//
// #include "cuda.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "utils.hpp"

#include "cpu_renderer.hpp"
// Library used for sleep function
#include <unistd.h>

#define RENDERED_FRAMES 100

//------------------------------------------------------------------------------------------//
// Convert CPU Objects to GPU Objects
//------------------------------------------------------------------------------------------//

void cpu_objects_to_gpu_objects(std::vector<Shape*>& objects, const object_to_gpu& tab_pos) {
    int n = objects.size();

    for (int i = 0;i < n;i++) {
        switch (tab_pos.type[i]) {
            case CUBE:
                tab_pos.pos[i].x = objects[i].massCenter.x;
                tab_pos.pos[i].y = objects[i].massCenter.y;
                tab_pos.pos[i].z = objects[i].massCenter.z;
                tab_pos.rot[i].theta_x = objects[i].angle.x;
                tab_pos.rot[i].theta_y = objects[i].angle.y;
                tab_pos.rot[i].theta_z = objects[i].angle.z;

            case SPHERE:
                tab_pos.pos[i].x = objects[i].massCenter.x;
                tab_pos.pos[i].y = objects[i].massCenter.y;
                tab_pos.pos[i].z = objects[i].massCenter.z;

            default:
                throw std::runtime_error("Error: Object type not recognized");
        }
    }
}

//------------------------------------------------------------------------------------------//
// GPU Librairies
//------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------//
// GPU Functions (Scene rendering)
//------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------//
// Main
//------------------------------------------------------------------------------------------//
int main(int argc, char **argv)
{
    // Performance debug values
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::_V2::system_clock::time_point before_image_draw;
    double time_table[RENDERED_FRAMES];
    int index_min_time;
    double min_time = std::numeric_limits<double>::infinity();
    int index_max_time;
    double max_time = 0.0;
    double mean_time = 0.0;

    Shape* s1 = new Cube({50.0,50.0,10.0}, 50.0, {0,0,0}, {0,0,0}, {0,0,0});
    Shape* s2 = new Cube({50.0,50.0,40.0}, 100.0, {0,0,0}, {0,0,0}, {0,0,0});
    Shape* s3 = new Cube({200.0,200.0,20.0}, 200.0, {0,0,0}, {0,0,0}, {0,0,0});
    Shape* s4 = new Sphere({200.0,200.0,10.0}, 150.0, {0,0,0});
    Shape* s5 = new Cube({100.0,100.0,50.0}, 400.0, {0,0,0}, {0,0,0}, {0,0,0});

    std::vector<Shape*> objects = {s1, s2, s3, s4, s5};

    // Test objects positions
    object_to_gpu tab_pos;

    // First object (cube of side R=50.0)
    tab_pos.type[0] = CUBE;
    tab_pos.pos[0].x = 50.0;
    tab_pos.pos[0].y = 50.0;
    tab_pos.pos[0].z = 10.0;
    tab_pos.rot[0].theta_x = 0.0;
    tab_pos.rot[0].theta_y = 0.0;
    tab_pos.rot[0].theta_z = 0.0;
    tab_pos.dimension[0][0] = 50.0;
    tab_pos.is_single_color[0] = true;
    tab_pos.col[0][0].red = 255;
    tab_pos.col[0][0].blue = 255;
    tab_pos.col[0][0].green = 255;

    // Second object (cube of side R=100.0)
    tab_pos.type[1] = CUBE;
    tab_pos.pos[1].x = 50.0;
    tab_pos.pos[1].y = 50.0;
    tab_pos.pos[1].z = 40.0;
    tab_pos.rot[1].theta_x = 0.0;
    tab_pos.rot[1].theta_y = 0.0;
    tab_pos.rot[1].theta_z = 0.0;
    tab_pos.dimension[1][0] = 100.0;
    tab_pos.is_single_color[1] = true;
    tab_pos.col[1][0].red = 255;
    tab_pos.col[1][0].blue = 0;
    tab_pos.col[1][0].green = 0;

    // Third object (cube of side R=200.0)
    tab_pos.type[2] = CUBE;
    tab_pos.pos[2].x = 200.0;
    tab_pos.pos[2].y = 200.0;
    tab_pos.pos[2].z = 20.0;
    tab_pos.rot[2].theta_x = 0.0;
    tab_pos.rot[2].theta_y = 0.0;
    tab_pos.rot[2].theta_z = 0.0;
    tab_pos.dimension[2][0] = 200.0;
    tab_pos.is_single_color[2] = true;
    tab_pos.col[2][0].red = 0;
    tab_pos.col[2][0].blue = 255;
    tab_pos.col[2][0].green = 0;

    // Fourth object (sphere of side R=150.0)
    tab_pos.type[3] = SPHERE;
    tab_pos.pos[3].x = 200.0;
    tab_pos.pos[3].y = 200.0;
    tab_pos.pos[3].z = 10.0;
    tab_pos.rot[3].theta_x = 0.0;
    tab_pos.rot[3].theta_y = 0.0;
    tab_pos.rot[3].theta_z = 0.0;
    tab_pos.dimension[3][0] = 150.0;
    tab_pos.is_single_color[3] = true;
    tab_pos.col[3][0].red = 0;
    tab_pos.col[3][0].blue = 0;
    tab_pos.col[3][0].green = 255;

    // Fifth object (cube of side R=400.0)
    tab_pos.type[4] = CUBE;
    tab_pos.pos[4].x = 100.0;
    tab_pos.pos[4].y = 100.0;
    tab_pos.pos[4].z = 50.0;
    tab_pos.rot[4].theta_x = 0.0;
    tab_pos.rot[4].theta_y = 0.0;
    tab_pos.rot[4].theta_z = 0.0;
    tab_pos.dimension[4][0] = 400.0;
    tab_pos.is_single_color[4] = false;
    //// Top
    tab_pos.col[4][0].red = 255;
    tab_pos.col[4][0].blue = 255;
    tab_pos.col[4][0].green = 255;
    //// Right
    tab_pos.col[4][1].red = 0;
    tab_pos.col[4][1].blue = 0;
    tab_pos.col[4][1].green = 255;
    //// Front
    tab_pos.col[4][2].red = 255;
    tab_pos.col[4][2].blue = 0;
    tab_pos.col[4][2].green = 0;
    //// Back
    tab_pos.col[4][3].red = 255;
    tab_pos.col[4][3].blue = 0;
    tab_pos.col[4][3].green = 255;
    //// Left
    tab_pos.col[4][4].red = 0;
    tab_pos.col[4][4].blue = 255;
    tab_pos.col[4][4].green = 0;
    //// Bottom
    tab_pos.col[4][5].red = 255;
    tab_pos.col[4][5].blue = 255;
    tab_pos.col[4][5].green = 255;

    // Test image arrays
    image_array image;
    image.red = new unsigned char[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];
    image.green = new unsigned char[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];
    image.blue = new unsigned char[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];
    image.alpha = new unsigned char[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];

    printf("Waiting to start\n");
    sleep(5);

    auto after_init = std::chrono::high_resolution_clock::now();
    if (DEBUG_PERF)
    {
        std::chrono::duration<double, std::milli> duration_after_init = after_init - start;
        printf("Execution time after initialisation: %f ms\n", duration_after_init.count());
    }
    printf("--------------Start Rendering---------------\n");
    for (int i = 0; i < RENDERED_FRAMES; i++)
    {
        // Main function to use in order to draw an image from the set of objects
        if (DEBUG_PERF)
        {
            before_image_draw = std::chrono::high_resolution_clock::now();
        }
        draw_image(tab_pos, image);
        if (DEBUG_PERF)
        {
            auto after_image_draw = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> temp = after_image_draw - before_image_draw;
            time_table[i] = temp.count();
            if (i > 0)
            {
                printf("\x1b[1F"); // Move to beginning of previous line
                printf("\x1b[2K"); // Clear entire line
            }
            printf("Image %d / %d took %f ms to render\n", i + 1, RENDERED_FRAMES, time_table[i]);
            mean_time += time_table[i];
            if (min_time > time_table[i])
            {
                min_time = time_table[i];
                index_min_time = i;
            }
            if (max_time < time_table[i])
            {
                max_time = time_table[i];
                index_max_time = i;
            }
        }

        // Image output temporary function
        if (save_as_bmp(image, "test_image_cpu.bmp") == false)
        {
            printf("Image saving error, leaving loop\n");
            break;
        }

        // Temporary positions updates for testing rendering techniques
        for (auto obj : objects) {
            obj->update(dt);
            obj->printPosition();
        }

        // Exemple collision simple pour RigidBody
        RigidBody* r1 = dynamic_cast<RigidBody*>(s1);
        RigidBody* r2 = dynamic_cast<RigidBody*>(s2);
        if(r1 && r2 && checkRigidNoAngleCollision(*r1, *r2)) {
            cout << "Collision Rigid Bodies!\n";
        }

        cout << "----------\n";

        cpu_objects_to_gpu_objects(tab_pos, objects);

        // usleep(100000);
    }
    printf("--------------End of Rendering--------------\n");

    // Output performance metrics
    printf("\n--------------Run Parameters Recap---------------\n");
    printf("Image resolution : %d * %d\n", IMAGE_RESOLUTION_WIDTH, IMAGE_RESOLUTION_HEIGHT);
    printf("Number of pixels to compute : %d\n", RESOLUTION);
    printf("Number of objects in simulation : %d\n", NB_OBJECT);
    printf("Number of rendered frames : %d\n", RENDERED_FRAMES);
    printf("-------------------------------------------------\n");

    auto end = std::chrono::high_resolution_clock::now();
    if (DEBUG_PERF)
    {
        std::chrono::duration<double, std::milli> duration_end = end - start;
        printf("\n--------------Performance Metrics---------------\n");
        printf("Total execution time: %f ms\n", duration_end.count());
        printf("Mean image drawing time : %f ms\n", mean_time / ((float)RENDERED_FRAMES));
        printf("Maximum image drawing time (%d): %f\n", index_max_time, max_time);
        printf("Minimum image drawing time (%d): %f\n", index_min_time, min_time);
        printf("Mean FPS : %f\n", 1000.0 * ((float)RENDERED_FRAMES) / mean_time);
        printf("-------------------------------------------------\n");
    }
    return 0;
}