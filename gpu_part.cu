#include "utils.hpp"

//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//
#include "cuda.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Library used for sleep function
#include <unistd.h>

#define RENDERED_FRAMES 100

//------------------------------------------------------------------------------------------//
// GPU Functions (Scene rendering)
//------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------//
// General Functions
//------------------------------------------------------------------------------------------//
float dist2D(float A_x, float A_y, float B_x, float B_y) {
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return sqrt(dx * dx + dy * dy);
}
float dist3D(position A, position B) {
    float dx = B.x - A.x;
    float dy = B.y - A.y;
    float dz = B.z - A.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

float squareDist2D(float A_x, float A_y, float B_x, float B_y) {
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return dx * dx + dy * dy;
}

void rotate2D_x(float& new_y, float& new_z, float old_y, float old_z, float theta) {
    new_y = old_y * cos(theta) - old_z * sin(theta);
    new_z = old_y * sin(theta) + old_z * cos(theta);
}

void rotate2D_y(float& new_x, float& new_z, float old_x, float old_z, float theta) {
    new_z = old_z * cos(theta) - old_x * sin(theta);
    new_x = old_z * sin(theta) + old_x * cos(theta);
}

void rotate2D_z(float& new_x, float& new_y, float old_x, float old_y, float theta) {
    new_x = old_x * cos(theta) - old_y * sin(theta);
    new_y = old_x * sin(theta) + old_y * cos(theta);
}

position rotate3D(position pos, rotation rot, position pos_center) {
    position new_pos;
    rotate2D_x(new_pos.y, new_pos.z, pos.y, pos.z, rot.theta_x);
    rotate2D_y(new_pos.x, new_pos.z, pos.x, new_pos.z, rot.theta_y);
    rotate2D_z(new_pos.x, new_pos.y, new_pos.x, new_pos.y, rot.theta_z);
    return new_pos;
}

bool belongs_2D_4side_convex_polygone(float x, float y, position A0, position A1, position A2, position A3, float D) {
    float det1 = (A1.x - A0.x) * (y - A0.y) - (A1.y - A0.y) * (x - A0.x);
    float det2 = (A2.x - A1.x) * (y - A1.y) - (A2.y - A1.y) * (x - A1.x);
    if (det1 * det2 <= 0) return false;
    float det3 = (A3.x - A2.x) * (y - A2.y) - (A3.y - A2.y) * (x - A2.x);
    if (det2 * det3 <= 0) return false;
    float det4 = (A0.x - A3.x) * (y - A3.y) - (A0.y - A3.y) * (x - A3.x);
    if (det3 * det4 <= 0) return false;
    else return true;
}

bool belongs_2D_4side_convex_polygone_with_sides(float x, float y, position A0, position A1, position A2, position A3, float D) {
    // Point is equal to one of the points of the polygon
    if (x == A0.x and y == A0.y) return true;
    if (x == A1.x and y == A1.y) return true;
    if (x == A2.x and y == A2.y) return true;
    if (x == A3.x and y == A3.y) return true;
    // Check if it is inside/on the sides
    float det1 = (A1.x - A0.x) * (y - A0.y) - (A1.y - A0.y) * (x - A0.x);
    float det2 = (A2.x - A1.x) * (y - A1.y) - (A2.y - A1.y) * (x - A1.x);
    float det3 = (A3.x - A2.x) * (y - A2.y) - (A3.y - A2.y) * (x - A2.x);
    float det4 = (A0.x - A3.x) * (y - A3.y) - (A0.y - A3.y) * (x - A3.x);
    // Point is on the sides
    if (det1 == 0.0) {
        if (det2 * det3 <= 0) return false;
        if (det3 * det4 <= 0) return false;
    } else if (det2 == 0.0) {
        if (det1 * det3 <= 0) return false;
        if (det3 * det4 <= 0) return false;
    } else if (det3 == 0.0) {
        if (det1 * det2 <= 0) return false;
        if (det2 * det4 <= 0) return false;
    } else if (det4 == 0.0) {
        if (det1 * det2 <= 0) return false;
        if (det2 * det3 <= 0) return false;
    // Point is inside
    } else {
        if (det1 * det2 < 0) return false;
        if (det2 * det3 < 0) return false;
        if (det3 * det4 < 0) return false;
    }
    return true;
}


int is_in_cube(float x, float y, position pos, rotation rot, float* dimensions) {
    // Compute positions of cube summits
    //    E----F
    //   /    /|
    //  A----B G
    //  |    |/
    //  D----C
    position A;
    A.x = -dimensions[0]/2.0;
    A.y = -dimensions[0]/2.0;
    A.z = -dimensions[0]/2.0;
    A = rotate3D(A, rot, pos);
    A.x += pos.x;
    A.y += pos.y;
    A.z += pos.z;
    position B;
    B.x = dimensions[0]/2.0;
    B.y = -dimensions[0]/2.0;
    B.z = -dimensions[0]/2.0;
    B = rotate3D(B, rot, pos);
    B.x += pos.x;
    B.y += pos.y;
    B.z += pos.z;
    position C;
    C.x = dimensions[0]/2.0;
    C.y = dimensions[0]/2.0;
    C.z = -dimensions[0]/2.0;
    C = rotate3D(C, rot, pos);
    C.x += pos.x;
    C.y += pos.y;
    C.z += pos.z;
    position D;
    D.x = -dimensions[0]/2.0;
    D.y = +dimensions[0]/2.0;
    D.z = -dimensions[0]/2.0;
    D = rotate3D(D, rot, pos);
    D.x += pos.x;
    D.y += pos.y;
    D.z += pos.z;
    position E;
    E.x = -dimensions[0]/2.0;
    E.y = -dimensions[0]/2.0;
    E.z = dimensions[0]/2.0;
    E = rotate3D(E, rot, pos);
    E.x += pos.x;
    E.y += pos.y;
    E.z += pos.z;
    position F;
    F.x = dimensions[0]/2.0;
    F.y = -dimensions[0]/2.0;
    F.z = dimensions[0]/2.0;
    F = rotate3D(F, rot, pos);
    F.x += pos.x;
    F.y += pos.y;
    F.z += pos.z;
    position G;
    G.x = dimensions[0]/2.0;
    G.y = dimensions[0]/2.0;
    G.z = dimensions[0]/2.0;
    G = rotate3D(G, rot, pos);
    G.x += pos.x;
    G.y += pos.y;
    G.z += pos.z;
    position H;
    H.x = -dimensions[0]/2.0;
    H.y = dimensions[0]/2.0;
    H.z = dimensions[0]/2.0;
    H = rotate3D(H, rot, pos);
    H.x += pos.x;
    H.y += pos.y;
    H.z += pos.z;
    // Compute belonging in front or back plan
    float front_z = (A.z + B.z + C.z + D.z)/4.0;
    float back_z = (E.z + F.z + G.z + H.z)/4.0;
    if (back_z > front_z) {
        // Front
        if (belongs_2D_4side_convex_polygone_with_sides(x, y, A, B, C, D, dimensions[0]) == true) {
            return 2;
        }
    } else {
        // Back
        if (belongs_2D_4side_convex_polygone_with_sides(x, y, E, F, G, H, dimensions[0]) == true) {
            return 3;
        }
    }
    // Compute belonging in top or bottom plan
    float top_z = (A.z + B.z + F.z + E.z)/4.0;
    float bottom_z = (D.z + C.z + G.z + H.z)/4.0;
    if (bottom_z > top_z) {
        // Top
        if (belongs_2D_4side_convex_polygone_with_sides(x, y, A, B, F, E, dimensions[0]) == true) {
            return 0;
        }
    } else {
        // Bottom
        if (belongs_2D_4side_convex_polygone_with_sides(x, y, D, C, G, H, dimensions[0]) == true) {
            return 5;
        }
    }
    //// Compute belonging in right or left plan
    float right_z = (B.z + F.z + G.z + C.z)/4.0;
    float left_z = (A.z + E.z + H.z + D.z)/4.0;
    if (left_z > right_z) {
        // Right
        if (belongs_2D_4side_convex_polygone_with_sides(x, y, B, F, G, C, dimensions[0]) == true) {
            return 1;
        }
    } else {
        // Left
        if (belongs_2D_4side_convex_polygone_with_sides(x, y, A, E, H, D, dimensions[0]) == true) {
            return 4;
        }
    }
    return -1;
}

bool is_in_sphere(float x, float y, position pos, float* dimensions) {
    return (squareDist2D(x, y, pos.x, pos.y) <= dimensions[0]*dimensions[0]);
}

int is_in_object(float x, float y, unsigned char id, position pos, rotation rot, float* dimensions) {
    bool side;
    if (id == CUBE) {
        return is_in_cube(x, y, pos, rot, dimensions);
    } else if (id == SPHERE) {
        side = is_in_sphere(x, y, pos, dimensions);
        if (side == true) return 0;
        else return -1;
    } else {
        return -1;
    }
}

void update_identifiers(object_to_gpu tab_pos, id_array* identifier_array) {
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        for(int width=0; width<IMAGE_RESOLUTION_WIDTH; width++) {
            float x = IMAGE_OFFSET_WIDTH + PIXEL_WIDTH_SIZE/2.0 + width * PIXEL_WIDTH_SIZE;
            float y = IMAGE_OFFSET_HEIGHT + PIXEL_HEIGHT_SIZE/2.0 + height * PIXEL_HEIGHT_SIZE;
            identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].id = -1;
            identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].side = -1;
            for(int i=0; i<NB_OBJECT; i++) {
                int is_in = is_in_object(x, y, tab_pos.type[i], tab_pos.pos[i], tab_pos.rot[i], tab_pos.dimension[i]);
                if (is_in != -1) {
                    if (identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].id != -1) {
                        if (tab_pos.pos[i].z < tab_pos.pos[identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].id].z) {
                            identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].id = i;
                            identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].side = is_in;
                        }
                    } else {
                        identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].id = i;
                        identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].side = is_in;
                    }
                }
            }
        }
    }
}

colors get_colors(int id, int side, object_to_gpu tab_pos) {
    colors col;
    col.red = 0;
    col.green = 0;
    col.blue = 0;
    if (id == -1) return col;
    else {
        if (tab_pos.is_single_color[id] == true) {
            col.red = tab_pos.col[id][0].red;
            col.green = tab_pos.col[id][0].green;
            col.blue = tab_pos.col[id][0].blue;
        } else {
            col.red = tab_pos.col[id][side].red;
            col.green = tab_pos.col[id][side].green;
            col.blue = tab_pos.col[id][side].blue;
        }
        return col;
    }
}

void update_image(id_array* identifier_array, object_to_gpu tab_pos, image_array& image) {
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        for(int width=0; width<IMAGE_RESOLUTION_WIDTH; width++) {
            colors col;
            col = get_colors(identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].id, identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].side, tab_pos);
            image.red[height*IMAGE_RESOLUTION_HEIGHT + width] = col.red;
            image.green[height*IMAGE_RESOLUTION_HEIGHT + width] = col.green;
            image.blue[height*IMAGE_RESOLUTION_HEIGHT + width] = col.blue;
            image.alpha[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
        }
    }
}

void print_identifier_array(id_array* identifier_array) {
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        printf(" ");
        for(int width=0; width<IMAGE_RESOLUTION_WIDTH; width++) {
            int carac = identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].id;
            int side = identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width].side;
            if (carac == -1 or carac>=10) {
                printf("%d", carac);
            } else {
                printf(" %d", carac);
            }
            if (side == -1 or side>=10) {
                printf(",%d | ", side);
            } else {
                printf(", %d | ", side);
            }
        }
        printf("\n");
    }
}

void reinit_terminal() {
    // Clear message from BMP image
    printf("\x1b[1F"); // Move to beginning of previous line
    printf("\x1b[2K"); // Clear entire line
    // Clear """image""" from identifiers
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        printf("\x1b[1F"); // Move to beginning of previous line
        printf("\x1b[2K"); // Clear entire line
    }
}

//------------------------------------------------------------------------------------------//
// Draw image function
//------------------------------------------------------------------------------------------//
void draw_image(object_to_gpu tab_pos, image_array& image) {
    // Array of 2D Object identifier (0 : index of object, 1 : hit side)
    id_array identifier_array[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];

    // Compute which object is visible (and which face can we see) for each pixel
    update_identifiers(tab_pos, identifier_array);

    // """Image preview"""
    if (DEBUG_VALUES and IMAGE_RESOLUTION_HEIGHT<32 and IMAGE_RESOLUTION_WIDTH<32) {
        print_identifier_array(identifier_array);
        reinit_terminal();
    }

    // Assigns colors to each pixel, simply based on which object is visible (no light computations)
    update_image(identifier_array, tab_pos, image);
}

//------------------------------------------------------------------------------------------//
// Main
//------------------------------------------------------------------------------------------//
int main (int argc, char** argv) {
    // Performance debug values
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::_V2::system_clock::time_point before_image_draw;
    double time_table[RENDERED_FRAMES];
    int index_min_time;
    double min_time = std::numeric_limits<double>::infinity();
    int index_max_time;
    double max_time = 0.0;
    double mean_time = 0.0;

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

    printf("Waiting to start\n");
    sleep(5);

    auto after_init = std::chrono::high_resolution_clock::now();
    if (DEBUG_PERF) {
        std::chrono::duration<double, std::milli> duration_after_init = after_init - start;
        printf("Execution time after initialisation: %f ms\n", duration_after_init.count());
    }
    printf("--------------Start Rendering---------------\n");
    for (int i=0; i<RENDERED_FRAMES; i++) {
        // Main function to use in order to draw an image from the set of objects
        if (DEBUG_PERF) {
            before_image_draw = std::chrono::high_resolution_clock::now();
        }
        draw_image(tab_pos, image);
        if (DEBUG_PERF) {
            auto after_image_draw = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> temp = after_image_draw - before_image_draw;
            time_table[i] = temp.count();
            if (i>0) {
                printf("\x1b[1F"); // Move to beginning of previous line
                printf("\x1b[2K"); // Clear entire line
            }
            printf("Image %d / %d took %f ms to render\n", i+1, RENDERED_FRAMES, time_table[i]);
            mean_time += time_table[i];
            if (min_time > time_table[i]) {
                min_time = time_table[i];
                index_min_time = i;
            }
            if (max_time < time_table[i]) {
                max_time = time_table[i];
                index_max_time = i;
            }
        }

        // Image output temporary function
        if (save_as_bmp(image, "test_image.bmp") == false) {
            printf("Image saving error, leaving loop\n");
            break;
        }

        // Temporary positions updates for testing rendering techniques
        tab_pos.pos[0].x += 40.0;
        tab_pos.pos[0].y += 40.0;
        tab_pos.pos[1].y += 40.0;
        tab_pos.pos[2].x += 40.0;
        tab_pos.pos[3].x += 10.0;
        tab_pos.pos[4].x += 5.0;
        tab_pos.pos[4].y += 5.0;
        tab_pos.rot[4].theta_x += 0.1;
        tab_pos.rot[4].theta_y += 0.1;
        tab_pos.rot[4].theta_z += 0.1;

        //usleep(100000);
    }
    printf("--------------End of Rendering--------------\n");

    // Output performance metrics
    printf("\n--------------Run Parameters Recap---------------\n");
    printf("Image resolution : %d * %d\n", IMAGE_RESOLUTION_WIDTH, IMAGE_RESOLUTION_HEIGHT);
    printf("Number of objects in simulation : %d\n", NB_OBJECT);
    printf("Number of rendered frames : %d\n", RENDERED_FRAMES);
    printf("-------------------------------------------------\n");

    auto end = std::chrono::high_resolution_clock::now();
    if (DEBUG_PERF) {
        std::chrono::duration<double, std::milli> duration_end = end - start;
        printf("\n--------------Performance Metrics---------------\n");
        printf("Total execution time: %f ms\n", duration_end.count());
        printf("Mean image drawing time : %f ms\n", mean_time/((float) RENDERED_FRAMES));
        printf("Maximum image drawing time (%d): %f\n", index_max_time, max_time);
        printf("Minimum image drawing time (%d): %f\n", index_min_time, min_time);
        printf("Mean FPS : %f\n", 1000.0 * ((float) RENDERED_FRAMES)/mean_time);
        printf("-------------------------------------------------\n");
    }
}