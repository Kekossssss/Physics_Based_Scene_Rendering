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

float squareDist2D(float A_x, float A_y, float B_x, float B_y) {
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return dx * dx + dy * dy;
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
    // Compute positions of object points (TODO: rotation)
    position A;
    A.x = pos.x - dimensions[0]/2.0;
    A.y = pos.y - dimensions[0]/2.0;
    A.z = pos.z - dimensions[0]/2.0;
    position B;
    B.x = pos.x + dimensions[0]/2.0;
    B.y = pos.y - dimensions[0]/2.0;
    B.z = pos.z - dimensions[0]/2.0;
    position C;
    C.x = pos.x + dimensions[0]/2.0;
    C.y = pos.y + dimensions[0]/2.0;
    C.z = pos.z - dimensions[0]/2.0;
    position D;
    D.x = pos.x - dimensions[0]/2.0;
    D.y = pos.y + dimensions[0]/2.0;
    D.z = pos.z - dimensions[0]/2.0;
    position E;
    E.x = pos.x - dimensions[0]/2.0;
    E.y = pos.y - dimensions[0]/2.0;
    E.z = pos.z + dimensions[0]/2.0;
    position F;
    F.x = pos.x + dimensions[0]/2.0;
    F.y = pos.y - dimensions[0]/2.0;
    F.z = pos.z + dimensions[0]/2.0;
    position G;
    G.x = pos.x + dimensions[0]/2.0;
    G.y = pos.y + dimensions[0]/2.0;
    G.z = pos.z + dimensions[0]/2.0;
    position H;
    H.x = pos.x - dimensions[0]/2.0;
    H.y = pos.y + dimensions[0]/2.0;
    H.z = pos.z + dimensions[0]/2.0;
    // Compute belonging in front plan
    if (belongs_2D_4side_convex_polygone_with_sides(x, y, A, B, C, D, dimensions[0]) == true) {
        return 2;
    }
    // Compute belonging in top plan
    if (belongs_2D_4side_convex_polygone_with_sides(x, y, A, B, F, E, dimensions[0]) == true) {
        return 0;
    }
    //// Compute belonging in bottom plan
    if (belongs_2D_4side_convex_polygone_with_sides(x, y, D, C, G, H, dimensions[0]) == true) {
        return 5;
    }
    //// Compute belonging in back plan
    if (belongs_2D_4side_convex_polygone_with_sides(x, y, E, F, G, H, dimensions[0]) == true) {
        return 3;
    }
    //// Compute belonging in left plan
    if (belongs_2D_4side_convex_polygone_with_sides(x, y, A, E, H, D, dimensions[0]) == true) {
        return 4;
    }
    //// Compute belonging in right plan
    if (belongs_2D_4side_convex_polygone_with_sides(x, y, B, F, G, C, dimensions[0]) == true) {
        return 1;
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
// Main
//------------------------------------------------------------------------------------------//
int main (int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    // Test objects positions
    object_to_gpu tab_pos;

    // First object (cube of side R=50.0)
    tab_pos.type[0] = 0;
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
    tab_pos.type[1] = 0;
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
    tab_pos.type[2] = 0;
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
    tab_pos.type[3] = 1;
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

    // Test image arrays
    image_array image;

    printf("Waiting to start\n");
    sleep(10);

    // Array of 2D Object identifier (0 : index of object, 1 : hit side)
    id_array identifier_array[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    for (int i=0; i<30; i++) {
        update_identifiers(tab_pos, identifier_array);

        // """Image preview"""
        if (i>0) {
            //reinit_terminal();
        }
        //print_identifier_array(identifier_array);

        // Update image colors
        update_image(identifier_array, tab_pos, image);

        if (save_as_bmp(image, "test_image.bmp")) {
            std::cout << "BMP Image " << i << " succesfully created !" << std::endl;
        }

        tab_pos.pos[0].x += 40.0;
        tab_pos.pos[0].y += 40.0;
        tab_pos.pos[1].y += 40.0;
        tab_pos.pos[2].x += 40.0;

        sleep(1);
    }    
}