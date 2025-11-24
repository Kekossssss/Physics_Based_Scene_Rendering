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
bool is_in_cube(position pos, rotation rot, unsigned char* dimensions) {

    return true;
}

bool is_in_sphere(position pos, rotation rot, unsigned char* dimensions) {

    return true;
}

bool is_in_object(unsigned char id, position pos, rotation rot, unsigned char* dimensions) {
    if (id == CUBE) {
        return is_in_cube(pos, rot, dimensions);
    } else if (id == SPHERE) {
        return is_in_sphere(pos, rot, dimensions);
    } else {
        return false;
    }
}

void update_identifiers(object_to_gpu tab_pos, int* identifier_array) {
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        for(int width=0; width<IMAGE_RESOLUTION_WIDTH; width++) {
            float x = IMAGE_OFFSET_WIDTH + PIXEL_WIDTH_SIZE/2 + width * PIXEL_WIDTH_SIZE;
            float y = IMAGE_OFFSET_HEIGHT + PIXEL_HEIGHT_SIZE/2 + height * PIXEL_HEIGHT_SIZE;
            identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] = -1;
            for(int i=0; i<3; i++) {
                if (((tab_pos.pos[i].x - tab_pos.dimension[i][0]/2) <= x) and ((tab_pos.pos[i].x + tab_pos.dimension[i][0]/2) >= x)) {
                    if (((tab_pos.pos[i].y - tab_pos.dimension[i][0]/2) <= y) and ((tab_pos.pos[i].y + tab_pos.dimension[i][0]/2) >= y)) {
                        if (identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] != -1) {
                            if (tab_pos.pos[i].z < tab_pos.pos[identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width]].z) {
                                identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] = i;
                            }
                        } else {
                            identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] = i;
                        }
                    }
                }
            }
        }
    }
}

void update_image(int* identifier_array, image_array& image) {
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        for(int width=0; width<IMAGE_RESOLUTION_WIDTH; width++) {
            image.red[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
            if (identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] == 0) {
                image.red[height*IMAGE_RESOLUTION_HEIGHT + width] = 255;
            } 
            image.green[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
            if (identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] == 1) {
                image.green[height*IMAGE_RESOLUTION_HEIGHT + width] = 255;
            }
            image.blue[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
            if (identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] == 2) {
                image.blue[height*IMAGE_RESOLUTION_HEIGHT + width] = 255;
            }
            image.alpha[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
        }
    }
}

void print_identifier_array(int* identifier_array) {
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        printf(" ");
        for(int width=0; width<IMAGE_RESOLUTION_WIDTH; width++) {
            int carac = identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width];
            if (carac == -1 or carac>=10) {
                printf("%d ", carac);
            } else {
                printf(" %d ", carac);
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

    // First object (cube of side R=5.0)
    tab_pos.id[0][0] = 0;
    tab_pos.id[0][1] = 0;
    tab_pos.pos[0].x = 50.0;
    tab_pos.pos[0].y = 50.0;
    tab_pos.pos[0].z = 10.0;
    tab_pos.rot[0].theta_x = 0.0;
    tab_pos.rot[0].theta_y = 0.0;
    tab_pos.rot[0].theta_z = 0.0;
    tab_pos.dimension[0][0] = 50.0;

    // Second object (cube of side R=10.0)
    tab_pos.id[1][0] = 0;
    tab_pos.id[1][1] = 1;
    tab_pos.pos[1].x = 50.0;
    tab_pos.pos[1].y = 50.0;
    tab_pos.pos[1].z = 40.0;
    tab_pos.rot[1].theta_x = 0.0;
    tab_pos.rot[1].theta_y = 0.0;
    tab_pos.rot[1].theta_z = 0.0;
    tab_pos.dimension[1][0] = 100.0;

    // Third object (cube of side R=20.0)
    tab_pos.id[2][0] = 0;
    tab_pos.id[2][1] = 2;
    tab_pos.pos[2].x = 200.0;
    tab_pos.pos[2].y = 200.0;
    tab_pos.pos[2].z = 20.0;
    tab_pos.rot[2].theta_x = 0.0;
    tab_pos.rot[2].theta_y = 0.0;
    tab_pos.rot[2].theta_z = 0.0;
    tab_pos.dimension[2][0] = 200.0;

    // Test image arrays
    image_array image;

    // Array of 2D Object identifier
    int identifier_array[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    for (int i=0; i<30; i++) {
        update_identifiers(tab_pos, identifier_array);

        // """Image preview"""
        if (i>0) {
            reinit_terminal();
        }
        print_identifier_array(identifier_array);

        // Update image colors
        update_image(identifier_array, image);

        if (save_as_bmp(image, "test_image.bmp")) {
            std::cout << "Image BMP " << i << " créée avec succès!" << std::endl;
        }

        tab_pos.pos[1].y += 40.0;
        tab_pos.pos[2].x += 40.0;

        sleep(1);
    }    
}