//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//
#include "cuda.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "utils.hpp"


//------------------------------------------------------------------------------------------//
// GPU Functions (Scene rendering)
//------------------------------------------------------------------------------------------//
    bool is_in_square() {

    }

    bool is_in_object() {

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
    tab_pos.pos[0].x = 10.0;
    tab_pos.pos[0].y = 10.0;
    tab_pos.pos[0].z = 10.0;
    tab_pos.rot[0].theta_x = 0.0;
    tab_pos.rot[0].theta_y = 0.0;
    tab_pos.rot[0].theta_z = 0.0;
    tab_pos.dimension[0][0] = 5.0;

    // Second object (cube of side R=10.0)
    tab_pos.id[1][0] = 0;
    tab_pos.id[1][1] = 1;
    tab_pos.pos[1].x = 10.0;
    tab_pos.pos[1].y = 10.0;
    tab_pos.pos[1].z = 40.0;
    tab_pos.rot[1].theta_x = 0.0;
    tab_pos.rot[1].theta_y = 0.0;
    tab_pos.rot[1].theta_z = 0.0;
    tab_pos.dimension[1][0] = 10.0;

    // Third object (cube of side R=20.0)
    tab_pos.id[2][0] = 0;
    tab_pos.id[2][1] = 2;
    tab_pos.pos[2].x = 20.0;
    tab_pos.pos[2].y = 20.0;
    tab_pos.pos[2].z = 20.0;
    tab_pos.rot[2].theta_x = 0.0;
    tab_pos.rot[2].theta_y = 0.0;
    tab_pos.rot[2].theta_z = 0.0;
    tab_pos.dimension[2][0] = 20.0;

    // Test image arrays
    image_array image;

    // Array of 2D Object identifier
    int identifier_array[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        for(int width=0; width<IMAGE_RESOLUTION_WIDTH; width++) {
            float x = TOP_IMAGE_OFFSET_WIDTH + width * PIXEL_WIDTH_SIZE/2;
            float y = TOP_IMAGE_OFFSET_HEIGHT + height * PIXEL_HEIGHT_SIZE/2;
            identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] = -1;
            for(int i=0; i<3; i++) {
                if (tab_pos.pos[i].x - tab_pos.dimension[i][0]/2 < x and tab_pos.pos[i].x + tab_pos.dimension[i][0]/2 > x) {
                    if (tab_pos.pos[i].y - tab_pos.dimension[i][0]/2 < y and tab_pos.pos[i].y + tab_pos.dimension[i][0]/2 > y) {
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
            if (identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] == 0) {
                image.red[height*IMAGE_RESOLUTION_HEIGHT + width] = 255;
            } else {
                image.red[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
            }
            if (identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] == 1) {
                image.green[height*IMAGE_RESOLUTION_HEIGHT + width] = 255;
            } else {
                image.green[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
            }
            if (identifier_array[height*IMAGE_RESOLUTION_HEIGHT + width] == 2) {
                image.blue[height*IMAGE_RESOLUTION_HEIGHT + width] = 255;
            } else {
                image.blue[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
            }
            image.alpha[height*IMAGE_RESOLUTION_HEIGHT + width] = 0;
        }
    }

    // """Image preview"""
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

    if (save_as_bmp(image, "test_image.bmp")) {
        std::cout << "Image BMP créée avec succès!" << std::endl;
    }
}