//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//
#include "stddef.h"
#include <fstream>
#include <iostream>
#include <cstring>

//------------------------------------------------------------------------------------------//
// Global parameters definition
//------------------------------------------------------------------------------------------//
// Configurable DEBUG parameters
#define DEBUG_PERF true
#define DEBUG_VALUES true

// Images resolution for the GPU to render
#define IMAGE_RESOLUTION_WIDTH 500
#define IMAGE_RESOLUTION_HEIGHT 500

// Image real size in the space
#define IMAGE_WIDTH 2000.0
#define IMAGE_HEIGHT 2000.0

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
#define PIXEL_WIDTH_SIZE IMAGE_WIDTH/IMAGE_RESOLUTION_WIDTH
#define PIXEL_HEIGHT_SIZE IMAGE_HEIGHT/IMAGE_RESOLUTION_HEIGHT

// Maximum simulation time before end of program (ms)
#define MAX_SIMU_TIME 120.0

// Maximum number of rendered images before end of program
#define MAX_RENDERED_FRAMES 12

// Fixed time between 2 rendered frame/New position of objects (ms)
#define INTERVAL_TIME 10.0

// Correspondance between ID of objects and their types
#define CUBE 0
#define SPHERE 1

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
    int id;
    int side;
};


// Structure to gather all of the objects at a defined time with all of their characteristics (with only GPU related values)
struct object_to_gpu {
    unsigned char type[NB_OBJECT]; // Object type
    position pos[NB_OBJECT];
    rotation rot[NB_OBJECT];
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
    unsigned char red[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    unsigned char green[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    unsigned char blue[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    unsigned char alpha[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];  //Opacity of the color
};

// Small AI generated function to output an image from the image array (does not currently support alpha values)
bool save_as_bmp(const image_array& img, const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Erreur: impossible d'ouvrir le fichier " << filename << std::endl;
        return false;
    }
    
    int width = IMAGE_RESOLUTION_WIDTH;
    int height = IMAGE_RESOLUTION_HEIGHT;
    int padding = (4 - (width * 3) % 4) % 4;
    int fileSize = 54 + (width * 3 + padding) * height;
    
    // En-tête BMP
    unsigned char bmpFileHeader[14] = {
        'B', 'M',           // Signature
        0, 0, 0, 0,         // Taille du fichier
        0, 0, 0, 0,         // Réservé
        54, 0, 0, 0         // Offset vers les données
    };
    
    unsigned char bmpInfoHeader[40] = {
        40, 0, 0, 0,        // Taille de cet en-tête
        0, 0, 0, 0,         // Largeur
        0, 0, 0, 0,         // Hauteur
        1, 0,               // Nombre de plans
        24, 0,              // Bits par pixel
        0, 0, 0, 0,         // Compression (0 = aucune)
        0, 0, 0, 0,         // Taille de l'image
        0, 0, 0, 0,         // Résolution horizontale
        0, 0, 0, 0,         // Résolution verticale
        0, 0, 0, 0,         // Couleurs dans la palette
        0, 0, 0, 0          // Couleurs importantes
    };
    
    // Remplir les valeurs
    bmpFileHeader[2] = (unsigned char)(fileSize);
    bmpFileHeader[3] = (unsigned char)(fileSize >> 8);
    bmpFileHeader[4] = (unsigned char)(fileSize >> 16);
    bmpFileHeader[5] = (unsigned char)(fileSize >> 24);
    
    bmpInfoHeader[4] = (unsigned char)(width);
    bmpInfoHeader[5] = (unsigned char)(width >> 8);
    bmpInfoHeader[6] = (unsigned char)(width >> 16);
    bmpInfoHeader[7] = (unsigned char)(width >> 24);
    
    bmpInfoHeader[8] = (unsigned char)(height);
    bmpInfoHeader[9] = (unsigned char)(height >> 8);
    bmpInfoHeader[10] = (unsigned char)(height >> 16);
    bmpInfoHeader[11] = (unsigned char)(height >> 24);
    
    // Écrire les en-têtes
    file.write((char*)bmpFileHeader, 14);
    file.write((char*)bmpInfoHeader, 40);
    
    // Écrire les pixels (BGR, de bas en haut)
    unsigned char padding_bytes[3] = {0, 0, 0};
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            file.put(img.blue[i]);   // B
            file.put(img.green[i]);  // G
            file.put(img.red[i]);    // R
        }
        file.write((char*)padding_bytes, padding);
    }
    
    file.close();
    return true;
}