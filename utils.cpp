#include "utils.hpp"

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
    
    unsigned char bmpFileHeader[14] = {
        'B', 'M',   
        0, 0, 0, 0,       
        0, 0, 0, 0,       
        54, 0, 0, 0         
    };
    
    unsigned char bmpInfoHeader[40] = {
        40, 0, 0, 0,    
        0, 0, 0, 0,       
        0, 0, 0, 0,       
        1, 0,        
        24, 0,           
        0, 0, 0, 0,    
        0, 0, 0, 0,   
        0, 0, 0, 0,       
        0, 0, 0, 0,       
        0, 0, 0, 0,        
        0, 0, 0, 0          
    };
    
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
    

    file.write((char*)bmpFileHeader, 14);
    file.write((char*)bmpInfoHeader, 40);
    
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