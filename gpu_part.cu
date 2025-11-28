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
// CPU Functions (Video Memory Management)
//------------------------------------------------------------------------------------------//
void initiate_video_memory(id_array& gpu_id_array, image_array& gpu_image, gpu_object_pointers& gpu_obj_pointers) {
    // Allocate references to variables in GPU memory
    //// GPU only id_array variable
    if (cudaMalloc(&gpu_id_array.id, sizeof(int) * RESOLUTION)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_id_array.side, sizeof(int) * RESOLUTION)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    //// GPU Image variable
    if (cudaMalloc(&gpu_image.red, sizeof(unsigned char) * RESOLUTION)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_image.green, sizeof(unsigned char) * RESOLUTION)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_image.blue, sizeof(unsigned char) * RESOLUTION)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_image.alpha, sizeof(unsigned char) * RESOLUTION)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    //// GPU objects variable allocation
    if (cudaMalloc(&gpu_obj_pointers.type, sizeof(unsigned char) * NB_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.pos_x, sizeof(float) * NB_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.pos_y, sizeof(float) * NB_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.pos_z, sizeof(float) * NB_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.rot_x, sizeof(float) * NB_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.rot_y, sizeof(float) * NB_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.rot_z, sizeof(float) * NB_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.dimension, sizeof(float) * NB_OBJECT * MAX_DIMENSIONS_OBJECTS)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.red, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.green, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.blue, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    if (cudaMalloc(&gpu_obj_pointers.is_single_color, sizeof(bool) * NB_OBJECT)!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
}

void copy_initial_data_to_video_memory(gpu_object_pointers& gpu_obj_pointers, object_to_gpu& obj) {
    // Copy data that doesn't need to be refetch after initialisation
    unsigned char col_r[NB_OBJECT * MAX_FACES_OBJECT];
    unsigned char col_g[NB_OBJECT * MAX_FACES_OBJECT];
    unsigned char col_b[NB_OBJECT * MAX_FACES_OBJECT];
    float dimension[NB_OBJECT * MAX_DIMENSIONS_OBJECTS];
    for (int i=0; i<NB_OBJECT; i++) {
        if (obj.is_single_color[i] == true) {
            col_r[i*MAX_FACES_OBJECT] = obj.col[i][0].red;
            col_g[i*MAX_FACES_OBJECT] = obj.col[i][0].green;
            col_b[i*MAX_FACES_OBJECT] = obj.col[i][0].blue;
            for (int j=1; j<MAX_FACES_OBJECT; j++) {
                col_r[i*MAX_FACES_OBJECT+j] = 0;
                col_g[i*MAX_FACES_OBJECT+j] = 0;
                col_b[i*MAX_FACES_OBJECT+j] = 0;
            }
        } else {
            for (int j=0; j<MAX_FACES_OBJECT; j++) {
                col_r[i*MAX_FACES_OBJECT+j] = obj.col[i][j].red;
                col_g[i*MAX_FACES_OBJECT+j] = obj.col[i][j].green;
                col_b[i*MAX_FACES_OBJECT+j] = obj.col[i][j].blue;
            }
        }
        for (int j=0; j<MAX_DIMENSIONS_OBJECTS; j++) {
            if (obj.nb_dim[i] > j) {
                dimension[i*MAX_DIMENSIONS_OBJECTS+j] = obj.dimension[i][j];
            } else {
                dimension[i*MAX_DIMENSIONS_OBJECTS+j] = 0.0;
            }
        }
    }
    // Copy data that doesn't need to be refetch after initialisation
    if (cudaMemcpy(gpu_obj_pointers.type, obj.type, sizeof(unsigned char) * NB_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.dimension, dimension, sizeof(float) * NB_OBJECT * MAX_DIMENSIONS_OBJECTS, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.red, col_r, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.green, col_g, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.blue, col_b, sizeof(unsigned char) * NB_OBJECT * MAX_FACES_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.is_single_color, obj.is_single_color, sizeof(bool) * NB_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
}

void copy_data_to_video_memory(gpu_object_pointers& gpu_obj_pointers, object_to_gpu& obj) {
    // Puts positional and rotation data in adequate arrays for GPU
    float pos_x[NB_OBJECT];
    float pos_y[NB_OBJECT];
    float pos_z[NB_OBJECT];
    float rot_x[NB_OBJECT];
    float rot_y[NB_OBJECT];
    float rot_z[NB_OBJECT];
    for (int i=0; i<NB_OBJECT; i++) {
        pos_x[i] = obj.pos[i].x;
        pos_y[i] = obj.pos[i].y;
        pos_z[i] = obj.pos[i].z;
        rot_x[i] = obj.rot[i].theta_x;
        rot_y[i] = obj.rot[i].theta_y;
        rot_z[i] = obj.rot[i].theta_z;
    }
    // Copy positional data which has been computed by the CPU
    if (cudaMemcpy(gpu_obj_pointers.pos_x, pos_x, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.pos_y, pos_y, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.pos_z, pos_z, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.rot_x, rot_x, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.rot_y, rot_y, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
    if (cudaMemcpy(gpu_obj_pointers.rot_z, rot_z, sizeof(float) * NB_OBJECT, ::cudaMemcpyHostToDevice)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to device)\n");
    }
}

void copy_data_from_video_memory(image_array& gpu_image, image_array& img) {
    // Copy image data from video memory
    if (cudaMemcpy(img.red, gpu_image.red, sizeof(unsigned char) * RESOLUTION, ::cudaMemcpyDeviceToHost)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to host)\n");
    }
    if (cudaMemcpy(img.green, gpu_image.green, sizeof(unsigned char) * RESOLUTION, ::cudaMemcpyDeviceToHost)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to host)\n");
    }
    if (cudaMemcpy(img.blue, gpu_image.blue, sizeof(unsigned char) * RESOLUTION, ::cudaMemcpyDeviceToHost)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to host)\n");
    }
    if (cudaMemcpy(img.alpha, gpu_image.alpha, sizeof(unsigned char) * RESOLUTION, ::cudaMemcpyDeviceToHost)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to host)\n");
    }
}

void clean_video_memory(id_array& gpu_id_array, image_array& gpu_image, gpu_object_pointers& gpu_obj_pointers) {
    // Allocate references to variables in GPU memory
    //// GPU only id_array variable
    if (cudaFree(gpu_id_array.id)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_id_array.side)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    //// GPU Image variable
    if (cudaFree(gpu_image.red)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_image.green)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_image.blue)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_image.alpha)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    //// GPU objects variable allocation
    if (cudaFree(gpu_obj_pointers.type)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.pos_x)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.pos_y)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.pos_z)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.rot_x)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.rot_y)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.rot_z)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.dimension)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.red)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.green)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.blue)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    if (cudaFree(gpu_obj_pointers.is_single_color)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
}

//------------------------------------------------------------------------------------------//
// CPU Functions (Block/Thread allocation management)
//------------------------------------------------------------------------------------------//
__global__ void get_warpSize(int* size) {
    *size = warpSize;
}

bool allocate_gpu_thread(dim3& numBlocks, dim3& threadsPerBlock) {
    // Getting Warp Size
    int* warp_size_device;
    int warp_size;
    if (cudaMalloc(&warp_size_device, sizeof(int))!=cudaSuccess) {
        printf("Cuda Malloc Failed\n");
    }
    get_warpSize<<<1, 1>>>(warp_size_device);
    if (cudaMemcpy(&warp_size, warp_size_device, sizeof(int), ::cudaMemcpyDeviceToHost)!=cudaSuccess) {
        printf("Cuda Memcpy failed (to host)\n");
    }
    if (cudaFree(warp_size_device)!=cudaSuccess) {
        printf("Cuda Free Failed\n");
    }
    // Compute thread allocation
    if (RESOLUTION <= 1024) {
        numBlocks.x = 1;
        numBlocks.y = 1;
        threadsPerBlock.x = IMAGE_RESOLUTION_HEIGHT;
        threadsPerBlock.y = IMAGE_RESOLUTION_WIDTH;
    } else {
        if ((IMAGE_RESOLUTION_HEIGHT%warp_size == 0) and (IMAGE_RESOLUTION_WIDTH%warp_size == 0) and (warp_size<=32)) {
            numBlocks.x = IMAGE_RESOLUTION_HEIGHT/warp_size;
            numBlocks.y = IMAGE_RESOLUTION_WIDTH/warp_size;
            threadsPerBlock.x = warp_size;
            threadsPerBlock.y = warp_size;
        } else if ((IMAGE_RESOLUTION_HEIGHT%(warp_size/2) == 0) and (IMAGE_RESOLUTION_WIDTH%(warp_size/2) == 0) and (warp_size<=64)) {
            numBlocks.x = IMAGE_RESOLUTION_HEIGHT/(warp_size/2);
            numBlocks.y = IMAGE_RESOLUTION_WIDTH/(warp_size/2);
            threadsPerBlock.x = (warp_size/2);
            threadsPerBlock.y = (warp_size/2);
        } else if ((IMAGE_RESOLUTION_HEIGHT%(warp_size/4) == 0) and (IMAGE_RESOLUTION_WIDTH%(2*warp_size) == 0) and (warp_size<=64)) {
            numBlocks.x = IMAGE_RESOLUTION_HEIGHT/(warp_size/4);
            numBlocks.y = IMAGE_RESOLUTION_WIDTH/(2*warp_size);
            threadsPerBlock.x = (warp_size/4);
            threadsPerBlock.y = (2*warp_size);
        } else {
            printf("Resolutions that are not a multiple of %d/%d are not supported yet\n", warp_size, 2*warp_size);
            return 1;
        }
    }
    numBlocks.z = 1;
    numBlocks.z = 1;
    return 0;
}

//------------------------------------------------------------------------------------------//
// GPU Functions (General Geometry functions)
//------------------------------------------------------------------------------------------//
__device__ float dist2D(float A_x, float A_y, float B_x, float B_y) {
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return sqrt(dx * dx + dy * dy);
}

__device__ float dist3D(float A_x, float A_y, float A_z, float B_x, float B_y, float B_z) {
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    float dz = B_z - A_z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__device__ float squareDist2D(float A_x, float A_y, float B_x, float B_y) {
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return dx * dx + dy * dy;
}

__device__ void rotate2D_x(float& new_y, float& new_z, float old_y, float old_z, float theta) {
    new_y = old_y * cos(theta) - old_z * sin(theta);
    new_z = old_y * sin(theta) + old_z * cos(theta);
}

__device__ void rotate2D_y(float& new_x, float& new_z, float old_x, float old_z, float theta) {
    new_z = old_z * cos(theta) - old_x * sin(theta);
    new_x = old_z * sin(theta) + old_x * cos(theta);
}

__device__ void rotate2D_z(float& new_x, float& new_y, float old_x, float old_y, float theta) {
    new_x = old_x * cos(theta) - old_y * sin(theta);
    new_y = old_x * sin(theta) + old_y * cos(theta);
}

__device__ void rotate3D(position& new_pos, position pos
                       , float theta_x, float theta_y, float theta_z) {
    rotate2D_x(new_pos.y, new_pos.z, pos.y, pos.z, theta_x);
    rotate2D_y(new_pos.x, new_pos.z, pos.x, new_pos.z, theta_y);
    rotate2D_z(new_pos.x, new_pos.y, new_pos.x, new_pos.y, theta_z);
}

__device__ bool belongs_2D_4side_convex_polygone(float x, float y, position A0, position A1, position A2, position A3, float D) {
    bool result = true;
    float det1 = (A1.x - A0.x) * (y - A0.y) - (A1.y - A0.y) * (x - A0.x);
    float det2 = (A2.x - A1.x) * (y - A1.y) - (A2.y - A1.y) * (x - A1.x);
    if (det1 * det2 <= 0) result = false;
    float det3 = (A3.x - A2.x) * (y - A2.y) - (A3.y - A2.y) * (x - A2.x);
    if (det2 * det3 <= 0) result = false;
    float det4 = (A0.x - A3.x) * (y - A3.y) - (A0.y - A3.y) * (x - A3.x);
    if (det3 * det4 <= 0) result = false;
    return result;
}

__device__ bool belongs_2D_4side_convex_polygone_with_sides(float x, float y, position A0, position A1, position A2, position A3, float D) {
    bool result_bypass;
    bool result = true;
    // Point is equal to one of the points of the polygon
    result_bypass = (x == A0.x and y == A0.y);
    result_bypass = result_bypass or (x == A1.x and y == A1.y);
    result_bypass = result_bypass or (x == A2.x and y == A2.y);
    result_bypass = result_bypass or (x == A3.x and y == A3.y);
    // Check if it is inside/on the sides
    float det1 = (A1.x - A0.x) * (y - A0.y) - (A1.y - A0.y) * (x - A0.x);
    float det2 = (A2.x - A1.x) * (y - A1.y) - (A2.y - A1.y) * (x - A1.x);
    float det3 = (A3.x - A2.x) * (y - A2.y) - (A3.y - A2.y) * (x - A2.x);
    float det4 = (A0.x - A3.x) * (y - A3.y) - (A0.y - A3.y) * (x - A3.x);
    // Point is on the sides
    if (det1 == 0.0) {
        if (det2 * det3 <= 0) result = false;
        if (det3 * det4 <= 0) result = false;
    } else if (det2 == 0.0) {
        if (det1 * det3 <= 0) result = false;
        if (det3 * det4 <= 0) result = false;
    } else if (det3 == 0.0) {
        if (det1 * det2 <= 0) result = false;
        if (det2 * det4 <= 0) result = false;
    } else if (det4 == 0.0) {
        if (det1 * det2 <= 0) result = false;
        if (det2 * det3 <= 0) result = false;
    // Point is inside
    } else {
        if (det1 * det2 < 0) result = false;
        if (det2 * det3 < 0) result = false;
        if (det3 * det4 < 0) result = false;
    }
    return (result_bypass or result);
}

//------------------------------------------------------------------------------------------//
// GPU Functions (Object detection)
//------------------------------------------------------------------------------------------//
__device__ int is_in_cube(float x, float y
                        , float pos_x, float pos_y, float pos_z
                        , float rot_x, float rot_y, float rot_z
                        , float* dimensions) {
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
    rotate3D(A, A, rot_x, rot_y, rot_z);
    A.x += pos_x;
    A.y += pos_y;
    A.z += pos_z;
    position B;
    B.x = dimensions[0]/2.0;
    B.y = -dimensions[0]/2.0;
    B.z = -dimensions[0]/2.0;
    rotate3D(B, B, rot_x, rot_y, rot_z);
    B.x += pos_x;
    B.y += pos_y;
    B.z += pos_z;
    position C;
    C.x = dimensions[0]/2.0;
    C.y = dimensions[0]/2.0;
    C.z = -dimensions[0]/2.0;
    rotate3D(C, C, rot_x, rot_y, rot_z);
    C.x += pos_x;
    C.y += pos_y;
    C.z += pos_z;
    position D;
    D.x = -dimensions[0]/2.0;
    D.y = +dimensions[0]/2.0;
    D.z = -dimensions[0]/2.0;
    rotate3D(D, D, rot_x, rot_y, rot_z);
    D.x += pos_x;
    D.y += pos_y;
    D.z += pos_z;
    position E;
    E.x = -dimensions[0]/2.0;
    E.y = -dimensions[0]/2.0;
    E.z = dimensions[0]/2.0;
    rotate3D(E, E, rot_x, rot_y, rot_z);
    E.x += pos_x;
    E.y += pos_y;
    E.z += pos_z;
    position F;
    F.x = dimensions[0]/2.0;
    F.y = -dimensions[0]/2.0;
    F.z = dimensions[0]/2.0;
    rotate3D(F, F, rot_x, rot_y, rot_z);
    F.x += pos_x;
    F.y += pos_y;
    F.z += pos_z;
    position G;
    G.x = dimensions[0]/2.0;
    G.y = dimensions[0]/2.0;
    G.z = dimensions[0]/2.0;
    rotate3D(G, G, rot_x, rot_y, rot_z);
    G.x += pos_x;
    G.y += pos_y;
    G.z += pos_z;
    position H;
    H.x = -dimensions[0]/2.0;
    H.y = dimensions[0]/2.0;
    H.z = dimensions[0]/2.0;
    rotate3D(H, H, rot_x, rot_y, rot_z);
    H.x += pos_x;
    H.y += pos_y;
    H.z += pos_z;
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

__device__ bool is_in_sphere(float x, float y, float pos_x, float pos_y, float pos_z, float dimensions) {
    return (squareDist2D(x, y, pos_x, pos_y) <= dimensions*dimensions);
}

__device__ int is_in_object(float x, float y, unsigned char id
                          , float pos_x, float pos_y, float pos_z
                          , float rot_x, float rot_y, float rot_z
                          , float* dimensions) {
    bool side;
    if (id == CUBE) {
        return is_in_cube(x, y, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, dimensions);
    } else if (id == SPHERE) {
        side = is_in_sphere(x, y, pos_x, pos_y, pos_z, dimensions[0]);
        if (side == true) return 0;
        else return -1;
    } else {
        return -1;
    }
}

//------------------------------------------------------------------------------------------//
// GPU Functions (Color related)
//------------------------------------------------------------------------------------------//
__device__ colors get_colors(int id, int side, gpu_object_pointers& gpu_obj_pointers) {
    colors col;
    int single = (gpu_obj_pointers.is_single_color[id] == false) ? side: 0;
    unsigned char is_valid = (id != -1) ? 1: 0;
    col.red = is_valid * gpu_obj_pointers.red[id*MAX_FACES_OBJECT+single];
    col.green = is_valid * gpu_obj_pointers.green[id*MAX_FACES_OBJECT+single];
    col.blue = is_valid * gpu_obj_pointers.blue[id*MAX_FACES_OBJECT+single];
    return col;
}

//------------------------------------------------------------------------------------------//
// GPU Functions (Scene rendering)
//------------------------------------------------------------------------------------------//
__global__ void update_identifiers(gpu_object_pointers gpu_obj_pointers, id_array identifier_array) {
    int height = blockIdx.x * blockDim.x + threadIdx.x;
    int width = blockIdx.y * blockDim.y + threadIdx.y;
    float x = IMAGE_OFFSET_WIDTH + PIXEL_WIDTH_SIZE/2.0 + ((float) width) * PIXEL_WIDTH_SIZE;
    float y = IMAGE_OFFSET_HEIGHT + PIXEL_HEIGHT_SIZE/2.0 + ((float) height) * PIXEL_HEIGHT_SIZE;
    identifier_array.id[height*IMAGE_RESOLUTION_WIDTH + width] = -1;
    identifier_array.side[height*IMAGE_RESOLUTION_WIDTH + width] = -1;
    for(int i=0; i<NB_OBJECT; i++) {
        float dim[MAX_DIMENSIONS_OBJECTS];
        #pragma unroll
        for (int j=0; j<MAX_DIMENSIONS_OBJECTS; j++) {
            dim[j] = gpu_obj_pointers.dimension[i*MAX_DIMENSIONS_OBJECTS+j];
        }
        int is_in = is_in_object(x, y, gpu_obj_pointers.type[i]
                               , gpu_obj_pointers.pos_x[i], gpu_obj_pointers.pos_y[i], gpu_obj_pointers.pos_z[i]
                               , gpu_obj_pointers.rot_x[i], gpu_obj_pointers.rot_y[i], gpu_obj_pointers.rot_z[i]
                               , dim);
        if (is_in != -1) {
            if (identifier_array.id[height*IMAGE_RESOLUTION_WIDTH + width] != -1) {
                if (gpu_obj_pointers.pos_z[i] < gpu_obj_pointers.pos_z[identifier_array.id[height*IMAGE_RESOLUTION_WIDTH + width]]) {
                    identifier_array.id[height*IMAGE_RESOLUTION_WIDTH + width] = i;
                    identifier_array.side[height*IMAGE_RESOLUTION_WIDTH + width] = is_in;
                }
            } else {
                identifier_array.id[height*IMAGE_RESOLUTION_WIDTH + width] = i;
                identifier_array.side[height*IMAGE_RESOLUTION_WIDTH + width] = is_in;
            }
        }
    }
}

__global__ void update_image(id_array identifier_array, gpu_object_pointers gpu_obj_pointers, image_array image) {
    int height = blockIdx.x * blockDim.x + threadIdx.x;
    int width = blockIdx.y * blockDim.y + threadIdx.y;
    colors col = get_colors(identifier_array.id[height*IMAGE_RESOLUTION_WIDTH + width], identifier_array.side[height*IMAGE_RESOLUTION_WIDTH + width], gpu_obj_pointers);
    image.red[height*IMAGE_RESOLUTION_WIDTH + width] = col.red;
    image.green[height*IMAGE_RESOLUTION_WIDTH + width] = col.green;
    image.blue[height*IMAGE_RESOLUTION_WIDTH + width] = col.blue;
    image.alpha[height*IMAGE_RESOLUTION_WIDTH + width] = 0;
}

__global__ void simple_anti_aliasing(image_array image) {
    int height = blockIdx.x * blockDim.x + threadIdx.x;
    int width = blockIdx.y * blockDim.y + threadIdx.y;
    int AA_result_red = 0;
    int AA_result_green = 0;
    int AA_result_blue = 0;
    if ((height > AA_SIMPLE_SURROUNDING_PIXEL-1) and (height < IMAGE_RESOLUTION_HEIGHT-AA_SIMPLE_SURROUNDING_PIXEL)) {
        if ((width > AA_SIMPLE_SURROUNDING_PIXEL-1) and (width < IMAGE_RESOLUTION_WIDTH-AA_SIMPLE_SURROUNDING_PIXEL)) {
            for (int i=-AA_SIMPLE_SURROUNDING_PIXEL; i<=AA_SIMPLE_SURROUNDING_PIXEL; i++) {
                for (int j=-AA_SIMPLE_SURROUNDING_PIXEL; j<=AA_SIMPLE_SURROUNDING_PIXEL; j++) {
                    AA_result_red += image.red[(height+i)*IMAGE_RESOLUTION_WIDTH + (width+j)];
                    AA_result_green += image.green[(height+i)*IMAGE_RESOLUTION_WIDTH + (width+j)];
                    AA_result_blue += image.blue[(height+i)*IMAGE_RESOLUTION_WIDTH + (width+j)];
                }
            }
            AA_result_red /= (2*AA_SIMPLE_SURROUNDING_PIXEL+1)*(2*AA_SIMPLE_SURROUNDING_PIXEL+1);
            image.red[height*IMAGE_RESOLUTION_WIDTH + width] = AA_result_red;
            AA_result_green /= (2*AA_SIMPLE_SURROUNDING_PIXEL+1)*(2*AA_SIMPLE_SURROUNDING_PIXEL+1);
            image.green[height*IMAGE_RESOLUTION_WIDTH + width] = AA_result_green;
            AA_result_blue /= (2*AA_SIMPLE_SURROUNDING_PIXEL+1)*(2*AA_SIMPLE_SURROUNDING_PIXEL+1);
            image.blue[height*IMAGE_RESOLUTION_WIDTH + width] = AA_result_blue;
        }
    }
}

//------------------------------------------------------------------------------------------//
// GPU Functions (Debug)
//------------------------------------------------------------------------------------------//
__global__ void print_identifier_array(id_array identifier_array) {
    for(int height=0; height<IMAGE_RESOLUTION_HEIGHT; height++) {
        printf(" ");
        for(int width=0; width<IMAGE_RESOLUTION_WIDTH; width++) {
            int carac = identifier_array.id[height*IMAGE_RESOLUTION_WIDTH + width];
            int side = identifier_array.side[height*IMAGE_RESOLUTION_WIDTH + width];
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

//------------------------------------------------------------------------------------------//
// Draw image function
//------------------------------------------------------------------------------------------//
void draw_image(object_to_gpu& tab_pos, image_array& image
              , id_array& identifier_array, image_array& gpu_image, gpu_object_pointers& gpu_obj_pointers
              , dim3 numBlocks, dim3 threadsPerBlock) {
    // Copy data to video memory
    copy_data_to_video_memory(gpu_obj_pointers, tab_pos);

    // Compute which object is visible (and which face can we see) for each pixel
    update_identifiers<<<numBlocks, threadsPerBlock>>>(gpu_obj_pointers, identifier_array);
    //print_identifier_array<<<1, 1>>>(identifier_array);

    // Assigns colors to each pixel, simply based on which object is visible (no light computations)
    update_image<<<numBlocks, threadsPerBlock>>>(identifier_array, gpu_obj_pointers, gpu_image);

    // AntiAliasing
    if (AA == "simple") {
        simple_anti_aliasing<<<numBlocks, threadsPerBlock>>>(gpu_image);
    }

    // Copy data from video memory (TODO: Needs improvement as this is the slowest point of the "compute")
    copy_data_from_video_memory(gpu_image, image);
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
    tab_pos.nb_dim[0] = 1;
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
    tab_pos.nb_dim[1] = 1;
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
    tab_pos.nb_dim[2] = 1;
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
    tab_pos.nb_dim[3] = 1;
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
    tab_pos.nb_dim[4] = 1;
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
    image.red = new unsigned char[RESOLUTION];
    image.green = new unsigned char[RESOLUTION];
    image.blue = new unsigned char[RESOLUTION];
    image.alpha = new unsigned char[RESOLUTION];

    // Video Memory initialisation
    dim3 numBlocks, threadsPerBlock;
    id_array gpu_id_array;
    image_array gpu_image;
    gpu_object_pointers gpu_obj_pointers;
    if (allocate_gpu_thread(numBlocks, threadsPerBlock)) return 1;
    initiate_video_memory(gpu_id_array, gpu_image, gpu_obj_pointers);
    copy_initial_data_to_video_memory(gpu_obj_pointers, tab_pos);

    printf("Initialisation finished, Waiting to start\n");
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
        draw_image(tab_pos, image, gpu_id_array, gpu_image, gpu_obj_pointers, numBlocks, threadsPerBlock);
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
        if (save_as_bmp(image, "test_image_gpu.bmp") == false) {
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

        usleep(100000);
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
    clean_video_memory(gpu_id_array, gpu_image, gpu_obj_pointers);
}