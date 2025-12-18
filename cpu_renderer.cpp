#include "utils.hpp"
#include "cpu_renderer.hpp"

//------------------------------------------------------------------------------------------//
// Librairies
//------------------------------------------------------------------------------------------//
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Library used for sleep function
#include <unistd.h>

#define RENDERED_FRAMES 100

//------------------------------------------------------------------------------------------//
// General Functions
//------------------------------------------------------------------------------------------//
float dist2D(float A_x, float A_y, float B_x, float B_y)
{
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return sqrt(dx * dx + dy * dy);
}

float dist3D(position A, position B)
{
    float dx = B.x - A.x;
    float dy = B.y - A.y;
    float dz = B.z - A.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

float squareDist2D(float A_x, float A_y, float B_x, float B_y)
{
    float dx = B_x - A_x;
    float dy = B_y - A_y;
    return dx * dx + dy * dy;
}

float squareDist3D(position A, position B)
{
    float dx = B.x - A.x;
    float dy = B.y - A.y;
    float dz = B.z - A.z;
    return dx * dx + dy * dy + dz * dz;
}

void rotate2D_x(float &new_y, float &new_z, float old_y, float old_z, float theta)
{
    new_y = old_y * cos(theta) - old_z * sin(theta);
    new_z = old_y * sin(theta) + old_z * cos(theta);
}

void rotate2D_y(float &new_x, float &new_z, float old_x, float old_z, float theta)
{
    new_z = old_z * cos(theta) - old_x * sin(theta);
    new_x = old_z * sin(theta) + old_x * cos(theta);
}

void rotate2D_z(float &new_x, float &new_y, float old_x, float old_y, float theta)
{
    new_x = old_x * cos(theta) - old_y * sin(theta);
    new_y = old_x * sin(theta) + old_y * cos(theta);
}

void rotate3D(position &new_pos, position pos, rotation rot)
{
    rotate2D_x(new_pos.y, new_pos.z, pos.y, pos.z, rot.theta_x);
    rotate2D_y(new_pos.x, new_pos.z, pos.x, new_pos.z, rot.theta_y);
    rotate2D_z(new_pos.x, new_pos.y, new_pos.x, new_pos.y, rot.theta_z);
}

position update_camera_perspective(position old_pos)
{
    position new_pos;
    float ratio;
    if (old_pos.z < 0.0)
    {
        ratio = old_pos.z / (-CAMERA_Z);
    }
    else
    {
        ratio = old_pos.z / (old_pos.z - CAMERA_Z);
    }
    new_pos.x = (CAMERA_X - old_pos.x) * ratio + old_pos.x;
    new_pos.y = (CAMERA_Y - old_pos.y) * ratio + old_pos.y;
    new_pos.z = old_pos.z;
    return new_pos;
}

int sort_dist_list(float *list_dist, int *list_faces, int size)
{
    float temp_f;
    int temp_c;
    for (int i = 0; i < size - 1; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            if (list_dist[i] > list_dist[j])
            {
                temp_f = list_dist[i];
                temp_c = list_faces[i];
                list_dist[i] = list_dist[j];
                list_faces[i] = list_faces[j];
                list_dist[j] = temp_f;
                list_faces[j] = temp_c;
            }
        }
    }
    for (int i = 0; i < size; i++)
    {
        if (list_faces[i] != -1)
            return list_faces[i];
    }
    return -1;
}

int belongs_2D_4side_convex_polygone(float x, float y, position A0, position A1, position A2, position A3, float D, int face)
{
    float det1 = (A1.x - A0.x) * (y - A0.y) - (A1.y - A0.y) * (x - A0.x);
    float det2 = (A2.x - A1.x) * (y - A1.y) - (A2.y - A1.y) * (x - A1.x);
    if (det1 * det2 <= 0)
        return -1;
    float det3 = (A3.x - A2.x) * (y - A2.y) - (A3.y - A2.y) * (x - A2.x);
    if (det2 * det3 <= 0)
        return -1;
    float det4 = (A0.x - A3.x) * (y - A3.y) - (A0.y - A3.y) * (x - A3.x);
    if (det3 * det4 <= 0)
        return -1;
    else
        return face;
}

int belongs_2D_4side_convex_polygone_with_sides(float x, float y, position A0, position A1, position A2, position A3, float D, int face)
{
    // Point is equal to one of the points of the polygon
    if (x == A0.x and y == A0.y)
        return face;
    if (x == A1.x and y == A1.y)
        return face;
    if (x == A2.x and y == A2.y)
        return face;
    if (x == A3.x and y == A3.y)
        return face;
    // Check if it is inside/on the sides
    float det1 = (A1.x - A0.x) * (y - A0.y) - (A1.y - A0.y) * (x - A0.x);
    float det2 = (A2.x - A1.x) * (y - A1.y) - (A2.y - A1.y) * (x - A1.x);
    float det3 = (A3.x - A2.x) * (y - A2.y) - (A3.y - A2.y) * (x - A2.x);
    float det4 = (A0.x - A3.x) * (y - A3.y) - (A0.y - A3.y) * (x - A3.x);
    // Point is on the sides
    if (det1 == 0.0)
    {
        if (det2 * det3 <= 0)
            return -1;
        if (det3 * det4 <= 0)
            return -1;
    }
    else if (det2 == 0.0)
    {
        if (det1 * det3 <= 0)
            return -1;
        if (det3 * det4 <= 0)
            return -1;
    }
    else if (det3 == 0.0)
    {
        if (det1 * det2 <= 0)
            return -1;
        if (det2 * det4 <= 0)
            return -1;
    }
    else if (det4 == 0.0)
    {
        if (det1 * det2 <= 0)
            return -1;
        if (det2 * det3 <= 0)
            return -1;
        // Point is inside
    }
    else
    {
        if (det1 * det2 < 0)
            return -1;
        if (det2 * det3 < 0)
            return -1;
        if (det3 * det4 < 0)
            return -1;
    }
    return face;
}

int is_in_cube(float x, float y, position pos, rotation rot, float *dimensions)
{
    // Compute positions of cube summits
    //    E----F
    //   /    /|
    //  A----B G
    //  |    |/
    //  D----C
    position A;
    A.x = -dimensions[0] / 2.0;
    A.y = -dimensions[0] / 2.0;
    A.z = -dimensions[0] / 2.0;
    rotate3D(A, A, rot);
    A.x += pos.x;
    A.y += pos.y;
    A.z += pos.z;
    position B;
    B.x = dimensions[0] / 2.0;
    B.y = -dimensions[0] / 2.0;
    B.z = -dimensions[0] / 2.0;
    rotate3D(B, B, rot);
    B.x += pos.x;
    B.y += pos.y;
    B.z += pos.z;
    position C;
    C.x = dimensions[0] / 2.0;
    C.y = dimensions[0] / 2.0;
    C.z = -dimensions[0] / 2.0;
    rotate3D(C, C, rot);
    C.x += pos.x;
    C.y += pos.y;
    C.z += pos.z;
    position D;
    D.x = -dimensions[0] / 2.0;
    D.y = dimensions[0] / 2.0;
    D.z = -dimensions[0] / 2.0;
    rotate3D(D, D, rot);
    D.x += pos.x;
    D.y += pos.y;
    D.z += pos.z;
    position E;
    E.x = -dimensions[0] / 2.0;
    E.y = -dimensions[0] / 2.0;
    E.z = dimensions[0] / 2.0;
    rotate3D(E, E, rot);
    E.x += pos.x;
    E.y += pos.y;
    E.z += pos.z;
    position F;
    F.x = dimensions[0] / 2.0;
    F.y = -dimensions[0] / 2.0;
    F.z = dimensions[0] / 2.0;
    rotate3D(F, F, rot);
    F.x += pos.x;
    F.y += pos.y;
    F.z += pos.z;
    position G;
    G.x = dimensions[0] / 2.0;
    G.y = dimensions[0] / 2.0;
    G.z = dimensions[0] / 2.0;
    rotate3D(G, G, rot);
    G.x += pos.x;
    G.y += pos.y;
    G.z += pos.z;
    position H;
    H.x = -dimensions[0] / 2.0;
    H.y = dimensions[0] / 2.0;
    H.z = dimensions[0] / 2.0;
    rotate3D(H, H, rot);
    H.x += pos.x;
    H.y += pos.y;
    H.z += pos.z;
    // Compute position of faces centers
    position camera;
    camera.x = CAMERA_X;
    camera.y = CAMERA_Y;
    camera.z = CAMERA_Z;
    position center_front;
    center_front.x = (A.x + B.x + C.x + D.x) / 4.0;
    center_front.y = (A.y + B.y + C.y + D.y) / 4.0;
    center_front.z = (A.z + B.z + C.z + D.z) / 4.0;
    position center_back;
    center_back.x = (E.x + F.x + G.x + H.x) / 4.0;
    center_back.y = (E.y + F.y + G.y + H.y) / 4.0;
    center_back.z = (E.z + F.z + G.z + H.z) / 4.0;
    position center_top;
    center_top.x = (A.x + B.x + F.x + E.x) / 4.0;
    center_top.y = (A.y + B.y + F.y + E.y) / 4.0;
    center_top.z = (A.z + B.z + F.z + E.z) / 4.0;
    position center_bottom;
    center_bottom.x = (D.x + C.x + G.x + H.x) / 4.0;
    center_bottom.y = (D.y + C.y + G.y + H.y) / 4.0;
    center_bottom.z = (D.z + C.z + G.z + H.z) / 4.0;
    position center_right;
    center_right.x = (B.x + F.x + G.x + C.x) / 4.0;
    center_right.y = (B.y + F.y + G.y + C.y) / 4.0;
    center_right.z = (B.z + F.z + G.z + C.z) / 4.0;
    position center_left;
    center_left.x = (A.x + E.x + H.x + D.x) / 4.0;
    center_left.y = (A.y + E.y + H.y + D.y) / 4.0;
    center_left.z = (A.z + E.z + H.z + D.z) / 4.0;
    // Compute distance to faces
    float dist_to_cam[6];
    dist_to_cam[0] = squareDist3D(center_front, camera);
    dist_to_cam[1] = squareDist3D(center_back, camera);
    dist_to_cam[2] = squareDist3D(center_top, camera);
    dist_to_cam[3] = squareDist3D(center_bottom, camera);
    dist_to_cam[4] = squareDist3D(center_right, camera);
    dist_to_cam[5] = squareDist3D(center_left, camera);
    // Update Points position on screen relative to camera viewpoint
    A = update_camera_perspective(A);
    B = update_camera_perspective(B);
    C = update_camera_perspective(C);
    D = update_camera_perspective(D);
    E = update_camera_perspective(E);
    F = update_camera_perspective(F);
    G = update_camera_perspective(G);
    H = update_camera_perspective(H);
    // Check hit faces
    int list_faces[6];
    // Front
    list_faces[0] = belongs_2D_4side_convex_polygone_with_sides(x, y, A, B, C, D, dimensions[0], 2);
    // Back
    list_faces[1] = belongs_2D_4side_convex_polygone_with_sides(x, y, E, F, G, H, dimensions[0], 3);
    // Top
    list_faces[2] = belongs_2D_4side_convex_polygone_with_sides(x, y, A, B, F, E, dimensions[0], 0);
    // Bottom
    list_faces[3] = belongs_2D_4side_convex_polygone_with_sides(x, y, D, C, G, H, dimensions[0], 5);
    // Right
    list_faces[4] = belongs_2D_4side_convex_polygone_with_sides(x, y, B, F, G, C, dimensions[0], 1);
    // Left
    list_faces[5] = belongs_2D_4side_convex_polygone_with_sides(x, y, A, E, H, D, dimensions[0], 4);
    // Return closest hit face
    return sort_dist_list(dist_to_cam, list_faces, 6);
}

bool is_in_sphere(float x, float y, position pos, float *dimensions)
{
    position new_pos;
    float ratio;
    float new_dim;
    if (pos.z < 0.0)
    {
        ratio = pos.z / (-CAMERA_Z);
    }
    else
    {
        ratio = pos.z / (pos.z - CAMERA_Z);
    }
    new_dim = (1.0 - ratio) * dimensions[0];
    new_pos.x = (CAMERA_X - pos.x) * ratio + pos.x;
    new_pos.y = (CAMERA_Y - pos.y) * ratio + pos.y;
    return (squareDist2D(x, y, new_pos.x, new_pos.y) <= new_dim * new_dim);
}

int is_in_object(float x, float y, unsigned char id, position pos, rotation rot, float *dimensions)
{
    bool side;
    if (id == CUBE)
    {
        return is_in_cube(x, y, pos, rot, dimensions);
    }
    else if (id == SPHERE)
    {
        side = is_in_sphere(x, y, pos, dimensions);
        if (side == true)
            return 0;
        else
            return -1;
    }
    else
    {
        return -1;
    }
}

void update_identifiers(object_to_gpu &tab_pos, id_array &identifier_array)
{
    for (int height = 0; height < IMAGE_RESOLUTION_HEIGHT; height++)
    {
        for (int width = 0; width < IMAGE_RESOLUTION_WIDTH; width++)
        {
            float x = IMAGE_OFFSET_WIDTH + PIXEL_WIDTH_SIZE / 2.0 + ((float)width) * PIXEL_WIDTH_SIZE;
            float y = IMAGE_OFFSET_HEIGHT + PIXEL_HEIGHT_SIZE / 2.0 + ((float)height) * PIXEL_HEIGHT_SIZE;
            identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = -1;
            identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = -1;
            for (int i = 0; i < NB_OBJECT; i++)
            {
                if (identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] == -1)
                {
                    if (is_in_sphere(x, y, tab_pos.pos[i], tab_pos.dimension[i]))
                    {
                        if (tab_pos.type[i] == SPHERE)
                        {
                            identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = i;
                            identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = 0;
                        }
                        else
                        {
                            int is_in = is_in_object(x, y, tab_pos.type[i], tab_pos.pos[i], tab_pos.rot[i], tab_pos.dimension[i]);
                            if (is_in != -1)
                            {
                                identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = i;
                                identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = is_in;
                            }
                        }
                    }
                }
                else if (tab_pos.pos[i].z < tab_pos.pos[identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width]].z)
                {
                    if (is_in_sphere(x, y, tab_pos.pos[i], tab_pos.dimension[i]))
                    {
                        if (tab_pos.type[i] == SPHERE)
                        {
                            identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = i;
                            identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = 0;
                        }
                        else
                        {
                            int is_in = is_in_object(x, y, tab_pos.type[i], tab_pos.pos[i], tab_pos.rot[i], tab_pos.dimension[i]);
                            if (is_in != -1)
                            {
                                identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = i;
                                identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = is_in;
                            }
                        }
                    }
                }
            }
        }
    }
}

void update_identifiers_gold(object_to_gpu &tab_pos, id_array &identifier_array)
{
    for (int height = 0; height < IMAGE_RESOLUTION_HEIGHT; height++)
    {
        for (int width = 0; width < IMAGE_RESOLUTION_WIDTH; width++)
        {
            float x = IMAGE_OFFSET_WIDTH + PIXEL_WIDTH_SIZE / 2.0 + ((float)width) * PIXEL_WIDTH_SIZE;
            float y = IMAGE_OFFSET_HEIGHT + PIXEL_HEIGHT_SIZE / 2.0 + ((float)height) * PIXEL_HEIGHT_SIZE;
            identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = -1;
            identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = -1;
            for (int i = 0; i < NB_OBJECT; i++)
            {
                if (identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] == -1)
                {
                    int is_in = is_in_object(x, y, tab_pos.type[i], tab_pos.pos[i], tab_pos.rot[i], tab_pos.dimension[i]);
                    if (is_in != -1)
                    {
                        identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = i;
                        identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = is_in;
                    }
                }
                else if (tab_pos.pos[i].z < tab_pos.pos[identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width]].z)
                {
                    int is_in = is_in_object(x, y, tab_pos.type[i], tab_pos.pos[i], tab_pos.rot[i], tab_pos.dimension[i]);
                    if (is_in != -1)
                    {
                        identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width] = i;
                        identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width] = is_in;
                    }
                }
            }
        }
    }
}

colors get_colors(int id, int side, object_to_gpu &tab_pos)
{
    colors col;
    int single = (tab_pos.is_single_color[id] == false) ? side : 0;
    unsigned char is_valid = (id != -1) ? 1 : 0;
    col.red = is_valid * tab_pos.col[id][single].red;
    col.green = is_valid * tab_pos.col[id][single].green;
    col.blue = is_valid * tab_pos.col[id][single].blue;
    return col;
}

void update_image(id_array &identifier_array, object_to_gpu &tab_pos, image_array &image)
{
    for (int height = 0; height < IMAGE_RESOLUTION_HEIGHT; height++)
    {
        for (int width = 0; width < IMAGE_RESOLUTION_WIDTH; width++)
        {
            colors col;
            col = get_colors(identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width], identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width], tab_pos);
            image.red[height * IMAGE_RESOLUTION_WIDTH + width] = col.red;
            image.green[height * IMAGE_RESOLUTION_WIDTH + width] = col.green;
            image.blue[height * IMAGE_RESOLUTION_WIDTH + width] = col.blue;
            image.alpha[height * IMAGE_RESOLUTION_WIDTH + width] = 0;
        }
    }
}

void print_identifier_array(id_array &identifier_array)
{
    for (int height = 0; height < IMAGE_RESOLUTION_HEIGHT; height++)
    {
        printf(" ");
        for (int width = 0; width < IMAGE_RESOLUTION_WIDTH; width++)
        {
            int carac = identifier_array.id[height * IMAGE_RESOLUTION_WIDTH + width];
            int side = identifier_array.side[height * IMAGE_RESOLUTION_WIDTH + width];
            if (carac == -1 or carac >= 10)
            {
                printf("%d", carac);
            }
            else
            {
                printf(" %d", carac);
            }
            if (side == -1 or side >= 10)
            {
                printf(",%d | ", side);
            }
            else
            {
                printf(", %d | ", side);
            }
        }
        printf("\n");
    }
}

void simple_anti_aliasing(image_array image)
{
    for (int height = 0; height < IMAGE_RESOLUTION_HEIGHT; height++)
    {
        for (int width = 0; width < IMAGE_RESOLUTION_WIDTH; width++)
        {
            int AA_result_red = 0;
            int AA_result_green = 0;
            int AA_result_blue = 0;
            if ((height > AA_SIMPLE_SURROUNDING_PIXEL - 1) and (height < IMAGE_RESOLUTION_HEIGHT - AA_SIMPLE_SURROUNDING_PIXEL))
            {
                if ((width > AA_SIMPLE_SURROUNDING_PIXEL - 1) and (width < IMAGE_RESOLUTION_WIDTH - AA_SIMPLE_SURROUNDING_PIXEL))
                {
                    for (int i = -AA_SIMPLE_SURROUNDING_PIXEL; i <= AA_SIMPLE_SURROUNDING_PIXEL; i++)
                    {
                        for (int j = -AA_SIMPLE_SURROUNDING_PIXEL; j <= AA_SIMPLE_SURROUNDING_PIXEL; j++)
                        {
                            AA_result_red += image.red[(height + i) * IMAGE_RESOLUTION_WIDTH + (width + j)];
                            AA_result_green += image.green[(height + i) * IMAGE_RESOLUTION_WIDTH + (width + j)];
                            AA_result_blue += image.blue[(height + i) * IMAGE_RESOLUTION_WIDTH + (width + j)];
                        }
                    }
                    AA_result_red /= (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1) * (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1);
                    image.red[height * IMAGE_RESOLUTION_WIDTH + width] = AA_result_red;
                    AA_result_green /= (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1) * (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1);
                    image.green[height * IMAGE_RESOLUTION_WIDTH + width] = AA_result_green;
                    AA_result_blue /= (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1) * (2 * AA_SIMPLE_SURROUNDING_PIXEL + 1);
                    image.blue[height * IMAGE_RESOLUTION_WIDTH + width] = AA_result_blue;
                }
            }
        }
    }
}

void reinit_terminal()
{
    // Clear message from BMP image
    printf("\x1b[1F"); // Move to beginning of previous line
    printf("\x1b[2K"); // Clear entire line
    // Clear """image""" from identifiers
    for (int height = 0; height < IMAGE_RESOLUTION_HEIGHT; height++)
    {
        printf("\x1b[1F"); // Move to beginning of previous line
        printf("\x1b[2K"); // Clear entire line
    }
}

bool check_image(image_array &img1, image_array &img2)
{
    for (int height = 0; height < IMAGE_RESOLUTION_HEIGHT; height++)
    {
        for (int width = 0; width < IMAGE_RESOLUTION_WIDTH; width++)
        {
            if (img1.red[height * IMAGE_RESOLUTION_WIDTH + width] != img2.red[height * IMAGE_RESOLUTION_WIDTH + width])
            {
                return false;
            }
            if (img1.green[height * IMAGE_RESOLUTION_WIDTH + width] != img2.green[height * IMAGE_RESOLUTION_WIDTH + width])
            {
                return false;
            }
            if (img1.blue[height * IMAGE_RESOLUTION_WIDTH + width] != img2.blue[height * IMAGE_RESOLUTION_WIDTH + width])
            {
                return false;
            }
        }
    }
    return true;
}

//------------------------------------------------------------------------------------------//
// Draw image function
//------------------------------------------------------------------------------------------//
void draw_image(object_to_gpu &tab_pos, image_array &image)
{
    // Array of 2D Object identifier (0 : index of object, 1 : hit side)
    id_array identifier_array;
    identifier_array.id = new int[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];
    identifier_array.side = new int[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];

    // Compute which object is visible (and which face can we see) for each pixel
    update_identifiers(tab_pos, identifier_array);

    // """Image preview"""
    if (DEBUG_VALUES and IMAGE_RESOLUTION_HEIGHT < 32 and IMAGE_RESOLUTION_WIDTH < 32)
    {
        print_identifier_array(identifier_array);
        reinit_terminal();
    }

    // Assigns colors to each pixel, simply based on which object is visible (no light computations)
    update_image(identifier_array, tab_pos, image);

    if (AA == "simple")
    {
        simple_anti_aliasing(image);
    }
}

void draw_image_gold(object_to_gpu &tab_pos, image_array &image)
{
    // Array of 2D Object identifier (0 : index of object, 1 : hit side)
    id_array identifier_array;
    identifier_array.id = new int[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];
    identifier_array.side = new int[IMAGE_RESOLUTION_WIDTH * IMAGE_RESOLUTION_HEIGHT];

    // Compute which object is visible (and which face can we see) for each pixel
    update_identifiers_gold(tab_pos, identifier_array);

    // """Image preview"""
    if (DEBUG_VALUES and IMAGE_RESOLUTION_HEIGHT < 32 and IMAGE_RESOLUTION_WIDTH < 32)
    {
        print_identifier_array(identifier_array);
        reinit_terminal();
    }

    // Assigns colors to each pixel, simply based on which object is visible (no light computations)
    update_image(identifier_array, tab_pos, image);

    if (AA == "simple")
    {
        simple_anti_aliasing(image);
    }
}

// --- Conversion wrappers: draw directly from CPU shapes ---
#include "cpu_renderer.hpp" // pulls in cpu_converter.hpp so we can convert shapes

void draw_image(const std::vector<Shape*> &shapes, image_array &image, bool randomColors)
{
    object_to_gpu tab_pos;
    convertSceneToGPU(shapes, tab_pos, randomColors);
    draw_image(tab_pos, image);
}

void draw_image_gold(const std::vector<Shape*> &shapes, image_array &image, bool randomColors)
{
    object_to_gpu tab_pos;
    convertSceneToGPU(shapes, tab_pos, randomColors);
    draw_image_gold(tab_pos, image);
}

/*
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
    tab_pos.is_single_color[2] = false;
    //// Top
    tab_pos.col[2][0].red = 255;
    tab_pos.col[2][0].blue = 255;
    tab_pos.col[2][0].green = 255;
    //// Right
    tab_pos.col[2][1].red = 0;
    tab_pos.col[2][1].blue = 0;
    tab_pos.col[2][1].green = 255;
    //// Front
    tab_pos.col[2][2].red = 255;
    tab_pos.col[2][2].blue = 0;
    tab_pos.col[2][2].green = 0;
    //// Back
    tab_pos.col[2][3].red = 255;
    tab_pos.col[2][3].blue = 0;
    tab_pos.col[2][3].green = 255;
    //// Left
    tab_pos.col[2][4].red = 0;
    tab_pos.col[2][4].blue = 255;
    tab_pos.col[2][4].green = 0;
    //// Bottom
    tab_pos.col[2][5].red = 255;
    tab_pos.col[2][5].blue = 255;
    tab_pos.col[2][5].green = 255;

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
    image.red = new unsigned char[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    image.green = new unsigned char[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    image.blue = new unsigned char[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    image.alpha = new unsigned char[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];

    image_array image_gold;
    image_gold.red = new unsigned char[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    image_gold.green = new unsigned char[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    image_gold.blue = new unsigned char[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];
    image_gold.alpha = new unsigned char[IMAGE_RESOLUTION_WIDTH*IMAGE_RESOLUTION_HEIGHT];

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
        if (ONLY_FINAL_FRAME == false) {
            if (save_as_bmp(image, "test_image_cpu.bmp") == false) {
                printf("Image saving error, leaving loop\n");
                break;
            }
        }

        // Temporary positions updates for testing rendering techniques
        tab_pos.pos[0].x += 40.0;
        tab_pos.pos[0].y += 40.0;
        tab_pos.pos[1].y += 40.0;
        tab_pos.pos[2].x += 40.0;
        tab_pos.pos[2].z += 10.0;
        tab_pos.pos[3].x += 10.0;
        tab_pos.pos[3].z += 10.0;
        tab_pos.pos[4].x += 5.0;
        tab_pos.pos[4].y += 5.0;
        tab_pos.rot[4].theta_x += 0.1;
        tab_pos.rot[4].theta_y += 0.1;
        tab_pos.rot[4].theta_z += 0.1;

        //usleep(100000);
    }
    printf("--------------End of Rendering--------------\n");

    // Final image output
    if (ONLY_FINAL_FRAME) {
        if (save_as_bmp(image, "test_image_cpu.bmp") == false) {
            printf("Image saving error, leaving loop\n");
        }
    }

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
}*/