#ifndef CPU_RENDERER_HPP // include guard
#define CPU_RENDERER_HPP

#include "utils.hpp"
#include "cpu_converter.hpp"
#include <vector>

// Draw an image from GPU-style data
void draw_image(object_to_gpu &tab_pos, image_array &image);
void draw_image_gold(object_to_gpu &tab_pos, image_array &image);

// Draw an image directly from CPU shapes (converts to GPU format internally)
void draw_image(const std::vector<Shape*> &shapes, image_array &image, bool randomColors = true);
void draw_image_gold(const std::vector<Shape*> &shapes, image_array &image, bool randomColors = true);

#endif
