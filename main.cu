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
// Librairies
//------------------------------------------------------------------------------------------//



//------------------------------------------------------------------------------------------//
// Main
//------------------------------------------------------------------------------------------//
int main (int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    // Start of program
    //...



    // Beginning of loop initialization
    //...

    
    // Main loop
    auto loop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_loop = loop - start;
    while (duration_loop.count()<MAX_SIMU_TIME) {
        // Start of rendering loop

        // End of rendering loop
        loop = std::chrono::high_resolution_clock::now();
        duration_loop = loop - start;
    }
    // End of loop initialization
    //...

    

    // End of program
    if (DEBUG_PERF) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        printf("Total execution time: %f ms\n", duration.count());
    }
    return 0;
}