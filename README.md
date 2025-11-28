## Physics Based Scene Rendering

This project aims at implementing a scene rendering application for N objects in a frame of fixed resolution using various but yet not specified rendering techniques.

We use physics to compute the new positions of the objects in the scene at fixed time intervals, on the CPU by using multithreading to improve performance.

We use various techniques in order to output an image based on these positions by using a GPU (for now only a CUDA enabled GPU).