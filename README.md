## Physics Based Scene Rendering

This project aims at implementing a scene rendering application for N objects in a frame of fixed resolution using various but yet not specified rendering techniques.

We use physics to compute the new positions of the objects in the scene at fixed time intervals, on the CPU by using multithreading to improve performance.


We use various techniques in order to output an image based on these positions by using a GPU (for now only a CUDA enabled GPU).

## Building the project

# Dependencies

In order to execute the full programm, you will need to have installed a few things :

 - [Singularity][https://docs.sylabs.io/guides/3.0/user-guide/installation.html]: Used for the docker that is used to output the video

```bash
sudo apt-get install -y singularity-container
```
 - FFMpeg: Used for outputing the final video

```bash
singularity pull docker://jrottenberg/ffmpeg
```
