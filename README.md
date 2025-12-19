## Physics Based Scene Rendering

This project aims at implementing a scene rendering application for N objects in a frame of fixed resolution using various but yet not specified rendering techniques.

We use physics to compute the new positions of the objects in the scene at fixed time intervals, on the CPU by using multithreading to improve performance.


We use various techniques in order to output an image based on these positions by using a GPU (for now only a CUDA enabled GPU).

## Building the project

1. Clone the repository in your directory :
```bash
git clone https://github.com/Kekossssss/Physics_Based_Scene_Rendering.git
```

Install the following dependencies : Singularity and ffmpeg.
You can find proper links below.
Make sure to add Singularity in your path.

You can compile and run a few versions :

CPU Version :
```bash
make main_cpu
```

GPU Version :
```bash
make main_cpu
```

Now you can run the following different version :

CPU Version :
```bash
./main_cpu
```

GPU Version :
```bash
./main_gpu
```

Objects can be added by directly modifying the shapes vector in both versions of the code. Objects are instantiated as follows:

Sphere:
Sphere(Point3D(x, y, z), double diameter, Point3D(vx, vy, vz), double mass, double e, double g)

Cube:
Cube(Point3D(x, y, z), double side, Point3D(vx, vy, vz), Point3D(ax, ay, az), Point3D(avx, avy, avz), double mass, double e, double g)

Here, (x, y, z) represent the spatial coordinates of the object, and (vx, vy, vz) define its linear velocity. The parameter mass corresponds to the object’s mass, e is the coefficient of restitution, and g represents the gravity constant.
For cubes, (ax, ay, az) define the linear acceleration, while (avx, avy, avz) correspond to the angular velocity.

The gravity constant g can be modified to make an object independent from gravity, or set to the standard value of 9.81.

The video resolution can be adjusted in utils.hpp using the variables IMAGE_RESOLUTION_WIDTH, IMAGE_RESOLUTION_WIDTH_FLOAT, IMAGE_RESOLUTION_HEIGHT, and IMAGE_RESOLUTION_HEIGHT_FLOAT.

The total number of rendered frames can be configured using RENDERED_FRAMES. In our implementation, the video is rendered at 60 frames per second; therefore, changing RENDERED_FRAMES directly affects the video duration. For example, setting RENDERED_FRAMES to 60 produces a video with a duration of one second.

# Dependencies

In order to execute the full programm, you will need to have installed a few things :

 - [Singularity][https://docs.sylabs.io/guides/3.0/user-guide/installation.html]: Used for the docker that is used to output the video

```bash
sudo apt-get install -y singularity-container
```
 - FFMpeg: Used for outputing the final video

This command should be executed in ./Physics_Based_Scene_Rendering
```bash
singularity pull docker://jrottenberg/ffmpeg
```
The following video demonstrates the capabilities of our implementation:

[![IMAGE ALT TEXT HERE](https://i.ytimg.com/vi/-D36CBSIemU/hqdefault.jpg?sqp=-oaymwE2CNACELwBSFXyq4qpAygIARUAAIhCGAFwAcABBvABAfgB_gmAAtAFigIMCAAQARgRIHIoETAP&rs=AOn4CLCSCvqLHNROWM9I-84iaEzEh3WClw)](https://www.youtube.com/watch?v=-D36CBSIemU)



