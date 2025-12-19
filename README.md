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
./main_cpu [NUMBER OF OBJECTS] [fps] [WIDTH] [HEIGHT]
```

GPU Version :
```bash
./main_gpu [NUMBER OF OBJECTS] [fps] [WIDTH] [HEIGHT]
```

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
