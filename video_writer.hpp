struct ThreadData
{
    std::vector<Shape *> *shapes;
    int start_idx;
    int end_idx;
    double dt;
};

struct MP4FrameData
{
    image_array *image;
    unsigned char *rgb_buffer;
    bool ready;
    bool done;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

struct RGBConversionData
{
    const image_array *image;
    unsigned char *rgb_buffer;
    int start_idx;
    int end_idx;
};

class MP4VideoEncoder
{
private:
    FILE *ffmpeg_pipe;
    int width;
    int height;
    int fps;
    bool is_open;

public:
    MP4VideoEncoder(const char *filename, int w, int h, bool debug, int framerate);

    bool writeFrame(const image_array &image);

    bool writeFrameRGB(const unsigned char *rgb_buffer);

    ~MP4VideoEncoder();
};

struct VideoEncoderThreadData
{
    MP4VideoEncoder *encoder;
    MP4FrameData *frame_data;
};

void *updateShapesThread(void *arg);
void updateShapesParallel(std::vector<Shape *> &shapes, double dt, int num_threads);
void *convertToRGBThread(void *arg);
void convertToRGBParallel(const image_array &image, unsigned char *rgb_buffer, int num_threads = 4);
void *asyncVideoEncoderThread(void *arg);
void copyImageArray(const image_array &src, image_array &dst);
