#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#define FRAMEIDX(F) F - 1
using namespace cv;
using namespace std;
VideoCapture vcap;
Mat image;
string filename = "0X1A0A263B22CCD966.avi";
int main()
{

    string filepath = "../../" + filename;

    vcap.open(filepath);

    int framerate = vcap.get(CAP_PROP_FPS);
    int frames = vcap.get(CAP_PROP_FRAME_COUNT);
    int height = vcap.get(CAP_PROP_FRAME_HEIGHT);
    int width = vcap.get(CAP_PROP_FRAME_WIDTH);
    cout << "Video framerate:" << framerate << " height:" << height << " width:" << width << " Num Frames" << frames << '\n';

    int slider_position = 0;

    namedWindow(filename, WINDOW_NORMAL);
    resizeWindow(filename, width * 5, height * 5);
    createTrackbar("Position", filename, &slider_position, FRAMEIDX(frames),
        [](int pos, void*) { vcap.set(CAP_PROP_POS_FRAMES, pos); vcap >> image; imshow(filename,image); });

    while (1) {

        vcap >> image;
        if (image.empty())
            break;
        imshow(filename, image);
        waitKey(1000 / framerate);
    }
    char c = (char)waitKey(0);
    destroyAllWindows();
    return 0;
}
