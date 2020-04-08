#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
    string filepath = "../../QuickMask_00_subsample.mov";
    VideoCapture vcap;
    vcap.open(filepath);

    namedWindow("Test");
    Mat image;
    while (1) {
        vcap >> image;
        if (image.empty()) {
            break;
        }
        imshow("Test", image);
        waitKey(30);

    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}
