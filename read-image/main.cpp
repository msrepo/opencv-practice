#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    string filepath = "../2020-03-28-Scene.png";

    Mat file = imread(filepath);

    namedWindow("TestWindow");
    imshow("TestWindow",file);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
