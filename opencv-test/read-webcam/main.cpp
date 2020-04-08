#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    string camWindow = "Camera";
    namedWindow(camWindow);
    VideoCapture capturer;
    capturer.open(0);
    if(!capturer.isOpened()){
        cout <<"could not open camera."<<endl;
        exit(-1);
    }

    Mat image;
    while(1){
        capturer >>image;
        imshow(camWindow, image);
        if (waitKey(10) == (char) 's') break;
    }

    destroyAllWindows();
    return 0;
}
