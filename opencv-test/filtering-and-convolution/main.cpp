#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
Mat image;

int maxThresh = 0, minThresh = 0;

void on_trackbar_threshold(int , void *);
void showInWindow(char *, Mat);
Mat resizeWithAspectRatio(Mat);

Mat resizeWithAspectRatio(Mat img){
    Size sz = img.size();
    Mat returnImg;
    int height = 600;
    float ratio = sz.height * 1.0 / sz.width;
    Size resized(height / ratio ,height);
    resize(img,returnImg,resized,0,0,INTER_AREA);
    return returnImg;
}

void on_trackbar_threshold(int , void*)
{
    Mat threshMaxImg,threshMinImg;
    threshold(image,threshMaxImg,maxThresh,255,THRESH_BINARY_INV);
    threshold(image,threshMinImg,minThresh,255,THRESH_BINARY);
    Mat result;
    bitwise_and(threshMaxImg,threshMinImg,result);
    showInWindow("Threshold",resizeWithAspectRatio(result));
}
void showInWindow(char* name, Mat img)
{
    namedWindow(name,WINDOW_KEEPRATIO);
    imshow(name, img);
}



int main()
{
    string filepath = "../../sunhl-1th-02-Jan-2017-162 A AP.jpg";
    image = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);

    double min, max;
    minMaxLoc(image, &min, &max);
    cout << min << " " << max << endl;

    showInWindow("Original", resizeWithAspectRatio(image));

    showInWindow("Threshold", resizeWithAspectRatio(image));

    createTrackbar("Max", "Threshold", &maxThresh, max, on_trackbar_threshold);
    createTrackbar("Min", "Threshold", &minThresh, max, on_trackbar_threshold);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
