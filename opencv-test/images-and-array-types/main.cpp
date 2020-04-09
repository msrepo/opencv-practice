#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
int main()
{
    string filepath = "../../Lenna.png";

    Mat m = imread(filepath,CV_LOAD_IMAGE_GRAYSCALE);

    Size sz = m.size();
    cout <<m.size().width<<" "<<m.size().height<<" Depth:"<<m.depth()
         <<" Channels:"<<m.channels()<<endl;
    cout<<(int)m.at<uchar>(0,0)<<" "<<(int)m.at<uchar>(200,0)<<endl;

    double min,max;
    minMaxLoc(m,&min,&max);
    cout<<"Min:"<<min<<" Max:"<<max<<endl;

    Mat integralImage;
    integralImage.convertTo(integralImage,CV_32FC1);
    integral(m,integralImage);

    namedWindow("test");
    imshow("test",integralImage);
    imwrite("output.png",m);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
