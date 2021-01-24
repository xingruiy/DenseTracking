#include "DenseTracking.h"
#include <opencv2/opencv.hpp>

cv::Mat GetResidualImage(cv::Mat src_img, cv::Mat ref_img,
                         cv::Mat ref_depth, Sophus::SE3d T)
{
    cv::Mat res_img(src_img.size(), src_img.type());
    for (int y = 0; y < ref_img.rows; ++y)
    {
        for (int x = 0; x < ref_img.cols; ++x)
        {
        }
    }
}

int main(int argc, char **argv)
{
    int w = 640;
    int h = 480;
    Eigen::Matrix3f K;
    K << 580, 0, 320, 0, 580, 240, 0, 0, 1;
    dt::DenseTracker tracker(
        w, h, K, 5, false, true, 5.0, 16);

    cv::Mat ref_img = cv::imread("test/ref_rgb.jpg", -1);
    cv::Mat src_img = cv::imread("test/src_rgb.jpg", -1);
    cv::Mat ref_depth = cv::imread("test/ref_depth.png", -1);
    cv::Mat src_depth = cv::imread("test/src_depth.png", -1);

    ref_depth.convertTo(ref_depth, CV_32FC1, 1 / 1000.0);
    src_depth.convertTo(src_depth, CV_32FC1, 1 / 1000.0);

    tracker.SetReferenceDepth(ref_depth);
    tracker.SetReferenceImage(ref_img);
    tracker.SetTrackingDepth(src_depth);
    tracker.SetTrackingImage(src_img);
    auto rt = tracker.ComputeSE3(Sophus::SE3d(), false);

    auto res_img = GetResidualImage(src_img, ref_img, ref_depth, rt);

    cv::imshow("res_img", res_img);
    cv::imshow("ref_img", ref_img);
    cv::imshow("src_img", src_img);
    cv::imshow("ref_depth", ref_depth);
    cv::imshow("src_depth", src_depth);
    cv::waitKey(0);

    std::cout << rt.matrix3x4() << std::endl;
}