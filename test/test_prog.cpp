#include "DenseTracking.h"
#include <opencv2/opencv.hpp>

float bilinear_interp(
    const cv::Mat &map,
    const float &x,
    const float &y)
{
    int u = static_cast<int>(std::floor(x));
    int v = static_cast<int>(std::floor(y));
    float cox = x - u;
    float coy = y - v;
    return (map.ptr<uchar>(v)[u] * (1 - cox) + map.ptr<uchar>(v)[u + 1] * cox) * (1 - coy) +
           (map.ptr<uchar>(v + 1)[u] * (1 - cox) + map.ptr<uchar>(v + 1)[u + 1] * cox) * coy;
}

cv::Mat GetResidualImage(cv::Mat src_img, cv::Mat ref_img,
                         cv::Mat ref_depth, Eigen::Matrix3f K,
                         Sophus::SE3d T)
{
    cv::Mat res_img(src_img.size(), src_img.type());
    res_img.setTo(0);
    for (int y = 0; y < ref_img.rows; ++y)
    {
        uchar *ref_img_row = ref_img.ptr<uchar>(y);
        uchar *src_img_row = src_img.ptr<uchar>(y);
        float *ref_depth_row = ref_depth.ptr<float>(y);

        for (int x = 0; x < ref_img.cols; ++x)
        {
            float z = ref_depth_row[x];
            if (z > 0)
            {
                Eigen::Vector3d pt;
                pt[0] = (x - K(0, 2)) / K(0, 0) * z;
                pt[1] = (y - K(1, 2)) / K(1, 1) * z;
                pt[2] = z;
                auto ptWarped = T * pt;
                float u = K(0, 0) * ptWarped[0] / ptWarped[2] + K(0, 2);
                float v = K(1, 1) * ptWarped[1] / ptWarped[2] + K(1, 2);
                if (u >= 0 && v >= 0 && u < src_img.cols && v < src_img.rows)
                {
                    res_img.ptr(y)[x] = fabs(ref_img_row[x] - bilinear_interp(src_img, u, v));
                }
            }
        }
    }

    return res_img;
}

void GetGradientCD(cv::Mat img, cv::Mat &gx, cv::Mat &gy)
{
    if (gx.empty())
        gx.create(img.size(), CV_32FC1);

    if (gy.empty())
        gy.create(img.size(), CV_32FC1);

    gx.setTo(0);
    gy.setTo(0);

    for (int y = 1; y < img.rows - 1; ++y)
    {
        for (int x = 1; x < img.cols - 1; ++x)
        {
            gx.ptr<float>(y)[x] = 0.5 * ((float)img.ptr<uchar>(y)[x + 1] - img.ptr<uchar>(y)[x - 1]);
            gy.ptr<float>(y)[x] = 0.5 * ((float)img.ptr<uchar>(y + 1)[x] - img.ptr<uchar>(y - 1)[x]);
        }
    }
}

int main(int argc, char **argv)
{
    int w = 640;
    int h = 480;
    Eigen::Matrix3f K;
    K << 577.59, 0, 318.90, 0, 578.729, 242.68, 0, 0, 1;
    dt::DenseTracker tracker(
        w, h, K, 5, true, false, 5.0, 8, 0.7);

    cv::Mat ref_img = cv::imread("test/ref_rgb.jpg", -1);
    cv::Mat src_img = cv::imread("test/src_rgb.jpg", -1);
    cv::Mat ref_depth = cv::imread("test/ref_depth.png", -1);
    cv::Mat src_depth = cv::imread("test/src_depth.png", -1);

    ref_depth.convertTo(ref_depth, CV_32FC1, 1 / 5000.0);
    src_depth.convertTo(src_depth, CV_32FC1, 1 / 5000.0);

    tracker.SetReferenceDepth(ref_depth);
    tracker.SetReferenceImage(ref_img);
    tracker.SetTrackingDepth(src_depth);
    tracker.SetTrackingImage(src_img);
    auto rt = tracker.ComputeSE3(Sophus::SE3d(), false);
    tracker.DisplayDebugImages();

    auto res_img = GetResidualImage(src_img, ref_img, ref_depth, K, rt);

    cv::Mat gx, gy;
    GetGradientCD(src_img, gx, gy);
    cv::imshow("res_img_grad_x", gx);
    cv::imshow("res_img_grad_y", gy);

    cv::imshow("res_img", res_img);
    cv::imshow("ref_img", ref_img);
    cv::imshow("src_img", src_img);
    cv::imshow("ref_depth", ref_depth);
    cv::imshow("src_depth", src_depth);
    cv::waitKey(0);

    std::cout << rt.matrix3x4() << std::endl;
}
