#pragma once

#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

namespace dt
{
    namespace internal
    {
        struct DenseTrackingImpl;
    }

    class DenseTracker
    {
    public:
        DenseTracker(int w, int h, Eigen::Matrix3f K,
                     int maxLvl, bool bRGB, bool bIcp,
                     float depthClipping, float gradTh);
        void SetReferenceImage(const cv::Mat &imGray);
        void SetReferenceDepth(const cv::Mat &imDepth);
        void SetTrackingImage(const cv::Mat &imGray);
        void SetTrackingDepth(const cv::Mat &imDepth);
        void SetReferenceModel(const cv::cuda::GpuMat vmap);

        Sophus::SE3d ComputeSE3(Sophus::SE3d init, bool swap, int maxLevel = 0);
        Eigen::Matrix<double, 6, 6> GetCovarianceMatrix();

        void SwapFrameBuffer();
        bool TrackingFailed();
        void DisplayDebugImages(int waitTime = 0);

    private:
        std::shared_ptr<internal::DenseTrackingImpl> impl;
        DenseTracker(const DenseTracker &) = default;
    };

} // namespace dt