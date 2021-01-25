#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

namespace dt
{

    void PyrDownDepth(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
    void ComputeVertexMap(const cv::cuda::GpuMat depth, cv::cuda::GpuMat vmap,
                          const float invfx, const float invfy, const float cx, const float cy,
                          const float cut_off);
    void ComputeNormalMap(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat nmap);
    void DepthToInvDepth(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &invDepth);
    void RenderScene(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat &image);
    void PyrDownImage(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
    void PyrDownVec4f(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);

    void GetGradientSobel(cv::cuda::GpuMat img, cv::cuda::GpuMat &gx, cv::cuda::GpuMat &gy);
    void ComputeImageGradientCD(const cv::cuda::GpuMat image, cv::cuda::GpuMat &gx, cv::cuda::GpuMat &gy);

} // namespace dt