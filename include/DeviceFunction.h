#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

namespace dt
{
    void ComputeSingleStepRGB(
        cv::cuda::GpuMat image_src,
        cv::cuda::GpuMat image_ref,
        cv::cuda::GpuMat image_gx,
        cv::cuda::GpuMat image_gy,
        cv::cuda::GpuMat vmap_ref,
        Sophus::SE3d T,
        Eigen::Matrix3f K,
        float gradThresh,
        cv::cuda::GpuMat RESVec,
        cv::cuda::GpuMat VAR_sum,
        cv::cuda::GpuMat VAR_out,
        cv::cuda::GpuMat RES_sum,
        cv::cuda::GpuMat RES_out,
        cv::cuda::GpuMat SE3_sum,
        cv::cuda::GpuMat SE3_out,
        double *res,
        double *hessian,
        double *residual);

    void ComputeSingleStepDepth(
        cv::cuda::GpuMat vmap_src,
        cv::cuda::GpuMat nmap_src,
        cv::cuda::GpuMat vmap_ref,
        cv::cuda::GpuMat nmap_ref,
        Sophus::SE3d init,
        Eigen::Matrix3f K,
        cv::cuda::GpuMat SE3_sum,
        cv::cuda::GpuMat SE3_out,
        double *res,
        double *hessian,
        double *residual);

} // namespace dt