#include "ImageProc.h"
#include "DenseTracking.h"
#include "DeviceFunction.h"

namespace dt
{
    namespace internal
    {
        struct RGBDFrameStruct
        {
            size_t id;
            std::vector<cv::cuda::GpuMat> vmap;
            std::vector<cv::cuda::GpuMat> nmap;
            std::vector<cv::cuda::GpuMat> depth;
            std::vector<cv::cuda::GpuMat> intensity;
            std::vector<cv::cuda::GpuMat> warpedPoints;
            std::vector<cv::cuda::GpuMat> intensityGradX;
            std::vector<cv::cuda::GpuMat> intensityGradY;
        };

        struct DenseTrackingImpl
        {
            cv::cuda::GpuMat bufferFloat96x29;
            cv::cuda::GpuMat bufferFloat96x3;
            cv::cuda::GpuMat bufferFloat96x2;
            cv::cuda::GpuMat bufferFloat96x1;
            cv::cuda::GpuMat bufferFloat1x29;
            cv::cuda::GpuMat bufferFloat1x3;
            cv::cuda::GpuMat bufferFloat1x2;
            cv::cuda::GpuMat bufferFloat1x1;
            cv::cuda::GpuMat bufferVec4HW;
            cv::cuda::GpuMat bufferFloatHW;

            bool trackingGood;
            Eigen::Matrix<double, 6, 6> mHessian;

            int lvl_pyr;
            bool useRGB, useICP;
            float gradThresh;
            float depthClipping;
            std::vector<int> w_pyr, h_pyr;
            std::vector<Eigen::Matrix3f> K_pyr;

            RGBDFrameStruct source;
            RGBDFrameStruct reference;
            float depthWeight;

            DenseTrackingImpl(int w, int h, Eigen::Matrix3f K,
                              int maxLvl, bool bRGB, bool bIcp,
                              float clipping, float gradTh,
                              float weight)
                : lvl_pyr(maxLvl), useRGB(bRGB), useICP(bIcp),
                  depthClipping(clipping), gradThresh(gradTh),
                  depthWeight(weight)
            {
                w_pyr.push_back(w);
                h_pyr.push_back(h);
                K_pyr.push_back(K);
                for (int lvl = 1; lvl < lvl_pyr; ++lvl)
                {
                    w_pyr.push_back(w_pyr[lvl - 1] / 2);
                    h_pyr.push_back(h_pyr[lvl - 1] / 2);
                    Eigen::Matrix3f _K = K_pyr[lvl - 1] / 2;
                    _K(2, 2) = 1;
                    K_pyr.push_back(_K);
                }

                InitializeFrame(source);
                InitializeFrame(reference);

                bufferFloat96x29.create(96, 29, CV_32FC1);
                bufferFloat96x3.create(96, 3, CV_32FC1);
                bufferFloat96x2.create(96, 2, CV_32FC1);
                bufferFloat96x1.create(96, 1, CV_32FC1);
                bufferFloat1x29.create(1, 29, CV_32FC1);
                bufferFloat1x3.create(1, 2, CV_32FC1);
                bufferFloat1x2.create(1, 2, CV_32FC1);
                bufferFloat1x1.create(1, 1, CV_32FC1);
                bufferVec4HW.create(h, w, CV_32FC4);
                bufferFloatHW.create(h, w, CV_32FC1);
            }

            void InitializeFrame(RGBDFrameStruct &out)
            {
                out.depth.resize(lvl_pyr);
                out.intensity.resize(lvl_pyr);
                out.intensityGradX.resize(lvl_pyr);
                out.intensityGradY.resize(lvl_pyr);
                out.warpedPoints.resize(lvl_pyr);
                out.vmap.resize(lvl_pyr);
                out.nmap.resize(lvl_pyr);

                for (int lvl = 0; lvl < lvl_pyr; ++lvl)
                {
                    auto hLvl = h_pyr[lvl];
                    auto wLvl = w_pyr[lvl];

                    out.depth[lvl].create(hLvl, wLvl, CV_32FC1);
                    out.intensity[lvl].create(hLvl, wLvl, CV_32FC1);
                    out.intensityGradX[lvl].create(hLvl, wLvl, CV_32FC1);
                    out.intensityGradY[lvl].create(hLvl, wLvl, CV_32FC1);
                    out.warpedPoints[lvl].create(hLvl, wLvl, CV_32FC4);
                    out.vmap[lvl].create(hLvl, wLvl, CV_32FC4);
                    out.nmap[lvl].create(hLvl, wLvl, CV_32FC4);
                }
            }

            void SetReferenceImage(cv::Mat img)
            {
                cv::Mat imgF;
                img.convertTo(imgF, CV_32FC1);

                for (int lvl = 0; lvl < lvl_pyr; ++lvl)
                {
                    if (lvl == 0)
                        reference.intensity[0].upload(imgF);
                    else
                        PyrDownImage(reference.intensity[lvl - 1], reference.intensity[lvl]);
                }
            }

            void SetReferenceDepth(cv::Mat depth)
            {
                for (int lvl = 0; lvl < lvl_pyr; ++lvl)
                {
                    if (lvl == 0)
                    {
                        bufferFloatHW.upload(depth);
                        DepthToInvDepth(bufferFloatHW, reference.depth[lvl]);
                    }
                    else
                        PyrDownDepth(reference.depth[lvl - 1], reference.depth[lvl]);

                    float invfx = 1.0 / K_pyr[lvl](0, 0);
                    float invfy = 1.0 / K_pyr[lvl](1, 1);
                    float cx = K_pyr[lvl](0, 2);
                    float cy = K_pyr[lvl](1, 2);

                    ComputeVertexMap(reference.depth[lvl], reference.vmap[lvl], invfx, invfy, cx, cy, depthClipping);
                    ComputeNormalMap(reference.vmap[lvl], reference.nmap[lvl]);
                }
            }

            void SetTrackingImage(cv::Mat img)
            {
                cv::Mat imgF;
                img.convertTo(imgF, CV_32FC1);

                for (int lvl = 0; lvl < lvl_pyr; ++lvl)
                {
                    if (lvl == 0)
                        source.intensity[lvl].upload(imgF);
                    else
                        PyrDownImage(source.intensity[lvl - 1], source.intensity[lvl]);

                    // ComputeImageGradientCD(source.intensity[lvl], source.intensityGradX[lvl], source.intensityGradY[lvl]);
                    GetGradientSobel(source.intensity[lvl], source.intensityGradX[lvl], source.intensityGradY[lvl]);
                }
            }

            void SetTrackingDepth(cv::Mat depth)
            {
                for (int lvl = 0; lvl < lvl_pyr; ++lvl)
                {
                    if (lvl == 0)
                    {
                        bufferFloatHW.upload(depth);
                        DepthToInvDepth(bufferFloatHW, source.depth[lvl]);
                    }
                    else
                        PyrDownDepth(source.depth[lvl - 1], source.depth[lvl]);

                    float invfx = 1.0 / K_pyr[lvl](0, 0);
                    float invfy = 1.0 / K_pyr[lvl](1, 1);
                    float cx = K_pyr[lvl](0, 2);
                    float cy = K_pyr[lvl](1, 2);

                    ComputeVertexMap(source.depth[lvl], source.vmap[lvl], invfx, invfy, cx, cy, depthClipping);
                    ComputeNormalMap(source.vmap[lvl], source.nmap[lvl]);
                }
            }

            void SetReferenceModel(cv::cuda::GpuMat vmap)
            {
                vmap.copyTo(reference.vmap[0]);
                for (int lvl = 0; lvl < lvl_pyr; ++lvl)
                {
                    if (lvl != 0)
                        PyrDownVec4f(reference.vmap[lvl - 1], reference.vmap[lvl]);
                    ComputeNormalMap(reference.vmap[lvl], reference.nmap[lvl]);
                }
            }

            Sophus::SE3d ComputeSE3(Sophus::SE3d init, bool swap, int maxLevel)
            {
                int nIteration = 0;
                trackingGood = false;
                int nSuccessfulIteration = 0;

                Sophus::SE3d estimate = init;
                std::vector<int> vIterations = {15, 15, 15, 15, 15};
                std::vector<float> incIterMin = {1e-8, 1e-8, 1e-8, 1e-8, 1e-8};
                std::vector<float> LMStepInit = {0, 0, 0, 0, 0};

                for (int lvl = lvl_pyr - 1; lvl >= maxLevel; --lvl)
                {
                    float LMStep = LMStepInit[lvl];
                    float lastError = std::numeric_limits<float>::max();
                    for (int iter = 0; iter < vIterations[lvl]; ++iter)
                    {
                        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
                        Eigen::Matrix<double, 6, 1> residual = Eigen::Matrix<double, 6, 1>::Zero();
                        double resSum[2] = {0, 0};

                        if (useRGB && !useICP)
                            ComputeSingleStepRGB(
                                source.intensity[lvl],
                                reference.intensity[lvl],
                                source.intensityGradX[lvl],
                                source.intensityGradY[lvl],
                                reference.vmap[lvl],
                                estimate,
                                K_pyr[lvl],
                                gradThresh,
                                bufferVec4HW,
                                bufferFloat96x1,
                                bufferFloat1x1,
                                bufferFloat96x2,
                                bufferFloat1x2,
                                bufferFloat96x29,
                                bufferFloat1x29,
                                resSum,
                                hessian.data(),
                                residual.data());
                        else if (!useRGB && useICP)
                            ComputeSingleStepDepth(
                                source.vmap[lvl],
                                source.nmap[lvl],
                                reference.vmap[lvl],
                                reference.nmap[lvl],
                                estimate,
                                K_pyr[lvl],
                                bufferFloat96x29,
                                bufferFloat1x29,
                                resSum,
                                hessian.data(),
                                residual.data());

                        else
                            ComputeSingleStepRGBDLinear(
                                lvl,
                                estimate,
                                resSum,
                                hessian.data(),
                                residual.data());

                        for (int i = 0; i < 6; i++)
                            hessian(i, i) *= 1 + LMStep;

                        float error = sqrt(resSum[0]) / (resSum[1] + 6);
                        auto update = hessian.ldlt().solve(residual);

                        if (std::isnan(update(0)))
                        {
                            trackingGood = false;
                            return Sophus::SE3d();
                        }

                        if (error < lastError)
                        {
                            estimate = Sophus::SE3d::exp(update) * estimate;
                            if (update.dot(update) < incIterMin[lvl])
                                break;
                            lastError = error;

                            if (LMStep < 0.2)
                                LMStep = 0;
                            else
                                LMStep *= 0.5;
                        }
                        else
                        {
                            if (LMStep == 0)
                                LMStep = 0.2;
                            else
                                LMStep *= 2;
                        }

                        nIteration++;
                    }
                }

                if (swap)
                {
                    std::swap(reference, source);
                }

                trackingGood = true;
                return estimate;
            }

            void SwapFrameBuffer()
            {
                std::swap(reference, source);
            }

            void ComputeSingleStepRGBDLinear(
                const int lvl,
                const Sophus::SE3d &T,
                double *resSum,
                double *hessian,
                double *residual)
            {
                double w = 0.01;

                Eigen::Map<Eigen::Matrix<double, 6, 6>> hessianMapped(hessian);
                Eigen::Map<Eigen::Matrix<double, 6, 1>> residualMapped(residual);

                Eigen::Matrix<double, 6, 6> hessianBuffer;
                Eigen::Matrix<double, 6, 1> residualBuffer;

                float dw = depthWeight;
                float rgbw = 1 - depthWeight;

                ComputeSingleStepDepth(
                    source.vmap[lvl],
                    source.nmap[lvl],
                    reference.vmap[lvl],
                    reference.nmap[lvl],
                    T,
                    K_pyr[lvl],
                    bufferFloat96x29,
                    bufferFloat1x29,
                    resSum,
                    hessianBuffer.data(),
                    residualBuffer.data());

                hessianMapped += dw * dw * hessianBuffer;
                residualMapped += dw * residualBuffer;

                ComputeSingleStepRGB(
                    source.intensity[lvl],
                    reference.intensity[lvl],
                    source.intensityGradX[lvl],
                    source.intensityGradY[lvl],
                    reference.vmap[lvl],
                    T,
                    K_pyr[lvl],
                    gradThresh,
                    bufferVec4HW,
                    bufferFloat96x1,
                    bufferFloat1x1,
                    bufferFloat96x2,
                    bufferFloat1x2,
                    bufferFloat96x29,
                    bufferFloat1x29,
                    resSum,
                    hessianBuffer.data(),
                    residualBuffer.data());

                hessianMapped += rgbw * rgbw * hessianBuffer;
                residualMapped += rgbw * residualBuffer;

                mHessian = hessianMapped;
            }

            Eigen::Matrix<double, 6, 6> GetCovarianceMatrix()
            {
                return mHessian.cast<double>().lu().inverse();
            }

            void DisplayDebugImages(int waitTime)
            {
                cv::Mat out;
                cv::Mat(source.intensity[0]).convertTo(out, CV_8UC1);
                cv::imshow("source image", out);
                cv::imshow("source depth", cv::Mat(source.depth[0]));
                cv::Mat(reference.intensity[0]).convertTo(out, CV_8UC1);
                cv::imshow("reference image", out);
                cv::imshow("reference vmap", cv::Mat(reference.vmap[0]));
                cv::imshow("source intensity grad x", cv::Mat(source.intensityGradX[0]));
                cv::imshow("source intensity grad y", cv::Mat(source.intensityGradY[0]));
                cv::waitKey(waitTime);
            }
        };
    } // namespace internal

    DenseTracker::DenseTracker(int w, int h, Eigen::Matrix3f K,
                               int maxLvl, bool bRGB, bool bIcp,
                               float depthClipping, float gradTh,
                               float depthWeight)
        : impl(new internal::DenseTrackingImpl(w, h, K, maxLvl, bRGB,
                                               bIcp, depthClipping, gradTh, depthWeight))
    {
    }

    void DenseTracker::SetReferenceImage(const cv::Mat &imGray)
    {
        impl->SetReferenceImage(imGray);
    }

    void DenseTracker::SetReferenceDepth(const cv::Mat &imDepth)
    {
        impl->SetReferenceDepth(imDepth);
    }

    void DenseTracker::SetTrackingImage(const cv::Mat &imGray)
    {
        impl->SetTrackingImage(imGray);
    }

    void DenseTracker::SetTrackingDepth(const cv::Mat &imDepth)
    {
        impl->SetTrackingDepth(imDepth);
    }

    void DenseTracker::SetReferenceModel(const cv::cuda::GpuMat vmap)
    {
        impl->SetReferenceModel(vmap);
    }

    Sophus::SE3d DenseTracker::ComputeSE3(Sophus::SE3d init, bool swap, int maxLevel)
    {
        return impl->ComputeSE3(init, swap, maxLevel);
    }

    void DenseTracker::SwapFrameBuffer()
    {
        impl->SwapFrameBuffer();
    }

    Eigen::Matrix<double, 6, 6> DenseTracker::GetCovarianceMatrix()
    {
        return impl->GetCovarianceMatrix();
    }

    bool DenseTracker::TrackingFailed()
    {
        return !impl->trackingGood;
    }

    void DenseTracker::DisplayDebugImages(int waitTime)
    {
        impl->DisplayDebugImages(waitTime);
    }

} // namespace dt