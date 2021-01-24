#include "CudaUtils.h"
#include "DeviceFunction.h"

namespace dt
{

#define WarpSize 32

    template <typename T, int size>
    __device__ inline void WarpReduceSum(T *val)
    {
#pragma unroll
        for (int offset = WarpSize / 2; offset > 0; offset /= 2)
        {
#pragma unroll
            for (int i = 0; i < size; ++i)
            {
                val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
            }
        }
    }

    template <typename T, int size>
    __device__ inline void BlockReduceSum(T *val)
    {
        static __shared__ T shared[32 * size];
        int lane = threadIdx.x % WarpSize;
        int wid = threadIdx.x / WarpSize;

        WarpReduceSum<T, size>(val);

        if (lane == 0)
            memcpy(&shared[wid * size], val, sizeof(T) * size);

        __syncthreads();

        if (threadIdx.x < blockDim.x / WarpSize)
            memcpy(val, &shared[lane * size], sizeof(T) * size);
        else
            memset(val, 0, sizeof(T) * size);

        if (wid == 0)
            WarpReduceSum<T, size>(val);
    }

    template <int rows, int cols>
    void inline RankUpdateHessian(float *hostData, double *hessian, double *residual)
    {
        int shift = 0;
        for (int i = 0; i < rows; ++i)
            for (int j = i; j < cols; ++j)
            {
                float value = hostData[shift++];
                if (j == rows)
                    residual[i] = value;
                else
                    hessian[j * rows + i] = hessian[i * rows + j] = value;
            }
    }

    template <typename T>
    __device__ __forceinline__ T interpolateBiLinear(
        const cv::cuda::PtrStep<T> &map,
        const float &x, const float &y)
    {
        int u = static_cast<int>(std::floor(x));
        int v = static_cast<int>(std::floor(y));
        float cox = x - u;
        float coy = y - v;
        return (map.ptr(v)[u] * (1 - cox) + map.ptr(v)[u + 1] * cox) * (1 - coy) +
               (map.ptr(v + 1)[u] * (1 - cox) + map.ptr(v + 1)[u + 1] * cox) * coy;
    }

    struct se3StepRGBResidualFunctor
    {
        int w, h, n;
        float gradTh;
        float fx, fy, cx, cy;
        Sophus::SE3f T;
        cv::cuda::PtrStep<Eigen::Vector4f> refVert;
        // cv::cuda::PtrStep<Eigen::Vector4f> refPtWarped;
        cv::cuda::PtrStep<float> refInt;
        cv::cuda::PtrStep<float> currInt;
        cv::cuda::PtrStep<float> currGx;
        cv::cuda::PtrStep<float> currGy;

        mutable cv::cuda::PtrStep<float> out;
        mutable cv::cuda::PtrStep<Eigen::Vector4f> refResidual;

        __device__ __forceinline__ bool findCorresp(
            const int &x, const int &y,
            float &residual,
            float &gx,
            float &gy) const
        {
            Eigen::Vector4f pt = refVert.ptr(y)[x];

            if (pt(3) > 0)
            {
                Eigen::Vector3f ptWarped = T * pt.head<3>();

                float u = fx * ptWarped(0) / ptWarped(2) + cx;
                float v = fy * ptWarped(1) / ptWarped(2) + cy;

                if (u > 0 && v > 0 && u < w - 1 && v < h - 1)
                {
                    auto refVal = refInt.ptr(y)[x];
                    auto currVal = interpolateBiLinear(currInt, u, v);

                    residual = currVal - refVal;
                    gx = interpolateBiLinear(currGx, u, v);
                    gy = interpolateBiLinear(currGy, u, v);

                    return sqrt(gx * gx) > gradTh &&
                           sqrt(gy * gy) > gradTh &&
                           isfinite(residual);
                }
            }

            return false;
        }

        __device__ __forceinline__ void computeResidual(const int &k, float *res) const
        {
            const int y = k / w;
            const int x = k - y * w;

            float residual = 0.f;
            float gx, gy;

            bool correspFound = findCorresp(x, y, residual, gx, gy);

            res[0] = correspFound ? residual : 0.f;
            res[1] = correspFound ? 1.0 : 0.f;

            refResidual.ptr(y)[x] = Eigen::Vector4f(residual, gx, gy, (float)correspFound - 0.5f);
        }

        __device__ __forceinline__ void operator()() const
        {
            float sum[2] = {0, 0};
            float res[2];
            for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
            {
                computeResidual(k, res);

                sum[0] += res[0];
                sum[1] += res[1];
            }

            BlockReduceSum<float, 2>(sum);

            if (threadIdx.x == 0)
            {
                out.ptr(blockIdx.x)[0] = sum[0];
                out.ptr(blockIdx.x)[1] = sum[1];
            }
        }
    };

    struct se3StepRGBFunctor
    {
        int w, h, n;
        float fx, fy;
        float huberTh;
        Sophus::SE3f T;
        cv::cuda::PtrStep<Eigen::Vector4f> refVert;
        cv::cuda::PtrStep<Eigen::Vector4f> refResidual;

        mutable cv::cuda::PtrStep<float> out;

        __device__ __forceinline__ void computeJacobian(const int &k, float *sum) const
        {
            const int y = k / w;
            const int x = k - y * w;
            const Eigen::Vector4f &res = refResidual.ptr(y)[x];
            Eigen::Matrix<float, 7, 1> row = Eigen::Matrix<float, 7, 1>::Zero();

            if (res(3) > 0)
            {
                Eigen::Vector3f pt = T * refVert.ptr(y)[x].head<3>();
                float zInv = 1.0f / pt(2);
                float wt = 1.0f;

                if (abs(res(0)) > huberTh)
                {
                    wt = huberTh / abs(res(0));
                }

                float dx = wt * res(1);
                float dy = wt * res(2);
                float r = wt * res(0);

                row[0] = dx * fx * zInv;
                row[1] = dy * fy * zInv;
                row[2] = -(row[0] * pt(0) + row[1] * pt(1)) * zInv;
                row[3] = row[2] * pt(1) - dy * fy;
                row[4] = -row[2] * pt(0) + dx * fx;
                row[5] = -row[0] * pt(1) + row[1] * pt(0);
                row[6] = -r;
            }

            int count = 0;
#pragma unroll
            for (int i = 0; i < 7; ++i)
#pragma unroll
                for (int j = i; j < 7; ++j)
                    sum[count++] = row[i] * row[j];
        }

        __device__ __forceinline__ void operator()() const
        {
            float sum[29];
            memset(&sum[0], 0, sizeof(float) * 29);

            float temp[29];
            for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
            {
                computeJacobian(k, temp);
#pragma unroll
                for (int i = 0; i < 29; ++i)
                    sum[i] += temp[i];
            }

            BlockReduceSum<float, 29>(sum);

            if (threadIdx.x == 0)
#pragma unroll
                for (int i = 0; i < 29; ++i)
                    out.ptr(blockIdx.x)[i] = sum[i];
        }
    };

    struct VarianceEstimator
    {
        int w, h, n;
        float meanEstimated;
        cv::cuda::PtrStep<Eigen::Vector4f> residual;

        mutable cv::cuda::PtrStep<float> out;

        __device__ __forceinline__ void operator()() const
        {
            float sum[1] = {0};
            for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < n; k += gridDim.x * blockDim.x)
            {
                int y = k / w;
                int x = k - y * w;

                const Eigen::Vector4f &res = residual.ptr(y)[x];

                if (res(3) > 0)
                {
                    sum[0] += ((res(0) - meanEstimated) * (res(0) - meanEstimated));
                }
            }

            BlockReduceSum<float, 1>(sum);

            if (threadIdx.x == 0)
                out.ptr(blockIdx.x)[0] = sum[0];
        }
    };

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
        double *residual)
    {
        const int w = image_src.cols;
        const int h = image_src.rows;

        se3StepRGBResidualFunctor functor;
        functor.w = w;
        functor.h = h;
        functor.n = w * h;
        functor.refInt = image_ref;
        functor.currInt = image_src;
        functor.currGx = image_gx;
        functor.currGy = image_gy;
        functor.refVert = vmap_ref;
        functor.gradTh = gradThresh;
        functor.T = T.cast<float>();
        functor.refResidual = RESVec;
        functor.fx = K(0, 0);
        functor.fy = K(1, 1);
        functor.cx = K(0, 2);
        functor.cy = K(1, 2);
        functor.out = RES_sum;

        callDeviceFunctor<<<96, 224>>>(functor);
        cv::cuda::reduce(RES_sum, SE3_out, 0, cv::REDUCE_SUM);
        cv::Mat hostData(SE3_out);

        double residualSum = hostData.ptr<float>(0)[0];
        double numResidual = hostData.ptr<float>(0)[1];

        VarianceEstimator estimator;
        estimator.w = w;
        estimator.h = h;
        estimator.n = w * h;
        estimator.meanEstimated = (float)(residualSum / numResidual);
        estimator.residual = RESVec;
        estimator.out = VAR_sum;

        callDeviceFunctor<<<96, 224>>>(estimator);
        cv::cuda::reduce(VAR_sum, VAR_out, 0, cv::REDUCE_SUM);
        VAR_out.download(hostData);

        double squaredDeviationSum = hostData.ptr<float>(0)[0];
        double varEstimated = sqrt(squaredDeviationSum / (numResidual - 1));

        se3StepRGBFunctor sfunctor;
        sfunctor.w = w;
        sfunctor.h = h;
        sfunctor.n = w * h;
        sfunctor.huberTh = 4.685 * varEstimated;
        sfunctor.refVert = vmap_ref;
        sfunctor.T = T.cast<float>();
        sfunctor.refResidual = RESVec;
        sfunctor.fx = K(0, 0);
        sfunctor.fy = K(1, 1);
        sfunctor.out = SE3_sum;

        callDeviceFunctor<<<96, 224>>>(sfunctor);
        cv::cuda::reduce(SE3_sum, SE3_out, 0, cv::REDUCE_SUM);

        SE3_out.download(hostData);
        RankUpdateHessian<6, 7>(hostData.ptr<float>(0), hessian, residual);

        res[0] = hostData.ptr<float>(0)[27];
        res[1] = hostData.ptr<float>(0)[28];
    }

    struct IcpStepFunctor
    {
        cv::cuda::PtrStep<Eigen::Vector4f> vmap_curr;
        cv::cuda::PtrStep<Eigen::Vector4f> nmap_curr;
        cv::cuda::PtrStep<Eigen::Vector4f> vmap_last;
        cv::cuda::PtrStep<Eigen::Vector4f> nmap_last;
        cv::cuda::PtrStep<float> curv_last;
        cv::cuda::PtrStep<float> curv_curr;
        int cols, rows, N;
        float fx, fy, cx, cy;
        float angleTH, distTH;
        Sophus::SE3f T_last_curr;
        mutable cv::cuda::PtrStep<float> out;

        __device__ __forceinline__ bool ProjectPoint(
            int &x, int &y,
            Eigen::Vector3f &v_curr,
            Eigen::Vector3f &n_last,
            Eigen::Vector3f &v_last) const
        {
            Eigen::Vector4f v_last_c = vmap_last.ptr(y)[x];
            if (v_last_c(3) < 0)
                return false;

            v_last = T_last_curr * v_last_c.head<3>();

            float invz = 1.0 / v_last(2);
            int u = __float2int_rd(fx * v_last(0) * invz + cx + 0.5);
            int v = __float2int_rd(fy * v_last(1) * invz + cy + 0.5);
            if (u < 1 || v < 1 || u >= cols - 1 || v >= rows - 1)
                return false;

            Eigen::Vector4f v_curr_c = vmap_curr.ptr(v)[u];
            v_curr = v_curr_c.head<3>();
            if (v_curr_c(3) < 0)
                return false;

            Eigen::Vector4f n_last_c = nmap_last.ptr(y)[x];
            n_last = T_last_curr.so3() * n_last_c.head<3>();

            Eigen::Vector4f n_curr_c = nmap_curr.ptr(v)[u];

            float dist = (v_last - v_curr).norm();
            float angle = n_curr_c.head<3>().cross(n_last).norm();

            return (angle < angleTH && dist < distTH && n_last_c(3) > 0 && n_curr_c(3) > 0);
        }

        __device__ __forceinline__ void GetProduct(int &k, float *sum) const
        {
            int y = k / cols;
            int x = k - (y * cols);

            Eigen::Vector3f v_curr, n_last, v_last;
            float row[7] = {0, 0, 0, 0, 0, 0, 0};
            bool found = ProjectPoint(x, y, v_curr, n_last, v_last);

            if (found)
            {
                row[6] = n_last.dot(v_curr - v_last);
                float hw = 1; //fabs(row[6]) < 0.3 ? 1 : 0.3 / fabs(row[6]);
                row[6] *= hw;
                *(Eigen::Vector3f *)&row[0] = hw * n_last;
                *(Eigen::Vector3f *)&row[3] = hw * v_last.cross(n_last);
            }

            int count = 0;
#pragma unroll
            for (int i = 0; i < 7; ++i)
#pragma unroll
                for (int j = i; j < 7; ++j)
                    sum[count++] = row[i] * row[j];
            sum[count] = (float)found;
        }

        __device__ __forceinline__ void operator()() const
        {
            float sum[29] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            float val[29];
            for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
            {
                GetProduct(k, val);

#pragma unroll
                for (int i = 0; i < 29; ++i)
                {
                    sum[i] += val[i];
                }
            }

            BlockReduceSum<float, 29>(sum);

            if (threadIdx.x == 0)
            {
#pragma unroll
                for (int i = 0; i < 29; ++i)
                    out.ptr(blockIdx.x)[i] = sum[i];
            }
        }
    };

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
        double *residual)
    {
        int cols = vmap_src.cols;
        int rows = vmap_src.rows;

        IcpStepFunctor icpStep;
        icpStep.out = SE3_sum;
        icpStep.vmap_curr = vmap_src;
        icpStep.nmap_curr = nmap_src;
        icpStep.vmap_last = vmap_ref;
        icpStep.nmap_last = nmap_ref;
        icpStep.cols = cols;
        icpStep.rows = rows;
        icpStep.N = cols * rows;
        icpStep.T_last_curr = init.cast<float>();
        icpStep.angleTH = 0.6;
        icpStep.distTH = 0.1;
        icpStep.fx = K(0, 0);
        icpStep.fy = K(1, 1);
        icpStep.cx = K(0, 2);
        icpStep.cy = K(1, 2);

        callDeviceFunctor<<<96, 224>>>(icpStep);
        cv::cuda::reduce(SE3_sum, SE3_out, 0, cv::REDUCE_SUM);

        cv::Mat hostData(SE3_out);
        RankUpdateHessian<6, 7>(hostData.ptr<float>(0), hessian, residual);

        res[0] = hostData.ptr<float>(0)[27];
        res[1] = hostData.ptr<float>(0)[27];
    }

} // namespace slam