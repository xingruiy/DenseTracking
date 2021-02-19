#include "CudaUtils.h"
#include "ImageProc.h"
#include <opencv2/cudafilters.hpp>

#define DEPTH_MAX 8.f
#define DEPTH_MIN 0.2f

namespace dt
{

    __global__ void ComputeImageGradientCD_kernel(
        const cv::cuda::PtrStepSz<float> src,
        cv::cuda::PtrStep<float> gx,
        cv::cuda::PtrStep<float> gy)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x > src.cols - 1 || y > src.rows - 1)
            return;

        if (x < 1 || y < 1 || x > src.cols - 2 || y > src.rows - 2)
        {
            gx.ptr(y)[x] = gy.ptr(y)[x] = 0;
        }
        else
        {
            gx.ptr(y)[x] = (src.ptr(y)[x + 1] - src.ptr(y)[x - 1]) * 0.5f;
            gy.ptr(y)[x] = (src.ptr(y + 1)[x] - src.ptr(y - 1)[x]) * 0.5f;
        }
    }

    void ComputeImageGradientCD(const cv::cuda::GpuMat image,
                                cv::cuda::GpuMat &gx,
                                cv::cuda::GpuMat &gy)
    {
        if (gx.empty())
            gx.create(image.size(), CV_32FC1);
        if (gy.empty())
            gy.create(image.size(), CV_32FC1);

        dim3 block(8, 8);
        dim3 grid(cv::divUp(image.cols, block.x), cv::divUp(image.rows, block.y));

        ComputeImageGradientCD_kernel<<<grid, block>>>(image, gx, gy);
    }

    void GetGradientSobel(cv::cuda::GpuMat img, cv::cuda::GpuMat &gx, cv::cuda::GpuMat &gy)
    {
        auto sobelX = cv::cuda::createSobelFilter(CV_32FC1, CV_32FC1, 1, 0);
        auto sobelY = cv::cuda::createSobelFilter(CV_32FC1, CV_32FC1, 0, 1);
        sobelX->apply(img, gx);
        sobelY->apply(img, gy);
    }

    __device__ __forceinline__ Eigen::Vector<uchar, 4> RenderPoint(
        const Eigen::Vector3f &point,
        const Eigen::Vector3f &normal,
        const Eigen::Vector3f &image,
        const Eigen::Vector3f &lightPos)
    {
        Eigen::Vector3f colour(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
        if (!isnan(point(0)))
        {
            const float Ka = 0.3f;     // ambient coeff
            const float Kd = 0.5f;     // diffuse coeff
            const float Ks = 0.2f;     // specular coeff
            const float n = 20.f;      // specular power
            const float Ax = image(0); // ambient color
            const float Dx = image(1); // diffuse color
            const float Sx = image(2); // specular color
            const float Lx = 1.f;      // light color

            Eigen::Vector3f L = (lightPos - point).normalized();
            Eigen::Vector3f V = (Eigen::Vector3f(0.f, 0.f, 0.f) - point).normalized();
            Eigen::Vector3f R = (2 * normal * (normal.dot(L)) - L).normalized();

            float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, (normal.dot(L))) + Lx * Ks * Sx * pow(fmax(0.f, (R.dot(V))), n);
            colour = Eigen::Vector3f(Ix, Ix, Ix);
        }

        return Eigen::Vector<uchar, 4>(static_cast<uchar>(__saturatef(colour(0)) * 255.f),
                                       static_cast<uchar>(__saturatef(colour(1)) * 255.f),
                                       static_cast<uchar>(__saturatef(colour(2)) * 255.f),
                                       255);
    }

    __global__ void RenderScene_kernel(
        const cv::cuda::PtrStep<Eigen::Vector4f> vmap,
        const cv::cuda::PtrStep<Eigen::Vector4f> nmap,
        const Eigen::Vector3f lightPos,
        cv::cuda::PtrStepSz<Eigen::Vector<uchar, 4>> dst)
    {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= dst.cols || y >= dst.rows)
            return;

        Eigen::Vector3f point = vmap.ptr(y)[x].head<3>();
        Eigen::Vector3f normal = nmap.ptr(y)[x].head<3>();
        Eigen::Vector3f pixel(1.f, 1.f, 1.f);

        dst.ptr(y)[x] = RenderPoint(point, normal, pixel, lightPos);
    }

    void RenderScene(const cv::cuda::GpuMat vmap,
                     const cv::cuda::GpuMat nmap,
                     cv::cuda::GpuMat &image)
    {
        if (image.empty())
            image.create(vmap.size(), CV_8UC4);

        dim3 block(8, 8);
        dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

        RenderScene_kernel<<<grid, block>>>(vmap, nmap, Eigen::Vector3f(5, 5, 5), image);
    }

    __global__ void DepthToInvDepth_kernel(const cv::cuda::PtrStep<float> depth, cv::cuda::PtrStepSz<float> depth_inv)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x > depth_inv.cols - 1 || y > depth_inv.rows - 1)
            return;

        const float z = depth.ptr(y)[x];
        if (z > DEPTH_MIN && z < DEPTH_MAX)
            depth_inv.ptr(y)[x] = 1.0 / z;
        else
            depth_inv.ptr(y)[x] = 0;
    }

    void DepthToInvDepth(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &depth_inv)
    {
        if (depth_inv.empty())
            depth_inv.create(depth.size(), depth.type());

        dim3 block(8, 8);
        dim3 grid(cv::divUp(depth.cols, block.x), cv::divUp(depth.rows, block.y));

        DepthToInvDepth_kernel<<<grid, block>>>(depth, depth_inv);
    }

    __global__ void PyrDownDepth_kernel(
        const cv::cuda::PtrStep<float> src,
        cv::cuda::PtrStepSz<float> dst)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= dst.cols - 1 || y >= dst.rows - 1)
            return;

        dst.ptr(y)[x] = src.ptr(2 * y)[2 * x];
    }

    void PyrDownDepth(const cv::cuda::GpuMat src,
                      cv::cuda::GpuMat &dst)
    {
        if (dst.empty())
            dst.create(src.size(), CV_32FC1);

        dim3 block(8, 8);
        dim3 grid(cv::divUp(src.cols, block.x), cv::divUp(src.rows, block.y));

        PyrDownDepth_kernel<<<grid, block>>>(src, dst);
    }

    __global__ void ComputeVertexMap_kernel(
        const cv::cuda::PtrStepSz<float> depth_inv,
        cv::cuda::PtrStep<Eigen::Vector4f> vmap,
        const float invfx, const float invfy,
        const float cx, const float cy, const float cut_off)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= depth_inv.cols || y >= depth_inv.rows)
            return;

        const float invz = depth_inv.ptr(y)[x];
        const float z = 1.0 / invz;
        if (invz > 0 && z < cut_off)
        {
            vmap.ptr(y)[x] = Eigen::Vector4f(z * (x - cx) * invfx, z * (y - cy) * invfy, z, 1.0);
        }
        else
        {
            vmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.f);
        }
    }

    void ComputeVertexMap(const cv::cuda::GpuMat depth_inv, cv::cuda::GpuMat vmap, const float invfx, const float invfy, const float cx, const float cy, const float cut_off)
    {
        if (vmap.empty())
            vmap.create(depth_inv.size(), CV_32FC4);

        dim3 block(8, 8);
        dim3 grid(cv::divUp(depth_inv.cols, block.x), cv::divUp(depth_inv.rows, block.y));

        ComputeVertexMap_kernel<<<grid, block>>>(depth_inv, vmap, invfx, invfy, cx, cy, cut_off);
    }

    __global__ void ComputeNormalMap_kernel(
        const cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
        cv::cuda::PtrStep<Eigen::Vector4f> nmap)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= vmap.cols || y >= vmap.rows)
            return;

        nmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.f);

        if (x <= 1 || y <= 1 || x >= vmap.cols - 1 || y >= vmap.rows - 1)
        {
            return;
        }

        Eigen::Vector4f v00 = vmap.ptr(y)[x - 1];
        Eigen::Vector4f v01 = vmap.ptr(y)[x + 1];
        Eigen::Vector4f v10 = vmap.ptr(y - 1)[x];
        Eigen::Vector4f v11 = vmap.ptr(y + 1)[x];

        if (v00(3) > 0 && v01(3) > 0 && v10(3) > 0 && v11(3) > 0)
        {
            nmap.ptr(y)[x].head<3>() = (v11 - v10).head<3>().cross((v01 - v00).head<3>()).normalized();
            nmap.ptr(y)[x](3) = 1.f;
        }
        else
        {
            nmap.ptr(y)[x](3) = -1.f;
        }
    }

    void ComputeNormalMap(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat nmap)
    {
        if (nmap.empty())
            nmap.create(vmap.size(), CV_32FC4);

        dim3 block(8, 8);
        dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

        ComputeNormalMap_kernel<<<grid, block>>>(vmap, nmap);
    }

    __global__ void PyrDownImage_kernel(const cv::cuda::PtrStep<float> src, cv::cuda::PtrStepSz<float> dst)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= dst.cols || y >= dst.rows)
            return;

        dst.ptr(y)[x] = 0.25 * (src.ptr(y * 2)[x * 2] + src.ptr(y * 2)[x * 2 + 1] + src.ptr(y * 2 + 1)[x * 2] + src.ptr(y * 2 + 1)[x * 2 + 1]);
    }

    void PyrDownImage(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
    {
        if (dst.empty())
            dst.create(src.rows / 2, src.cols / 2, CV_32FC1);

        dim3 block(8, 8);
        dim3 grid(cv::divUp(dst.cols, block.x), cv::divUp(dst.rows, block.y));

        PyrDownImage_kernel<<<grid, block>>>(src, dst);
    }

    __global__ void PyrDownVec4f_kernel(const cv::cuda::PtrStep<Eigen::Vector4f> src, cv::cuda::PtrStepSz<Eigen::Vector4f> dst)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= dst.cols || y >= dst.rows)
            return;

        Eigen::Vector4f v;
        Eigen::Vector3f vsum(0, 0, 0);
        int vcount = 0;

        v = src.ptr(y * 2)[x * 2];
        if (v(3) > 0)
        {
            vcount++;
            vsum += v.head<3>();
        }

        v = src.ptr(y * 2)[x * 2 + 1];
        if (v(3) > 0)
        {
            vcount++;
            vsum += v.head<3>();
        }

        v = src.ptr(y * 2 + 1)[x * 2];
        if (v(3) > 0)
        {
            vcount++;
            vsum += v.head<3>();
        }

        v = src.ptr(y * 2 + 1)[x * 2 + 1];
        if (v(3) > 0)
        {
            vcount++;
            vsum += v.head<3>();
        }

        if (vcount == 0)
        {
            dst.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1);
        }
        else
        {
            v.head<3>() = vsum / vcount;
            v(3) = 1.f;
            dst.ptr(y)[x] = v;
        }
    }

    void PyrDownVec4f(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
    {
        if (dst.empty())
            dst.create(src.rows / 2, src.cols / 2, CV_32FC1);

        dim3 block(8, 8);
        dim3 grid(cv::divUp(dst.cols, block.x), cv::divUp(dst.rows, block.y));

        PyrDownVec4f_kernel<<<grid, block>>>(src, dst);
    }

    __device__ bool TryLock(int *addr)
    {
        int prev = atomicExch(addr, 1);
        if (prev == 0)
            return true;
        else
            return false;
    }

    __global__ void ComputeVertMap_kernel(
        const cv::cuda::PtrStepSz<float> depth,
        cv::cuda::PtrStep<Eigen::Vector4f> vmap,
        const float invfx, const float invfy,
        const float cx, const float cy,
        const float cut_off)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= depth.cols || y >= depth.rows)
            return;

        const float z = depth.ptr(y)[x];
        if (z > 0 && z < cut_off)
        {
            vmap.ptr(y)[x] = Eigen::Vector4f(z * (x - cx) * invfx, z * (y - cy) * invfy, z, 1.0);
        }
        else
        {
            vmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, -1.f);
        }
    }

    void ComputeVertMap(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap,
                        const Eigen::Matrix3f K, const float cut_off)
    {
        if (vmap.empty())
            vmap.create(depth.size(), CV_32FC4);

        dim3 block(8, 8);
        dim3 grid(cv::divUp(depth.cols, block.x), cv::divUp(depth.rows, block.y));

        ComputeVertMap_kernel<<<grid, block>>>(
            depth, vmap, 1.0 / K(0, 0), 1.0 / K(1, 1), K(0, 2), K(1, 2), cut_off);
        SafeCall(cudaDeviceSynchronize());
        SafeCall(cudaGetLastError());
    }

    struct CUDAThreadLock
    {
        int *d_state;
        CUDAThreadLock()
        {
            SafeCall(cudaMalloc((void **)&d_state, sizeof(int)));
            SafeCall(cudaMemset(d_state, 0, sizeof(int)));
        }

        void release()
        {
            SafeCall(cudaFree(d_state));
        }

        __device__ void lock()
        {
            while (atomicCAS(d_state, 0, 1) != 0)
                ;
        }

        __device__ void unlock()
        {
            atomicExch(d_state, 0);
        }
    };

    struct CUDACounter
    {
        uint *d_count;
        CUDACounter()
        {
            SafeCall(cudaMalloc((void **)&d_count, sizeof(int)));
            SafeCall(cudaMemset(d_count, 0, sizeof(int)));
        }

        void release()
        {
            SafeCall(cudaFree(d_count));
        }

        uint get()
        {
            uint h_count = 0;
            SafeCall(cudaMemcpy(&h_count, d_count, sizeof(uint), cudaMemcpyDeviceToHost));
            return h_count;
        }

        __device__ void inc()
        {
            atomicAdd(d_count, 1);
        }

        __device__ void sub()
        {
            atomicSub(d_count, 1);
        }
    };

    template <typename T>
    __device__ __forceinline__ T InterpolateBilinear(
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

    template <int BlockDimX, int BlockDimY>
    __global__ void SamplePixelsZbuffed_kernel(
        const cv::cuda::PtrStepSz<float> src_depth,
        const cv::cuda::PtrStepSz<Eigen::Vector3f> src_color,
        const cv::cuda::PtrStepSz<Eigen::Vector4f> pts,
        cv::cuda::PtrStep<Eigen::Vector3f> sampling_grid,
        cv::cuda::PtrStep<float> depth_samples,
        cv::cuda::PtrStep<Eigen::Vector3f> color_samples,
        const Sophus::SE3f Tdst2src, const Eigen::Matrix3f K,
        CUDAThreadLock cudaLock)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= pts.cols || y >= pts.rows)
            return;

        Eigen::Vector4f pt = pts.ptr(y)[x];
        Eigen::Vector3f ptWarped = Tdst2src * pt.head<3>();
        float new_depth = ptWarped[2];
        const float &fx = K(0, 0);
        const float &fy = K(1, 1);
        const float &cx = K(0, 2);
        const float &cy = K(1, 2);
        float u = fx * ptWarped[0] / ptWarped[2] + cx;
        float v = fy * ptWarped[1] / ptWarped[2] + cy;
        int iu = __float2int_rd(u + 0.5);
        int iv = __float2int_rd(v + 0.5);
        if (iu >= 0 && iv >= 0 && iu <= pts.cols - 1 && iv <= pts.rows - 1)
        {
            float depth_sample = src_depth.ptr(iv)[iu];
            if (depth_sample == 0 || fabs(new_depth - depth_sample) > 0.1)
                return;

            Eigen::Vector3f color_sample = InterpolateBilinear(src_color, u, v);
            Eigen::Vector3f &grid = sampling_grid.ptr(iv)[iu];

            // printf("entering critical seciton.\n");
            //======== critical section begin =========
            if (threadIdx.x == 0)
            {
                cudaLock.lock();
            }

            for (int ii = 0; ii < BlockDimX; ++ii)
                for (int jj = 0; jj < BlockDimY; ++jj)
                {
                    if (threadIdx.x == ii && threadIdx.y == jj)
                    {
                        if (grid[2] == 0)
                        {
                            grid = Eigen::Vector3f(x, y, new_depth);
                            depth_samples.ptr(y)[x] = depth_sample;
                            color_samples.ptr(y)[x] = color_sample;
                        }
                        else if (new_depth < grid[2])
                        {
                            depth_samples.ptr((int)grid[1])[(int)grid[0]] = 0;
                            color_samples.ptr((int)grid[1])[(int)grid[0]] = Eigen::Vector3f::Zero();
                            grid = Eigen::Vector3f(x, y, new_depth);
                            depth_samples.ptr(y)[x] = depth_sample;
                            color_samples.ptr(y)[x] = color_sample;
                        }
                    }
                }

            if (threadIdx.x == 0)
            {
                cudaLock.unlock();
            }
            // printf("exit critical seciton.\n");
            //======== critical section end =========
        }
    }

    void SamplePixelsZbuffed(cv::cuda::GpuMat src_depth, cv::cuda::GpuMat src_color, cv::cuda::GpuMat dst_pts,
                             cv::cuda::GpuMat &sampling_grid, cv::cuda::GpuMat &depth_samples,
                             cv::cuda::GpuMat &color_samples, const Sophus::SE3d Tdst2src, const Eigen::Matrix3f K)
    {
        if (sampling_grid.empty())
            sampling_grid.create(src_depth.size(), CV_32FC3);
        if (depth_samples.empty())
            depth_samples.create(src_depth.size(), CV_32FC1);
        if (color_samples.empty())
            depth_samples.create(src_depth.size(), CV_32FC3);

        const int BlockDimX = 8;
        const int BlockDimY = 8;

        dim3 block(BlockDimX, BlockDimY);
        dim3 grid(cv::divUp(src_depth.cols, block.x), cv::divUp(src_depth.rows, block.y));

        CUDAThreadLock lock;
        depth_samples.setTo(0);
        color_samples.setTo(0);
        sampling_grid.setTo(0);

        SamplePixelsZbuffed_kernel<BlockDimX, BlockDimY><<<grid, block>>>(
            src_depth, src_color, dst_pts, sampling_grid,
            depth_samples, color_samples, Tdst2src.cast<float>(), K, lock);

        SafeCall(cudaDeviceSynchronize());
        SafeCall(cudaGetLastError());

        lock.release();
    }

    __global__ void CUDACountImageNonZero_kernel(
        const cv::cuda::PtrStepSz<float> image,
        CUDACounter counter)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= image.cols || y >= image.rows)
            return;

        if (image.ptr(y)[x] != 0)
            counter.inc();
    }

    uint CUDACountImageNonZero(const cv::cuda::GpuMat image)
    {
        dim3 block(8, 8);
        dim3 grid(cv::divUp(image.cols, block.x), cv::divUp(image.rows, block.y));

        CUDACounter counter;
        CUDACountImageNonZero_kernel<<<grid, block>>>(image, counter);

        uint hcount = counter.get();
        counter.release();
        return hcount;
    }

} // namespace dt