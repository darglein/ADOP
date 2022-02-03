/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

//#undef CUDA_DEBUG
//#define CUDA_NDEBUG

//#include "saiga/colorize.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "PointRenderer.h"
#include "PointRendererHelper.h"

#include "cooperative_groups.h"

#ifdef CUDA_DEBUG
#    define CUDA_DEBUG_ASSERT(_x) CUDA_KERNEL_ASSERT(_x)
#else
#    define CUDA_DEBUG_ASSERT(_x)
#endif


// Blog: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
// Code: https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
__device__ inline vec3 colorizeTurbo(float x)
{
    const vec4 kRedVec4   = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4  = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2   = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2  = vec2(-89.90310912, 27.34824973);

    x       = saturate(x);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    // vec2 v2 = v4.zw * v4.z;
    vec2 v2 = vec2(v4[2], v4[3]) * v4[2];
    return vec3(dot(v4, kRedVec4) + dot(v2, kRedVec2), dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
                dot(v4, kBlueVec4) + dot(v2, kBlueVec2));
}

static constexpr int default_block_size = 256;

struct DeviceRenderParams
{
    int num_texture_channels;
    bool check_normal;
    float dropout;
    bool ghost_gradients;
    float dist_cutoff;
    int num_layers;
    float depth_accept;
    float drop_out_radius_threshold;
    bool drop_out_points_by_radius;
    int test_backward_mode;
    float distortion_gradient_factor;
    float K_gradient_factor;

    // For every layer a batch of images
    // [layers, batches, 1, height_of_layer, width_of_layer]
    StaticDeviceTensor<float, 4> depth[max_layers];
    StaticDeviceTensor<float, 4> weight[max_layers];

    StaticDeviceTensor<float, 2> in_texture;
    StaticDeviceTensor<float, 3> tmp_projections;

    // for every image one pose
    Sophus::SE3d* _poses;

    HD inline Sophus::SE3f Pose(int image_index) { return _poses[image_index].cast<float>(); }

    // [num_cameras, num_model_params]
    StaticDeviceTensor<float, 2> intrinsics;

    HD inline thrust::pair<IntrinsicsPinholef, Distortionf> PinholeIntrinsics(int camera_index)
    {
        float* ptr             = &intrinsics(camera_index, 0);
        IntrinsicsPinholef K   = ((vec5*)ptr)[0];
        Distortionf distortion = ((vec8*)(ptr + 5))[0];

        return {K, distortion};
    }

    HD inline thrust::pair<Vector<float, 5>, ArrayView<const float>> OcamIntrinsics(int camera_index)
    {
        float* ptr = &intrinsics(camera_index, 0);
        int count  = intrinsics.sizes[1];

        Vector<float, 5> aff = ((vec5*)ptr)[0];
        ArrayView<const float> poly((ptr + 5), count - 5);

        return {aff, poly};
    }

    // vec5* intrinsics_pinhole;
    // vec8* intrinsics_distortion;

    DeviceRenderParams() {}

    DeviceRenderParams(RenderParams params)
    {
        num_texture_channels       = params.num_texture_channels;
        check_normal               = params.check_normal;
        dropout                    = params.dropout;
        ghost_gradients            = params.ghost_gradients;
        dist_cutoff                = params.dist_cutoff;
        depth_accept               = params.depth_accept;
        drop_out_points_by_radius  = params.drop_out_points_by_radius;
        drop_out_radius_threshold  = params.drop_out_radius_threshold;
        test_backward_mode         = params.test_backward_mode;
        distortion_gradient_factor = params.distortion_gradient_factor;
        K_gradient_factor          = params.K_gradient_factor;
    }
};

struct DeviceForwardParams
{
    StaticDeviceTensor<float, 4> neural_out[max_layers];
};

struct DeviceBackwardParams
{
    Vec6* out_gradient_pose;
    float* out_gradient_pose_count;

    vec4* out_gradient_points;
    float* out_gradient_points_count;

    StaticDeviceTensor<float, 2> out_gradient_intrinsics;
    float* out_gradient_intrinsics_count;

    StaticDeviceTensor<float, 2> out_gradient_texture;
    StaticDeviceTensor<float, 4> in_gradient_image[max_layers];
};

static __device__ __constant__ DeviceRenderParams d_render_params;
static __device__ __constant__ DeviceForwardParams d_forward_params;
static __device__ __constant__ DeviceBackwardParams d_backward_params;

__global__ void Clear(ImageView<Packtype> src, ImageView<float> dsrc)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= src.width || gy >= src.height) return;


    CUDA_DEBUG_ASSERT(dsrc(gy, gx) == 100000);
    src(gy, gx)  = PackIndex(1000000, 0);
    dsrc(gy, gx) = 100000;
}

#ifdef _CG_HAS_MATCH_COLLECTIVE

template <int GROUP_SIZE = 32>
__device__ cooperative_groups::coalesced_group subgroupPartitionNV(ivec2 p)
{
    using namespace cooperative_groups;
    thread_block block                   = this_thread_block();
    thread_block_tile<GROUP_SIZE> tile32 = tiled_partition<GROUP_SIZE>(block);

    coalesced_group g1 = labeled_partition(tile32, p(0));
    coalesced_group g2 = labeled_partition(tile32, p(1));

    details::_coalesced_group_data_access acc;
    return acc.construct_from_mask<coalesced_group>(acc.get_mask(g1) & acc.get_mask(g2));
}


template <typename T, int GROUP_SIZE = 32>
__device__ T subgroupPartitionedAddNV(T value, cooperative_groups::coalesced_group group)
{
    int s = group.size();
    int r = group.thread_rank();

    for (int offset = GROUP_SIZE / 2; offset > 0; offset /= 2)
    {
        auto v = group.template shfl_down(value, offset);
        if (r + offset < s) value += v;
    }
    return value;
}

template <typename T, int GROUP_SIZE = 32>
__device__ T subgroupPartitionedMinNV(T value, cooperative_groups::coalesced_group group)
{
    int s = group.size();
    int r = group.thread_rank();

    for (int offset = GROUP_SIZE / 2; offset > 0; offset /= 2)
    {
        auto v = group.template shfl_down(value, offset);
        if (r + offset < s) value = min(value, v);
    }
    return value;
}

#endif

template <int num_layers, bool opt_test, bool opt_ballot, bool opt_early_z>
__global__ void DepthPrepassMulti(DevicePointCloud point_cloud, ReducedImageInfo cam, int batch)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
        vec2 ip;
        float z;
        float radius_pixels;

        if (opt_test)
        {
            float* dst    = &d_render_params.tmp_projections.Get({batch, point_id, 0});
            float4 res    = ((float4*)dst)[0];
            ip(0)         = res.x;
            ip(1)         = res.y;
            z             = res.z;
            radius_pixels = res.w;

            if (z <= 0) continue;
        }
        else
        {
            vec3 position;
            vec3 normal;
            vec2 image_p_a;
            float drop_out_radius;

            CUDA_KERNEL_ASSERT(cam.image_index >= 0);

            Sophus::SE3f V                                 = d_render_params.Pose(cam.image_index);
            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
                auto [K, distortion]      = d_render_params.PinholeIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) = ProjectPointPinhole(
                    position, normal, V, K, distortion, d_render_params.check_normal, d_render_params.dist_cutoff);
                radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff, poly]          = d_render_params.OcamIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) = ProjectPointOcam(position, normal, V, aff, poly,
                                                             d_render_params.check_normal, d_render_params.dist_cutoff);

                radius_pixels = d_render_params.depth[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
            }

            if (z == 0) continue;
            ip = cam.crop_transform.normalizedToImage(image_p_a);
        }

#pragma unroll
        for (int layer = 0; layer < num_layers; ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
        {
            if (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold)
            {
                break;
            }

            ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

            // Check in image
            if (!d_render_params.depth[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;



            float* dst_pos = &(d_render_params.depth[layer](batch, 0, p_imgi(1), p_imgi(0)));

            if (opt_early_z)
            {
                // earlyz
                if (z >= *dst_pos) continue;
            }

            int i_depth = reinterpret_cast<int*>(&z)[0];

#ifdef _CG_HAS_MATCH_COLLECTIVE
            if constexpr (opt_ballot)
            {
                auto ballot   = subgroupPartitionNV(p_imgi);
                int min_depth = subgroupPartitionedMinNV(i_depth, ballot);
                if (ballot.thread_rank() == 0)
                {
                    atomicMin((int*)dst_pos, min_depth);
                }
            }
            else
#endif

            {
                atomicMin((int*)dst_pos, i_depth);
            }
        }
    }
}

template <int num_layers, bool opt_test>
__global__ void RenderForwardMulti(DevicePointCloud point_cloud, float* dropout_p, ReducedImageInfo cam, int batch)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
        //        if (point_id == 0)
        //        {
        //            printf("dist cutoff %f\n", d_render_params.dist_cutoff);
        //        }
        bool drop_out = dropout_p[point_id] == 1;
        if (drop_out) return;

        vec2 ip;
        float z;
        float radius_pixels;

        if (opt_test)
        {
            float* dst    = &d_render_params.tmp_projections.Get({batch, point_id, 0});
            float4 res    = ((float4*)dst)[0];
            ip(0)         = res.x;
            ip(1)         = res.y;
            z             = res.z;
            radius_pixels = res.w;

            if (z <= 0) continue;
        }
        else
        {
            vec3 position;
            vec3 normal;
            vec2 image_p_a;
            float drop_out_radius;

            Sophus::SE3f V                                 = d_render_params.Pose(cam.image_index);
            thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K, distortion]      = d_render_params.PinholeIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) = ProjectPointPinhole(
                    position, normal, V, K, distortion, d_render_params.check_normal, d_render_params.dist_cutoff);
                radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff, poly]          = d_render_params.OcamIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) = ProjectPointOcam(position, normal, V, aff, poly,
                                                             d_render_params.check_normal, d_render_params.dist_cutoff);
                radius_pixels = d_render_params.depth[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
            }

            if (z == 0) continue;
            ip = cam.crop_transform.normalizedToImage(image_p_a);
        }

        int texture_index = point_cloud.GetIndex(point_id);
        CUDA_KERNEL_ASSERT(texture_index >= 0 && texture_index < d_render_params.in_texture.sizes[1]);

#pragma unroll
        for (int layer = 0; layer < num_layers; ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
        // for (int layer = num_layers - 1; layer >= 0; --layer, radius_pixels *= 2.f, ip *= 2.f)
        {
            if (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold)
            {
                break;
            }

            ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

            // Check in image
            if (!d_render_params.depth[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;


            float image_depth = d_render_params.depth[layer](batch, 0, p_imgi(1), p_imgi(0));
            if (z > image_depth * (d_render_params.depth_accept + 1)) continue;


            for (int ci = 0; ci < d_render_params.in_texture.sizes[0]; ++ci)
            {
                float t = d_render_params.in_texture(ci, texture_index);
                atomicAdd(&d_forward_params.neural_out[layer](batch, ci, p_imgi(1), p_imgi(0)), t);
            }

            auto* dst_pos_weight = &(d_render_params.weight[layer](batch, 0, p_imgi(1), p_imgi(0)));
            atomicAdd(dst_pos_weight, 1);
        }
    }
}


__global__ void RenderBackward(DevicePointCloud point_cloud, float* dropout_p, ReducedImageInfo cam, int layer,
                               int batch)
{
    int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_id >= point_cloud.Size()) return;
    bool drop_out = dropout_p[point_id] == 1;



    vec2 ip;
    float z;
    float radius_pixels;

    vec3 position;
    vec3 normal;
    Sophus::SE3f V = d_render_params.Pose(cam.image_index);

    {
        vec2 image_p_a;
        float drop_out_radius;

        thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

        if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
        {
            CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
            auto [K, distortion]      = d_render_params.PinholeIntrinsics(cam.camera_index);
            thrust::tie(image_p_a, z) = ProjectPointPinhole(position, normal, V, K, distortion,
                                                            d_render_params.check_normal, d_render_params.dist_cutoff);
            radius_pixels             = K.fx * cam.crop_transform.fx * drop_out_radius / z;
        }
        else if (cam.camera_model_type == CameraModel::OCAM)
        {
            auto [aff, poly]          = d_render_params.OcamIntrinsics(cam.camera_index);
            thrust::tie(image_p_a, z) = ProjectPointOcam(position, normal, V, aff, poly, d_render_params.check_normal,
                                                         d_render_params.dist_cutoff);

            radius_pixels = d_render_params.depth[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
        }

        if (z == 0) return;
        ip = cam.crop_transform.normalizedToImage(image_p_a);
    }

    auto texture_index = point_cloud.GetIndex(point_id);
    float scale        = 1;

#pragma unroll
    for (int layer = 0; layer < max_layers; ++layer, scale *= 0.5f, radius_pixels *= 0.5f, ip *= 0.5f)
    {
        if (layer < d_render_params.num_layers)
        {
            // ip            = cam.crop_transform.scale(scale).normalizedToImage(image_p_a, nullptr, nullptr);
            // radius_pixels = scale * cam.base_K.fx * cam.crop_transform.fx * drop_out_radius / z;

            if (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold)
            {
                continue;
            }

            ivec2 p_imgi = ivec2(__float2int_rn(ip(0)), __float2int_rn(ip(1)));

            // Check in image
            if (!d_render_params.depth[layer].Image().inImage2(p_imgi(1), p_imgi(0))) continue;

            float image_depth = d_render_params.depth[layer](batch, 0, p_imgi(1), p_imgi(0));
            if (z > image_depth * (d_render_params.depth_accept + 1)) break;

            auto* dst_pos_weight = &(d_render_params.weight[layer](batch, 0, p_imgi(1), p_imgi(0)));
            float w              = *dst_pos_weight;



            if (!drop_out)
            {
                // This is currently necessary because somehow the results cam.crop_transform transformation gives
                // different results here compared to the forwrad function even though the inputs are the same.
                if (w == 0) continue;

                float iw = 1.f / w;
                CUDA_DEBUG_ASSERT(w > 0);

                for (int ci = 0; ci < d_backward_params.out_gradient_texture.sizes[0]; ++ci)
                {
                    float g = iw * d_backward_params.in_gradient_image[layer](batch, ci, p_imgi.y(), p_imgi.x());
                    CUDA_KERNEL_ASSERT(isfinite(g));
                    // g = clamp(g * 100, -10, 10);
                    atomicAdd(&d_backward_params.out_gradient_texture(ci, texture_index), g);
                }
            }

            bool point_grad =
                (d_render_params.ghost_gradients && drop_out) | (!d_render_params.ghost_gradients && !drop_out);
            if (point_grad && d_render_params.depth[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
            {
                // We have the following pixel constellation were p is the pixel of the current point and px0, px1,...
                // are the left, right, bottom and top neighbors.
                //
                //        py1
                //         |
                //   px0 - p - px1
                //         |
                //        py0
                //
                // The output neural image of the render operation is I.
                // The intensity of a pixel in I is for example I(px1). The texture of a point is T(p). The background
                // color is B(p). We now compute the gradient of p w.r.t. I.
                //
                // If the point moves from p to px1 and I(px1) was previously the background color
                // then I(px1) is now colored by the texture of p. The new value NV of px1 is then:
                //   NV(px1) = T(p)
                //
                // The change of I at px1 is then:
                // Motion p -> px1 (positive X):
                //   dI(x)/dp|x=px1 = NV(px1) - I(px1)
                //
                // There is a special case that if I(px1) is already colored by one or more points which have a similar
                // z-value then the point is blended into the neighbors instead of overriding them. This change is then
                // defined using the weight at that pixel W(p).
                //
                // New value if p -> px1:
                //   NV(px1) = W(px1)/(W(px1)+1)*I(px1)+1/(W(px1)+1)*T(p)
                // The gradient is therefore:
                //   dI(x)/dp|x=px1 = NV(px1) - I(px1)
                //

                ivec2 px0 = p_imgi + ivec2(-1, 0);
                ivec2 px1 = p_imgi + ivec2(1, 0);
                ivec2 py0 = p_imgi + ivec2(0, -1);
                ivec2 py1 = p_imgi + ivec2(0, 1);

                float iw = 1.f / (w + 1);

                float dR_dpx = 0;
                float dR_dpy = 0;

                auto sample_grad = [&](int ci, ivec2 p) -> float
                { return d_backward_params.in_gradient_image[layer](batch, ci, p.y(), p.x()); };
                auto sample_forward = [&](int ci, ivec2 p) -> float
                { return d_forward_params.neural_out[layer](batch, ci, p.y(), p.x()); };
                auto sample_tex = [&](int ci, int uv) -> float { return d_render_params.in_texture(ci, uv); };

                auto compute_dR_dp_at_x = [&](ivec2 x, int ci, float W_x, float D_x, float T_p) -> float
                {
                    auto I_x = sample_forward(ci, x);
                    auto G_x = sample_grad(ci, x);

                    if (d_render_params.test_backward_mode == 4)
                    {
                        float dI_dp_at_x;
                        if (W_x == 0 || z * (d_render_params.depth_accept + 1) < D_x)
                        {
                            // Full override
                            dI_dp_at_x = T_p - I_x;
                        }
                        else if (z > D_x * (d_render_params.depth_accept + 1))
                        {
                            // Discard
                            dI_dp_at_x = 0;
                            //                             dI_dp_at_x = T_p - I_x;
                        }
                        else
                        {
                            // Blend
                            dI_dp_at_x = (W_x / (W_x + 1) * I_x + 1 / (W_x + 1) * T_p - I_x);
                            //                            dI_dp_at_x = T_p - I_x;
                        }

                        float dR_dp_at_x = dI_dp_at_x * G_x;
                        return dR_dp_at_x;
                    }
                    else
                    {
                        float dI_dp_at_x = T_p - I_x;

                        float dR_dp_at_x = dI_dp_at_x * G_x;
                        return dR_dp_at_x;
                    }
                };

                float W_px0 = d_render_params.weight[layer](batch, 0, px0(1), px0(0));
                float W_px1 = d_render_params.weight[layer](batch, 0, px1(1), px1(0));
                float W_py0 = d_render_params.weight[layer](batch, 0, py0(1), py0(0));
                float W_py1 = d_render_params.weight[layer](batch, 0, py1(1), py1(0));

                float D_px0 = d_render_params.depth[layer](batch, 0, px0(1), px0(0));
                float D_px1 = d_render_params.depth[layer](batch, 0, px1(1), px1(0));
                float D_py0 = d_render_params.depth[layer](batch, 0, py0(1), py0(0));
                float D_py1 = d_render_params.depth[layer](batch, 0, py1(1), py1(0));

                int texture_channels = d_backward_params.out_gradient_texture.sizes[0];

#pragma unroll
                for (int ci = 0; ci < texture_channels; ++ci)
                {
                    auto g = sample_grad(ci, p_imgi);

                    auto T_p = sample_tex(ci, texture_index);

                    // The spatial derivatives at the neighboring points.
                    float dI_dp_at_px0 = 0;  //-(T_p - I_px0);
                    float dI_dp_at_px1 = 0;  // T_p - I_px1;
                    float dI_dp_at_py0 = 0;  //-(T_p - I_py0);
                    float dI_dp_at_py1 = 0;  // T_p - I_py1;

                    dI_dp_at_px0 = -compute_dR_dp_at_x(px0, ci, W_px0, D_px0, T_p);
                    dI_dp_at_px1 = compute_dR_dp_at_x(px1, ci, W_px1, D_px1, T_p);
                    dI_dp_at_py0 = -compute_dR_dp_at_x(py0, ci, W_py0, D_py0, T_p);
                    dI_dp_at_py1 = compute_dR_dp_at_x(py1, ci, W_py1, D_py1, T_p);

                    // Average between forward and backward diff. to get symmetric central diff.
                    dR_dpx += 0.5f * (dI_dp_at_px0 + dI_dp_at_px1);
                    dR_dpy += 0.5f * (dI_dp_at_py0 + dI_dp_at_py1);
                }

                vec2 dR_dp = vec2(dR_dpx, dR_dpy) / float(texture_channels);


                float grad_scale    = 1.f;
                auto cam2           = cam;
                cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);


                if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
                {
                    auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
                    auto [g_point, g_pose, g_k, g_dis] =
                        ProjectPointPinholeBackward(position, normal, dR_dp, V, K, cam2.crop_transform, distortion,
                                                    d_render_params.check_normal, d_render_params.dist_cutoff);

                    if (d_backward_params.out_gradient_points)
                    {
                        for (int k = 0; k < g_point.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                    }

                    if (d_backward_params.out_gradient_pose)
                    {
                        // Extrinsics
                        for (int k = 0; k < g_pose.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], g_pose(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1);
                    }

                    if (d_backward_params.out_gradient_intrinsics_count)
                    {
                        float k_factor = d_render_params.K_gradient_factor;
                        // Intrinsics
                        // g_k(2) *= 0.5;
                        // g_k(3) *= 0.5;

                        // sheer
                        g_k(4) *= 0.1;
                        g_k(4) *= 0;  // remove sheer (so that we can use it in colmap)
                        for (int k = 0; k < 5; ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k),
                                      k_factor * g_k(k));
                        }

                        float distortion_factor = d_render_params.distortion_gradient_factor;

                        // k3
                        g_dis(2) *= 0.25;

                        // k4 - 6
                        g_dis(3) *= 0.1;
                        g_dis(4) *= 0.1;
                        g_dis(5) *= 0.1;

                        // tangential distortion
                        g_dis(6) *= 0.1;
                        g_dis(7) *= 0.1;
                        for (int k = 0; k < 8; ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k + 5),
                                      distortion_factor * g_dis(k));
                        }
                        // Note we add a value less than 1 to increase float precision
                        float factor = 1.f / 1024.f;
                        atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], factor);
                    }
                }
                else if (cam.camera_model_type == CameraModel::OCAM)
                {
                    auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
                    auto [g_point, g_pose, g_affine] =
                        ProjectPointOcamBackward(position, normal, dR_dp, V, cam2.crop_transform, aff, poly,
                                                 d_render_params.check_normal, d_render_params.dist_cutoff);

                    if (d_backward_params.out_gradient_points)
                    {
                        // Points
                        for (int k = 0; k < g_point.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_points_count[point_id], 1);
                    }

                    if (d_backward_params.out_gradient_pose)
                    {
                        // Extrinsics
                        for (int k = 0; k < g_pose.rows(); ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], g_pose(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_pose_count[cam.image_index], 1);
                    }

                    if (d_backward_params.out_gradient_intrinsics_count)
                    {
                        // Extrinsics
                        for (int k = 0; k < 5; ++k)
                        {
                            atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                        }
                        atomicAdd(&d_backward_params.out_gradient_intrinsics_count[cam.camera_index], 1);
                    }
                }
            }
        }
    }
}

template <typename IndexType>
__global__ void CombineAndFill(float* background_color, ImageView<float> weight,
                               StaticDeviceTensor<float, 3> out_neural_image)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= weight.width || gy >= weight.height) return;

    auto cou              = weight(gy, gx);
    auto texture_channels = out_neural_image.sizes[0];

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < texture_channels; ++ci)
        {
            out_neural_image(ci, gy, gx) = background_color[ci];
        }
    }
    else
    {
        // divide by weight
        for (int ci = 0; ci < texture_channels; ++ci)
        {
            out_neural_image(ci, gy, gx) /= cou;
        }
    }
}


__global__ void DebugWeightToColor(ImageView<float> weight, StaticDeviceTensor<float, 3> out_neural_image,
                                   float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= weight.width || gy >= weight.height) return;

    auto cou = weight(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        // float t = ::saturate(x);
        // vec3 c  = saturate(vec3(sqrt(t), t * t * t, std::max(sin(3.1415 * 1.75 * t), pow(t, 12.0))));

        vec3 c = colorizeTurbo(x);

        // divide by weight
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}


__global__ void DebugDepthToColor(ImageView<float> depth, StaticDeviceTensor<float, 3> out_neural_image,
                                  float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= depth.width || gy >= depth.height) return;

    auto cou = depth(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        vec3 c  = vec3(x, x, x);
        // divide by weight
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}

__global__ void CreateMask(StaticDeviceTensor<float, 4> in_weight, StaticDeviceTensor<float, 4> out_mask,
                           float background_value, int b)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;


    if (!in_weight.Image().template inImage(gy, gx)) return;

    auto w = in_weight.At({b, 0, gy, gx});

    if (w == 0)
    {
        out_mask.At({b, 0, gy, gx}) = background_value;
    }
    else
    {
        out_mask(b, 0, gy, gx) = 1;
    }
}



template <typename IndexType>
__global__ void CombineAndFillBackward(StaticDeviceTensor<float, 3> image_gradient, ImageView<float> weight,
                                       float* out_background_gradient)
{
    int gx        = blockIdx.x * blockDim.x + threadIdx.x;
    int gy        = blockIdx.y * blockDim.y + threadIdx.y;
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;

    bool in_image = gx < weight.width && gy < weight.height;
    // if (!in_image) return;

    gx = min(gx, weight.width - 1);
    gy = min(gy, weight.height - 1);

    __shared__ float bg_grad[20];

    int num_channels = image_gradient.sizes[0];

    if (local_tid < num_channels)
    {
        bg_grad[local_tid] = 0;
    }
    __syncthreads();


    auto w = weight(gy, gx);

    float factor = (w == 0 & in_image);

    // if (w == 0)
    {
        for (int ci = 0; ci < num_channels; ++ci)
        {
            float g = factor * image_gradient(ci, gy, gx);

            g = CUDA::warpReduceSum<float>(g);
            if (local_tid % 32 == 0)
            {
                atomicAdd(&bg_grad[ci], g);
            }
        }
    }

    __syncthreads();

    if (local_tid < num_channels)
    {
        atomicAdd(&out_background_gradient[local_tid], bg_grad[local_tid]);
    }
}

void PointRendererCache::Build(NeuralRenderInfo* info, bool forward)
{
    this->info        = info;
    this->num_batches = info->images.size();
    SAIGA_OPTIONAL_TIME_MEASURE("Build Cache", info->timer_system);
    static_assert(sizeof(Packtype) == 8);

    SAIGA_ASSERT(num_batches > 0);


    {
        SAIGA_OPTIONAL_TIME_MEASURE("Allocate", info->timer_system);
        Allocate(info, forward);
    }

    if (forward)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Initialize", info->timer_system);
        InitializeData(forward);
    }
    else
    {
        output_gradient_texture    = torch::zeros_like(info->scene->texture->texture);
        output_gradient_background = torch::zeros_like(info->scene->texture->background_color);


        if (info->scene->point_cloud_cuda->t_position.requires_grad())
        {
            output_gradient_points = torch::zeros_like(info->scene->point_cloud_cuda->t_position);
            output_gradient_point_count =
                torch::zeros({output_gradient_points.size(0)}, output_gradient_points.options());
        }

        if (info->scene->poses->tangent_poses.requires_grad())
        {
            output_gradient_pose_tangent = torch::zeros_like(info->scene->poses->tangent_poses);
            output_gradient_pose_tangent_count =
                torch::zeros({info->scene->poses->tangent_poses.size(0)},
                             info->scene->poses->tangent_poses.options().dtype(torch::kFloat32));
        }

        if (info->scene->intrinsics->intrinsics.requires_grad())
        {
            output_gradient_intrinsics       = torch::zeros_like(info->scene->intrinsics->intrinsics);
            output_gradient_intrinsics_count = torch::zeros({info->scene->intrinsics->intrinsics.size(0)},
                                                            info->scene->intrinsics->intrinsics.options());
        }
    }
}

void PointRendererCache::Allocate(NeuralRenderInfo* info, bool forward)
{
    auto& fd = info->images.front();
    int h    = fd.h;
    int w    = fd.w;

    SAIGA_ASSERT(info->scene->point_cloud_cuda);
    SAIGA_ASSERT(info->scene->texture);

    std::vector<int> new_cache_size = {(int)info->scene->texture->texture.size(0),
                                       info->scene->point_cloud_cuda->Size(),
                                       info->num_layers,
                                       num_batches,
                                       h,
                                       w};


    bool size_changed = new_cache_size != cache_size;

    if (size_changed)
    {
        cache_has_forward  = false;
        cache_has_backward = false;
    }

    bool need_allocate_forward  = !cache_has_forward && forward;
    bool need_allocate_backward = !cache_has_backward && !forward;

    if (!need_allocate_forward && !need_allocate_backward)
    {
        // std::cout << "skip allocate" << std::endl;
        return;
    }

    // std::cout << "allocate render cache " << need_allocate_forward << " " << need_allocate_backward << " "
    //          << size_changed << std::endl;



    if (size_changed)
    {
        layers_cuda.resize(info->num_layers);
    }

    float scale = 1;
    for (int i = 0; i < info->num_layers; ++i)
    {
        SAIGA_ASSERT(w > 0 && h > 0);
        auto& l = layers_cuda[i];

        if (need_allocate_forward || need_allocate_backward)
        {
            l.depth  = torch::empty({num_batches, 1, h, w},
                                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            l.weight = torch::empty({num_batches, 1, h, w},
                                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        }



        l.size  = {w, h};
        l.scale = scale;

        h /= 2;
        w /= 2;
        scale *= 0.5;
    }



    if (need_allocate_forward)
    {
        dropout_points = torch::empty({num_batches, info->scene->point_cloud_cuda->Size()},
                                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }

    cache_size = new_cache_size;
    if (forward)
    {
        cache_has_forward = true;
    }
    else
    {
        cache_has_backward = true;
    }
}

void PointRendererCache::InitializeData(bool forward)
{
    if (forward)
    {
        for (auto& l : layers_cuda)
        {
            l.depth.fill_(100000);
            l.weight.zero_();
        }


        // This is created every frame, because we 'move' it to the output
        output_forward.resize(info->num_layers);
        for (int i = 0; i < info->num_layers; ++i)
        {
            int w                = layers_cuda[i].size(0);
            int h                = layers_cuda[i].size(1);
            int texture_channels = info->scene->texture->texture.size(0);
            output_forward[i]    = torch::zeros({num_batches, texture_channels, h, w},
                                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        }


        if (info->params.output_background_mask)
        {
            output_forward_background_mask.resize(info->num_layers);
            for (int i = 0; i < info->num_layers; ++i)
            {
                auto& l = layers_cuda[i];
                output_forward_background_mask[i] =
                    torch::zeros({num_batches, 1, l.size.y(), l.size.x()},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            }
        }

#if 0
        for (auto& t : output_forward)
        {
            t.zero_();
        }
#endif

        if (info->params.dropout > 0)
        {
            dropout_points.bernoulli_(info->params.dropout);
        }
        else
        {
            dropout_points.zero_();
        }
    }
}


void PointRendererCache::PushParameters(bool forward)
{
    SAIGA_OPTIONAL_TIME_MEASURE("Param Upload", info->timer_system);
    if (forward)
    {
        static DeviceForwardParams dfp;
        for (int i = 0; i < info->num_layers; ++i)
        {
            dfp.neural_out[i] = output_forward[i];
        }
        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_forward_params, &dfp, sizeof(dfp)));
    }

    if (!forward)
    {
        DeviceBackwardParams dbp = {0};

        if (output_gradient_pose_tangent.defined())
        {
            SAIGA_ASSERT(output_gradient_pose_tangent.size(1) == 6);
            dbp.out_gradient_pose       = (Vec6*)output_gradient_pose_tangent.data_ptr<double>();
            dbp.out_gradient_pose_count = output_gradient_pose_tangent_count.data_ptr<float>();
        }

        if (output_gradient_points.defined())
        {
            SAIGA_ASSERT(output_gradient_points.size(1) == 4);
            dbp.out_gradient_points       = (vec4*)output_gradient_points.data_ptr<float>();
            dbp.out_gradient_points_count = output_gradient_point_count.data_ptr<float>();
        }

        if (output_gradient_intrinsics.defined())
        {
            dbp.out_gradient_intrinsics       = output_gradient_intrinsics;
            dbp.out_gradient_intrinsics_count = output_gradient_intrinsics_count.data_ptr<float>();
        }

        dbp.out_gradient_texture = output_gradient_texture;
        SAIGA_ASSERT(image_gradients.size() == info->num_layers);
        for (int i = 0; i < info->num_layers; ++i)
        {
            SAIGA_ASSERT(image_gradients[i].dim() == 4);
            dbp.in_gradient_image[i] = image_gradients[i];
        }

        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_backward_params, &dbp, sizeof(dbp)));
    }


    {
        static DeviceRenderParams drp;

        drp            = DeviceRenderParams(info->params);
        drp._poses     = (Sophus::SE3d*)info->scene->poses->poses_se3.data_ptr<double>();
        drp.intrinsics = info->scene->intrinsics->intrinsics;
        drp.num_layers = info->num_layers;

        for (int i = 0; i < info->num_layers; ++i)
        {
            drp.depth[i]  = layers_cuda[i].depth;
            drp.weight[i] = layers_cuda[i].weight;
        }
        drp.in_texture = info->scene->texture->texture;



        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_render_params, &drp, sizeof(drp)));
        CUDA_SYNC_CHECK_ERROR();
    }
}



void PointRendererCache::DepthPrepassMulti(int batch, NeuralPointCloudCuda point_cloud)
{
    SAIGA_ASSERT(point_cloud);

    {
        auto cam = info->images[batch];
        SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);


        int c = iDivUp(point_cloud->Size(), default_block_size);

        if (info->num_layers == 1)
        {
            ::DepthPrepassMulti<1, false, false, true><<<c, default_block_size>>>(point_cloud, cam, batch);
        }
        else if (info->num_layers == 2)
        {
            ::DepthPrepassMulti<2, false, false, true><<<c, default_block_size>>>(point_cloud, cam, batch);
        }
        else if (info->num_layers == 3)
        {
            ::DepthPrepassMulti<3, false, false, true><<<c, default_block_size>>>(point_cloud, cam, batch);
        }
        else if (info->num_layers == 4)
        {
            ::DepthPrepassMulti<4, false, false, true><<<c, default_block_size>>>(point_cloud, cam, batch);
        }
        else if (info->num_layers == 5)
        {
            ::DepthPrepassMulti<5, false, false, true><<<c, default_block_size>>>(point_cloud, cam, batch);
        }
        else
        {
            SAIGA_EXIT_ERROR("invalid number of layers");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

#if 0


void PointRendererCache::ProjectPoints(int batch, NeuralPointCloudCuda point_cloud)
{
    SAIGA_ASSERT(point_cloud);
    int c = iDivUp(point_cloud->Size(), default_block_size);
    ::ComputeProjections<<<c, default_block_size>>>(point_cloud, info->images[batch], batch);
    CUDA_SYNC_CHECK_ERROR();
}

void PointRendererCache::CombinedForward(int batch, NeuralPointCloudCuda point_cloud)
{
    SAIGA_ASSERT(point_cloud);
    CUDA_SYNC_CHECK_ERROR();


    auto cam = info->images[batch];

    int device = 0;

    const int block_size = default_block_size;

    int c = iDivUp(point_cloud->Size(), block_size);

    int numBlocksPerSm = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, ::CombinedDepthColor<true>, block_size, 0);
    int max_active_blocks = numBlocksPerSm * deviceProp.multiProcessorCount;


    c = std::min(c, max_active_blocks);
    // std::cout << "active " << numBlocksPerSm << " mps " << deviceProp.multiProcessorCount << " total " << c
    //           << std::endl;



    float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * batch;
    DevicePointCloud dpc(point_cloud);

    std::array<void*, 4> args;
    args[0] = &dpc;
    args[1] = &dropout;
    args[2] = &cam;
    args[3] = &batch;
    // initialize, then launch
    CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel((void*)::CombinedDepthColor<true>, c, block_size, args.data()));

    //    if (info->params.cuda_use_ballot)
    //    {
    //        ::CombinedDepthColor<true><<<c, block_size>>>(point_cloud, cam, batch);
    //    }
    //    else
    //    {
    //        ::CombinedDepthColor<false><<<c, block_size>>>(point_cloud, cam, batch);
    //    }

    CUDA_SYNC_CHECK_ERROR();
}
#endif


void PointRendererCache::CombineAndFill(int batch, torch::Tensor background_color)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    SAIGA_ASSERT(output_forward.size() == info->num_layers);


    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);
        SAIGA_ASSERT(batch < output_forward[i].size(0));
        auto in_out_neural_image = output_forward[i][batch];

        auto weights = l.BatchViewWeights(batch);
        ::CombineAndFill<unsigned int><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(background, weights, in_out_neural_image);
    }
    CUDA_SYNC_CHECK_ERROR();
}


void PointRendererCache::RenderForwardMulti(int batch, NeuralPointCloudCuda point_cloud)
{
    SAIGA_ASSERT(point_cloud);
    auto& cam = info->images[batch];

    SAIGA_ASSERT(info->scene->texture->texture.is_cuda());


    // std::cout << "render forwrad " << cam.w << "x" << cam.h << " " << cam.camera_index << " " << cam.image_index << "
    // "
    //           << cam.crop_transform << std::endl;

    {
        float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * batch;
        int c          = iDivUp(point_cloud->Size(), default_block_size);
        if (info->num_layers == 1)
        {
            ::RenderForwardMulti<1, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 2)
        {
            ::RenderForwardMulti<2, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 3)
        {
            ::RenderForwardMulti<3, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 4)
        {
            ::RenderForwardMulti<4, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else if (info->num_layers == 5)
        {
            ::RenderForwardMulti<5, false><<<c, default_block_size>>>(point_cloud, dropout, cam, batch);
        }
        else
        {
            SAIGA_EXIT_ERROR("sdf");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

void PointRendererCache::RenderBackward(int batch, NeuralPointCloudCuda point_cloud)
{
    SAIGA_ASSERT(point_cloud);

    {
        auto cam       = info->images[batch];
        float* dropout = dropout_points.data_ptr<float>() + dropout_points.stride(0) * batch;
        int c          = iDivUp(point_cloud->Size(), default_block_size);
        ::RenderBackward<<<c, default_block_size>>>(point_cloud, dropout, cam, 0, batch);
    }
    CUDA_SYNC_CHECK_ERROR();
}

void PointRendererCache::CombineAndFillBackward(int batch, torch::Tensor background_color,
                                                std::vector<torch::Tensor> image_gradient)
{
    for (int i = 0; i < info->num_layers; ++i)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("CombineAndFillBackward " + std::to_string(i), info->timer_system);
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);

        auto out_info            = output_gradient_background.data_ptr<float>();
        auto image_gradient_info = image_gradient[i][batch];

        ::CombineAndFillBackward<unsigned int>
            <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(image_gradient_info, l.BatchViewWeights(batch), out_info);
    }
    CUDA_SYNC_CHECK_ERROR();
}


void PointRendererCache::CreateMask(int batch, float background_value)
{
    SAIGA_ASSERT(output_forward_background_mask.size() == info->num_layers);
    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);

        SAIGA_ASSERT(output_forward_background_mask[i].size(2) == l.size.y());
        ::CreateMask<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(l.weight, output_forward_background_mask[i],
                                                           background_value, batch);
    }
    CUDA_SYNC_CHECK_ERROR();
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> BlendPointCloudForward(
    torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info)
{
    SAIGA_ASSERT(info->cache);
    int num_batches     = info->images.size();
    NeuralScene& scene  = *info->scene;
    RenderParams params = info->params;

    PointRendererCache& cache = *info->cache;
    cache.Build(info, true);

    cache.PushParameters(true);

    if (params.render_outliers)
    {
        if (!scene.outlier_point_cloud_cuda)
        {
            scene.BuildOutlierCloud(params.outlier_count);
        }
    }



    {

        {SAIGA_OPTIONAL_TIME_MEASURE("DepthPrepass", info->timer_system);
    for (int b = 0; b < num_batches; ++b)
    {
        if (params.render_points)
        {
            cache.DepthPrepassMulti(b, scene.point_cloud_cuda);
        }
        if (params.render_outliers)
        {
            cache.DepthPrepassMulti(b, scene.outlier_point_cloud_cuda);
        }
    }
}


{
    SAIGA_OPTIONAL_TIME_MEASURE("RenderForward", info->timer_system);
    for (int b = 0; b < num_batches; ++b)
    {
        if (params.render_points)
        {
            cache.RenderForwardMulti(b, scene.point_cloud_cuda);
        }
        if (params.render_outliers)
        {
            cache.RenderForwardMulti(b, scene.outlier_point_cloud_cuda);
        }
    }
}
}


{
    SAIGA_OPTIONAL_TIME_MEASURE("CombineAndFill", info->timer_system);
    for (int b = 0; b < num_batches; ++b)
    {
        cache.CombineAndFill(b, scene.texture->background_color);
    }
}

if (info->params.output_background_mask)
{
    for (int b = 0; b < num_batches; ++b)
    {
        cache.CreateMask(b, info->params.output_background_mask_value);
    }
}

if (info->params.debug_weight_color && info->params.num_texture_channels == 4)
{
    for (int b = 0; b < num_batches; ++b)
    {
        for (int i = 0; i < info->num_layers; ++i)
        {
            // Allocate result tensor
            auto& l = cache.layers_cuda[i];
            int bx  = iDivUp(l.size.x(), 16);
            int by  = iDivUp(l.size.y(), 16);
            SAIGA_ASSERT(bx > 0 && by > 0);
            auto in_out_neural_image = cache.output_forward[i][b];

            auto weights = l.BatchViewWeights(b);
            ::DebugWeightToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(weights, in_out_neural_image,
                                                                       info->params.debug_max_weight);
        }
    }
}

if (info->params.debug_depth_color && info->params.num_texture_channels == 4)
{
    for (int b = 0; b < num_batches; ++b)
    {
        for (int i = 0; i < info->num_layers; ++i)
        {
            // Allocate result tensor
            auto& l = cache.layers_cuda[i];
            int bx  = iDivUp(l.size.x(), 16);
            int by  = iDivUp(l.size.y(), 16);
            SAIGA_ASSERT(bx > 0 && by > 0);
            auto in_out_neural_image = cache.output_forward[i][b];

            auto depths = l.BatchViewDepth(b);
            ::DebugDepthToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(depths, in_out_neural_image,
                                                                      info->params.debug_max_weight);
        }
    }
}

if (info->params.debug_print_num_rendered_points)
{
    double weight_sum = 0;
    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = cache.layers_cuda[i];
        weight_sum += l.weight.sum().item().toFloat();
    }
    std::cout << "# Rasterized Points = " << (int)weight_sum << std::endl;
}

if (ctx)
{
    SAIGA_OPTIONAL_TIME_MEASURE("Save in Graph", info->timer_system);
    std::vector<torch::Tensor> save_variables;
    for (auto l : cache.layers_cuda)
    {
        save_variables.push_back(l.depth);
        save_variables.push_back(l.weight);
    }
    save_variables.insert(save_variables.end(), cache.output_forward.begin(), cache.output_forward.end());
    save_variables.push_back(cache.dropout_points);
    ctx->save_for_backward(save_variables);
    CUDA_SYNC_CHECK_ERROR();
}

// cudaDeviceSynchronize();
return {std::move(cache.output_forward), std::move(cache.output_forward_background_mask)};
}


template <typename T, int N>
__global__ void NormalizeGradient(Vector<T, N>* tangent, float* tangent_count, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vector<T, N> t = tangent[tid];
    float c        = tangent_count[tid];

    if (c > 0)
    {
        tangent[tid] = t / c;
    }
}

torch::autograd::variable_list BlendPointCloudBackward(torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info,
                                                       torch::autograd::variable_list _image_gradients)
{
    SAIGA_ASSERT(info->cache);
    for (auto& ig : _image_gradients)
    {
        SAIGA_ASSERT(ig.dtype() == torch::kFloat32);
    }

    int num_batches     = info->images.size();
    NeuralScene& scene  = *info->scene;
    RenderParams params = info->params;

    // PointRendererCache cache;
    PointRendererCache& cache = *info->cache;
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Prepare Backward", info->timer_system);
        cache.Build(info, false);

        // The first [num_layers] gradients are the actual neural image gradients. After that we get the gradients of
        // the mask which does not help us much
        cache.image_gradients =
            std::vector<torch::Tensor>(_image_gradients.begin(), _image_gradients.begin() + info->num_layers);


        auto save_variables = ctx->get_saved_variables();
        SAIGA_ASSERT(save_variables.size() == info->num_layers * 3 + 1);

        cache.output_forward.resize(info->num_layers);
        for (int i = 0; i < info->num_layers; ++i)
        {
            cache.layers_cuda[i].depth  = save_variables[i * 2 + 0];
            cache.layers_cuda[i].weight = save_variables[i * 2 + 1];

            cache.output_forward[i] = save_variables[info->num_layers * 2 + i];
        }
        cache.dropout_points = save_variables.back();



        SAIGA_ASSERT(cache.image_gradients.size() == info->num_layers);

        cache.PushParameters(false);
    }
    {
        SAIGA_OPTIONAL_TIME_MEASURE("CombineAndFillBackward", info->timer_system);
        for (int b = 0; b < num_batches; ++b)
        {
            cache.CombineAndFillBackward(b, scene.texture->background_color, cache.image_gradients);
        }
    }

    {
        SAIGA_OPTIONAL_TIME_MEASURE("RenderBackward", info->timer_system);
        for (int b = 0; b < num_batches; ++b)
        {
            cache.RenderBackward(b, scene.point_cloud_cuda);
        }
    }

    {
        SAIGA_OPTIONAL_TIME_MEASURE("Post Process Gradient", info->timer_system);
        if (cache.output_gradient_pose_tangent.defined())
        {
            // Average pose gradient over all measurements
            int n = cache.output_gradient_pose_tangent.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<double, 6><<<c, 128>>>((Vec6*)cache.output_gradient_pose_tangent.data_ptr<double>(),
                                                     cache.output_gradient_pose_tangent_count.data_ptr<float>(), n);
        }

        if (cache.output_gradient_points.defined())
        {
            // Average point gradient over all measurements
            int n = cache.output_gradient_points.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<float, 3><<<c, 128>>>((vec3*)cache.output_gradient_points.data_ptr<float>(),
                                                    cache.output_gradient_point_count.data_ptr<float>(), n);
        }
        if (cache.output_gradient_intrinsics.defined())
        {
            // Average intrinsics/distortion gradient over all measurements
            int n = cache.output_gradient_intrinsics.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<float, 13>
                <<<c, 128>>>((Vector<float, 13>*)cache.output_gradient_intrinsics.data_ptr<float>(),
                             cache.output_gradient_intrinsics_count.data_ptr<float>(), n);
        }
    }

    CUDA_SYNC_CHECK_ERROR();

    return {std::move(cache.output_gradient_texture), std::move(cache.output_gradient_background),
            std::move(cache.output_gradient_points), std::move(cache.output_gradient_pose_tangent),
            std::move(cache.output_gradient_intrinsics)};
}

__global__ void ApplyTangent(Vec6* tangent, Sophus::SE3d* pose, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vec6 t = tangent[tid];
    auto p = pose[tid];
    p      = Sophus::se3_expd(t) * p;

    pose[tid]    = p;
    tangent[tid] = Vec6::Zero();
}

void ApplyTangentToPose(torch::Tensor tangent, torch::Tensor pose)
{
    SAIGA_ASSERT(tangent.is_contiguous() && pose.is_contiguous());
    int n = tangent.size(0);
    int c = iDivUp(n, 128);
    ApplyTangent<<<c, 128>>>((Vec6*)tangent.data_ptr<double>(), (Sophus::SE3d*)pose.data_ptr<double>(), n);
}