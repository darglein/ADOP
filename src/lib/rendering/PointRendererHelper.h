/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/cameraModel/OCam.h"
#include "saiga/vision/kernels/BA.h"

#include "config.h"
#include "data/Dataset.h"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <THC/THCGeneral.h>

HD inline thrust::pair<float, uint32_t> ExtractIndex(Packtype i)
{
    uint32_t depthi = i >> 32;
    float depth     = reinterpret_cast<float*>(&depthi)[0];
    uint32_t index  = i & 0xFFFFFFFFUL;
    return {depth, index};
}


HD inline Packtype PackIndex(float depth, uint32_t index)
{
    uint32_t depthi = reinterpret_cast<uint32_t*>(&depth)[0];
    return (Packtype(depthi) << 32) | Packtype(index);
}


struct RenderImages
{
    ImageView<Packtype> depth_index[max_layers];
};


struct OutputImages
{
    ImageView<long> output[max_layers];
};


HD inline thrust::pair<vec2, float> ProjectPointPinhole(vec3 p, vec3 n, Sophus::SE3f V, IntrinsicsPinholef K,
                                                        Distortionf distortion, bool check_normal, float dist_cutoff)
{
    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 world_n = vec3(n(0), n(1), n(2));

    CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));

    vec3 view_p = TransformPoint<float>(V, world_p);
    float z     = view_p.z();
    z           = fmax(z, 0);

    vec3 view_n = V.so3() * world_n;
    if (check_normal & dot(view_p, view_n) > 0)
    {
        z = 0;
    }

    vec2 norm_p = DivideByZ<float>(view_p);

    vec2 dist_p = distortNormalizedPoint<float>(norm_p, distortion, nullptr, nullptr, dist_cutoff);

    if (dist_p(0) == 100000)
    {
        z = 0;
    }

    vec2 image_p = K.normalizedToImage(dist_p, nullptr, nullptr);

    return {image_p, z};
}


HD inline thrust::pair<vec2, float> ProjectPointOcam(vec3 p, vec3 n, Sophus::SE3f V, Vector<float, 5> a,
                                                     ArrayView<const float> poly, bool check_normal, float dist_cutoff)
{
    CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));

    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 world_n = vec3(n(0), n(1), n(2));

    vec3 view_p = TransformPoint<float>(V, world_p);

    vec3 ip_z    = ProjectOCam(view_p, a, poly, dist_cutoff);
    vec2 image_p = ip_z.head<2>();
    float z      = ip_z(2);

    vec3 view_n = V.so3() * world_n;
    if (check_normal & dot(view_p, view_n) > 0)
    {
        z = 0;
    }

    return {image_p, z};
}


struct BackwardOutputPinhole
{
    vec3 g_point = vec3::Zero();
    vec6 g_pose  = vec6::Zero();
    vec5 g_k     = vec5::Zero();
    vec8 g_dis   = vec8::Zero();
};

// Backpropagates the image space gradient to the point position
//
// Return [gradient_point, gradient_pose]
HD inline BackwardOutputPinhole ProjectPointPinholeBackward(vec3 p, vec3 n, vec2 grad, Sophus::SE3f V,
                                                            IntrinsicsPinholef K, IntrinsicsPinholef crop_transform,
                                                            Distortionf distortion, bool check_normal,
                                                            float dist_cutoff)
{
    using T = float;

    CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));

    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 world_n = vec3(n(0), n(1), n(2));

    Matrix<T, 3, 3> J_point;
    Matrix<T, 3, 6> J_pose;
    vec3 view_p = TransformPoint<float>(V, world_p, &J_pose, &J_point);
    float z     = view_p.z();
    CUDA_KERNEL_ASSERT(z > 0);
    if (z <= 0) return {};

    vec3 view_n = V.so3() * world_n;

    if (!isfinite(n(0)) || (check_normal && dot(view_p, view_n) > 0))
    {
        printf("invalid normal %f %f %f \n", n(0), n(1), n(2));
    }
    CUDA_KERNEL_ASSERT(!check_normal || dot(view_p, view_n) <= 0);
    if (check_normal && dot(view_p, view_n) > 0) return {};

    Matrix<T, 2, 3> J_p_div;
    vec2 norm_p = DivideByZ<float>(view_p, &J_p_div);

    Matrix<T, 2, 2> J_p_dis;
    Matrix<T, 2, 8> J_dis_dis;
    vec2 dist_p = distortNormalizedPoint<float>(norm_p, distortion, &J_p_dis, &J_dis_dis, dist_cutoff);

    Matrix<T, 2, 2> J_p_K1, J_p_K2;
    Matrix<T, 2, 5> J_k_K1;
    vec2 image_p = K.normalizedToImage(dist_p, &J_p_K1, &J_k_K1);
    image_p      = crop_transform.normalizedToImage(image_p, &J_p_K2, nullptr);

    vec2 grad_p_k2 = J_p_K2.transpose() * grad;

    vec3 g_point =
        J_point.transpose() * (J_p_div.transpose() * (J_p_dis.transpose() * (J_p_K1.transpose() * grad_p_k2)));

    vec6 g_pose = J_pose.transpose() * (J_p_div.transpose() * (J_p_dis.transpose() * (J_p_K1.transpose() * grad_p_k2)));


    vec5 g_k   = J_k_K1.transpose() * grad_p_k2;
    vec8 g_dis = J_dis_dis.transpose() * (J_p_K1.transpose() * grad_p_k2);

    return {g_point, g_pose, g_k, g_dis};
}



struct BackwardOutputOcam
{
    vec3 g_point  = vec3::Zero();
    vec6 g_pose   = vec6::Zero();
    vec5 g_affine = vec5::Zero();
};

// Backpropagates the image space gradient to the point position
//
// Return [gradient_point, gradient_pose]
HD inline BackwardOutputOcam ProjectPointOcamBackward(vec3 p, vec3 n, vec2 grad, Sophus::SE3f V,
                                                      IntrinsicsPinholef crop_transform, Vector<float, 5> a,
                                                      ArrayView<const float> poly, bool check_normal, float dist_cutoff)
{
    using T = float;

    CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));

    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 world_n = vec3(n(0), n(1), n(2));

    Matrix<T, 3, 3> J_point;
    Matrix<T, 3, 6> J_pose;
    vec3 view_p = TransformPoint<float>(V, world_p, &J_pose, &J_point);

    vec3 view_n = V.so3() * world_n;

    if (!isfinite(n(0)) || (check_normal && dot(view_p, view_n) > 0))
    {
        printf("invalid normal %f %f %f \n", n(0), n(1), n(2));
    }
    CUDA_KERNEL_ASSERT(!check_normal || dot(view_p, view_n) <= 0);
    if (check_normal && dot(view_p, view_n) > 0) return {};


    Matrix<T, 2, 3> J_p_ocam;
    Matrix<T, 2, 5> J_affine_ocam;
    vec3 ip_z    = ProjectOCam<T>(view_p, a, poly, dist_cutoff, &J_p_ocam, &J_affine_ocam);
    vec2 image_p = ip_z.head<2>();
    float z      = ip_z(2);


    Matrix<T, 2, 2> J_p_crop;
    image_p = crop_transform.normalizedToImage(image_p, &J_p_crop, nullptr);


    vec2 grad_p_k2 = J_p_crop.transpose() * grad;

    vec3 g_point  = J_point.transpose() * (J_p_ocam.transpose() * grad_p_k2);
    vec6 g_pose   = J_pose.transpose() * (J_p_ocam.transpose() * grad_p_k2);
    vec5 g_affine = J_affine_ocam.transpose() * grad_p_k2;

    return {g_point, g_pose, g_affine};
}
