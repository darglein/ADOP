/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */


// #define CUDA_NDEBUG

#include "saiga/vision/torch/CudaHelper.h"

#include "EnvironmentMap.h"


EnvironmentMapImpl::EnvironmentMapImpl(int channels, int h, int w, bool log_texture)
{
    if (log_texture)
    {
        texture = torch::ones({1, channels, h, w}) * -1;
    }
    else
    {
        texture = torch::zeros({1, channels, h, w});
    }
    SAIGA_ASSERT(texture.dim() == 4);
    register_parameter("texture", texture);
}

EnvironmentMapImpl::EnvironmentMapImpl(torch::Tensor tex)
{
    if (tex.dim() == 3)
    {
        tex = tex.unsqueeze(0);
    }
    texture = tex;
    SAIGA_ASSERT(texture.dim() == 4);
    register_parameter("texture", texture);
}


HD inline vec2 SphericalCoordinates(vec3 r)
{
    const float pi = 3.14159265358979323846;

    vec2 lookup_coord = vec2(r[0], r[1]);
    // Note, coordinate system is (mathematical [x,y,z]) => (here: [x,-z,y])
    // also Note, r is required to be normalized for theta.
    // spheric
    float phi = atan2(-r[2], r[0]);
    float x   = phi / (2.0 * pi);  // in [-.5,.5]
    x         = fract(x);  // uv-coord in [0,1]    // is not needed. just for convenience (but it changes seam-position)

    float theta = acos(r[1]);  // [1,-1] ->  [0,Pi]    // acos(r.y/length(r)), if r not normalized
    float y     = theta / pi;  // uv in [0,1]
    y           = 1 - y;       // texture-coordinate-y is flipped in opengl

    lookup_coord = vec2(x, y);

    return lookup_coord;
}

__global__ void BuildSphericalUV(StaticDeviceTensor<double, 2> poses, StaticDeviceTensor<float, 2> intrinsics,
                                 StaticDeviceTensor<float, 3> uvs_out, ReducedImageInfo cam)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= uvs_out.sizes[1] || gy >= uvs_out.sizes[0]) return;


    vec2 ip(gx, gy);



    Sophus::SE3f V = (((Sophus::SE3d*)(&poses(cam.image_index, 0)))[0]).cast<float>();

    float* ptr             = &intrinsics(cam.camera_index, 0);
    IntrinsicsPinholef K   = ((vec5*)ptr)[0];
    Distortionf distortion = ((vec8*)(ptr + 5))[0];


    vec2 dist_p = K.unproject2(cam.crop_transform.unproject2(ip));
    vec2 np     = undistortNormalizedPointSimple<float>(dist_p, distortion);

    vec3 inp(np(0), np(1), 1);

    vec3 wp = V.inverse() * inp;

    vec3 dir = (wp - V.inverse().translation()).normalized();

    // CV -> opengl
    // dir(1) *= -1;
    // dir(2) *= -1;

    vec2 uv = SphericalCoordinates(dir);

    uv = uv * 2 - vec2(1, 1);

    uvs_out(gy, gx, 0) = uv[0];
    uvs_out(gy, gx, 1) = uv[1];


    // uvs_out.At({gy, gx, 0}) = dir(0);
    // uvs_out.At({gy, gx, 1}) = dir(1);
}

std::vector<torch::Tensor> EnvironmentMapImpl::Sample(torch::Tensor poses, torch::Tensor intrinsics,
                                                      ArrayView<ReducedImageInfo> info_batch, int num_layers)
{
    int num_batches = info_batch.size();
    std::vector<torch::Tensor> uv(num_layers);

    int h       = info_batch.front().h;
    int w       = info_batch.front().w;
    float scale = 1;
    for (int i = 0; i < num_layers; ++i)
    {
        uv[i] =
            torch::empty({num_batches, h, w, 2}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        uv[i].uniform_(-1, 1);

        int bx = iDivUp(w, 16);
        int by = iDivUp(h, 16);


        for (int b = 0; b < num_batches; ++b)
        {
            auto cam = info_batch[b];
            SAIGA_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);

            cam.crop_transform = cam.crop_transform.scale(scale);
            BuildSphericalUV<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(poses, intrinsics, uv[i][b], cam);
        }

        scale /= 2;
        h /= 2;
        w /= 2;
    }

    std::vector<torch::Tensor> result(num_layers);


    torch::nn::functional::GridSampleFuncOptions options;
    options.align_corners(true);
    options.padding_mode(torch::kBorder);
    options.mode(torch::kBilinear);



    auto batched_texture = texture.repeat({(long)info_batch.size(), 1, 1, 1});

    for (int i = 0; i < num_layers; ++i)
    {
        result[i] = torch::nn::functional::grid_sample(batched_texture, uv[i], options);
    }
    return result;
}