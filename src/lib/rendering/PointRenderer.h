/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/imgui_cuda.h"

#include "NeuralPointCloudCuda.h"
#include "RenderInfo.h"
#include "config.h"
#include "data/Dataset.h"
#include "data/NeuralScene.h"
#include "data/Settings.h"


using Packtype           = unsigned long long;
constexpr int max_layers = 5;


class PointRendererCache;

class NeuralRenderInfo : public torch::CustomClassHolder
{
   public:
    NeuralScene* scene;
    std::vector<ReducedImageInfo> images;
    RenderParams params;
    int num_layers;
    CUDA::CudaTimerSystem* timer_system = nullptr;
    PointRendererCache* cache           = nullptr;
};


namespace torch::autograd
{
struct PointRender : public Function<PointRender>
{
    // returns a tensor for every layer
    static variable_list forward(AutogradContext* ctx, Variable texture, Variable background_color, Variable points,
                                 Variable pose_tangents, Variable intrinsics, IValue info);

    static variable_list backward(AutogradContext* ctx, variable_list grad_output);
};
}  // namespace torch::autograd


// Render the scene into a batch of images
// Every image is a pyramid of layers in different resolutions
std::vector<torch::Tensor> BlendPointCloud(NeuralRenderInfo* info);


// ==== Internal ====
std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> BlendPointCloudForward(
    torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info);
// derivative towards the texture
torch::autograd::variable_list BlendPointCloudBackward(torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info,
                                                       torch::autograd::variable_list image_gradients);


void ApplyTangentToPose(torch::Tensor tangent, torch::Tensor pose);

struct LayerCuda
{
    ivec2 size  = ivec2(0, 0);
    float scale = 1;


    ImageView<Packtype> BatchView(int batch)
    {
        return ImageView<Packtype>(size(1), size(0),
                                   depth_index_tensor.data_ptr<long>() + batch * depth_index_tensor.stride(0));
    }

    ImageView<float> BatchViewDepth(int batch)
    {
        return ImageView<float>(size(1), size(0), depth.data_ptr<float>() + batch * depth.stride(0));
    }

    ImageView<float> BatchViewWeights(int batch)
    {
        return ImageView<float>(size(1), size(0), weight.data_ptr<float>() + batch * weight.stride(0));
    }


    // for new rendering
    torch::Tensor depth;
    torch::Tensor weight;

    // for old rendering
    torch::Tensor depth_index_tensor;
};

class PointRendererCache
{
   public:
    PointRendererCache() {}

    void Build(NeuralRenderInfo* info, bool forward);

    // Allocates cuda memory in tensors. Does not initialize them!!!
    // Call InitializeData() below for that
    void Allocate(NeuralRenderInfo* info, bool forward);
    void InitializeData(bool forward);

    void PushParameters(bool forward);

    void ProjectPoints(int batch, NeuralPointCloudCuda point_cloud);
    void DepthPrepassMulti(int batch, NeuralPointCloudCuda point_cloud);
    void RenderForwardMulti(int batch, NeuralPointCloudCuda point_cloud);
    void CombinedForward(int batch, NeuralPointCloudCuda point_cloud);

    void RenderBackward(int batch, NeuralPointCloudCuda point_cloud);


    void CreateMask(int batch, float background_value);

    void CombineAndFill(int batch, torch::Tensor background_color);
    void CombineAndFillBackward(int batch, torch::Tensor background_color, std::vector<torch::Tensor> gradient);


    std::vector<LayerCuda> layers_cuda;
    NeuralRenderInfo* info;
    int num_batches;

    // [batch, num_points]
    torch::Tensor dropout_points;

    std::vector<torch::Tensor> output_forward;
    std::vector<torch::Tensor> output_forward_background_mask;

    // [batches, num_points, 2]
    torch::Tensor tmp_point_projections;

    torch::Tensor output_gradient_texture;
    torch::Tensor output_gradient_background;
    torch::Tensor output_gradient_points;

    torch::Tensor output_gradient_pose_tangent;
    torch::Tensor output_gradient_pose_tangent_count;
    torch::Tensor output_gradient_point_count;

    torch::Tensor output_gradient_intrinsics;
    torch::Tensor output_gradient_intrinsics_count;

    std::vector<torch::Tensor> image_gradients;



    // Variables used to check if we have to reallocate
    // [num_points, layers, batch_size, h, w]
    std::vector<int> cache_size = {0, 0, 0, 0};
    bool cache_has_forward      = false;
    bool cache_has_backward     = false;
};