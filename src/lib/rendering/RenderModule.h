/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "PointRenderer.h"

class PointRenderModuleImpl : public torch::nn::Module
{
   public:
    PointRenderModuleImpl(std::shared_ptr<CombinedParams> params);

    // Renders the point cloud image in multiple scale levels and returns all of them.
    // If masks are required, the first n images are the color images followed by n mask images
    //
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward(NeuralRenderInfo* nri);

    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward(
        NeuralScene& scene, const std::vector<NeuralTrainData>& batch, CUDA::CudaTimerSystem* timer_system = nullptr);

    std::shared_ptr<CombinedParams> params;
    int num_layers;
    std::shared_ptr<PointRendererCache> cache;
};


TORCH_MODULE(PointRenderModule);