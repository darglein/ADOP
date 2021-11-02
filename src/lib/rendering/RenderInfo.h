/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/imageProcessing/image.h"

#include "NeuralPointCloudCuda.h"
#include "config.h"
#include "data/SceneData.h"

#include <torch/torch.h>



struct TorchFrameData
{
    // FrameData fd;
    ReducedImageInfo img;

    torch::Tensor target;

    // binary float tensor, where the loss should only be taken if target_mask == 1
    torch::Tensor target_mask;

    torch::Tensor uv, uv_local;

    // long index for the camera
    // used for training camera specific parameters
    torch::Tensor camera_index;
    torch::Tensor scale;

    int scene_id;
    void to(torch::Device device)
    {
        if (target.defined()) target = target.to(device);
        if (target_mask.defined()) target_mask = target_mask.to(device);
        if (uv.defined()) uv = uv.to(device);
        if (camera_index.defined()) camera_index = camera_index.to(device);
    }
};