/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
*/

#pragma once
#include "saiga/core/model/UnifiedMesh.h"
#include "saiga/core/util/file.h"
#include "saiga/cuda/cudaTimer.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "saiga/core/util/FileSystem.h"

using namespace Saiga;

// [num_channels, num_points]
class NeuralPointTextureImpl : public torch::nn::Module
{
   public:
    NeuralPointTextureImpl(int num_channels, int num_points, bool random_init, bool log_texture);

    NeuralPointTextureImpl(const Saiga::UnifiedMesh& model, int channels = 4);

    std::vector<float> GetDescriptorSlow(int i)
    {
        std::vector<float> desc;
        for (int j = 0; j < texture.size(0); ++j)
        {
            float f = texture[j][i].item().toFloat();
            desc.push_back(f);
        }
        return desc;
    }

    std::vector<float> GetBackgroundColor()
    {
        std::vector<float> desc;
        for (int j = 0; j < background_color.size(0); ++j)
        {
            float f = background_color[j].item().toFloat();
            desc.push_back(f);
        }
        return desc;
    }

    void SetBackgroundColor(std::vector<float> col)
    {
        torch::NoGradGuard ngg;
        background_color.set_(torch::from_blob(col.data(), {(long)col.size()}).to(background_color.device()).clone());
    }

    int NumPoints() { return texture.size(1); }
    int TextureChannels()
    {
        SAIGA_ASSERT(texture.dim() == 2);
        return texture.size(0);
    }

    bool log_texture;

    // [channels, points]
    torch::Tensor texture;
    // [channels]
    torch::Tensor background_color;
};

TORCH_MODULE(NeuralPointTexture);
