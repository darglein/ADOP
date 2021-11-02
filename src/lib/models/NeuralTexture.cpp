/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
*/

#include "NeuralTexture.h"

using namespace Saiga;

NeuralPointTextureImpl::NeuralPointTextureImpl(int num_channels, int num_points, bool random_init, bool log_texture)
    : log_texture(log_texture)
{
    if (random_init)
    {
        // Random init in the range [-factor/2, factor/2]
        float factor = 2;
        texture      = (torch::rand({num_channels, num_points}) * factor);
        texture      = texture - torch::ones_like(texture) * (factor / 2);

        texture = torch::empty({num_channels, num_points});
        texture.uniform_(0, 1);
    }
    else
    {
        texture = torch::ones({num_channels, num_points}) * 0.5f;
    }

    background_color = torch::ones({num_channels}) * -1;


    if (log_texture)
    {
        texture = torch::log(torch::ones({num_channels, num_points}) * 0.5);
        background_color = torch::ones({num_channels}) * -1;
    }

    register_parameter("texture", texture);
    register_parameter("background_color", background_color);

    std::cout << "GPU memory - Texture: " << (texture.nbytes() + background_color.nbytes()) / 1000000.0 << "MB"
              << std::endl;
}
NeuralPointTextureImpl::NeuralPointTextureImpl(const UnifiedMesh& model, int channels)
    : NeuralPointTextureImpl(channels, model.NumVertices(), false, false)
{
    if (channels == 3)
    {
        std::vector<vec3> colors;
        for (auto c : model.color)
        {
            colors.push_back(c.head<3>());
        }

        auto t =
            torch::from_blob(colors.data(), {(long)colors.size(), 3}, torch::TensorOptions().dtype(torch::kFloat32))
                .to(texture.device())
                .permute({1, 0})
                .contiguous();
        SAIGA_ASSERT(t.sizes() == texture.sizes());

        {
            torch::NoGradGuard ngg;
            texture.set_(t);
        }

        SetBackgroundColor({0, 0, 0});
    }
    else if (channels == 4)
    {
        std::vector<vec4> colors;
        for (auto c : model.color)
        {
            c(3) = 1;
            colors.push_back(c.head<4>());
        }

        auto t =
            torch::from_blob(colors.data(), {(long)colors.size(), 4}, torch::TensorOptions().dtype(torch::kFloat32))
                .to(texture.device())
                .permute({1, 0})
                .contiguous();
        SAIGA_ASSERT(t.sizes() == texture.sizes());

        {
            torch::NoGradGuard ngg;
            texture.set_(t);
        }

        SetBackgroundColor({0, 0, 0, 1});
    }
}
