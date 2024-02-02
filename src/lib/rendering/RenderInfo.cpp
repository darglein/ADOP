/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RenderInfo.h"

#include "PointRenderer.h"

TORCH_LIBRARY(sdfgergsd, m)
{
    std::cout << "register neural render info" << std::endl;
    m.class_<NeuralRenderInfo>("NeuralRenderInfo").def(torch::init());
}



std::vector<torch::Tensor> BlendPointCloud(NeuralRenderInfo* info)
{
    torch::intrusive_ptr<NeuralRenderInfo> render_data = torch::make_intrusive<NeuralRenderInfo>(*info);

    SAIGA_ASSERT(render_data->scene);
    SAIGA_ASSERT(render_data->scene->texture);

    return torch::autograd::PointRender::apply(
        render_data->scene->texture->texture, render_data->scene->texture->background_color,
        render_data->scene->point_cloud_cuda->t_position, render_data->scene->poses->tangent_poses,
        render_data->scene->intrinsics->intrinsics, torch::IValue(render_data));
}

namespace torch::autograd
{
variable_list PointRender::forward(AutogradContext* ctx, Variable texture, Variable background_color, Variable points,
                                   Variable pose_tangents, Variable intrinsics, IValue info)
{
    ctx->saved_data["render_info"] = info;

    auto [color, mask] = BlendPointCloudForward(ctx, info.toCustomClass<NeuralRenderInfo>().get());
    color.insert(color.end(), mask.begin(), mask.end());
    return color;
}

variable_list PointRender::backward(AutogradContext* ctx, variable_list grad_output)
{
    IValue info  = ctx->saved_data["render_info"];
    auto in      = info.toCustomClass<NeuralRenderInfo>().get();
    auto derives = BlendPointCloudBackward(ctx, in, grad_output);

    // With enviroment map we have <num_layers> additonal inputs because we render them first
    // TODO: maybe move env-map computation into rendering kernel
    int expected_output = 5;
    //CHECK_EQ(derives.size(), expected_output);

    // The last empty grad is for the 'IValue info' of the forward function
    derives.push_back(Variable());

    return derives;
}
}  // namespace torch::autograd