/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/Core.h"

#include "SceneData.h"
#include "Settings.h"
#include "config.h"

#include <torch/torch.h>


using namespace Saiga;


class PoseModuleImpl : public torch::nn::Module
{
   public:
    PoseModuleImpl(std::shared_ptr<SceneData> scene);

    // Adds only a single pose
    PoseModuleImpl(Sophus::SE3d pose);

    std::vector<Sophus::SE3d> Download();
    void SetPose(int id, Sophus::SE3d pose);

    // double: [num_cameras, 8]
    torch::Tensor poses_se3;

    // double: [num_cameras, 6]
    torch::Tensor tangent_poses;

    void ApplyTangent();
};
TORCH_MODULE(PoseModule);


class IntrinsicsModuleImpl : public torch::nn::Module
{
   public:
    IntrinsicsModuleImpl(std::shared_ptr<SceneData> scene);

    // Adds only a single intrinsic
    IntrinsicsModuleImpl(IntrinsicsPinholef K);

    std::vector<Distortionf> DownloadDistortion();
    std::vector<IntrinsicsPinholef> DownloadK();


    void SetPinholeIntrinsics(int id, IntrinsicsPinholef K, Distortionf dis);

    // [num_cameras, num_model_params]
    // Pinhole + Distortion: [num_cameras, 5 + 8]
    // OCam:                 [num_cameras, 5 + world2cam_coefficients]
    torch::Tensor intrinsics;
};
TORCH_MODULE(IntrinsicsModule);
