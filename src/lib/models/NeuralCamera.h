/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
*/

#pragma once
#include "saiga/core/camera/HDR.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/file.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "data/Settings.h"

using namespace Saiga;

class VignetteNetImpl : public torch::nn::Module
{
   public:
    VignetteNetImpl(ivec2 image_size);

    void Set(vec3 params, vec2 center)
    {
        torch::NoGradGuard ngg;
        vignette_params[0] = params(0);
        vignette_params[1] = params(1);
        vignette_params[2] = params(1);

        vignette_center[0][0][0][0] = center(0);
        vignette_center[0][1][0][0] = center(1);
    }

    void PrintParams(const std::string& log_dir, std::string& name);

    void ApplyConstraints() {}

    torch::Tensor forward(torch::Tensor uv);

    ivec2 image_size;
    bool use_calibrated_center = false;
    vec2 calibrated_center;

    torch::Tensor vignette_params;
    torch::Tensor vignette_center;
};

TORCH_MODULE(VignetteNet);

// This is just test.
// It does not work very well.
class RollingShutterNetImpl : public torch::nn::Module
{
   public:
    RollingShutterNetImpl(int frames, int size_x = 32, int size_y = 32);

    void ApplyConstraints() { torch::NoGradGuard ngg; }

    torch::Tensor forward(torch::Tensor x, torch::Tensor frame_index, torch::Tensor uv);

    // [n_frames, size, 1 ,1]
    torch::Tensor transform_grid;


    torch::Tensor uv_local;

    torch::nn::functional::GridSampleFuncOptions options;
};
TORCH_MODULE(RollingShutterNet);

// Also a test
// Not really used
class MotionblurNetImpl : public torch::nn::Module
{
   public:
    MotionblurNetImpl(int frames, int radius = 3, float initial_blur = 0);

    void ApplyConstraints()
    {
        torch::NoGradGuard ngg;
        // Not exact 0,1 so we don't get zero gradients
        blur_values.clamp_(0.0001, 0.9999);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor frame_index, torch::Tensor scale);

    // [n, 1, 1 ,1]
    torch::Tensor blur_values;

    torch::Tensor kernel_raw_x, kernel_raw_y;

    int padding;
};
TORCH_MODULE(MotionblurNet);


class CameraResponseNetImpl : public torch::nn::Module
{
   public:
    CameraResponseNetImpl(int params, int num_channels, float initial_gamma, float leaky_clamp_value = 0);

    std::vector<Saiga::DiscreteResponseFunction<float>> GetCRF();

    void ApplyConstraints() {}

    torch::Tensor forward(torch::Tensor image);

    torch::Tensor ParamLoss();

    int NumParameters() { return response.size(3); }

    torch::nn::functional::GridSampleFuncOptions options;
    torch::Tensor response;
    torch::Tensor leaky_value;
};

TORCH_MODULE(CameraResponseNet);

class NeuralCameraImpl : public torch::nn::Module
{
   public:
    NeuralCameraImpl(ivec2 image_size, NeuralCameraParams params, int frames, std::vector<float> initial_exposure,
                     std::vector<vec3> initial_wb);

    torch::Tensor ParamLoss(torch::Tensor frame_index);

    torch::Tensor forward(torch::Tensor x, torch::Tensor frame_index, torch::Tensor uv, torch::Tensor scale,
                          float fixed_exposure, vec3 fixed_white_balance);

    float param_loss_exposure = 20.f;
    float param_loss_wb       = 20.f;

    // [n, 1, 1 ,1]
    torch::Tensor exposures_values;
    torch::Tensor exposures_values_reference;

    // [n, 3, 1, 1]
    torch::Tensor white_balance_values;
    torch::Tensor white_balance_values_reference;

    RollingShutterNet rolling_shutter = nullptr;
    VignetteNet vignette_net          = nullptr;
    CameraResponseNet camera_response = nullptr;
    NeuralCameraParams params;

    std::vector<float> DownloadExposure();

    // Interpolates exposure and wb for all passed indices from the neighbors.
    // Usually used for train/test split.
    void InterpolateFromNeighbors(std::vector<int> indices);

    void SaveCheckpoint(const std::string& checkpoint_dir);
    void LoadCheckpoint(const std::string& checkpoint_dir);

    void ApplyConstraints();
};
TORCH_MODULE(NeuralCamera);
