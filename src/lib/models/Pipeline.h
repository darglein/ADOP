/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
*/

#pragma once
#include "saiga/core/util/file.h"
#include "saiga/cuda/imgui_cuda.h"
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/PartialConvUnet2d.h"
#include "saiga/vision/torch/TorchHelper.h"
#include "saiga/vision/torch/VGGLoss.h"

#include "data/NeuralScene.h"
#include "models/NeuralTexture.h"
#include "models/Pipeline.h"
#include "rendering/PointRenderer.h"
#include "rendering/RenderModule.h"
using namespace Saiga;

struct LossResult
{
    float loss_vgg   = 0;
    float loss_l1    = 0;
    float loss_mse   = 0;
    float loss_psnr  = 0;
    float loss_ssim  = 0;
    float loss_lpips = 0;


    float loss_float       = 0;
    float loss_float_param = 0;

    int count = 0;

    LossResult& operator+=(const LossResult& other)
    {
        loss_vgg += other.loss_vgg;
        loss_l1 += other.loss_l1;
        loss_mse += other.loss_mse;
        loss_psnr += other.loss_psnr;
        loss_ssim += other.loss_ssim;
        loss_lpips += other.loss_lpips;
        loss_float += other.loss_float;
        loss_float_param += other.loss_float_param;
        count += other.count;
        return *this;
    }

    LossResult& operator/=(float value)
    {
        loss_vgg /= value;
        loss_l1 /= value;
        loss_mse /= value;
        loss_psnr /= value;
        loss_ssim /= value;
        loss_lpips /= value;
        loss_float /= value;
        loss_float_param /= value;
        return *this;
    }

    LossResult Average()
    {
        LossResult cpy = *this;
        cpy /= count;
        return cpy;
    }

    void AppendToFile(const std::string& file, int epoch)
    {
        std::ofstream strm(file, std::ios_base::app);

        Table tab({10, 15, 15, 15, 15, 15, 15, 15}, strm, ',');
        if (epoch == 0)
        {
            tab << "ep"
                << "vgg"
                << "lpips"
                << "l1"
                << "psrn"
                << "ssim"
                << "param"
                << "count";
        }

        tab << epoch << loss_vgg << loss_lpips << loss_l1 << loss_psnr << loss_ssim << loss_float_param << count;
    }

    void Print()
    {
        console << "Param " << loss_float_param << " VGG " << loss_vgg << " L1 " << loss_l1 << " MSE " << loss_mse
                << " PSNR " << loss_psnr << " SSIM " << loss_ssim << " LPIPS " << loss_lpips << " count " << count
                << std::endl;
    }
};

struct ForwardResult
{
    std::vector<TemplatedImage<ucvec3>> outputs;
    std::vector<TemplatedImage<ucvec3>> targets;
    std::vector<int> image_ids;

    torch::Tensor x;
    torch::Tensor loss;

    LossResult float_loss;
};

using RenderNetwork       = MultiScaleUnet2d;
using RenderNetworkParams = MultiScaleUnet2dParams;


class NeuralPipeline
{
   public:
    NeuralPipeline(std::shared_ptr<CombinedParams> params);

    void Train(bool train);

    void SaveCheckpoint(const std::string& dir) { torch::save(render_network, dir + "/render_net.pth"); }
    void LoadCheckpoint(const std::string& dir)
    {
        if (render_network && std::filesystem::exists(dir + "/render_net.pth"))
        {
            std::cout << "Load Checkpoint render" << std::endl;
            torch::load(render_network, dir + "/render_net.pth");
        }
    }

    void Log(const std::string& dir);

    void OptimizerStep(int epoch_id);
    void UpdateLearningRate(double factor);

    ForwardResult Forward(NeuralScene& scene, std::vector<NeuralTrainData>& batch, torch::Tensor global_mask,
                          bool loss_statistics, bool keep_image = false,
                          float fixed_exposure     = std::numeric_limits<float>::infinity(),
                          vec3 fixed_white_balance = vec3(std::numeric_limits<float>::infinity(), 0, 0));

    RenderNetwork render_network = nullptr;
    torch::DeviceType device     = torch::kCUDA;

    PointRenderModule render_module = nullptr;
    std::shared_ptr<CombinedParams> params;
    CUDA::CudaTimerSystem* timer_system = nullptr;

    std::shared_ptr<torch::optim::Optimizer> render_optimizer;

    // Loss stuff
    PretrainedVGG19Loss loss_vgg = nullptr;
    PSNR loss_psnr               = PSNR(0, 1);
    SSIM loss_ssim               = SSIM(2, 1);
    LPIPS loss_lpips             = LPIPS("loss/traced_lpips.pt");

};