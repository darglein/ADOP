/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/VGGLoss.h"

#include "config.h"
#include "data/Dataset.h"
#include "data/NeuralScene.h"
#include "data/Settings.h"
#include "rendering/NeuralPointCloudCuda.h"
#include "rendering/PointRenderer.h"
#include "rendering/RenderInfo.h"
using namespace Saiga;

#include "gtest/gtest.h"

struct LossTest
{
    LossTest()
    {
        TemplatedImage<ucvec3> o(500, 1000);
        TemplatedImage<ucvec3> t(500, 1000);
        for (int i : o.rowRange())
        {
            for (int j : o.colRange())
            {
                o(i, j) = ucvec3::Random();
                t(i, j) = ucvec3::Random();
            }
        }
        output = ImageViewToTensor(o.getImageView()).unsqueeze(0);
        target = ImageViewToTensor(t.getImageView()).unsqueeze(0);
    }

    torch::Tensor output;
    torch::Tensor target;
};

TEST(Loss, MSE)
{
    LossTest lt;
    std::cout << "mse " << torch::mse_loss(lt.output, lt.target).item().toFloat() << std::endl;
}

TEST(Loss, LPIPS)
{
    LossTest lt;
    LPIPS lpips(PROJECT_DIR.append("loss/traced_lpips.pt"));
    lpips.module.eval();
    std::cout << "LPIPS " << lpips.forward(lt.output, lt.target).item().toFloat() << std::endl;
}

TEST(Loss, LPIPSGPU)
{
    auto device = torch::kCUDA;
    LossTest lt;
    LPIPS lpips(PROJECT_DIR.append("loss/traced_lpips.pt"));
    lpips.module.eval();
    lpips.module.to(device);
    std::cout << "LPIPS " << lpips.forward(lt.output.to(device), lt.target.to(device)).item().toFloat() << std::endl;
}

TEST(Loss, PSNR)
{
    LossTest lt;

    auto loss_vgg = PSNR();
    loss_vgg->eval();

    std::cout << "PSNR " << loss_vgg->forward(lt.output, lt.target).item().toFloat() << std::endl;
}

TEST(Loss, SSIM)
{
    LossTest lt;

    auto loss_vgg = SSIM();
    loss_vgg->eval();

    std::cout << "SSIM " << loss_vgg->forward(lt.output, lt.target).item().toFloat() << std::endl;
}

TEST(Loss, VGGCaffe)
{
    LossTest lt;

    auto loss_vgg = Saiga::PretrainedVGG19Loss(PROJECT_DIR.append("loss/vgg_script_caffe.pth"), true, false);
    loss_vgg->eval();

    std::cout << "vgg " << loss_vgg->forward(lt.output, lt.target).item().toFloat() << std::endl;
}


TEST(Loss, VGGCaffeCuda)
{
    auto device = torch::kCUDA;
    LossTest lt;

    auto loss_vgg = Saiga::PretrainedVGG19Loss(PROJECT_DIR.append("loss/vgg_script_caffe.pth"), true, false);
    loss_vgg->eval();
    loss_vgg->to(device);

    std::cout << "vgg " << loss_vgg->forward(lt.output.to(device), lt.target.to(device)).item().toFloat() << std::endl;
}
