/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "NeuralCamera.h"

#include "saiga/vision/torch/ImageTensor.h"

#include "data/Dataset.h"


VignetteNetImpl::VignetteNetImpl(ivec2 image_size) : image_size(image_size)
{
    vignette_params = torch::zeros({3});
    vignette_center = torch::zeros({1, 2, 1, 1});
    register_parameter("vignette_params", vignette_params);
    register_parameter("vignette_center", vignette_center);
}

torch::Tensor VignetteNetImpl::forward(torch::Tensor uv)
{
    SAIGA_ASSERT(uv.dim() == 4);
    SAIGA_ASSERT(uv.dtype() == vignette_params.dtype());

    float aspect = float(image_size.x()) / image_size.y();


    // We subtract the center in uv-space so that after
    // the subtraction the center is at (0,0)
    torch::Tensor transformed_uv  = uv - vignette_center;
    transformed_uv.slice(1, 0, 1) = transformed_uv.slice(1, 0, 1) * aspect;
    transformed_uv                = transformed_uv * transformed_uv;

    torch::Tensor r2 = torch::sum(transformed_uv, 1, true);
    torch::Tensor r4 = r2 * r2;
    torch::Tensor r6 = r4 * r2;

    torch::Tensor factor = 1.f + vignette_params[0] * r2 + vignette_params[1] * r4 + vignette_params[2] * r6;
    return factor;
}
void VignetteNetImpl::PrintParams(const std::string& log_dir, std::string& name)
{
    vec3 vig;
    vig[0] = vignette_params.cpu()[0].item().toFloat();
    vig[1] = vignette_params.cpu()[1].item().toFloat();
    vig[2] = vignette_params.cpu()[2].item().toFloat();

    vec2 t;
    t[0] = vignette_center.cpu()[0][0][0][0].item().toFloat();
    t[1] = vignette_center.cpu()[0][1][0][0].item().toFloat();

    std::cout << "Vignette params: " << vig.transpose() << " | " << t.transpose() << std::endl;

    if (!log_dir.empty())
    {
        std::ofstream strm(log_dir + "vignette_" + name + ".txt", std::ios_base::app);
        Table tab({20, 20, 20, 20, 20}, strm);
        tab << vig(0) << vig(1) << vig(2) << t(0) << t(1);
    }
}

CameraResponseNetImpl::CameraResponseNetImpl(int params, int num_channels, float initial_gamma, float leaky_clamp_value)
{
    Saiga::DiscreteResponseFunction<float> crf;
    crf = Saiga::DiscreteResponseFunction<float>(params);
    crf.MakeGamma(initial_gamma);
    crf.normalize(1);

    options.align_corners(true);
    options.padding_mode(torch::kBorder);
    options.mode(torch::kBilinear);

    response = torch::from_blob(crf.irradiance.data(), {1, 1, 1, (long)crf.irradiance.size()}, torch::kFloat).clone();

    // repeat across channels
    response = response.repeat({1, num_channels, 1, 1});

    if (leaky_clamp_value > 0)
    {
        leaky_value = torch::empty({1}, torch::kFloat).fill_(leaky_clamp_value);
    }

    register_parameter("response", response);
}
torch::Tensor CameraResponseNetImpl::forward(torch::Tensor image)
{
    SAIGA_ASSERT(image.dtype() == response.dtype());
    SAIGA_ASSERT(image.dim() == 4);
    SAIGA_ASSERT(image.size(1) == response.size(1));

    torch::Tensor leak_add;
    if (this->is_training() && leaky_value.defined())
    {
        leaky_value              = leaky_value.to(image.device());
        torch::Tensor clamp_low  = image < 0;
        torch::Tensor clamp_high = image > 1;

        // below 0 leak
        leak_add = (image * leaky_value) * clamp_low;

        // above 1 leak
        leak_add += (-leaky_value / ((image.abs() + 1e-4).sqrt()) + leaky_value) * clamp_high;
    }


    int num_batches  = image.size(0);
    int num_channels = image.size(1);

    auto batched_response = response.repeat({num_batches, 1, 1, 1});

    // The grid sample uv space is from -1 to +1
    image = image * 2.f - 1.f;

    // Add zero-y coordinate because gridsample is only implemented for 2D and 3D
    auto yoffset = torch::zeros_like(image);
    auto x       = torch::cat({image.unsqueeze(4), yoffset.unsqueeze(4)}, 4);

    auto result = torch::ones_like(image);
    for (int i = 0; i < num_channels; ++i)
    {
        // Slice away the channel dimension
        auto sl                   = x.slice(1, i, i + 1).squeeze(1);
        auto response_sl          = batched_response.slice(1, i, i + 1);
        result.slice(1, i, i + 1) = torch::nn::functional::grid_sample(response_sl, sl, options);
    }

    if (leak_add.defined())
    {
        result += leak_add;
    }

    return result;
}

torch::Tensor CameraResponseNetImpl::ParamLoss()
{
    auto result = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));

    auto low = response.slice(3, 0, NumParameters() - 2);
    auto up  = response.slice(3, 2, NumParameters());

    torch::Tensor target = response.clone();  // torch::empty_like(response);

    // Set first value to zero and last value to 1
    target.slice(3, 0, 1).zero_();
    // target.slice(3, NumParameters() - 1, NumParameters()).fill_(1);

    // Set middle values to mean of neighbouring values
    // -> Force smoothness
    target.slice(3, 1, NumParameters() - 1) = (up + low) * 0.5f;

    double smoothness_factor = 1e-5;
    double factor            = NumParameters() * sqrt(smoothness_factor);
    return torch::mse_loss(response * factor, target * factor, torch::Reduction::Sum);
}
std::vector<Saiga::DiscreteResponseFunction<float>> CameraResponseNetImpl::GetCRF()
{
    auto r     = response.cpu().to(torch::kFloat32).contiguous();
    float* ptr = r.data_ptr<float>();


    std::vector<Saiga::DiscreteResponseFunction<float>> crfs;

    for (int c = 0; c < 3; ++c)
    {
        Saiga::DiscreteResponseFunction<float> crf(NumParameters());
        for (int i = 0; i < NumParameters(); ++i)
        {
            crf.irradiance[i] = ptr[i + c * NumParameters()];
        }
        crfs.push_back(crf);
    }
    return crfs;
}


RollingShutterNetImpl::RollingShutterNetImpl(int frames, int size_x, int size_y)
{
    transform_grid = torch::zeros({(long)frames, 2, size_y, size_x}, torch::TensorOptions().dtype(torch::kFloat32));
    register_parameter("transform_grid", transform_grid);

    options.align_corners(true);
    options.padding_mode(torch::kBorder);
    options.mode(torch::kBilinear);
}

torch::Tensor RollingShutterNetImpl::forward(torch::Tensor x, torch::Tensor frame_index, torch::Tensor uv)
{
    SAIGA_ASSERT(x.dim() == 4);
    SAIGA_ASSERT(uv.dim() == 4);

    if (uv_local.sizes() != uv.sizes())
    {
        int h    = uv.size(2);
        int w    = uv.size(3);
        uv_local = ImageViewToTensor(InitialUVImage(h, w).getImageView()).unsqueeze(0);
        uv_local = uv_local.repeat({uv.size(0), 1, 1, 1});
        uv_local = uv_local.to(x.options().requires_grad(false));
        SAIGA_ASSERT(uv_local.sizes() == uv.sizes());
    }

    auto transformer  = torch::index_select(transform_grid, 0, frame_index.view(-1));
    auto uv_per       = uv.permute({0, 2, 3, 1});
    auto offset       = torch::nn::functional::grid_sample(transformer, uv_per, options);
    auto local_offset = (uv_local + offset).permute({0, 2, 3, 1});

    x = torch::nn::functional::grid_sample(x, local_offset, options);

    return x;
}


MotionblurNetImpl::MotionblurNetImpl(int frames, int radius, float initial_blur)
{
    padding = radius;

    blur_values = torch::full({(long)frames, 1L, 1L, 1L}, initial_blur, torch::TensorOptions().dtype(torch::kFloat32));
    register_parameter("blur_values", blur_values);

    float sigma  = 5;
    kernel_raw_x = FilterTensor(gaussianBlurKernel1d(radius, sigma));
    kernel_raw_y = FilterTensor(gaussianBlurKernel1d(radius, sigma).transpose().eval());
    register_buffer("kernel_raw_x", kernel_raw_x);
    register_buffer("kernel_raw_y", kernel_raw_y);

    ApplyConstraints();
}
torch::Tensor MotionblurNetImpl::forward(torch::Tensor x, torch::Tensor frame_index, torch::Tensor scale)
{
    SAIGA_ASSERT(x.dim() == 4);

    auto kernel_x = kernel_raw_x.repeat({x.size(1), 1, 1, 1});
    auto kernel_y = kernel_raw_y.repeat({x.size(1), 1, 1, 1});

    auto padded_x = torch::replication_pad2d(x, {padding, padding, padding, padding});

    auto tmp       = torch::conv2d(padded_x, kernel_x, {}, 1, at::IntArrayRef(0), 1, padded_x.size(1));
    auto blurred_x = torch::conv2d(tmp, kernel_y, {}, 1, at::IntArrayRef(0), 1, padded_x.size(1));

    SAIGA_ASSERT(x.sizes() == blurred_x.sizes());


    auto blur_factor = torch::index_select(blur_values, 0, frame_index.view(-1));
    blur_factor      = torch::clamp(blur_factor * scale, 0, 1);
    return x * (1 - blur_factor) + blurred_x * blur_factor;
}

NeuralCameraImpl::NeuralCameraImpl(ivec2 image_size, NeuralCameraParams params, int frames,
                                   std::vector<float> initial_exposure, std::vector<vec3> initial_wb)
    : params(params)
{
    if (params.enable_response)
    {
        camera_response =
            CameraResponseNet(params.response_params, 3, params.response_gamma, params.response_leak_factor);
        register_module("camera_response", camera_response);
    }

    if (params.enable_motion_blur)
    {
        motion_blur = MotionblurNet(frames, 20, 0);
        register_module("motion_blur", motion_blur);
    }

    if (params.enable_rolling_shutter)
    {
        rolling_shutter = RollingShutterNet(frames, 2, 8);
        register_module("rolling_shutter", rolling_shutter);
    }

    if (params.enable_vignette)
    {
        vignette_net = VignetteNet(image_size);
        register_module("vignette_net", vignette_net);
    }
    if (params.enable_exposure)
    {
        exposures_values = torch::from_blob(initial_exposure.data(), {(long)initial_exposure.size(), 1L, 1L, 1L},
                                            torch::TensorOptions().dtype(torch::kFloat32))
                               .clone();
        exposures_values_reference = exposures_values.clone();
        register_parameter("exposures_values", exposures_values);
    }

    if (params.enable_white_balance)
    {
        for (auto& wb : initial_wb)
        {
            SAIGA_ASSERT(wb.y() == 1);
        }
        white_balance_values = torch::from_blob(initial_wb.data(), {(long)initial_wb.size(), 3L, 1L, 1L},
                                                torch::TensorOptions().dtype(torch::kFloat32))
                                   .clone();
        white_balance_values_reference = white_balance_values.clone();
        register_parameter("white_balance_values", white_balance_values);
    }
}
torch::Tensor NeuralCameraImpl::forward(torch::Tensor x, torch::Tensor frame_index, torch::Tensor uv,
                                        torch::Tensor scale, float fixed_exposure, vec3 fixed_white_balance)
{
    bool log_render = false;
    // From here on we have a RGB Image!
    SAIGA_ASSERT(x.size(1) == 3);

    if (params.enable_exposure)
    {
        torch::Tensor exposure;
        if (std::isfinite(fixed_exposure))
        {
            exposure = torch::ones({1L}, x.options()) * fixed_exposure;
        }
        else
        {
            exposure = torch::index_select(exposures_values, 0, frame_index.view(-1));
        }

        if (log_render)
        {
            x = x - exposure;
        }
        else
        {
            auto exposure_factor = 1.f / torch::exp2(exposure);
            x                    = x * exposure_factor;
        }
    }


    if (params.enable_white_balance)
    {
        torch::Tensor wb;
        if (std::isfinite(fixed_white_balance.x()))
        {
            wb = torch::from_blob(fixed_white_balance.data(), {1L, 3L, 1L, 1L})
                     .repeat({x.size(0), 1, 1, 1})
                     .to(x.options());
        }
        else
        {
            wb = torch::index_select(white_balance_values, 0, frame_index.view(-1));
        }

        x = wb * x;
    }

    if (params.enable_vignette)
    {
        auto v = vignette_net->forward(uv);
        x      = v * x;
    }

    if (camera_response)
    {
        x = camera_response->forward(x);
    }
    else
    {
        x = torch::clamp(x, 0, 1);
    }

    if (0)
    {
        // srgb gamma correction
        x = torch::clamp(x, 0, 1);
        x = torch::pow(x, 1. / 2.2);
    }

    if (rolling_shutter && frame_index.defined())
    {
        x = rolling_shutter->forward(x, frame_index, uv);
    }

    if (motion_blur && frame_index.defined())
    {
        SAIGA_ASSERT(scale.defined());
        x = motion_blur->forward(x, frame_index, scale);
    }

    return x;
}
torch::Tensor NeuralCameraImpl::ParamLoss(torch::Tensor frame_index)
{
    auto result = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));

    if (!this->is_training())
    {
        return result;
    }

    if (params.enable_response)
    {
        result += camera_response->ParamLoss();
    }
    return result;
}

void NeuralCameraImpl::ApplyConstraints()
{
    if (camera_response) camera_response->ApplyConstraints();
    if (vignette_net) vignette_net->ApplyConstraints();
    if (motion_blur) motion_blur->ApplyConstraints();
    if (rolling_shutter) rolling_shutter->ApplyConstraints();
    torch::NoGradGuard ngg;
    if (exposures_values.defined())
    {
        // fix first image
        // exposures_values.slice(0, 0, 1) = exposures_values_reference.slice(0, 0, 1);
    }
    if (white_balance_values.defined())
    {
        // fix first image
        white_balance_values.slice(0, 0, 1) = white_balance_values_reference.slice(0, 0, 1);

        // fix green channel
        white_balance_values.slice(1, 1, 2) = white_balance_values_reference.slice(1, 1, 2);
    }
}
void NeuralCameraImpl::LoadCheckpoint(const std::string& checkpoint_prefix)
{
    if (vignette_net && std::filesystem::exists(checkpoint_prefix + "vignette.pth"))
    {
        std::cout << "Load Checkpoint vignette" << std::endl;
        torch::load(vignette_net, checkpoint_prefix + "vignette.pth");
    }

    if (camera_response && std::filesystem::exists(checkpoint_prefix + "response.pth"))
    {
        std::cout << "Load Checkpoint response" << std::endl;
        torch::load(camera_response, checkpoint_prefix + "response.pth");
    }

    if (motion_blur && std::filesystem::exists(checkpoint_prefix + "mb.pth"))
    {
        std::cout << "Load Checkpoint motion blur" << std::endl;
        torch::load(motion_blur, checkpoint_prefix + "mb.pth");
    }

    if (white_balance_values.defined() && std::filesystem::exists(checkpoint_prefix + "wb.pth"))
    {
        std::cout << "Load Checkpoint white balance" << std::endl;
        torch::load(white_balance_values, checkpoint_prefix + "wb.pth");
    }

    if (exposures_values.defined() && std::filesystem::exists(checkpoint_prefix + "ex.pth"))
    {
        std::cout << "Load Checkpoint exposures_values" << std::endl;
        torch::load(exposures_values, checkpoint_prefix + "ex.pth");
    }
}

std::vector<float> NeuralCameraImpl::DownloadExposure()
{
    auto ex_cpu = exposures_values.cpu();
    std::vector<float> res;
    for (int i = 0; i < ex_cpu.size(0); ++i)
    {
        float ex;
        ex = ex_cpu[i][0][0][0].item().toFloat();
        res.push_back(ex);
    }
    return res;
}
void NeuralCameraImpl::InterpolateFromNeighbors(std::vector<int> indices)
{
    if (exposures_values.defined())
    {
        std::cout << "interpolating exposure/wb for " << indices.size() << " images" << std::endl;

        std::vector<float> exp = DownloadExposure();
        int n                  = exp.size();
        for (int k = 0; k < 10; ++k)
        {
            for (int i : indices)
            {
                exp[i] = 0.5f * (exp[max(i - 1, 0)] + exp[min(i + 1, n - 1)]);
            }
        }

        torch::Tensor new_exposure =
            torch::from_blob(exp.data(), {(long)exp.size(), 1L, 1L, 1L}, torch::TensorOptions().dtype(torch::kFloat32))
                .clone()
                .to(exposures_values.options());

        torch::NoGradGuard ngg;
        exposures_values.set_(new_exposure);
    }
}

void NeuralCameraImpl::SaveCheckpoint(const std::string& checkpoint_prefix)
{
    if (vignette_net)
    {
        torch::save(vignette_net, checkpoint_prefix + "vignette.pth");
    }

    if (camera_response)
    {
        torch::save(camera_response, checkpoint_prefix + "response.pth");
        auto crfs = camera_response->GetCRF();
        DiscreteResponseFunction<float>::RGBImage(crfs).save(checkpoint_prefix + "response.png");

        // create csv
        SAIGA_ASSERT(crfs.size() == 3);
        std::ofstream strm(checkpoint_prefix + "response.csv");
        strm << "alpha,r,g,b" << std::endl;
        for (int i = 0; i < crfs.front().irradiance.size(); ++i)
        {
            auto r = crfs[0].irradiance[i];
            auto g = crfs[1].irradiance[i];
            auto b = crfs[2].irradiance[i];

            double alpha = double(i) / (crfs.front().irradiance.size() - 1);

            strm << alpha << "," << r << "," << g << "," << b << std::endl;
        }
    }

    if (white_balance_values.defined())
    {
        std::ofstream file(checkpoint_prefix + "wb.txt");
        auto wb_cpu = white_balance_values.cpu();
        for (int i = 0; i < wb_cpu.size(0); ++i)
        {
            vec3 wb;
            wb(0) = wb_cpu[i][0][0][0].item().toFloat();
            wb(1) = wb_cpu[i][1][0][0].item().toFloat();
            wb(2) = wb_cpu[i][2][0][0].item().toFloat();
            file << std::setprecision(5) << std::setw(10) << wb(0) << std::setw(10) << wb(1) << std::setw(10) << wb(2)
                 << "\n";
        }
        torch::save(white_balance_values, checkpoint_prefix + "wb.pth");
    }

    if (exposures_values.defined())
    {
        std::ofstream file(checkpoint_prefix + "ex.txt");
        auto ex_cpu = exposures_values.cpu();
        for (int i = 0; i < ex_cpu.size(0); ++i)
        {
            float ex;
            ex = ex_cpu[i][0][0][0].item().toFloat();
            file << std::setprecision(5) << std::setw(10) << ex << "\n";
        }
        torch::save(exposures_values, checkpoint_prefix + "ex.pth");
    }
    if (motion_blur)
    {
        auto ex_cpu = motion_blur->blur_values.cpu();
        std::ofstream file(checkpoint_prefix + "mb.txt");
        for (int i = 0; i < ex_cpu.size(0); ++i)
        {
            float ex;
            ex = ex_cpu[i][0][0][0].item().toFloat();
            file << std::setprecision(5) << std::setw(10) << ex << "\n";
        }
        torch::save(motion_blur, checkpoint_prefix + "mb.pth");
    }
}
