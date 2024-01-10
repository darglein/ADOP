/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "NeuralScene.h"

NeuralScene::NeuralScene(std::shared_ptr<SceneData> scene, std::shared_ptr<CombinedParams> _params)
    : scene(scene), params(_params)
{
    params->Check();

    if (scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION)
    {
        params->render_params.dist_cutoff = scene->scene_cameras.front().distortion.MonotonicThreshold();
    }
    else if (scene->dataset_params.camera_model == CameraModel::OCAM)
    {
        params->render_params.dist_cutoff = scene->scene_cameras.front().ocam_cutoff;
    }

    SAIGA_ASSERT(scene);

    // ========== Create Modules ==========

    point_cloud_cuda = NeuralPointCloudCuda(scene->point_cloud);
    SAIGA_ASSERT(point_cloud_cuda->t_normal.defined() || !params->render_params.check_normal);


    std::vector<float> exposures;
    for (auto& f : scene->frames) exposures.push_back(f.exposure_value - scene->dataset_params.scene_exposure_value);

    std::vector<vec3> wbs;
    for (auto& f : scene->frames) wbs.push_back(f.white_balance);

    poses      = PoseModule(scene);
    intrinsics = IntrinsicsModule(scene);
    camera = NeuralCamera(ivec2(scene->scene_cameras.front().w, scene->scene_cameras.front().h), params->camera_params,
                          scene->frames.size(), exposures, wbs);

    if (params->pipeline_params.enable_environment_map)
    {
        environment_map = EnvironmentMap(params->pipeline_params.env_map_channels, params->pipeline_params.env_map_w,
                                         params->pipeline_params.env_map_h, params->pipeline_params.log_texture);
    }

    SAIGA_ASSERT(scene->point_cloud.NumVertices() > 0);
    if (params->train_params.texture_color_init)
    {
        std::cout << "Using point color as texture" << std::endl;
        texture = NeuralPointTexture(scene->point_cloud, 3);
    }
    else
    {
        texture = NeuralPointTexture(params->pipeline_params.num_texture_channels, scene->point_cloud.NumVertices(),
                                     params->train_params.texture_random_init, params->pipeline_params.log_texture);
    }

    LoadCheckpoint(params->train_params.checkpoint_directory);

    camera->eval();
    camera->to(device);

    if (params->net_params.half_float)
    {
        camera->to(torch::kFloat16);
    }



    // ========== Create Optimizers ==========

    {
        std::vector<torch::optim::OptimizerParamGroup> g;

        if (params->optimizer_params.texture_optimizer == "adam")
        {
            using TexOpt   = torch::optim::AdamOptions;
            using TexOptim = torch::optim::Adam;

            if (!params->optimizer_params.fix_texture)
            {
                std::cout << "Using Adam texture optimzier" << std::endl;
                std::cout << "optimizing texture with lr " << params->optimizer_params.lr_texture << "/"
                          << params->optimizer_params.lr_background_color << std::endl;
                {
                    auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_texture);
                    std::vector<torch::Tensor> ts;
                    ts.push_back(texture->texture);
                    g.emplace_back(ts, std::move(opt_t));
                }
                {
                    auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_background_color);
                    std::vector<torch::Tensor> ts;
                    ts.push_back(texture->background_color);
                    g.emplace_back(ts, std::move(opt_t));
                }
            }

            if (environment_map && !params->optimizer_params.fix_environment_map)
            {
                std::cout << "optimizing environment_map with lr " << params->optimizer_params.lr_environment_map
                          << std::endl;
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_environment_map);
                g.emplace_back(environment_map->parameters(), std::move(opt_t));
            }
            texture_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
        }
        else if (params->optimizer_params.texture_optimizer == "sgd")
        {
            using TexOpt   = torch::optim::SGDOptions;
            using TexOptim = torch::optim::SGD;


            if (!params->optimizer_params.fix_texture)
            {
                std::cout << "Using SGD texture optimzier" << std::endl;
                std::cout << "optimizing texture with lr " << params->optimizer_params.lr_texture << "/"
                          << params->optimizer_params.lr_background_color << std::endl;
                {
                    auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_texture);
                    std::vector<torch::Tensor> ts;
                    ts.push_back(texture->texture);
                    g.emplace_back(ts, std::move(opt_t));
                }
                {
                    auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_background_color);
                    std::vector<torch::Tensor> ts;
                    ts.push_back(texture->background_color);
                    g.emplace_back(ts, std::move(opt_t));
                }
            }

            if (environment_map && !params->optimizer_params.fix_environment_map)
            {
                std::cout << "optimizing environment_map with lr " << params->optimizer_params.lr_environment_map
                          << std::endl;
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_environment_map);
                g.emplace_back(environment_map->parameters(), std::move(opt_t));
            }
            texture_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
        }
        else
        {
            SAIGA_EXIT_ERROR("unknown optimizer");
        }
    }

    {
        std::vector<torch::optim::OptimizerParamGroup> g_cam_adam, g_cam_sgd;

        if (params->camera_params.enable_response && !params->optimizer_params.fix_response)
        {
            std::cout << "optimizing response with lr " << params->optimizer_params.lr_response << std::endl;
            auto opt = std::make_unique<torch::optim::AdamOptions>(params->optimizer_params.lr_response);
            g_cam_adam.emplace_back(camera->camera_response->parameters(), std::move(opt));
        }

        if (params->camera_params.enable_white_balance && !params->optimizer_params.fix_wb)
        {
            std::cout << "optimizing white balance with lr " << params->optimizer_params.lr_wb << std::endl;
            auto opt = std::make_unique<torch::optim::AdamOptions>(params->optimizer_params.lr_wb);
            std::vector<torch::Tensor> ts;
            ts.push_back(camera->white_balance_values);
            g_cam_adam.emplace_back(ts, std::move(opt));
        }

        if (params->camera_params.enable_exposure && !params->optimizer_params.fix_exposure)
        {
            std::cout << "optimizing exposure with lr " << params->optimizer_params.lr_exposure << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_exposure);
            std::vector<torch::Tensor> ts;
            ts.push_back(camera->exposures_values);
            g_cam_sgd.emplace_back(ts, std::move(opt));
        }


        if (params->camera_params.enable_vignette && !params->optimizer_params.fix_vignette)
        {
            std::cout << "optimizing vignette with lr " << params->optimizer_params.lr_vignette << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_vignette);
            g_cam_sgd.emplace_back(camera->vignette_net->parameters(), std::move(opt));
        }

        if (params->camera_params.enable_rolling_shutter && !params->optimizer_params.fix_rolling_shutter)
        {
            std::cout << "optimizing rolling shutter with lr " << params->optimizer_params.lr_rolling_shutter
                      << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_rolling_shutter);
            g_cam_sgd.emplace_back(camera->rolling_shutter->parameters(), std::move(opt));
        }

        if (!g_cam_adam.empty())
        {
            camera_adam_optimizer = std::make_shared<torch::optim::Adam>(g_cam_adam, torch::optim::AdamOptions(1));
        }
        if (!g_cam_sgd.empty())
        {
            camera_sgd_optimizer = std::make_shared<torch::optim::SGD>(g_cam_sgd, torch::optim::SGDOptions(1));
        }
    }

    {
        std::vector<torch::optim::OptimizerParamGroup> g_struc;
        if (!params->optimizer_params.fix_points)
        {
            std::cout << "optimizing 3D points with lr " << params->optimizer_params.lr_points << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_points);
            std::vector<torch::Tensor> ts;
            ts.push_back(point_cloud_cuda->t_position);
            g_struc.emplace_back(ts, std::move(opt));
        }
        else
        {
            point_cloud_cuda->t_position.set_requires_grad(false);
        }



        if (!params->optimizer_params.fix_poses)
        {
            std::cout << "optimizing poses with lr " << params->optimizer_params.lr_poses << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_poses);
            g_struc.emplace_back(poses->parameters(), std::move(opt));
        }

        if (!params->optimizer_params.fix_intrinsics)
        {
            std::cout << "optimizing pinhole intrinsics with lr " << params->optimizer_params.lr_intrinsics
                      << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_intrinsics);
            std::vector<torch::Tensor> ts;
            ts.push_back(intrinsics->intrinsics);
            g_struc.emplace_back(ts, std::move(opt));
        }

        if (!g_struc.empty())
        {
            structure_optimizer = std::make_shared<torch::optim::SGD>(g_struc, torch::optim::SGDOptions(1e-10));
        }
        else
        {
            std::cout << "no structure optimizer" << std::endl;
        }
    }
}
void NeuralScene::BuildOutlierCloud(int n)
{
    outlier_point_cloud_cuda = NeuralPointCloudCuda(scene->OutlierPointCloud(n, 0.1));
    outlier_point_cloud_cuda->MakeOutlier(texture->NumPoints() - 1);
}

void NeuralScene::LoadCheckpoint(const std::string& checkpoint_dir)
{
    std::string checkpoint_prefix = checkpoint_dir + "/scene_" + scene->scene_name + "_";

    if (point_cloud_cuda && std::filesystem::exists(checkpoint_prefix + "points.pth"))
    {
        torch::load(point_cloud_cuda, checkpoint_prefix + "points.pth");

        std::cout << "Load Checkpoint points " << point_cloud_cuda->t_position.size(0)
                  << " max uv: " << point_cloud_cuda->t_index.max().item().toInt() << std::endl;

        SAIGA_ASSERT(point_cloud_cuda->t_position.dtype() == torch::kFloat);
        SAIGA_ASSERT(point_cloud_cuda->t_index.dtype() == torch::kInt32);

        SAIGA_ASSERT(point_cloud_cuda->t_position.size(0) == point_cloud_cuda->t_index.size(0));
    }

    if (texture && std::filesystem::exists(checkpoint_prefix + "texture.pth"))
    {
        torch::load(texture, checkpoint_prefix + "texture.pth");
        std::cout << "Load Checkpoint texture. Texels: " << texture->NumPoints()
                  << " Channels: " << texture->TextureChannels() << std::endl;
        SAIGA_ASSERT(texture->NumPoints() == point_cloud_cuda->Size());
        SAIGA_ASSERT(texture->TextureChannels() == params->pipeline_params.num_texture_channels);
    }

    SAIGA_ASSERT(point_cloud_cuda->t_index.max().item().toInt() <= texture->NumPoints());

    if (std::filesystem::exists(checkpoint_prefix + "poses.pth"))
    {
        std::cout << "Load Checkpoint pose" << std::endl;

        std::cout << "First pose before " << poses->Download().front() << std::endl;

        torch::load(poses, checkpoint_prefix + "poses.pth");

        SAIGA_ASSERT(poses->poses_se3.size(0) == scene->frames.size());
        SAIGA_ASSERT(poses->poses_se3.dtype() == torch::kDouble);
        SAIGA_ASSERT(poses->tangent_poses.dtype() == torch::kDouble);

        std::cout << "First pose after " << poses->Download().front() << std::endl;

        DownloadPoses();
    }

    if (std::filesystem::exists(checkpoint_prefix + "intrinsics.pth"))
    {
        std::cout << "Load Checkpoint intrinsics" << std::endl;
        torch::load(intrinsics, checkpoint_prefix + "intrinsics.pth");
        DownloadIntrinsics();
    }


    if (environment_map && std::filesystem::exists(checkpoint_prefix + "env.pth"))
    {
        std::cout << "Load Checkpoint environment_map" << std::endl;
        torch::load(environment_map, checkpoint_prefix + "env.pth");
    }

    camera->LoadCheckpoint(checkpoint_prefix);
}
void NeuralScene::SaveCheckpoint(const std::string& checkpoint_dir, bool reduced)
{
    std::string checkpoint_prefix = checkpoint_dir + "/scene_" + scene->scene_name + "_";


    if (!reduced)
    {
        // These variables are very large in memory so you can disable the checkpoin write here.
        torch::save(texture, checkpoint_prefix + "texture.pth");

        if (environment_map)
        {
            torch::save(environment_map, checkpoint_prefix + "env.pth");
        }
        torch::save(point_cloud_cuda, checkpoint_prefix + "points.pth");
    }

    {
        torch::save(poses, checkpoint_prefix + "poses.pth");
        torch::save(intrinsics, checkpoint_prefix + "intrinsics.pth");

        auto all_poses = poses->Download();
        SceneData::SavePoses(all_poses, checkpoint_prefix + "poses.txt");
    }

    camera->SaveCheckpoint(checkpoint_prefix);
}
void NeuralScene::Log(const std::string& log_dir)
{
    std::cout << "Scene Log - Texture: ";
    PrintTensorInfo(texture->texture);
    {
        auto bg = texture->GetBackgroundColor();
        std::cout << "Background Desc:  ";
        for (auto b : bg)
        {
            std::cout << b << " ";
        }
        std::cout << std::endl;
    }
    if (environment_map)
    {
        std::cout << "Environment map: ";
        PrintTensorInfo(environment_map->texture);
    }

    if (scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION &&
        !params->optimizer_params.fix_intrinsics)
    {
        for (auto cam : scene->scene_cameras)
        {
            std::cout << "K:    " << cam.K << std::endl;
            std::cout << "Dist: " << cam.distortion.Coeffs().transpose() << std::endl;
        }
    }

    if (!params->optimizer_params.fix_poses)
    {
        std::cout << "Poses: ";
        PrintTensorInfo(poses->poses_se3);
    }

    if (!params->optimizer_params.fix_points)
    {
        std::cout << "Point Position: ";
        PrintTensorInfo(point_cloud_cuda->t_position);
    }

    if (camera->vignette_net && !params->optimizer_params.fix_vignette)
    {
        camera->vignette_net->PrintParams(log_dir, scene->scene_name);
    }
}
void NeuralScene::OptimizerStep(int epoch_id, bool structure_only)
{
    if (!structure_only && texture_optimizer)
    {
        texture_optimizer->step();
        texture_optimizer->zero_grad();
    }

    if (epoch_id > params->train_params.lock_camera_params_epochs)
    {
        if (camera_adam_optimizer)
        {
            camera_adam_optimizer->step();
            camera_adam_optimizer->zero_grad();
        }
        if (camera_sgd_optimizer)
        {
            camera_sgd_optimizer->step();
            camera_sgd_optimizer->zero_grad();
        }
        camera->ApplyConstraints();
    }

    if (structure_optimizer && epoch_id > params->train_params.lock_structure_params_epochs)
    {
        structure_optimizer->step();
        poses->ApplyTangent();
        structure_optimizer->zero_grad();
    }

    if (!params->optimizer_params.fix_intrinsics)
    {
        DownloadIntrinsics();
    }
}


void NeuralScene::Train(int epoch_id, bool train)
{
    if (texture) texture->train(train);
    if (camera) camera->train(train);


    if (camera_sgd_optimizer) camera_sgd_optimizer->zero_grad();
    if (camera_adam_optimizer) camera_adam_optimizer->zero_grad();
    if (texture_optimizer) texture_optimizer->zero_grad();
    if (structure_optimizer) structure_optimizer->zero_grad();
}

void NeuralScene::UpdateLearningRate(int epoch_id, double factor)
{
    SAIGA_ASSERT(factor > 0);

    double lr_update_adam = factor;

    double lr_update_sgd = factor;

    if (texture_optimizer)
    {
        if (params->optimizer_params.texture_optimizer == "adam")
        {
            UpdateLR(texture_optimizer.get(), lr_update_adam);
        }
        else if (params->optimizer_params.texture_optimizer == "sgd")
        {
            UpdateLR(texture_optimizer.get(), lr_update_sgd);
        }
        else
        {
            SAIGA_EXIT_ERROR("sldg");
        }
    }

    if (epoch_id > params->train_params.lock_camera_params_epochs)
    {
        if (camera_adam_optimizer)
        {
            UpdateLR(camera_adam_optimizer.get(), lr_update_adam);
        }
        if (camera_sgd_optimizer)
        {
            UpdateLR(camera_sgd_optimizer.get(), lr_update_sgd);
        }
    }

    if (structure_optimizer && epoch_id > params->train_params.lock_structure_params_epochs)
    {
        UpdateLR(structure_optimizer.get(), lr_update_sgd);
    }
}

void NeuralScene::DownloadIntrinsics()
{
    if (scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION)
    {
        auto Ks = intrinsics->DownloadK();
        auto ds = intrinsics->DownloadDistortion();
        SAIGA_ASSERT(Ks.size() == scene->scene_cameras.size());

        for (int i = 0; i < Ks.size(); ++i)
        {
            scene->scene_cameras[i].K          = Ks[i];
            scene->scene_cameras[i].distortion = ds[i];
        }

        // We have do download and update the intrinsic matrix
        // because the cropping has to have the latest version
        params->render_params.dist_cutoff = scene->scene_cameras.front().distortion.MonotonicThreshold();
    }
    else
    {
        return;
        SAIGA_EXIT_ERROR("todo");
    }
}
void NeuralScene::DownloadPoses()
{
    auto new_poses = poses->Download();

    SAIGA_ASSERT(new_poses.size() == scene->frames.size());
    for (int i = 0; i < new_poses.size(); ++i)
    {
        scene->frames[i].pose = new_poses[i].inverse();
    }
}
