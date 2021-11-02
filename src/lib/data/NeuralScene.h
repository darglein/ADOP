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
#include "data/NeuralStructure.h"
#include "models/NeuralCamera.h"
#include "models/NeuralTexture.h"
#include "rendering/EnvironmentMap.h"
#include "rendering/NeuralPointCloudCuda.h"


using namespace Saiga;

class NeuralScene
{
   public:
    NeuralScene(std::shared_ptr<SceneData> scene, std::shared_ptr<CombinedParams> params);


    void BuildOutlierCloud(int n);

    void Train(int epoch_id, bool train);

    void to(torch::Device device)
    {
        if (environment_map)
        {
            environment_map->to(device);
        }
        texture->to(device);
        camera->to(device);
        intrinsics->to(device);
        poses->to(device);
        point_cloud_cuda->to(device);
        if (outlier_point_cloud_cuda)
        {
            outlier_point_cloud_cuda->to(device);
        }
    }

    void SaveCheckpoint(const std::string& dir, bool reduced);
    void LoadCheckpoint(const std::string& dir);

    void Log(const std::string& log_dir);

    void OptimizerStep(int epoch_id, bool structure_only);
    void UpdateLearningRate(int epoch_id , double factor);

    // Download + Save in 'scene'
    void DownloadIntrinsics();
    void DownloadPoses();

   public:
    friend class NeuralPipeline;
    std::shared_ptr<SceneData> scene;

    NeuralPointCloudCuda point_cloud_cuda         = nullptr;
    NeuralPointCloudCuda outlier_point_cloud_cuda = nullptr;

    NeuralPointTexture texture     = nullptr;
    EnvironmentMap environment_map = nullptr;
    NeuralCamera camera            = nullptr;
    PoseModule poses               = nullptr;
    IntrinsicsModule intrinsics    = nullptr;

    std::shared_ptr<torch::optim::Optimizer> camera_adam_optimizer, camera_sgd_optimizer;
    std::shared_ptr<torch::optim::Optimizer> texture_optimizer;
    std::shared_ptr<torch::optim::Optimizer> structure_optimizer;

    torch::DeviceType device = torch::kCUDA;
    std::shared_ptr<CombinedParams> params;
};
