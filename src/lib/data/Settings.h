/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/util/ini/ini.h"
#include "saiga/vision/torch/PartialConvUnet2d.h"
#include "saiga/vision/torch/TrainParameters.h"

#include "config.h"


using namespace Saiga;

struct RenderParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(RenderParams);


    // only for debugging
    int test_backward_mode = 0;

    bool render_points   = true;
    bool render_outliers = false;
    int outlier_count    = 1000000;
    bool check_normal    = true;

    // double res rendering + average pool
    bool super_sampling = false;

    float dropout                   = 0.25;
    float depth_accept              = 0.01;
    bool ghost_gradients            = true;
    float drop_out_radius_threshold = 0.6;
    bool drop_out_points_by_radius  = false;

    // Writes the weight into the 4-channel output texture
    bool debug_weight_color = false;
    bool debug_depth_color  = false;
    float debug_max_weight  = 10;

    float distortion_gradient_factor = 0.005;
    float K_gradient_factor          = 1;

    // == parameters set by the system ==
    int num_texture_channels           = -1;
    float dist_cutoff                  = 2;
    bool output_background_mask        = false;
    float output_background_mask_value = 0;

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        SAIGA_PARAM(render_outliers);
        SAIGA_PARAM(check_normal);
        SAIGA_PARAM(ghost_gradients);
        SAIGA_PARAM(drop_out_points_by_radius);
        SAIGA_PARAM(outlier_count);
        SAIGA_PARAM(drop_out_radius_threshold);
        SAIGA_PARAM(dropout);
        SAIGA_PARAM(depth_accept);
        SAIGA_PARAM(test_backward_mode);
        SAIGA_PARAM(distortion_gradient_factor);
        SAIGA_PARAM(K_gradient_factor);
    }
};

struct NeuralCameraParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(NeuralCameraParams);

    bool enable_vignette = true;
    bool enable_exposure = true;
    bool enable_response = true;

    bool enable_white_balance   = false;
    bool enable_motion_blur     = false;
    bool enable_rolling_shutter = false;

    int response_params        = 25;
    float response_gamma       = 1.0 / 2.2;
    float response_leak_factor = 0.01;

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        SAIGA_PARAM(enable_vignette);
        SAIGA_PARAM(enable_exposure);
        SAIGA_PARAM(enable_response);

        SAIGA_PARAM(enable_white_balance);
        SAIGA_PARAM(enable_motion_blur);
        SAIGA_PARAM(enable_rolling_shutter);

        SAIGA_PARAM(response_params);
        SAIGA_PARAM(response_gamma);
        SAIGA_PARAM(response_leak_factor);
    }
};

struct OptimizerParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(OptimizerParams);
    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        SAIGA_PARAM(texture_optimizer);

        SAIGA_PARAM(fix_render_network);
        SAIGA_PARAM(fix_texture);
        SAIGA_PARAM(fix_environment_map);

        // structure
        SAIGA_PARAM(fix_points);
        SAIGA_PARAM(fix_poses);
        SAIGA_PARAM(fix_intrinsics);

        // camera
        SAIGA_PARAM(fix_vignette);
        SAIGA_PARAM(fix_response);
        SAIGA_PARAM(fix_wb);
        SAIGA_PARAM(fix_exposure);
        SAIGA_PARAM(fix_motion_blur);
        SAIGA_PARAM(fix_rolling_shutter);


        SAIGA_PARAM(lr_render_network);
        SAIGA_PARAM(lr_texture);
        SAIGA_PARAM(lr_background_color);
        SAIGA_PARAM(lr_environment_map);

        SAIGA_PARAM(lr_points);
        SAIGA_PARAM(lr_poses);
        SAIGA_PARAM(lr_intrinsics);


        SAIGA_PARAM(response_smoothness);
        SAIGA_PARAM(lr_vignette);
        SAIGA_PARAM(lr_response);
        SAIGA_PARAM(lr_wb);
        SAIGA_PARAM(lr_exposure);
        SAIGA_PARAM(lr_motion_blur);
        SAIGA_PARAM(lr_rolling_shutter);
    }

    std::string texture_optimizer = "adam";

    bool fix_render_network  = false;
    bool fix_texture         = false;
    bool fix_environment_map = false;

    // structure
    bool fix_points     = true;
    bool fix_poses      = true;
    bool fix_intrinsics = true;

    // camera
    bool fix_vignette        = true;
    bool fix_response        = true;
    bool fix_wb              = true;
    bool fix_exposure        = true;
    bool fix_motion_blur     = true;
    bool fix_rolling_shutter = true;

    double lr_render_network   = 2e-4;
    double lr_texture          = 5e-2;
    double lr_background_color = 1e-3;
    double lr_environment_map  = 5e-3;

    // structure
    double lr_points     = 0.0001;
    double lr_poses      = 0.008;
    double lr_intrinsics = 0.01;

    // camera
    double lr_vignette         = 1e-5;  // sgd: 5e-7, adam 1e-4
    double lr_response         = 0.001;
    double response_smoothness = 1;
    double lr_wb               = 5e-4;
    double lr_exposure         = 5e-4;  // sgd 5e-4, adam 1e-3
    double lr_motion_blur      = 0.005000;
    double lr_rolling_shutter  = 2e-6;
};

struct PipelineParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(PipelineParams);

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        SAIGA_PARAM(train);

        SAIGA_PARAM(verbose_eval);
        SAIGA_PARAM(log_render);
        SAIGA_PARAM(log_texture);
        SAIGA_PARAM(skip_neural_render_network);

        SAIGA_PARAM(enable_environment_map);
        SAIGA_PARAM(env_map_w);
        SAIGA_PARAM(env_map_h);
        SAIGA_PARAM(env_map_channels);
        SAIGA_PARAM(num_texture_channels);
        SAIGA_PARAM(cat_env_to_color);
        SAIGA_PARAM(cat_masks_to_color);
    }

    bool train = true;

    bool verbose_eval               = false;
    bool log_render                 = false;
    bool log_texture                = false;
    bool skip_neural_render_network = false;
    bool skip_sensor_model          = false;


    bool enable_environment_map = false;
    int env_map_w               = 1024;
    int env_map_h               = 512;
    int env_map_channels        = 4;
    int num_texture_channels    = 8;

    // Concats the mask/env_map along the channel dimension
    // This increases the number of input channels of the network
    bool cat_env_to_color   = false;
    bool cat_masks_to_color = false;
};

struct MyTrainParams : public TrainParams
{
    MyTrainParams() {}
    MyTrainParams(const std::string file) { Load(file); }

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        TrainParams::Params(ini, app);
        SAIGA_PARAM(train_crop_size);
        SAIGA_PARAM(train_mask_border);
        SAIGA_PARAM(reduced_check_point);
        SAIGA_PARAM(write_images_at_checkpoint);
        SAIGA_PARAM(keep_all_scenes_in_memory);

        SAIGA_PARAM(use_image_masks);
        SAIGA_PARAM(write_test_images);
        SAIGA_PARAM(texture_random_init);
        SAIGA_PARAM(texture_color_init);
        SAIGA_PARAM(train_use_crop);

        SAIGA_PARAM(experiment_dir);
        SAIGA_PARAM(scene_base_dir);
        SAIGA_PARAM_LIST(scene_names, ',');
        SAIGA_PARAM(checkpoint_directory);


        SAIGA_PARAM(loss_vgg);
        SAIGA_PARAM(loss_l1);
        SAIGA_PARAM(loss_mse);

        SAIGA_PARAM(min_zoom);
        SAIGA_PARAM(max_zoom);
        SAIGA_PARAM(crop_prefere_border);
        SAIGA_PARAM(optimize_eval_camera);
        SAIGA_PARAM(interpolate_eval_settings);

        SAIGA_PARAM(noise_pose_r);
        SAIGA_PARAM(noise_pose_t);
        SAIGA_PARAM(noise_intr_k);
        SAIGA_PARAM(noise_intr_d);
        SAIGA_PARAM(noise_point);

        SAIGA_PARAM(lr_decay_factor);
        SAIGA_PARAM(lr_decay_patience);
        SAIGA_PARAM(lock_camera_params_epochs);
        SAIGA_PARAM(lock_structure_params_epochs);
    }

    // transformation
    float noise_pose_r         = 0;  // in Degrees
    float noise_pose_t         = 0;  // in mm
    float noise_intr_k         = 0;
    float noise_intr_d         = 0;
    float noise_point          = 0;

    float min_zoom           = 0.75;
    float max_zoom           = 1.5f;
    bool crop_prefere_border = true;


    double loss_vgg = 1.0;
    double loss_l1  = 1.0;
    double loss_mse = 0.0;

    int max_eval_size = 200000;


    bool use_image_masks  = false;
    int train_crop_size   = 256;
    bool train_use_crop   = true;
    int train_mask_border = 16;

    bool keep_all_scenes_in_memory  = false;
    bool reduced_check_point        = false;
    bool write_images_at_checkpoint = true;
    bool write_test_images          = false;
    bool texture_random_init        = false;
    bool texture_color_init         = false;

    bool optimize_eval_camera = false;

    // Interpolate estimated exposure values for test-images from neighbor frames
    // Assumes that the images were captured sequentially
    bool interpolate_eval_settings = false;

    std::string experiment_dir           = "experiments/";
    std::string scene_base_dir           = "scenes/";
    std::string checkpoint_directory     = "default_checkpoint/";
    std::vector<std::string> scene_names = {"church"};

    // in epoch 1 the lr is x
    // in epoch <max_epoch> the lr is x / 10
    float lr_decay_factor = 0.75;
    int lr_decay_patience = 10;



    // In the first few iterations we do not optimize camera parameters
    // such as vignetting and CRF because the solution is still too far of a reasonable result
    int lock_camera_params_epochs    = 50;
    int lock_structure_params_epochs = 50;
};


struct CombinedParams
{
    MyTrainParams train_params;
    RenderParams render_params;
    PipelineParams pipeline_params;
    OptimizerParams optimizer_params;
    NeuralCameraParams camera_params;
    MultiScaleUnet2dParams net_params;

    CombinedParams() {}
    CombinedParams(const std::string& combined_file)
        : train_params(combined_file),
          render_params(combined_file),
          pipeline_params(combined_file),
          optimizer_params(combined_file),
          camera_params(combined_file),
          net_params(combined_file)
    {
    }

    void Save(const std::string file)
    {
        train_params.Save(file);
        render_params.Save(file);
        pipeline_params.Save(file);
        optimizer_params.Save(file);
        camera_params.Save(file);
        net_params.Save(file);
    }

    void Check();
    void imgui();
};
