/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/Core.h"
#include "saiga/core/camera/HDR.h"
#include "saiga/core/math/CoordinateSystems.h"
#include "saiga/core/sophus/Sophus.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/cameraModel/OCam.h"

#include "config.h"


using namespace Saiga;

class SceneData;

constexpr int default_point_block_size = 256;


enum class CameraModel
{
    PINHOLE_DISTORTION = 0,
    OCAM               = 1,
};

// The reduced image is actually passed to the render module
// and is used during training. Note that it does not contain the camera and pose parameters.
// These are loaded from the respective tensor only using the image/camera indices.
struct ReducedImageInfo
{
    int w = 0, h = 0;
    int image_index  = -1;
    int camera_index = -1;

    // 0: pinhole+dist
    // 1: ocam
    CameraModel camera_model_type = CameraModel::PINHOLE_DISTORTION;
    IntrinsicsPinholef crop_transform;
};

// This is the full info required for generating a novel frame.
// The realtime renderer automatically uploads the camera/pose params to the device.
struct ImageInfo : public ReducedImageInfo
{
    // camera->world transform
    Sophus::SE3d pose;

    // The exposure value of this frame
    // https://en.wikipedia.org/wiki/Exposure_value
    // Stored logarithmically
    float exposure_value = 0;

    vec3 white_balance = vec3(1, 1, 1);

    // The actual K is constructed from
    // K = crop_transform * base_K
    // This is used for training to learn on image crops
    IntrinsicsPinholef K;
    Distortionf distortion;
    OCam<float> ocam;
};

// Additional dataset information for the ground truth images.
// Includes, for example, the filenames for the images/masks.
struct FrameData : public ImageInfo
{
    std::string target_file;
    std::string mask_file;

    vec4 display_color = vec4(1, 0, 0, 1);

    mat4 OpenglModel() const { return pose.matrix().cast<float>() * CV2GLView(); }

    // Return [image_point, depth]
    std::pair<vec2, float> Project3(vec3 wp) const;

    bool inImage(vec2 ip) const { return ip(0) >= 0 && ip(1) >= 0 && ip(0) < w && ip(1) < h; }


    Saiga::Camera GLCamera() const;
};

struct SceneDatasetParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(SceneDatasetParams);

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override;

    std::string file_model;
    std::string image_dir          = "color/";
    std::string mask_dir           = "not set";

    std::string file_point_cloud            = "point_cloud.ply";
    std::string file_point_cloud_compressed = "point_cloud.bin";

    // can be multiple camera files
    std::vector<std::string> camera_files = {"camera.ini"};

    float znear = 0.1, zfar = 1000;
    float render_scale = 1;
    // this value will be subtracted from the frames' EV for better normalization
    float scene_exposure_value = 0;

    // Mainly used for camera control
    vec3 scene_up_vector = vec3(0, 1, 0);

    CameraModel camera_model = CameraModel::PINHOLE_DISTORTION;
};

struct SceneCameraParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(SceneCameraParams);
    virtual ~SceneCameraParams() {}

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app)  override;


    int w = 512, h = 512;
    IntrinsicsPinholef K;
    Distortionf distortion;

    float ocam_cutoff = 1;
    OCam<double> ocam;

    CameraModel camera_model_type = CameraModel::PINHOLE_DISTORTION;

    // Unprojection to normalized image space
    vec3 ImageToNormalized(vec2 image_point, float z);
    std::pair<vec2, float> NormalizedToImage(vec3 normalized_point);
};

class SceneData
{
   public:
    SceneData(std::string scene_path);

    SceneData(int w, int h, IntrinsicsPinholef K = IntrinsicsPinholef());

    mat4 GLProj()
    {
        auto scene_camera = scene_cameras[0];
        return CVCamera2GLProjectionMatrix(scene_camera.K.matrix(), ivec2(scene_camera.w, scene_camera.h),
                                           dataset_params.znear, dataset_params.zfar);
    }

    std::vector<SceneCameraParams> scene_cameras;

    SceneDatasetParams dataset_params;

    // We use the data array to store some additional attributes
    // data: [radius, density, rand, rand]
    Saiga::UnifiedMesh point_cloud;

    std::string file_dataset_base;
    std::string file_point_cloud, file_point_cloud_compressed;
    std::string file_intrinsics;
    std::string file_camera_indices;
    std::string file_pose;
    std::string file_exposure;
    std::string file_white_balance;
    std::string file_image_names, file_mask_names;

    std::string scene_name;

    std::string scene_path;
    std::vector<FrameData> frames;

    void Save(bool extended_save = false);
    static void SavePoses(std::vector<SE3> poses, std::string file);

    TemplatedImage<ucvec3> CPURenderFrame(int id, float scale);

    FrameData Frame(int id) { return frames[id]; }

    std::vector<int> Indices()
    {
        std::vector<int> r;
        for (int i = 0; i < frames.size(); ++i)
        {
            r.push_back(i);
        }
        return r;
    }

    void AddPointNoise(float sdev);
    void AddPoseNoise(float sdev_rot, float sdev_trans);
    void AddIntrinsicsNoise(float sdev_K, float sdev_dist);

    void DuplicatePoints(int factor, float dis);

    // after downsample
    // ~ n/factor points remain
    void DownsamplePoints(float factor);

    void PointDistanceToCamera();


    void RemoveClosePoints(float dis_th);
    void RemoveLonelyPoints(int n, float dis);


    // Searches the 4 closest neighbours and stores the distance of the furthest neighbour
    // in data[0]
    void ComputeRadius(int n = 4);

    void SortBlocksByRadius(int block_size);

    Saiga::UnifiedMesh OutlierPointCloud(int n, float distance);
};