/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/glfw/all.h"

#include "config.h"
#include "data/SceneData.h"
#include "opengl/NeuralPointCloudOpenGL.h"
#include "../config.h"

using namespace Saiga;

class SceneViewer : public glfw_KeyListener
{
   public:
    SceneViewer(std::shared_ptr<SceneData> scene);

    void OptimizePoints();

    void CreateMasks(bool mult_old_mask);

    void RemoveInvalidPoints();

    void SetRandomPointColor();

    // Render the shapes for the cameras
    void RenderDebug(Camera* cam);

    std::shared_ptr<ColoredAsset> capture_asset;
    std::shared_ptr<LineVertexColoredAsset> frustum_asset;

    void Select(Ray r);

    void UpdatePointCloudBuffer();

    FrameData& Current()
    {
        SAIGA_ASSERT(selected_capture >= 0 && selected_capture < scene->frames.size());
        return scene->frames[selected_capture];
    }

    void DeleteCurrent()
    {
        if (selected_capture != -1)
        {
            scene->frames[selected_capture] = scene->frames.back();
            scene->frames.pop_back();
            selected_capture = -1;
        }
    }

    void imgui();


    std::shared_ptr<NeuralPointCloudOpenGL> gl_points;

    std::shared_ptr<SceneData> scene;

    UnifiedModel model;

    ImageInfo CurrentFrameData() const
    {
        ImageInfo fd;
        fd.w              = scene->scene_cameras[0].w * scene->dataset_params.render_scale;
        fd.h              = scene->scene_cameras[0].h * scene->dataset_params.render_scale;
        fd.K              = scene->scene_cameras[0].K;
        fd.distortion     = scene->scene_cameras[0].distortion;
        fd.ocam           = scene->scene_cameras[0].ocam.cast<float>();
        fd.crop_transform = fd.crop_transform.scale(scene->dataset_params.render_scale);
        fd.pose           = Sophus::SE3f::fitToSE3(scene_camera.model * GL2CVView()).cast<double>();
        return fd;
    }
    Glfw_Camera<PerspectiveCamera> scene_camera;

    // temp values
    int selected_capture = -1;
    bool render_frustums = true;


    virtual void keyPressed(int key, int scancode, int mods) override
    {
        switch (key)
        {
            case GLFW_KEY_X:
            {
                DeleteCurrent();
                break;
            }
            default:
                break;
        };
    }
};
