/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/imgui_cuda.h"
#include "saiga/cuda/interop.h"
#include "saiga/opengl/all.h"
#include "saiga/opengl/rendering/deferredRendering/tone_mapper.h"
#include "saiga/opengl/rendering/lighting/bloom.h"

#include "config.h"
#include "data/Dataset.h"
#include "models/Pipeline.h"
#include "opengl/SceneViewer.h"

// Helper class which is able to use a pretrained network to render a neural point cloud
// The output is an RGB image of the given viewpoint
class RealTimeRenderer
{
   public:
    RealTimeRenderer(std::shared_ptr<SceneData> scene);

    void Forward(Camera* cam, ImageInfo fd);

    void Render(ImageInfo fd);

    // flags:
    // 0: color
    //
    void RenderColor(ImageInfo fd, int flags);

    void imgui();

    struct Experiment
    {
        // full (absolute) directory of the experimenet folder
        std::string dir;

        // only the name
        std::string name;

        struct EP
        {
            // full (absolute) directory of the epxxxx folder
            std::string dir;

            // only the name. for example "ep0005"
            std::string name;

            // for example "kemenate"
            std::string scene_name;

            // ep number
            int ep = 0;
        };
        std::vector<EP> eps;

        Experiment(std::string dir, std::string name, std::string scene_name, bool render_able = true);
    };


    std::string experiments_base = "experiments/";
    std::vector<Experiment> experiments;
    int current_ex      = 0;
    int current_ep      = 0;
    int current_best_gt = -1;
    int best_gt_counter = 0;

    bool mouse_on_view = false;
    void LoadNets();

    TemplatedImage<vec4> output_image;
    TemplatedImage<ucvec4> output_image_ldr;
    std::shared_ptr<Texture> output_texture, output_texture_ldr, output_color, best_gt_texture;
    std::shared_ptr<Saiga::CUDA::Interop> texure_interop, color_interop;

    NeuralPointTexture color_texture = nullptr;

    std::shared_ptr<SceneData> scene;
    std::shared_ptr<NeuralScene> ns;
    std::shared_ptr<NeuralPipeline> pipeline;

    // The real-time camera parameters for live viewing
    IntrinsicsModule rt_intrinsics = nullptr;
    PoseModule rt_extrinsics       = nullptr;

    torch::Tensor uv_tensor, uv_tensor_center;
    bool use_center_tensor = false;
    int debug_point_layer  = 0;

    bool use_gl_tonemapping = false;
    bool use_bloom          = false;
    bool render_color       = true;

    // The custom camera can be controlled by the user and might be a different model
    bool use_custom_camera = true;

    int color_layer   = 1;
    int color_flags   = 0;
    float color_scale = 1.f;

    // int color_flags   = 1;
    // float color_scale = 16.f;



    torch::DeviceType device = torch::kCUDA;
    CUDA::CudaTimerSystem timer_system;

    // Default saiga renderer uses 16-bit float for HDR content
    ToneMapper tone_mapper = {GL_RGBA32F};
    Bloom bloom            = {GL_RGBA32F};
    std::shared_ptr<CombinedParams> params;

    TemplatedImage<ucvec4> DownloadRender()
    {
        if (use_gl_tonemapping)
        {
            SAIGA_ASSERT(output_texture_ldr);
            TemplatedImage<ucvec4> tmp(output_texture_ldr->getHeight(), output_texture_ldr->getWidth());

            output_texture_ldr->bind();
            glGetTexImage(output_texture_ldr->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
            assert_no_glerror();
            output_texture_ldr->unbind();

            return tmp;
        }
        else
        {
            SAIGA_ASSERT(output_texture);
            TemplatedImage<ucvec4> tmp(output_texture->getHeight(), output_texture->getWidth());

            output_texture->bind();
            glGetTexImage(output_texture->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
            assert_no_glerror();
            output_texture->unbind();

            return tmp;
        }
    }

    TemplatedImage<ucvec4> DownloadColor()
    {
        SAIGA_ASSERT(output_color);
        TemplatedImage<ucvec4> tmp(output_color->getHeight(), output_color->getWidth());

        output_color->bind();
        glGetTexImage(output_color->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
        assert_no_glerror();
        output_color->unbind();
        return tmp;
    }

    TemplatedImage<ucvec4> DownloadGt()
    {
        SAIGA_ASSERT(best_gt_texture);
        TemplatedImage<ucvec4> tmp(best_gt_texture->getHeight(), best_gt_texture->getWidth());

        best_gt_texture->bind();
        glGetTexImage(best_gt_texture->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
        assert_no_glerror();
        best_gt_texture->unbind();
        return tmp;
    }
};