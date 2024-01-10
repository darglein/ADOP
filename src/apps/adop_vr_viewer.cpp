/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/rendering/VRRendering/VRRenderer.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/window/glfw_window.h"
#include "saiga/opengl/world/TextureDisplay.h"
#include "saiga/opengl/world/proceduralSkybox.h"

#include "viewer_base.h"
using namespace Saiga;

class ADOPVRViewer : public StandaloneWindow<WindowManagement::GLFW, VRRenderer>, public glfw_KeyListener, ViewerBase
{
   public:
    ADOPVRViewer(std::string scene_dir) : StandaloneWindow("config.ini")
    {
        main_menu.AddItem(
            "Saiga", "MODEL", [this]() { view_mode = ViewMode::MODEL; }, GLFW_KEY_F1, "F1");

        main_menu.AddItem(
            "Saiga", "NEURAL", [this]() { view_mode = ViewMode::NEURAL; }, GLFW_KEY_F2, "F2");

        LoadScene(scene_dir);
        LoadSceneImpl();

        auto& f = scene->scene->frames[18];
        camera->setModelMatrix(f.OpenglModel());
        camera->updateFromModel();

        view_mode = ViewMode::NEURAL;
        std::cout << "Program Initialized!" << std::endl;
    }
    ~ADOPVRViewer() {}

    void LoadSceneImpl()
    {
        ::camera = &scene->scene_camera;
        window->setCamera(camera);
        renderer->tone_mapper.params.exposure_value = scene->scene->dataset_params.scene_exposure_value;

        auto& f = scene->scene->frames.front();
        camera->setModelMatrix(f.OpenglModel());
        camera->updateFromModel();

        renderer->lighting.directionalLights.clear();
        sun                   = std::make_shared<DirectionalLight>();
        sun->ambientIntensity = exp2(scene->scene->dataset_params.scene_exposure_value);
        sun->intensity        = 0;
        renderer->lighting.AddLight(sun);
        renderer->tone_mapper.params.exposure_value = scene->scene->dataset_params.scene_exposure_value;
    }

    void update(float dt) override
    {
        int FORWARD = keyboard.getKeyState(GLFW_KEY_W) - keyboard.getKeyState(GLFW_KEY_S);
        {
            float speed = dt * FORWARD;
            if (keyboard.getKeyState(GLFW_KEY_LEFT_SHIFT))
            {
                speed *= 5;
            }
            vec3 dir = camera->rot * renderer->VR().LookingDirection();
            camera->position.head<3>() = camera->position.head<3>() + dir * speed;
            camera->calculateModel();
        }
    }
    void interpolate(float dt, float interpolation) override
    {
        if (renderer->use_mouse_input_in_3dview || renderer->use_mouse_input_in_3dview)
        {
            scene->scene_camera.interpolate(dt, interpolation);
        }
    }



    void render(RenderInfo render_info) override
    {
        if (view_mode == ViewMode::MODEL && render_info.render_pass == RenderPass::Deferred)
        {
            if (!object_tex)
            {
                object_tex = std::make_shared<TexturedAsset>(scene->model);
            }
            object_tex->render(camera, mat4::Identity());
        }

        if (render_info.render_pass == RenderPass::Final && view_mode == ViewMode::NEURAL)
        {
            auto fd = scene->CurrentFrameData();

            fd.w = renderer->viewport_size.x() * render_scale;
            fd.h = renderer->viewport_size.y() * render_scale;

            // fd.w = iAlignUp(fd.w, 32);
            // fd.h = iAlignUp(fd.h, 32);

            fd.distortion = Distortionf();

            fd.K = GLProjectionMatrix2CVCamera(camera->proj, fd.w, fd.h);



            fd.pose = Sophus::SE3f::fitToSE3(camera->model * GL2CVView()).cast<double>();

            fd.exposure_value = renderer->tone_mapper.params.exposure_value;
            fd.white_balance  = renderer->tone_mapper.params.white_point;

            if (!neural_renderer)
            {
                neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene);
                // neural_renderer->tone_mapper = &renderer->tone_mapper;
            }
            neural_renderer->tone_mapper.params      = renderer->tone_mapper.params;
            neural_renderer->tone_mapper.tm_operator = renderer->tone_mapper.tm_operator;
            neural_renderer->tone_mapper.params.exposure_value -= scene->scene->dataset_params.scene_exposure_value;
            neural_renderer->tone_mapper.params_dirty = true;
            neural_renderer->Render(fd);

            if (neural_renderer->use_gl_tonemapping)
            {
                display.render(neural_renderer->output_texture_ldr.get(), ivec2(0, 0), renderer->viewport_size, true);
            }
            else
            {
                display.render(neural_renderer->output_texture.get(), ivec2(0, 0), renderer->viewport_size, true);
            }
        }

        if (render_info.render_pass == RenderPass::GUI)
        {
            ImGui::Begin("Model Viewer");
            ImGui::SliderFloat("render_scale", &render_scale, 0.01, 1);
            ImGui::End();
            ViewerBase::imgui();
        }
    }


    void keyPressed(int key, int scancode, int mods) override
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                window->close();
                break;
            default:
                break;
        }
    }

   private:
    TextureDisplay display;
    std::shared_ptr<DirectionalLight> sun;

    float render_scale = 0.75;
    ViewMode view_mode = ViewMode::MODEL;
};




int main(int argc, char* argv[])
{
    std::string scene_dir;
    CLI::App app{"ADOP VR Viewer for Scenes", "adop_vr_viewer"};
    app.add_option("--scene_dir", scene_dir)->required();
    CLI11_PARSE(app, argc, argv);

    initSaigaSample();
    ADOPVRViewer window(scene_dir);
    window.run();
    return 0;
}
