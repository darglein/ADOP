/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "adop_viewer.h"

#include "saiga/core/geometry/cameraAnimation.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/exif/TinyEXIF.h"


ADOPViewer::ADOPViewer(std::string scene_dir, std::unique_ptr<DeferredRenderer> renderer_,
                       std::unique_ptr<WindowType> window_)
    : StandaloneWindow<wm, DeferredRenderer>(std::move(renderer_), std::move(window_))
{
    main_menu.AddItem(
        "Saiga", "MODEL", [this]() { view_mode = ViewMode::MODEL; }, GLFW_KEY_F1, "F1");


    main_menu.AddItem(
        "Saiga", "NEURAL", [this]() { view_mode = ViewMode::NEURAL; }, GLFW_KEY_F2, "F2");


    main_menu.AddItem(
        "Saiga", "SPLIT_NEURAL", [this]() { view_mode = ViewMode::SPLIT_NEURAL; }, GLFW_KEY_F3, "F3");

    std::cout << "Program Initialized!" << std::endl;

    std::cout << "Loading Scene " << scene_dir << std::endl;
    LoadScene(scene_dir);
    LoadSceneImpl();

    std::filesystem::create_directories("videos/");
    recording_dir = "videos/" + scene->scene->scene_name + "/";
    view_mode     = ViewMode::SPLIT_NEURAL;
}


void ADOPViewer::LoadSceneImpl()
{
    if (renderer->tone_mapper.auto_exposure || renderer->tone_mapper.auto_white_balance)
    {
        renderer->tone_mapper.download_tmp_values = true;
    }

    renderer->lighting.pointLights.clear();
    renderer->lighting.directionalLights.clear();
    renderer->lighting.spotLights.clear();

    ::camera = &scene->scene_camera;
    window->setCamera(camera);

    if (scene->scene->point_cloud.NumVertices() > 15000000)
    {
        // by default don't render very large point clouds in the viewport
        render_points = false;
    }
    auto& f = scene->scene->frames.front();
    camera->setModelMatrix(f.OpenglModel());
    camera->updateFromModel();

    renderer->params.useSSAO = false;
    sun                      = std::make_shared<DirectionalLight>();
    sun->ambientIntensity    = exp2(scene->scene->dataset_params.scene_exposure_value);
    sun->intensity           = 0;
    renderer->lighting.AddLight(sun);
    renderer->tone_mapper.params.exposure_value = scene->scene->dataset_params.scene_exposure_value;
}

void ADOPViewer::render(RenderInfo render_info)
{
    if (renderObject && render_info.render_pass == RenderPass::Deferred)
    {
        if (!scene->model.mesh.empty())
        {
            if (renderTexture && scene->model.mesh.front().HasTC())
            {
                if (!object_tex)
                {
                    object_tex = std::make_shared<TexturedAsset>(scene->model);
                }
                object_tex->render(render_info.camera, mat4::Identity());
            }
            else
            {
                if (!object_col)
                {
                    scene->model.ComputeColor();
                    auto mesh = scene->model.CombinedMesh(VERTEX_POSITION | VERTEX_NORMAL | VERTEX_COLOR).first;
                    mesh.RemoveDoubles(0.001);
                    // mesh.SmoothVertexColors(10, 0);
                    object_col = std::make_shared<ColoredAsset>(mesh);
                }
                object_col->render(render_info.camera, mat4::Identity());
            }
        }
    }

    if (render_info.render_pass == RenderPass::Final)
    {
        if (view_mode == ViewMode::NEURAL)
        {
            auto fd           = scene->CurrentFrameData();
            fd.w              = fd.w * render_scale;
            fd.h              = fd.h * render_scale;
            fd.K              = fd.K.scale(render_scale);
            fd.exposure_value = renderer->tone_mapper.params.exposure_value;
            fd.white_balance  = renderer->tone_mapper.params.white_point;

            if (!neural_renderer)
            {
                neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene);
            }
            neural_renderer->tone_mapper.params      = renderer->tone_mapper.params;
            neural_renderer->tone_mapper.tm_operator = renderer->tone_mapper.tm_operator;
            neural_renderer->tone_mapper.params.exposure_value -= scene->scene->dataset_params.scene_exposure_value;
            neural_renderer->tone_mapper.params_dirty = true;

            neural_renderer->timer_system.BeginFrame();
            neural_renderer->Render(fd);
            neural_renderer->timer_system.EndFrame();

            if (neural_renderer->use_gl_tonemapping)
            {
                display.render(neural_renderer->output_texture_ldr.get(), ivec2(0, 0), renderer->viewport_size, true);
            }
            else
            {
                display.render(neural_renderer->output_texture.get(), ivec2(0, 0), renderer->viewport_size, true);
            }
        }
    }

    // Debug view of point cloud + image frames
    if ((view_mode == ViewMode::SPLIT_NEURAL || view_mode == ViewMode::MODEL) &&
        render_info.render_pass == RenderPass::Forward)
    {
        if (render_debug)
        {
            scene->RenderDebug(render_info.camera);
        }


        if (spline_mesh)
        {
            glLineWidth(3);
            spline_mesh->renderForward(render_info.camera, mat4::Identity());
            glLineWidth(1);
        }

        if (render_points)
        {
            if (!scene->gl_points)
            {
                scene->gl_points = std::make_shared<NeuralPointCloudOpenGL>(scene->scene->point_cloud);
            }
            // Create a frame around the current gl-camera
            FrameData fd;
            fd.w              = renderer->viewport_size.x();
            fd.h              = renderer->viewport_size.y();
            fd.pose           = Sophus::SE3f::fitToSE3(camera->model * GL2CVView()).cast<double>();
            fd.K              = GLProjectionMatrix2CVCamera(camera->proj, fd.w, fd.h);
            fd.distortion     = scene->scene->scene_cameras.front().distortion;
            fd.exposure_value = scene->scene->dataset_params.scene_exposure_value;
            {
                auto tim = renderer->timer->Measure("GL_POINTS render");
                scene->gl_points->render(fd, 0);
            }
        }


        if (renderWireframe)
        {
            glEnable(GL_POLYGON_OFFSET_LINE);
            //        glLineWidth(1);
            glPolygonOffset(0, -500);

            // object.renderWireframe(cam);
            glDisable(GL_POLYGON_OFFSET_LINE);
        }
    }

    if (render_info.render_pass == RenderPass::GUI)
    {
        if (ImGui::Begin("Video Recording"))
        {
        }
        ImGui::End();
        ViewerBase::imgui();


        ImGui::Begin("Model Viewer");

        ImGui::SliderFloat("render_scale", &render_scale, 0.1, 2);


        if (neural_renderer)
        {
            if (ImGui::Button("Set to closest frame"))
            {
                auto& f = scene->scene->frames[neural_renderer->current_best_gt];
                ::camera->setModelMatrix(f.OpenglModel());
                ::camera->updateFromModel();
                renderer->tone_mapper.params.exposure_value = f.exposure_value;
                renderer->tone_mapper.params_dirty          = true;
            }
        }
        ImGui::End();

        auto fd = scene->CurrentFrameData();

        fd.w              = fd.w * render_scale;
        fd.h              = fd.h * render_scale;
        fd.K              = fd.K.scale(render_scale);
        fd.exposure_value = renderer->tone_mapper.params.exposure_value;
        fd.white_balance  = renderer->tone_mapper.params.white_point;


        if (view_mode == ViewMode::SPLIT_NEURAL)
        {
            if (!neural_renderer)
            {
                neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene);
            }
            mouse_in_gt                              = neural_renderer->mouse_on_view;
            neural_renderer->tone_mapper.params      = renderer->tone_mapper.params;
            neural_renderer->tone_mapper.tm_operator = renderer->tone_mapper.tm_operator;
            neural_renderer->tone_mapper.params.exposure_value -= scene->scene->dataset_params.scene_exposure_value;
            neural_renderer->tone_mapper.params_dirty = true;
            // neural_renderer->tone_mapper.params_dirty |= renderer->tone_mapper.params_dirty;
            neural_renderer->Forward(&scene->scene_camera, fd);
        }

        if (ImGui::Begin("Video Recording"))
        {
            Recording(fd);
        }
        ImGui::End();
    }
}

void ADOPViewer::Recording(ImageInfo& fd)
{
    std::string out_dir = recording_dir;

    static bool is_recording = false;
    static bool downscale_gt = false;

    static bool interpolate_exposure = false;

    static bool record_debug  = true;
    static bool record_gt     = true;
    static bool record_neural = true;
    ImGui::Checkbox("record_debug", &record_debug);
    ImGui::Checkbox("record_gt", &record_gt);
    ImGui::Checkbox("record_neural", &record_neural);
    ImGui::Checkbox("downscale_gt", &downscale_gt);


    static std::vector<SplineKeyframe> traj;


    auto insert = [this](int id)
    {
        SplineKeyframe kf;
        kf.user_index = id;
        kf.pose       = scene->scene->frames[id].pose;
        camera_spline.Insert(kf);
    };

    bool update_curve         = false;
    static bool hdr_video_gen = false;

    static int current_frame = 0;

    if (is_recording)
    {
        std::string frame_name = std::to_string(current_frame) + ".png";

        if (record_neural)
        {
            auto frame = neural_renderer->DownloadRender();
            frame.save(out_dir + "/neural/" + frame_name);
        }
        if (record_debug)
        {
            auto frame = neural_renderer->DownloadColor();
            frame.save(out_dir + "/debug/" + frame_name);
        }

        if (record_gt)
        {
            TemplatedImage<ucvec4> frame = neural_renderer->DownloadGt();

            if (downscale_gt)
            {
                TemplatedImage<ucvec4> gt_small(frame.h / 2, frame.w / 2);
                frame.getImageView().copyScaleDownPow2(gt_small.getImageView(), 2);
                frame = gt_small;
            }
            frame.save(out_dir + "/gt/" + frame_name);
        }

        current_frame++;


        if (current_frame == traj.size() || ImGui::Button("stop recording"))
        {
            is_recording = false;
        }
    }

    if (ImGui::Button("preset lighthouse"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 60;

        std::vector<int> kfs = {108, 108, 121, 138, 6, 43, 71, 90, 108, 108};
        for (auto i : kfs)
        {
            insert(i);
        }
        renderer->tone_mapper.params.exposure_value = 0;
        renderer->tone_mapper.params_dirty          = true;

        update_curve = true;
    }

    if (ImGui::Button("preset m60"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 60;

        std::vector<int> kfs = {0, 0, 148, 27, 190, 67, 84, 90, 96, 277, 290, 303, 0, 0};
        for (auto i : kfs)
        {
            insert(i);
        }
        renderer->tone_mapper.params.exposure_value = scene->scene->frames[kfs[0]].exposure_value;
        renderer->tone_mapper.params_dirty          = true;

        update_curve = true;
    }

    if (ImGui::Button("preset playground"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 60;

        std::vector<int> kfs = {0, 0, 23, 52, 73, 88, 98, 136, 157, 170, 0, 0};
        for (auto i : kfs)
        {
            insert(i);
        }
        renderer->tone_mapper.params.exposure_value = 0;
        renderer->tone_mapper.params_dirty          = true;

        update_curve = true;
    }

    if (ImGui::Button("preset train"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 60;

        std::vector<int> kfs = {0, 0, 13, 20, 30, 40, 50, 207, 95, 100, 110, 120, 130, 140, 150, 160, 0, 0};
        for (auto i : kfs)
        {
            insert(i);
        }
        renderer->tone_mapper.params.exposure_value = scene->scene->frames[kfs[0]].exposure_value;
        renderer->tone_mapper.params_dirty          = true;

        update_curve = true;
    }


    if (ImGui::Button("preset boat"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 60;

        std::vector<int> kfs = {// outer circle
                                568, 568, 568, 590, 679, 556, 68, 101, 146, 190, 224, 290, 346, 446, 490, 590, 194,
                                // inner
                                442, 447, 462, 474, 475, 477, 372,
                                // close up sign
                                543, 548, 548, 541, 398,
                                // end
                                396, 664, 663, 663, 663};
        for (auto i : kfs)
        {
            insert(i);
        }

        renderer->tone_mapper.params.exposure_value = 14.1;
        renderer->tone_mapper.params_dirty          = true;
        renderer->tone_mapper.tm_operator           = 4;

        neural_renderer->use_gl_tonemapping = true;

        camera_spline.time_in_seconds = 30;
        downscale_gt                  = true;

        update_curve = true;
    }

    if (ImGui::Button("preset boat hdr stuff"))
    {
        camera_spline.keyframes.clear();
        camera_spline.frame_rate = 30;

        std::vector<int> kfs = {663, 663, 663, 663};
        for (auto i : kfs)
        {
            insert(i);
        }
        renderer->tone_mapper.params.exposure_value = 14.1;
        renderer->tone_mapper.params_dirty          = true;
        renderer->tone_mapper.tm_operator           = 4;


        auto exp_ref    = scene->scene->frames[kfs[0]].exposure_value;
        auto exp_target = 13;

        Eigen::Matrix<double, -1, 1> x(1);
        for (auto& k : camera_spline.keyframes)
        {
            k.user_data = x;
        }


        camera_spline.keyframes[0].user_data(0) = exp_ref;
        camera_spline.keyframes[1].user_data(0) = exp_ref;
        camera_spline.keyframes[2].user_data(0) = exp_target;
        camera_spline.keyframes[3].user_data(0) = exp_target;

        hdr_video_gen = true;

        camera_spline.time_in_seconds = 15;

        update_curve = true;
    }


    update_curve |= camera_spline.imgui();

    if (ImGui::Button("add yellow frame"))
    {
        if (scene->selected_capture >= 0)
        {
            insert(scene->selected_capture);
            update_curve = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("remove yellow frame"))
    {
        int id = -1;
        for (int i = 0; i < camera_spline.keyframes.size(); ++i)
        {
            if (camera_spline.keyframes[i].user_index == scene->selected_capture)
            {
                id = i;
            }
        }
        std::cout << "remove id " << id << " selected " << scene->selected_capture << std::endl;
        if (id != -1)
        {
            camera_spline.keyframes.erase(camera_spline.keyframes.begin() + id);
            update_curve = true;
        }
    }

    if (ImGui::Button("add gl camera"))
    {
        SplineKeyframe kf;
        kf.user_index = 0;
        kf.pose       = Sophus::SE3f::fitToSE3(scene->scene_camera.model * GL2CVView()).cast<double>();
        camera_spline.Insert(kf);
        update_curve = true;
    }

    if (update_curve)
    {
        for (auto& f : scene->scene->frames)
        {
            f.display_color = vec4(1, 0, 0, 1);
        }
        for (auto& kf : camera_spline.keyframes)
        {
            scene->scene->frames[kf.user_index].display_color = vec4(0, 1, 0, 1);
        }
        camera_spline.updateCurve();

        auto mesh = camera_spline.ProxyMesh();
        if (mesh.NumVertices() > 0)
        {
            spline_mesh = std::make_shared<LineVertexColoredAsset>(
                mesh.SetVertexColor(exp2(scene->scene->dataset_params.scene_exposure_value) * vec4(0, 1, 0, 1)));
        }
    }

    if (!is_recording && ImGui::Button("start recording"))
    {
        neural_renderer->current_best_gt = -1;
        is_recording                     = true;
        traj                             = camera_spline.Trajectory();
        current_frame                    = 0;
        std::filesystem::create_directories(out_dir);
        if (record_debug)
        {
            std::filesystem::remove_all(out_dir + "/debug");
            std::filesystem::create_directories(out_dir + "/debug");
        }
        if (record_gt)
        {
            std::filesystem::remove_all(out_dir + "/gt");
            std::filesystem::create_directories(out_dir + "/gt");
        }
        if (record_neural)
        {
            std::filesystem::remove_all(out_dir + "/neural");
            std::filesystem::create_directories(out_dir + "/neural");
        }
    }

    if (is_recording && !traj.empty())
    {
        auto frame = traj[current_frame];

        mat4 model = frame.pose.matrix().cast<float>() * CV2GLView();

        if (interpolate_exposure)
        {
            float alpha  = 0.001;
            auto new_exp = scene->scene->frames[neural_renderer->current_best_gt].exposure_value;
            renderer->tone_mapper.params.exposure_value =
                (1 - alpha) * renderer->tone_mapper.params.exposure_value + alpha * new_exp;
            renderer->tone_mapper.params_dirty = true;
        }


        if (hdr_video_gen)
        {
            auto new_exp                                = frame.user_data(0);
            renderer->tone_mapper.params.exposure_value = new_exp;
            renderer->tone_mapper.params_dirty          = true;
        }

        ::camera->setModelMatrix(model);
        ::camera->updateFromModel();
    }
}

int main(int argc, char* argv[])
{
    float render_scale = 1.0f;
    std::string scene_dir;
    CLI::App app{"ADOP Viewer for Scenes", "adop_viewer"};
    app.add_option("--scene_dir", scene_dir)->required();
    app.add_option("--render_scale", render_scale);
    CLI11_PARSE(app, argc, argv);


    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    WindowParameters windowParameters;
    OpenGLParameters openglParameters;
    DeferredRenderingParameters rendererParameters;
    windowParameters.fromConfigFile("config.ini");
    rendererParameters.hdr = true;

    auto window   = std::make_unique<WindowType>(windowParameters, openglParameters);
    auto renderer = std::make_unique<DeferredRenderer>(*window, rendererParameters);


    MainLoopParameters mlp;
    ADOPViewer viewer(scene_dir, std::move(renderer), std::move(window));
    viewer.render_scale = render_scale;
    viewer.run(mlp);
}
