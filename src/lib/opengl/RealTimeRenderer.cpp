/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "RealTimeRenderer.h"

#include "saiga/colorize.h"
#include "saiga/opengl/imgui/imgui_opengl.h"

RealTimeRenderer::Experiment::Experiment(std::string dir, std::string name, std::string scene_name, bool render_able)
    : dir(dir), name(name)
{
    if (!std::filesystem::exists(dir + "params.ini"))
    {
        return;
    }

    Directory d(dir);
    auto ep_dirs = d.getDirectories();

    ep_dirs.erase(std::remove_if(ep_dirs.begin(), ep_dirs.end(), [](auto str) { return !hasPrefix(str, "ep"); }),
                  ep_dirs.end());

    if (ep_dirs.empty()) return;
    std::sort(ep_dirs.begin(), ep_dirs.end());

    std::cout << "Found experiment " << dir << " with " << ep_dirs.size() << " epochs" << std::endl;



    for (auto ep_dir : ep_dirs)
    {
        EP ep;
        ep.name       = ep_dir;
        ep.dir        = dir + ep_dir;
        ep.scene_name = scene_name;
        if (render_able)
        {
            if (!std::filesystem::exists(ep.dir + "/render_net.pth") ||
                !std::filesystem::exists(ep.dir + "/scene_" + scene_name + "_texture.pth"))
            {
                continue;
            }
        }

        eps.emplace_back(ep);
    }
}

RealTimeRenderer::RealTimeRenderer(std::shared_ptr<SceneData> scene) : scene(scene)
{
    Directory dir(experiments_base);
    auto ex_names = dir.getDirectories();
    std::sort(ex_names.begin(), ex_names.end(), std::greater<std::string>());

    for (auto n : ex_names)
    {
        Experiment e(experiments_base + "/" + n + "/", n, scene->scene_name);
        if (!e.eps.empty())
        {
            experiments.push_back(e);
        }
    }

    current_ex = 0;
    // Load last in list
    if (!experiments.empty()) current_ep = experiments[current_ex].eps.size() - 1;

    LoadNets();
    torch::set_num_threads(4);
}

void RealTimeRenderer::Forward(Camera* cam, ImageInfo fd)
{
    if (experiments.empty()) return;

    timer_system.BeginFrame();

    mouse_on_view = false;
    if (ImGui::Begin("Neural View"))
    {
        ImGui::BeginChild("neural_child", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
        mouse_on_view |= ImGui::IsWindowHovered();


        Render(fd);

        if (use_gl_tonemapping)
        {
            ImGui::Texture(output_texture_ldr.get(), ImGui::GetWindowContentRegionMax(), false);
        }
        else
        {
            ImGui::Texture(output_texture.get(), ImGui::GetWindowContentRegionMax(), false);
        }

        ImGui::EndChild();
    }
    ImGui::End();

    if (ImGui::Begin("Debug View"))
    {
        ImGui::BeginChild("dbg_child", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
        mouse_on_view |= ImGui::IsWindowHovered();
        if (render_color)
        {
            RenderColor(fd, color_flags);
            ImGui::Texture(output_color.get(), ImGui::GetWindowContentRegionMax(), false);
        }

        ImGui::EndChild();
    }
    ImGui::End();


    {
        // find closest gt image
        std::vector<std::pair<double, int>> err_t, err_r, score;

        for (int i = 0; i < scene->frames.size(); ++i)
        {
            auto& f  = scene->frames[i];
            auto err = translationalError(fd.pose, f.pose);

            Vec3 dir1 = fd.pose.so3() * Vec3(0, 0, 1);
            Vec3 dir2 = f.pose.so3() * Vec3(0, 0, 1);

            auto err_angle = degrees(acos(dir1.dot(dir2)));

            err_t.push_back({err, i});
            err_r.push_back({err_angle, i});
        }

        std::sort(err_t.begin(), err_t.end());
        std::sort(err_r.begin(), err_r.end());

        for (int i = 0; i < err_t.size(); ++i)
        {
            score.push_back({0, i});
        }

        for (int i = 0; i < err_t.size(); ++i)
        {
            score[err_t[i].second].first += i;
            score[err_r[i].second].first += i;
        }
        std::sort(score.begin(), score.end());

        int best_idx = score.front().second;



        if (best_idx != current_best_gt)
        {
            if (current_best_gt != -1 && best_gt_counter < 20)
            {
                // smooth out a bit so we change only after every 30 frames
                best_gt_counter++;
            }
            else
            {
                auto& f = scene->frames[best_idx];
                console << "Current Best (img,cam) = (" << f.image_index << "," << f.camera_index
                        << ") EV: " << f.exposure_value << std::endl;
                Image img;
                if (img.load(scene->dataset_params.image_dir + "/" + f.target_file))
                {
                    best_gt_texture = std::make_shared<Texture>(img);
                }
                else
                {
                    std::cout << "Failed to load Ground Truth image. Check if 'image_dir' in dataset.ini is correct!"
                              << std::endl;
                }
                current_best_gt = best_idx;
                best_gt_counter = 0;
            }
        }

        if (ImGui::Begin("Closest Ground Truth"))
        {
            ImGui::BeginChild("gt_child", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
            if (best_gt_texture)
            {
                ImGui::Texture(best_gt_texture.get(), ImGui::GetWindowContentRegionMax(), false);
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }


    timer_system.EndFrame();
}
void RealTimeRenderer::imgui()
{
    if (experiments.empty())
    {
        // std::cout << "no experiments :(" << std::endl;
        return;
    }
    if (ImGui::Begin("Neural Renderer"))
    {
        ImGui::SetNextItemWidth(400);
        if (ImGui::ListBoxHeader("###experiments", 10))
        {
            for (int i = 0; i < experiments.size(); ++i)
            {
                if (ImGui::Selectable(experiments[i].name.c_str(), current_ex == i))
                {
                    current_ex = i;
                    current_ep = experiments[i].eps.size() - 1;
                    LoadNets();
                }
            }
            ImGui::ListBoxFooter();
        }
        // ImGui::SameLine();
        ImGui::SetNextItemWidth(400);
        if (ImGui::ListBoxHeader("###Eps", 10))
        {
            auto& ex = experiments[current_ex];
            for (int i = 0; i < ex.eps.size(); ++i)
            {
                if (ImGui::Selectable(ex.eps[i].name.c_str(), current_ep == i))
                {
                    current_ep = i;
                    LoadNets();
                }
            }
            ImGui::ListBoxFooter();
        }

        ImGui::Checkbox("use_center_tensor", &use_center_tensor);
        ImGui::Checkbox("use_custom_camera", &use_custom_camera);

        if (pipeline)
        {
            if (ImGui::CollapsingHeader("Params"))
            {
                pipeline->params->imgui();
            }

            ImGui::SliderInt("color_layer", &color_layer, 1, 5);

            ImGui::Checkbox("enable_response", &ns->camera->params.enable_response);
            ImGui::Checkbox("enable_white_balance", &ns->camera->params.enable_white_balance);
            ImGui::Checkbox("enable_vignette", &ns->camera->params.enable_vignette);
            ImGui::Checkbox("enable_exposure", &ns->camera->params.enable_exposure);

            ImGui::SliderInt("color_flags", &color_flags, 0, 1);
            ImGui::SliderFloat("color_scale", &color_scale, 0, 16);



            if (ImGui::Button("write optimized camera poses"))
            {
                auto poses = ns->poses->Download();
                std::ofstream strm("poses_quat_wxyz_pos_xyz.txt");
                for (auto pf : poses)
                {
                    SE3 p  = pf.cast<double>().inverse();
                    auto q = p.unit_quaternion();
                    auto t = p.translation();

                    strm << std::setprecision(8) << std::scientific << q.w() << " " << q.x() << " " << q.y() << " "
                         << q.z() << " " << t.x() << " " << t.y() << " " << t.z() << "\n";
                }
            }

            static float max_dens = 5;
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("###max_dens", &max_dens);
            ImGui::SameLine();
            if (ImGui::Button("Vis. density"))
            {
                auto pc = scene->point_cloud;
                SAIGA_ASSERT(pc.data.size() == pc.color.size());

                for (int i = 0; i < pc.data.size(); ++i)
                {
                    float f               = pc.data[i](0) / max_dens;
                    vec3 c                = colorizeFusion(f);
                    pc.color[i].head<3>() = c;
                    pc.color[i](3)        = 1;
                }

                color_texture = NeuralPointTexture(pc);
                color_texture->to(device);
            }
            if (ImGui::Button("Vis. color"))
            {
                color_texture = NeuralPointTexture(scene->point_cloud);
                color_texture->to(device);
            }

            ImGui::Checkbox("use_gl_tonemapping", &use_gl_tonemapping);
            if (use_gl_tonemapping)
            {
                ImGui::Checkbox("use_bloom", &use_bloom);
                if (use_bloom)
                {
                    bloom.imgui();
                }
            }

            ImGui::Checkbox("channels_last", &pipeline->params->net_params.channels_last);
            if (ImGui::Checkbox("half_float", &pipeline->params->net_params.half_float))
            {
                auto target_type = pipeline->params->net_params.half_float ? torch::kFloat16 : torch::kFloat32;

                pipeline->render_network->to(target_type);
                ns->camera->to(target_type);
            }
            if (ImGui::Button("random shuffle"))
            {
                scene->point_cloud.RandomShuffle();
                pipeline->Train(false);
            }

            if (ImGui::Button("save render gl tonemap"))
            {
                output_texture_ldr->download(output_image_ldr.data());
                output_image_ldr.save("out_render_gl.png");
            }

            if (ImGui::Button("save render"))
            {
                DownloadRender().save("out_render.png");
            }

            if (ImGui::Button("save gt"))
            {
                DownloadGt().save("out_gt.png");
            }


            if (ImGui::Button("save debug"))
            {
                TemplatedImage<ucvec4> tmp(output_color->getHeight(), output_color->getWidth());

                output_color->bind();
                glGetTexImage(output_color->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
                assert_no_glerror();
                output_color->unbind();

                tmp.save("out_debug_" + std::to_string(color_layer) + ".png");
            }

            if (ImGui::Button("morton shuffle"))
            {
                scene->point_cloud.ReorderMorton64();
                pipeline->Train(false);
            }

            if (ImGui::Button("RandomBlockShuffle shuffle"))
            {
                scene->point_cloud.RandomBlockShuffle(256);
                pipeline->Train(false);
            }
        }
    }
    ImGui::End();

    timer_system.Imgui();
}
void RealTimeRenderer::Render(ImageInfo fd)
{
    SAIGA_ASSERT(pipeline);
    if (!pipeline) return;

    std::vector<NeuralTrainData> batch(1);
    batch.front() = std::make_shared<TorchFrameData>();


    batch.front()->img                   = fd;
    batch.front()->img.camera_index      = 0;
    batch.front()->img.image_index       = 0;
    batch.front()->img.camera_model_type = scene->dataset_params.camera_model;

    if (current_best_gt != -1)
    {
        auto& f = scene->frames[current_best_gt];
        SAIGA_ASSERT(current_best_gt == f.image_index);
        batch.front()->img.camera_index = f.camera_index;
        batch.front()->img.image_index  = f.image_index;
    }

    if (!uv_tensor.defined() || uv_tensor.size(1) != fd.h || uv_tensor.size(2) != fd.w)
    {
        auto uv_image    = InitialUVImage(fd.h, fd.w);
        uv_tensor        = ImageViewToTensor(uv_image.getImageView()).to(device);
        uv_tensor_center = torch::zeros_like(uv_tensor);
    }


    if (fd.h != output_image.h || fd.w != output_image.w)
    {
        output_image.create(fd.h, fd.w);
        output_image.getImageView().set(vec4(1, 1, 1, 1));
        output_image_ldr.create(output_image.dimensions());
        output_texture     = std::make_shared<Texture>(output_image);
        output_texture_ldr = std::make_shared<Texture>(output_image_ldr);

        texure_interop = std::make_shared<Saiga::CUDA::Interop>();
        texure_interop->initImage(output_texture->getId(), output_texture->getTarget());

        std::cout << "Setting Neural Render Size to " << fd.w << "x" << fd.h << std::endl;
    }

    batch.front()->uv = use_center_tensor ? uv_tensor_center : uv_tensor;
    SAIGA_ASSERT(batch.front()->uv.size(1) == fd.h);
    SAIGA_ASSERT(batch.front()->uv.size(2) == fd.w);


    float old_cutoff = 0;
    if (use_custom_camera)
    {
        rt_extrinsics->SetPose(0, fd.pose);
        rt_intrinsics->SetPinholeIntrinsics(0, fd.K, fd.distortion);
        std::swap(ns->intrinsics, rt_intrinsics);
        std::swap(ns->poses, rt_extrinsics);

        old_cutoff                        = params->render_params.dist_cutoff;
        params->render_params.dist_cutoff = 100;

        batch.front()->img.camera_index      = 0;
        batch.front()->img.image_index       = 0;
        batch.front()->img.camera_model_type = CameraModel::PINHOLE_DISTORTION;
    }


    pipeline->params->pipeline_params.skip_sensor_model = use_gl_tonemapping;

    auto debug_weight_color_old = pipeline->render_module->params->render_params.debug_weight_color;
    pipeline->render_module->params->render_params.debug_weight_color = false;

    torch::Tensor x;
    {
        auto timer                 = timer_system.Measure("Forward");
        auto neural_exposure_value = fd.exposure_value - scene->dataset_params.scene_exposure_value;
        auto f_result = pipeline->Forward(*ns, batch, {}, false, false, neural_exposure_value, fd.white_balance);
        x             = f_result.x;
        // batchsize == 1 !
        SAIGA_ASSERT(x.dim() == 4 && x.size(0) == 1);
        SAIGA_ASSERT(x.size(1) == 3);
    }

    auto timer = timer_system.Measure("Post Process");
    x          = x.squeeze();

    // x has size [c, h, w]
    torch::Tensor alpha_channel =
        torch::ones({1, x.size(1), x.size(2)}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    x = torch::cat({x, alpha_channel}, 0);


    x = x.permute({1, 2, 0});
    // x = x.clamp(0.f, 1.f);
    x = x.contiguous();

    texure_interop->mapImage();
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(texure_interop->array, 0, 0, x.data_ptr(), x.stride(0) * sizeof(float),
                                         x.size(1) * x.size(2) * sizeof(float), x.size(0), cudaMemcpyDeviceToDevice));
    texure_interop->unmap();

    if (use_gl_tonemapping)
    {
        tone_mapper.MapLinear(output_texture.get());
        if (use_bloom)
        {
            bloom.Render(output_texture.get());
        }
        tone_mapper.Map(output_texture.get(), output_texture_ldr.get());
    }

    pipeline->render_module->params->render_params.debug_weight_color = debug_weight_color_old;


    if (use_custom_camera)
    {
        std::swap(ns->intrinsics, rt_intrinsics);
        std::swap(ns->poses, rt_extrinsics);
        params->render_params.dist_cutoff = old_cutoff;
    }


    SAIGA_ASSERT(x.is_contiguous() && x.is_cuda());
}
void RealTimeRenderer::RenderColor(ImageInfo fd, int flags)
{
    if (!color_texture)
    {
        color_texture = NeuralPointTexture(scene->point_cloud);
        color_texture->to(device);
    }


    NeuralRenderInfo nri;
    nri.scene          = ns.get();
    nri.num_layers     = color_layer;
    nri.timer_system   = nullptr;
    nri.params         = pipeline->render_module->params->render_params;
    nri.params.dropout = false;

    fd.image_index  = 0;
    fd.camera_index = 0;


    auto old_tex = nri.scene->texture;
    auto old_env = nri.scene->environment_map;



    if (flags == 0)
    {
        // Render the point color
        nri.params.num_texture_channels = 4;
        nri.scene->texture              = color_texture;
        nri.scene->environment_map      = nullptr;
    }

    nri.images.push_back(fd);

    nri.images.front().camera_model_type = scene->dataset_params.camera_model;


    if (current_best_gt != -1)
    {
        auto& f                         = scene->frames[current_best_gt];
        nri.images.front().camera_index = f.camera_index;
        nri.images.front().image_index  = f.image_index;
    }

    float old_cutoff = 0;
    if (use_custom_camera)
    {
        rt_extrinsics->SetPose(0, fd.pose);
        rt_intrinsics->SetPinholeIntrinsics(0, fd.K, fd.distortion);
        std::swap(ns->intrinsics, rt_intrinsics);
        std::swap(ns->poses, rt_extrinsics);

        old_cutoff                        = params->render_params.dist_cutoff;
        params->render_params.dist_cutoff = 100;

        nri.images.front().camera_index      = 0;
        nri.images.front().image_index       = 0;
        nri.images.front().camera_model_type = CameraModel::PINHOLE_DISTORTION;
    }


    torch::Tensor x            = pipeline->render_module->forward(&nri).first.back();
    nri.scene->texture         = old_tex;
    nri.scene->environment_map = old_env;

    x = x.squeeze();
    x = x.permute({1, 2, 0});

    if (flags == 1)
    {
        x = x.slice(2, 0, 4);
        x = x.abs();
    }

    x = x * (1.f / color_scale);
    x = x.clamp(0.f, 1.f);

    x.slice(2, 3, 4).fill_(1);

    x = x.contiguous();

    if (!output_color || x.size(0) != output_color->getHeight() || x.size(1) != output_color->getWidth())
    {
        TemplatedImage<vec4> tmp(x.size(0), x.size(1));
        output_color = std::make_shared<Texture>(tmp);
        output_color->setFiltering(GL_NEAREST);

        color_interop = std::make_shared<Saiga::CUDA::Interop>();
        color_interop->initImage(output_color->getId(), output_color->getTarget());

        std::cout << "Setting Debug Output Size to " << x.size(1) << "x" << x.size(0) << std::endl;
    }

    color_interop->mapImage();
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(color_interop->array, 0, 0, x.data_ptr(), x.stride(0) * sizeof(float),
                                         x.size(1) * x.size(2) * sizeof(float), x.size(0), cudaMemcpyDeviceToDevice));
    color_interop->unmap();


    if (use_custom_camera)
    {
        std::swap(ns->intrinsics, rt_intrinsics);
        std::swap(ns->poses, rt_extrinsics);
        params->render_params.dist_cutoff = old_cutoff;
    }
}


void RealTimeRenderer::LoadNets()
{
    if (experiments.empty()) return;

    ns            = nullptr;
    pipeline      = nullptr;
    color_texture = nullptr;

    //    texture = EmptyTexture(8, scene->point_cloud.NumVertices());
    auto ex = experiments[current_ex];
    auto ep = ex.eps[current_ep];

    std::cout << "loading checkpoint " << ex.name << " -> " << ep.name << std::endl;

    params = std::make_shared<CombinedParams>(ex.dir + "/params.ini");

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.major > 6)
    {
        std::cout << "Using half_float inference" << std::endl;
        params->net_params.half_float = true;
    }

    params->pipeline_params.train             = false;
    params->render_params.render_outliers     = false;
    params->train_params.checkpoint_directory = ep.dir;

    params->train_params.loss_vgg = 0;
    params->train_params.loss_l1  = 0;
    params->train_params.loss_mse = 0;


    current_best_gt = -1;

    pipeline = std::make_shared<NeuralPipeline>(params);
    pipeline->Train(false);
    pipeline->timer_system = &timer_system;



    ns = std::make_shared<NeuralScene>(scene, params);
    ns->to(device);
    ns->Train(0, false);

    rt_intrinsics = IntrinsicsModule(scene->scene_cameras.front().K);
    rt_extrinsics = PoseModule(scene->frames[0].pose);

    if (ns->camera->exposures_values.defined())
    {
        auto ex_cpu = ns->camera->exposures_values.cpu();
        SAIGA_ASSERT(scene->frames.size() == ex_cpu.size(0));
        for (int i = 0; i < scene->frames.size(); ++i)
        {
            float ex                        = ex_cpu[i][0][0][0].item().toFloat();
            scene->frames[i].exposure_value = ex + scene->dataset_params.scene_exposure_value;
        }
    }
}