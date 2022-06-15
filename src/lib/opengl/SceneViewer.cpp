/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SceneViewer.h"

#include "saiga/colorize.h"
#include "saiga/core/geometry/AccelerationStructure.h"
#include "saiga/core/util/ProgressBar.h"

float sphere_radius = 0.01;
float frustum_size  = 0.05;

// pose time view
// float sphere_radius = 0.0001;
// float frustum_size =  0.001;

SceneViewer::SceneViewer(std::shared_ptr<SceneData> scene) : scene(scene)
{
    if (std::filesystem::exists(scene->dataset_params.file_model))
    {
        model = UnifiedModel(scene->dataset_params.file_model).AddMissingDummyTextures();
        model.ComputeColor();
        std::cout << "[Scene] Model (Tris) " << model.TotalTriangles() << std::endl;
    }


    UpdatePointCloudBuffer();

    {
        auto obj = IcoSphereMesh(Sphere(vec3(0, 0, 0), sphere_radius), 3);
        obj.SetVertexColor(vec4(1, 1, 1, 1));
        capture_asset = std::make_shared<ColoredAsset>(obj);
    }


    scene_camera.proj                = scene->GLProj();
    scene_camera.zNear               = scene->dataset_params.znear;
    scene_camera.zFar                = scene->dataset_params.zfar;
    scene_camera.global_up           = scene->dataset_params.scene_up_vector;
    scene_camera.recompute_on_resize = false;

    scene_camera.enableInput();

    {
        auto obj = FrustumLineMesh(scene_camera.proj, frustum_size, false);
        obj.SetVertexColor(vec4(1, 1, 1, 1));
        frustum_asset = std::make_shared<LineVertexColoredAsset>(obj);
    }
}

void SceneViewer::OptimizePoints()
{
    {
        ScopedTimerPrintLine tim("ReorderMorton64");
        scene->point_cloud.ReorderMorton64();
    }
    {
        ScopedTimerPrintLine tim("RandomBlockShuffle");
        scene->point_cloud.RandomBlockShuffle(default_point_block_size);
    }

    UpdatePointCloudBuffer();
}


void SceneViewer::SetRandomPointColor()
{
    for (auto& c : scene->point_cloud.color)
    {
        vec3 nc     = Random::MatrixUniform<vec3>(0, 1);
        c.head<3>() = nc;
    }
    UpdatePointCloudBuffer();
}


void SceneViewer::RemoveInvalidPoints()
{
    console << "RemoveInvalidPoints" << std::endl;
    std::vector<int> to_erase;
    for (int i = 0; i < scene->point_cloud.NumVertices(); ++i)
    {
        if (scene->point_cloud.color[i].x() <= 0)
        {
            to_erase.push_back(i);
        }
    }

    scene->point_cloud.EraseVertices(to_erase);
    UpdatePointCloudBuffer();
}
void SceneViewer::Select(Ray r)
{
    float best_t = 10000;
    int best_id  = -1;


    for (int i = 0; i < scene->frames.size(); ++i)
    {
        auto& c = scene->frames[i];
        Sphere s(c.pose.translation().cast<float>(), sphere_radius);

        float t1, t2;

        if (Intersection::RaySphere(r, s, t1, t2))
        {
            if (t1 < best_t)
            {
                best_t  = t1;
                best_id = i;
            }
        }
    }
    selected_capture = best_id;
    std::cout << "Selected Frame " << selected_capture << std::endl;
}

void SceneViewer::imgui()
{
    if (ImGui::ListBoxHeader("###Frames", 10))
    {
        for (int i = 0; i < scene->frames.size(); ++i)
        {
            auto& f = scene->frames[i];

            auto str = "Frame " + std::to_string(i) + " exp: " + std::to_string(f.exposure_value) + " wp " +
                       std::to_string(f.white_balance(0)) + " " + std::to_string(f.white_balance(2));
            if (ImGui::Selectable(str.c_str(), i == selected_capture))
            {
                selected_capture = i;
                camera->setModelMatrix(f.OpenglModel());
                camera->updateFromModel();
            }
        }
        ImGui::ListBoxFooter();
    }
    ImGui::Text("Scene Exposure: %f", scene->dataset_params.scene_exposure_value);

    if (ImGui::CollapsingHeader("Rendering"))
    {
        ImGui::Checkbox("render_frustums", &render_frustums);

        if (ImGui::Button("Update Point Buffer"))
        {
            UpdatePointCloudBuffer();
        }

        if (gl_points)
        {
            gl_points->imgui();
        }
    }

    if (ImGui::CollapsingHeader("Augmentation"))
    {
        if (ImGui::Button("invert camera poses"))
        {
            for (auto& f : scene->frames)
            {
                f.pose = f.pose.inverse();
            }
        }


        if (ImGui::Button("duplicate"))
        {
            int n = scene->frames.size();
            for (int i = 0; i < n; ++i)
            {
                scene->frames.push_back(scene->frames[i]);
            }
        }

        static float downfac = 2;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###downfac", &downfac, 0, 0, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button("downsample points"))
        {
            scene->DownsamplePoints(downfac);
            OptimizePoints();
            scene->ComputeRadius();
            UpdatePointCloudBuffer();
        }

        static int dupfac = 2;
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("###dupfac", &dupfac);
        ImGui::SameLine();
        static float dupdis = 1;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###dupdis", &dupdis);
        ImGui::SameLine();
        if (ImGui::Button("dup points"))
        {
            scene->DuplicatePoints(dupfac, dupdis);
            scene->ComputeRadius();
            OptimizePoints();
            UpdatePointCloudBuffer();
        }

        static float rcdis = 0.0005;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###rcdis", &rcdis, 0, 0, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button("remove close z"))
        {
            scene->RemoveClosePoints(rcdis);
            OptimizePoints();
            scene->ComputeRadius();
            UpdatePointCloudBuffer();
        }



        static float ldis = 0.02;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###ldis", &ldis, 0, 0, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button("remove lonely z"))
        {
            scene->RemoveLonelyPoints(5, ldis);
            OptimizePoints();
            scene->ComputeRadius();
            UpdatePointCloudBuffer();
        }

        static float doudis = 0.0002;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###doudis", &doudis, 0, 0, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button("remove close"))
        {
            int bef = scene->point_cloud.NumVertices();
            scene->point_cloud.RemoveDoubles(doudis);
            int aft = scene->point_cloud.NumVertices();
            OptimizePoints();
            scene->ComputeRadius();
            UpdatePointCloudBuffer();
            std::cout << "remove close dis " << doudis << " Points " << bef << " -> " << aft << std::endl;
        }

        static float sdev_noise = 0.1;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###sdev_noise", &sdev_noise);
        ImGui::SameLine();
        if (ImGui::Button("pose noise"))
        {
            scene->AddPoseNoise(0, sdev_noise);
        }

        if (ImGui::Button("CreateMasks"))
        {
            CreateMasks(true);
        }
    }


    if (ImGui::CollapsingHeader("Point Cloud"))
    {
        static float point_noise = 0.01;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###point_noise", &point_noise);
        ImGui::SameLine();
        if (ImGui::Button("point noise"))
        {
            scene->AddPointNoise(point_noise);
            UpdatePointCloudBuffer();
        }


        if (ImGui::Button("color points by layout"))
        {
            vec4 c = Random::MatrixUniform<vec4>(0, 1);
            c(3)   = 1;

            for (int i = 0; i < scene->point_cloud.NumVertices(); ++i)
            {
                if (i % 256 == 0)
                {
                    c    = Random::MatrixUniform<vec4>(0, 1);
                    c(3) = 1;
                }
                scene->point_cloud.color[i] = c;
            }
            UpdatePointCloudBuffer();
        }

        if (ImGui::Button("OptimizePoints (memory layout)"))
        {
            OptimizePoints();
        }

        if (ImGui::Button("ComputeRadius"))
        {
            scene->ComputeRadius();
        }

        if (ImGui::Button("SetRandomPointColor")) SetRandomPointColor();
    }

    if (ImGui::Button("save scene"))
    {
        scene->Save();
    }

    if (ImGui::Button("save points ply"))
    {
        std::string file = scene->file_dataset_base + "/point_cloud_exported.ply";
        Saiga::UnifiedModel(scene->point_cloud).Save(file);
    }
}

void SceneViewer::RenderDebug(Camera* cam)
{
    if (capture_asset->forwardShader->bind())
    {
        if (scene)
        {
            if (render_frustums)
            {
                for (int i = 0; i < scene->frames.size(); ++i)
                {
                    auto& c = scene->frames[i];
                    if (i == selected_capture)
                    {
                        capture_asset->forwardShader->uploadColor(exp2(scene->dataset_params.scene_exposure_value) *
                                                                  vec4(1, 1, 0, 1));
                    }
                    else
                    {
                        capture_asset->forwardShader->uploadColor(exp2(scene->dataset_params.scene_exposure_value) *
                                                                  c.display_color);
                    }
                    capture_asset->forwardShader->uploadModel(c.OpenglModel());
                    capture_asset->renderRaw();
                    glLineWidth(3);
                    frustum_asset->renderRaw();
                    glLineWidth(1);
                }
            }
        }
        capture_asset->forwardShader->unbind();
    }
}
void SceneViewer::CreateMasks(bool mult_old_mask)
{
    std::filesystem::path out_dir = scene->dataset_params.image_dir;
    out_dir                       = out_dir.parent_path().parent_path();
    out_dir                       = out_dir.append("masks/");
    std::filesystem::create_directories(out_dir);

    std::cout << "CreateMasks into " << out_dir << std::endl;

    auto mesh = model.CombinedMesh(VERTEX_POSITION);
    // KDTree<3, vec3> tree(mesh.first.position);

    auto triangles = mesh.first.TriangleSoup();
    for (auto& t : triangles)
    {
        t.ScaleUniform(1.0005);
    }

    SAIGA_ASSERT(!triangles.empty());

    AccelerationStructure::ObjectMedianBVH bvh;

    {
        ScopedTimerPrintLine tim("Creating BVH");
        bvh                  = AccelerationStructure::ObjectMedianBVH(triangles);
        bvh.triangle_epsilon = 0.00;
        bvh.bvh_epsilon      = 0.00;
    }


    std::vector<std::vector<vec3>> directions(scene->scene_cameras.size());
    std::vector<ImageView<vec3>> dirs(scene->scene_cameras.size());

    {
        ScopedTimerPrintLine tim("Unproject image");

        for (int ic = 0; ic < scene->scene_cameras.size(); ++ic)
        {
            auto cam        = scene->scene_cameras[ic];
            auto& dirs_data = directions[ic];
            auto& dir       = dirs[ic];

            dirs_data.resize(cam.h * cam.w);
            dir = ImageView<vec3>(cam.h, cam.w, dirs_data.data());


            if (scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION)
            {
#pragma omp parallel for
                for (int y = 0; y < cam.h; ++y)
                {
                    for (int x = 0; x < cam.w; ++x)
                    {
                        vec2 ip(x, y);
                        vec2 dist   = cam.K.unproject2(ip);
                        vec2 undist = undistortNormalizedPointSimple(dist, cam.distortion);
                        vec3 np     = vec3(undist(0), undist(1), 1);
                        dir(y, x)   = np.normalized();
                    }
                }
            }
            else if (scene->dataset_params.camera_model == CameraModel::OCAM)
            {
#pragma omp parallel for
                for (int y = 0; y < cam.h; ++y)
                {
                    for (int x = 0; x < cam.w; ++x)
                    {
                        Vec2 ip(x, y);
                        Vec3 np   = UnprojectOCam<double>(ip, 1, cam.ocam.AffineParams(), cam.ocam.poly_cam2world);
                        dir(y, x) = np.normalized().cast<float>();
                    }
                }
            }
            else
            {
                SAIGA_EXIT_ERROR("unknown camera model");
            }
        }
    }

    std::vector<std::string> mask_files;

    for (int i = 0; i < scene->frames.size(); ++i)
    {
        std::cout << "Process image " << i << std::endl;
        auto& f = scene->frames[i];
        TemplatedImage<unsigned char> mask_img(f.h, f.w);
        mask_img.makeZero();


        auto dir = dirs[f.camera_index];

        vec3 center = f.pose.translation().cast<float>();

        quat R = f.pose.unit_quaternion().cast<float>();

        ProgressBar bar(std::cout, "Render Mask", mask_img.h);
#pragma omp parallel for schedule(dynamic)
        for (int y = 0; y < mask_img.h; ++y)
        {
            for (int x = 0; x < mask_img.w; ++x)
            {
                vec3 d = dir(y, x);

                Ray r;
                r.origin    = center;
                r.direction = R * d;
                auto inter  = bvh.getClosest(r);
                if (inter.valid)
                {
                    mask_img(y, x) = 255;
                }
            }
            bar.addProgress(1);
        }

        if (mult_old_mask)
        {
            if (std::filesystem::exists(scene->dataset_params.mask_dir + f.mask_file))
            {
                Saiga::TemplatedImage<unsigned char> img_mask_large(scene->dataset_params.mask_dir + f.mask_file);

                for (int y = 0; y < mask_img.h; ++y)
                {
                    for (int x = 0; x < mask_img.w; ++x)
                    {
                        if (img_mask_large(y, x) == 0)
                        {
                            mask_img(y, x) = 0;
                        }
                    }
                }
            }
        }

        std::string mask_file = leadingZeroString(i, 5) + ".png";
        mask_files.push_back(mask_file);
        auto dst_file = out_dir.string() + "/" + mask_file;
        mask_img.save(dst_file);
    }

    std::ofstream ostream3(out_dir.string() + "/masks.txt");
    for (auto m : mask_files)
    {
        ostream3 << m << "\n";
    }
}
void SceneViewer::UpdatePointCloudBuffer()
{
    gl_points = nullptr;
}
