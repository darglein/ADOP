/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/exif/TinyEXIF.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/json11.hpp"
#include "saiga/core/util/tinyxml2.h"
#include "saiga/vision/cameraModel/OCam.h"
#include "saiga/vision/util/ColmapReader.h"

#include "data/SceneData.h"


std::vector<float> ExposureValuesFromImages(std::vector<std::string> files, std::string image_dir)
{
    std::vector<float> exposures;

    for (auto f : files)
    {
        auto path = image_dir + "/" + f;

        auto data = File::loadFileBinary(path);
        TinyEXIF::EXIFInfo info;
        info.parseFrom((unsigned char*)data.data(), data.size());

        if (info.FNumber == 0 || info.ExposureTime == 0 || info.ISOSpeedRatings == 0)
        {
            std::cout << "No EXIF exposure value found for image " << f << std::endl;
            exposures.push_back(0);
        }
        else
        {
            double EV_log2 =
                log2((info.FNumber * info.FNumber) / info.ExposureTime) + log2(info.ISOSpeedRatings / 100.0);
            exposures.push_back(EV_log2);
            SAIGA_ASSERT(info.ExposureBiasValue == 0);
        }
    }

    std::cout << "EV Statistic:" << std::endl;
    Statistics stat(exposures);
    std::cout << stat << std::endl;
    std::cout << "dynamic range: " << exp2(stat.max - stat.min) << std::endl;

    return exposures;
}

static std::shared_ptr<SceneData> ColmapScene(std::string sparse_dir, std::string image_dir,
                                              std::string point_cloud_file, std::string output_scene_path,
                                              double scale_intrinsics = 1., double render_scale = 1.)
{
    std::cout << "Preprocessing Colmap scene " << sparse_dir << " -> " << output_scene_path << std::endl;
    std::filesystem::create_directories(output_scene_path);



    ColmapReader reader(sparse_dir);

    std::map<int, int> col_cam_to_id;

    std::vector<std::string> camera_files;
    std::vector<std::string> image_files;

    for (auto s : reader.images)
    {
        image_files.push_back(s.name);
    }

    std::vector<float> exposures = ExposureValuesFromImages(image_files, image_dir);

    {
        SceneCameraParams params;
        for (int i = 0; i < reader.cameras.size(); ++i)
        {
            auto c = reader.cameras[i];

            params.w = c.w * scale_intrinsics;
            params.h = c.h * scale_intrinsics;

            params.K          = c.K.scale(scale_intrinsics).cast<float>();
            params.distortion = c.dis.cast<float>();

            col_cam_to_id[c.camera_id] = i;

            auto f = "camera" + std::to_string(i) + ".ini";

            std::filesystem::remove(output_scene_path + "/" + f);
            params.Save(output_scene_path + "/" + f);

            camera_files.push_back(f);
        }
    }

    {
        std::filesystem::remove(output_scene_path + "/dataset.ini");
        SceneDatasetParams params;
        params.file_model           = "";
        params.image_dir            = image_dir;
        params.camera_files         = camera_files;
        params.scene_exposure_value = Statistics(exposures).mean;
        params.render_scale         = render_scale;

        params.scene_up_vector = vec3(0, -1, 0);
        params.Save(output_scene_path + "/dataset.ini");
    }


    {
        std::ofstream ostream1(output_scene_path + "/images.txt");
        std::ofstream ostream2(output_scene_path + "/camera_indices.txt");


        for (auto s : reader.images)
        {
            ostream1 << s.name << std::endl;
            ostream2 << col_cam_to_id[s.camera_id] << std::endl;
        }
    }

    {
        auto pc_in  = point_cloud_file;
        auto pc_out = output_scene_path + "/point_cloud.ply";

        std::filesystem::remove(pc_out);
        std::filesystem::remove(output_scene_path + "/point_cloud.bin");

        SAIGA_ASSERT(std::filesystem::exists(pc_in));
        SAIGA_ASSERT(std::filesystem::is_regular_file(pc_in));

        std::string command = "cp -v -n " + pc_in + " " + pc_out;
        auto res            = system(command.c_str());
        if (res != 0)
        {
            SAIGA_EXIT_ERROR("Copy failed!");
        }
    }

    std::vector<SE3> poses;
    for (int i = 0; i < reader.images.size(); ++i)
    {
        SE3 view(reader.images[i].q, reader.images[i].t);
        poses.push_back(view.inverse());
    }
    SceneData::SavePoses(poses, output_scene_path + "/poses.txt");


    std::shared_ptr<SceneData> sd = std::make_shared<SceneData>(output_scene_path);
    {
        SAIGA_ASSERT(sd->frames.size() == reader.images.size());

        for (int i = 0; i < reader.images.size(); ++i)
        {
            sd->frames[i].exposure_value = exposures[i];
        }
    }
    sd->Save();

    return sd;
}



int main(int argc, char* argv[])
{
    std::string sparse_dir;
    std::string image_dir;
    std::string point_cloud_file;
    std::string output_path;
    double scale_intrinsics = 1;
    double render_scale     = 1;

    CLI::App app{"COLMAP to ADOP Scene Converter", "colmap2adop"};
    app.add_option("--sparse_dir", sparse_dir)->required();
    app.add_option("--image_dir", image_dir)->required();
    app.add_option("--point_cloud_file", point_cloud_file)->required();
    app.add_option("--output_path", output_path)->required();
    app.add_option("--scale_intrinsics", scale_intrinsics)->required();
    app.add_option("--render_scale", render_scale)->required();


    CLI11_PARSE(app, argc, argv);


    ColmapScene(sparse_dir, image_dir, point_cloud_file, output_path, scale_intrinsics, render_scale);


    return 0;
}
