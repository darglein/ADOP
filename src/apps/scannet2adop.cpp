/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/exif/TinyEXIF.h"
#include "saiga/core/util/file.h"
#include <array>

#include "saiga/vision/cameraModel/OCam.h"
#define SAIGA_VISION_API //colmap reader is exported with dll declspec, this removes this

#include "saiga/vision/util/ColmapReader.h"

#include "data/SceneData.h"


static void ScannetScene(std::string input_dir, std::string output_scene_path)
{
    std::cout << "Preprocessing Scannet scene " << input_dir << " -> " << output_scene_path << std::endl;
    std::filesystem::create_directories(output_scene_path);

    std::vector<std::string> camera_files;
    std::vector<std::string> image_files;

    {
        SceneCameraParams params;

        // TODO
        params.w = 1296;
        params.h = 968;
        // Color Intrinsics
        std::array<double, 16> posearray;
        std::ifstream pstream(input_dir + "/intrinsic/intrinsic_color.txt");
        for (auto& pi : posearray)
        {
            pstream >> pi;
        }
        Mat4 intrinsics = Eigen::MatrixView<double, 4, 4, Eigen::ColMajor>(posearray.data(), 1, 4).eval().transpose();
        params.K        = IntrinsicsPinhole<float>(intrinsics.block<3, 3>(0, 0).eval().cast<float>());

        auto f = "camera" + std::to_string(0) + ".ini";
        std::filesystem::remove(output_scene_path + "/" + f);
        params.Save(output_scene_path + "/" + f);

        camera_files.push_back(f);
    }

    {
        std::filesystem::remove(output_scene_path + "/dataset.ini");
        SceneDatasetParams params;
        params.file_model      = "";
        params.image_dir       = input_dir + "/color/";
        params.camera_files    = camera_files;
        params.scene_up_vector = vec3(0, -1, 0);
        params.Save(output_scene_path + "/dataset.ini");
    }

    int num_images = 0;
    {
        std::ofstream ostream1(output_scene_path + "/images.txt");
        std::ofstream ostream2(output_scene_path + "/camera_indices.txt");

        while (true)
        {
            auto img_file = std::to_string(num_images) + ".jpg";
            if (std::filesystem::exists(input_dir + "/color/" + img_file))
            {
                ostream1 << img_file << "\n";
                ostream2 << 0 << "\n";
            }
            else
            {
                break;
            }
            num_images++;
        }
    }

    {
        auto pc_in  = input_dir + "/point_cloud.ply";
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

    {
        std::vector<SE3> poses;
        for (int i = 0; i < num_images; ++i)
        {
            auto pose_file = input_dir + "/pose/" + std::to_string(i) + ".txt";
            SAIGA_ASSERT(std::filesystem::exists(pose_file));
            // Color Intrinsics
            std::array<double, 16> posearray;
            std::ifstream pstream(pose_file);
            for (auto& pi : posearray)
            {
                pstream >> pi;
            }
            Mat4 pose_mat = Eigen::MatrixView<double, 4, 4, Eigen::ColMajor>(posearray.data(), 1, 4).eval().transpose();

            SE3 pose = SE3::fitToSE3(pose_mat);

            poses.push_back(pose);
        }
        SceneData::SavePoses(poses, output_scene_path + "/poses.txt");
    }


    std::shared_ptr<SceneData> sd = std::make_shared<SceneData>(output_scene_path);
    sd->Save();
}



int main(int argc, char* argv[])
{
    std::string input_dir;
    std::string output_path;

    CLI::App app{"Scannet to ADOP Scene Converter", "scannet2adop"};
    app.add_option("--input_dir", input_dir)->required();
    app.add_option("--output_path", output_path)->required();


    CLI11_PARSE(app, argc, argv);


    ScannetScene(input_dir, output_path);


    return 0;
}
