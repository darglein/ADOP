/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/time.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/file.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/LRScheduler.h"

#include "data/Dataset.h"
#include "models/Pipeline.h"

#include <torch/script.h>

#include "git_sha1.h"

std::string full_experiment_dir;

std::shared_ptr<CombinedParams> params;



torch::Device device = torch::kCUDA;


static torch::Tensor CropMask(int h, int w, int border)
{
    // Create a center crop mask so that the border is valued less during training.
    TemplatedImage<unsigned char> target_mask(h, w);
    target_mask.makeZero();

    int b     = border;
    auto crop = target_mask.getImageView().subImageView(b, b, h - b * 2, w - b * 2);
    crop.set(255);

    return ImageViewToTensor<unsigned char>(target_mask, true).unsqueeze(0);
}


class TrainScene
{
   public:
    TrainScene(std::vector<std::string> scene_dirs)
    {
        for (int i = 0; i < scene_dirs.size(); ++i)
        {
            auto scene = std::make_shared<SceneData>(params->train_params.scene_base_dir + scene_dirs[i]);

            // 1.  Separate indices
            auto all_indices                   = scene->Indices();
            auto [train_indices, test_indices] = params->train_params.Split(all_indices);

            if (std::filesystem::exists(params->train_params.split_index_file_train))
            {
                train_indices = params->train_params.ReadIndexFile(params->train_params.split_index_file_train);
            }


            if (std::filesystem::exists(params->train_params.split_index_file_test))
            {
                test_indices = params->train_params.ReadIndexFile(params->train_params.split_index_file_test);
            }


            if (params->train_params.duplicate_train_factor > 1)
            {
                // this multiplies the epoch size
                // increases performance for small epoch sizes
                auto cp = train_indices;
                for (int i = 1; i < params->train_params.duplicate_train_factor; ++i)
                {
                    train_indices.insert(train_indices.end(), cp.begin(), cp.end());
                }
            }


            {
                std::ofstream strm(full_experiment_dir + "/train_indices_" + scene->scene_name + ".txt");
                for (auto i : train_indices)
                {
                    strm << i << "\n";
                }
                std::ofstream strm2(full_experiment_dir + "/test_indices_" + scene->scene_name + ".txt");
                for (auto i : test_indices)
                {
                    strm2 << i << "\n";
                }
            }

            std::cout << "Train(" << train_indices.size() << "): " << array_to_string(train_indices, ' ') << std::endl;
            std::cout << "Test(" << test_indices.size() << "): " << array_to_string(test_indices, ' ') << std::endl;

            PerSceneData scene_data;

            scene_data.not_training_indices = all_indices;
            for (auto i : train_indices)
            {
                auto it = std::find(scene_data.not_training_indices.begin(), scene_data.not_training_indices.end(), i);
                if (it != scene_data.not_training_indices.end())
                {
                    scene_data.not_training_indices.erase(it);
                }
            }
            train_cropped_samplers.push_back(CroppedSampler(scene, train_indices));
            test_cropped_samplers.push_back(CroppedSampler(scene, test_indices));
            test_samplers.push_back(FullSampler(scene, test_indices));

            if (params->train_params.train_mask_border > 0)
            {
                int w              = test_samplers.back().image_size_crop(0);
                int h              = test_samplers.back().image_size_crop(1);
                torch::Tensor mask = CropMask(h, w, params->train_params.train_mask_border).to(device);
                TensorToImage<unsigned char>(mask).save(full_experiment_dir + "/eval_mask_" + scene->scene_name +
                                                        ".png");
                scene_data.eval_crop_mask = (mask);
            }

            std::vector<Sophus::SE3d> poses;
            for (auto& f : scene->frames)
            {
                poses.push_back(f.pose);
            }
            scene_data.old_poses = poses;

            scene->AddIntrinsicsNoise(params->train_params.noise_intr_k, params->train_params.noise_intr_d);

            auto ns = std::make_shared<NeuralScene>(scene, params);
            if (params->train_params.noise_pose_r > 0 || params->train_params.noise_pose_t > 0)
            {
                scene->AddPoseNoise(radians(params->train_params.noise_pose_r),
                                    params->train_params.noise_pose_t / 1000.);

                torch::NoGradGuard ngg;
                auto poses2 = PoseModule(scene);
                poses2->to(device);
                ns->poses->to(device);

                PrintTensorInfo(ns->poses->poses_se3);
                PrintTensorInfo(poses2->poses_se3);
                ns->poses->poses_se3.set_(poses2->poses_se3);
            }

            if (params->train_params.noise_point > 0)
            {
                torch::NoGradGuard ngg;
                auto noise =
                    torch::normal(0, params->train_params.noise_point, ns->point_cloud_cuda->t_position.sizes())
                        .to(device);
                noise.slice(1, 3, 4).zero_();

                ns->point_cloud_cuda->t_position += noise;
            }

            scene_data.scene = ns;
            data.push_back(scene_data);
        }
    }

    SceneDataTrainSampler CroppedSampler(std::shared_ptr<SceneData> scene, std::vector<int> indices)
    {
        ivec2 crop(params->train_params.train_crop_size, params->train_params.train_crop_size);

        SceneDataTrainSampler sampler(scene, indices, params->train_params.train_use_crop, crop,
                                      params->train_params.inner_batch_size, params->train_params.use_image_masks);
        sampler.min_max_zoom(0)   = params->train_params.min_zoom * scene->dataset_params.render_scale;
        sampler.min_max_zoom(1)   = params->train_params.max_zoom * scene->dataset_params.render_scale;
        sampler.prefere_border    = params->train_params.crop_prefere_border;
        sampler.inner_sample_size = params->train_params.inner_sample_size;
        return sampler;
    }

    SceneDataTrainSampler FullSampler(std::shared_ptr<SceneData> scene, std::vector<int> indices)
    {
        int w = scene->scene_cameras.front().w * scene->dataset_params.render_scale;
        int h = scene->scene_cameras.front().h * scene->dataset_params.render_scale;

        int max_eval_size = iAlignUp(params->train_params.max_eval_size, 32);

        std::cout << "full sampler " << w << "x" << h << " render scale " << scene->dataset_params.render_scale
                  << std::endl;

        int min_scene_size = std::min(w, h);
        if (min_scene_size > max_eval_size && params->train_params.train_use_crop)
        {
            w = std::min(w, max_eval_size);
            h = std::min(h, max_eval_size);

            SceneDataTrainSampler sdf(scene, indices, true, ivec2(w, h), 1, params->train_params.use_image_masks);
            sdf.random_zoom        = true;
            sdf.min_max_zoom(0)    = max_eval_size / double(min_scene_size);
            sdf.min_max_zoom(1)    = max_eval_size / double(min_scene_size);
            sdf.random_translation = false;
            return sdf;
        }
        else if (scene->dataset_params.render_scale != 1)
        {
            SceneDataTrainSampler sdf(scene, indices, true, ivec2(w, h), 1, params->train_params.use_image_masks);
            sdf.random_zoom        = true;
            sdf.min_max_zoom(0)    = scene->dataset_params.render_scale;
            sdf.min_max_zoom(1)    = scene->dataset_params.render_scale;
            sdf.random_translation = false;
            return sdf;
        }
        else
        {
            SceneDataTrainSampler sdf(scene, indices, false, ivec2(-1, -1), 1, params->train_params.use_image_masks);
            sdf.random_translation = false;
            return sdf;
        }
    }

    auto DataLoader(std::vector<SceneDataTrainSampler>& train_cropped_samplers, bool train)
    {
        std::vector<uint64_t> sizes;
        for (auto& t : train_cropped_samplers)
        {
            sizes.push_back(t.Size());
        }

        int batch_size  = train ? params->train_params.batch_size : 1;
        int num_workers = train ? params->train_params.num_workers_train : params->train_params.num_workers_eval;
        bool shuffle    = train ? params->train_params.shuffle_train_indices : false;
        auto options    = torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true).workers(num_workers);

        auto sampler = torch::MultiDatasetSampler(sizes, options.batch_size(), shuffle);
        int n        = sampler.Size();
        return std::pair(
            torch::data::make_data_loader(TorchSingleSceneDataset(train_cropped_samplers), std::move(sampler), options),
            n);
    }

    void Load(torch::DeviceType device, int scene)
    {
        if (scene == current_scene) return;
        Unload();
        current_scene = scene;
        data[current_scene].scene->to(device);
    }

    void Unload()
    {
        if (params->train_params.keep_all_scenes_in_memory) return;
        if (current_scene != -1)
        {
            data[current_scene].scene->to(torch::kCPU);
        }
        current_scene = -1;
    }

    void SetScene(int id) { current_scene = id; }

    void Train(int epoch, bool v)
    {
        SAIGA_ASSERT(current_scene != -1);
        data[current_scene].scene->Train(epoch, v);
    }

    void StartEpoch()
    {
        for (auto& sd : data)
        {
            sd.epoch_loss = {};
        }
    }

    int current_scene = -1;

    struct PerSceneData
    {
        std::shared_ptr<NeuralScene> scene;
        std::vector<Sophus::SE3d> old_poses;
        torch::Tensor eval_crop_mask;

        // all image indices that are not used during training.
        // -> we interpolate the meta data for them
        std::vector<int> not_training_indices;

        LossResult epoch_loss;
    };

    std::vector<PerSceneData> data;
    std::vector<SceneDataTrainSampler> train_cropped_samplers, test_samplers, test_cropped_samplers;
};



class NeuralTrainer
{
   public:
    torch::DeviceType device = torch::kCUDA;
    std::shared_ptr<NeuralPipeline> pipeline;

    torch::Tensor train_crop_mask;
    std::shared_ptr<TrainScene> train_scenes;
    std::string ep_dir;
    LRSchedulerPlateau lr_scheduler;

    ~NeuralTrainer() {}

    NeuralTrainer()
    {
        lr_scheduler = LRSchedulerPlateau(params->train_params.lr_decay_factor, params->train_params.lr_decay_patience);
        torch::set_num_threads(1);

        std::string experiment_name = Saiga::CurrentTimeString("%F_%T") + "_" + params->train_params.name;
        std::replace(experiment_name.begin(), experiment_name.end(), ':', '_'); // files with : forbiden in some linux distributives
        full_experiment_dir         = params->train_params.experiment_dir + "/" + experiment_name + "/";
        std::filesystem::create_directories(full_experiment_dir);
        console.setOutputFile(full_experiment_dir + "log.txt");
        SAIGA_ASSERT(console.rdbuf());
        std::cout.rdbuf(console.rdbuf());

        // Saiga::SaigaParameters saiga_params;
        // saiga_params.fromConfigFile("configs/train_saiga_config.ini");
        // Saiga::initSaiga(saiga_params);

        std::cout << "train" << std::endl;
        train_scenes = std::make_shared<TrainScene>(params->train_params.scene_names);

        // Save all paramters into experiment output dir
        params->Save(full_experiment_dir + "/params.ini");

        {
            std::ofstream strm(full_experiment_dir + "/git.txt");
            strm << GIT_SHA1 << std::endl;
        }
        pipeline = std::make_shared<NeuralPipeline>(params);

        for (int epoch_id = 0; epoch_id <= params->train_params.num_epochs; ++epoch_id)
        {
            std::cout << std::endl;
            std::cout << "=== Epoch " << epoch_id << " ===" << std::endl;
            std::string ep_str = Saiga::leadingZeroString(epoch_id, 4);

            bool last_ep         = epoch_id == params->train_params.num_epochs;
            bool save_checkpoint = epoch_id % params->train_params.save_checkpoints_its == 0 || last_ep;

            ep_dir = full_experiment_dir + "ep" + ep_str + "/";
            if (save_checkpoint)
            {
                std::filesystem::create_directory(ep_dir);
            }

            {
                if (params->train_params.do_train && epoch_id > 0)
                {
                    auto epoch_loss = TrainEpoch(epoch_id, train_scenes->train_cropped_samplers, false);

                    if (params->train_params.optimize_eval_camera)
                    {
                        std::cout << "Optimizing meta info from test cameras..." << std::endl;
                        TrainEpoch(epoch_id, train_scenes->test_cropped_samplers, true);
                    }

                    auto reduce_factor           = lr_scheduler.step(epoch_loss);
                    static double current_factor = 1;
                    current_factor *= reduce_factor;

                    if (reduce_factor < 1)
                    {
                        std::cout << "Reducing LR by " << reduce_factor << ". Current Factor: " << current_factor
                                  << std::endl;
                    }

                    // pipeline->UpdateLearningRate(epoch_id, params->train_params.num_epochs);
                    pipeline->UpdateLearningRate(reduce_factor);
                    for (auto& s : train_scenes->data)
                    {
                        // s.scene->UpdateLearningRate(epoch_id, params->train_params.num_epochs);
                        s.scene->UpdateLearningRate(epoch_id, reduce_factor);
                    }

                    for (auto& sd : train_scenes->data)
                    {
                        sd.epoch_loss.Average().AppendToFile(
                            full_experiment_dir + "loss_train_" + sd.scene->scene->scene_name + ".txt", epoch_id);
                    }
                }

                pipeline->Log(full_experiment_dir);
                for (auto& s : train_scenes->data)
                {
                    s.scene->Log(full_experiment_dir);
                }
                bool want_eval =
                    params->train_params.do_eval && (!params->train_params.eval_only_on_checkpoint || save_checkpoint);

                if (want_eval)
                {
                    EvalEpoch(epoch_id, save_checkpoint);

                    for (auto& sd : train_scenes->data)
                    {
                        sd.epoch_loss.Average().AppendToFile(
                            full_experiment_dir + "loss_eval_" + sd.scene->scene->scene_name + ".txt", epoch_id);
                    }
                }
            }

            if (save_checkpoint)
            {
                bool reduced_cp = params->train_params.reduced_check_point && !last_ep;
                // Save checkpoint
                console << "Saving checkpoint..." << std::endl;

                if (!reduced_cp)
                {
                    pipeline->SaveCheckpoint(ep_dir);
                }

                // for (auto scene : scenes)
                for (auto& s : train_scenes->data)
                {
                    s.scene->SaveCheckpoint(ep_dir, reduced_cp);
                }
            }
        }
    }


    double TrainEpoch(int epoch_id, std::vector<SceneDataTrainSampler>& data, bool structure_only)
    {
        train_scenes->StartEpoch();
        // Train
        float epoch_loss           = 0;
        int num_images             = 0;
        auto [loader, loader_size] = train_scenes->DataLoader(data, true);

        pipeline->Train(epoch_id);

        {
            Saiga::ProgressBar bar(
                std::cout, "Train " + std::to_string(epoch_id) + " |",
                loader_size * params->train_params.batch_size * params->train_params.inner_batch_size, 30, false, 5000);
            for (std::vector<NeuralTrainData>& batch : *loader)
            {
                SAIGA_ASSERT(batch.size() == params->train_params.batch_size * params->train_params.inner_batch_size);

                int scene_id_of_batch = batch.front()->scene_id;
                auto& scene_data      = train_scenes->data[scene_id_of_batch];

                train_scenes->Load(device, scene_id_of_batch);
                train_scenes->Train(epoch_id, true);
                if (!train_crop_mask.defined() && params->train_params.train_mask_border > 0)
                {
                    int h           = batch.front()->img.h;
                    int w           = batch.front()->img.w;
                    train_crop_mask = CropMask(h, w, params->train_params.train_mask_border).to(device);
                    TensorToImage<unsigned char>(train_crop_mask).save(full_experiment_dir + "/train_mask.png");
                }

                auto result = pipeline->Forward(*scene_data.scene, batch, train_crop_mask, false);
                scene_data.epoch_loss += result.float_loss;
                epoch_loss += result.float_loss.loss_float;
                num_images += batch.size();

                result.loss.backward();

                if (!structure_only)
                {
                    pipeline->OptimizerStep(epoch_id);
                }
                scene_data.scene->OptimizerStep(epoch_id, structure_only);
                bar.addProgress(batch.size());
                bar.SetPostfix(" Cur=" + std::to_string(result.float_loss.loss_float) + " Avg=" +
                               std::to_string(epoch_loss / num_images * params->train_params.batch_size *
                                              params->train_params.inner_batch_size));
            }
        }
        train_scenes->Unload();
        return epoch_loss;
    }

    void EvalEpoch(int epoch_id, bool save_checkpoint)
    {
        train_scenes->StartEpoch();

        if (params->train_params.interpolate_eval_settings)
        {
            SAIGA_ASSERT(params->train_params.optimize_eval_camera == false);
            for (int i = 0; i < train_scenes->data.size(); ++i)
            {
                auto indices = train_scenes->data[i].not_training_indices;
                train_scenes->data[i].scene->camera->InterpolateFromNeighbors(indices);
            }
        }

        // Eval
        torch::NoGradGuard ngg;
        float epoch_loss           = 0;
        int num_images             = 0;
        auto [loader, loader_size] = train_scenes->DataLoader(train_scenes->test_samplers, false);

        pipeline->Train(false);

        float best_loss  = 1000000;
        float worst_loss = 0;
        ForwardResult best_batch, worst_batch;

        bool write_test_images = save_checkpoint && params->train_params.write_test_images;

        Saiga::ProgressBar bar(std::cout, "Eval  " + std::to_string(epoch_id) + " |", loader_size, 30, false, 5000);
        for (std::vector<NeuralTrainData>& batch : *loader)
        {
            int scene_id_of_batch = batch.front()->scene_id;
            auto& scene_data      = train_scenes->data[scene_id_of_batch];
            train_scenes->Load(device, scene_id_of_batch);
            train_scenes->Train(epoch_id, false);

            SAIGA_ASSERT(!torch::GradMode::is_enabled());
            auto result = pipeline->Forward(*scene_data.scene, batch, scene_data.eval_crop_mask, true,
                                            write_test_images | params->train_params.write_images_at_checkpoint);

            if (params->train_params.write_images_at_checkpoint)
            {
                for (int i = 0; i < result.image_ids.size(); ++i)
                {
                    result.outputs[i].save(ep_dir + "/" + scene_data.scene->scene->scene_name + "_" +
                                           leadingZeroString(result.image_ids[i], 5) +
                                           params->train_params.output_file_type);

                    result.targets[i].save(ep_dir + "/" + scene_data.scene->scene->scene_name + "_" +
                                           leadingZeroString(result.image_ids[i], 5) + "_gt" +
                                           params->train_params.output_file_type);
                }
            }
            if (write_test_images)
            {
                if (result.float_loss.loss_float < best_loss)
                {
                    best_loss  = result.float_loss.loss_float;
                    best_batch = result;
                }

                if (result.float_loss.loss_float > worst_loss)
                {
                    worst_loss  = result.float_loss.loss_float;
                    worst_batch = result;
                }
            }

            epoch_loss += result.float_loss.loss_float;
            scene_data.epoch_loss += result.float_loss;
            num_images += batch.size();

            bar.addProgress(batch.size());
            bar.SetPostfix(" Cur=" + std::to_string(result.float_loss.loss_float) +
                           " Avg=" + std::to_string(epoch_loss / num_images));
        }
        train_scenes->Unload();

        bar.Quit();

        if (write_test_images)
        {
            console << "Best - Worst (Eval) [" << best_loss << ", " << worst_loss << "]" << std::endl;

            for (int i = 0; i < best_batch.targets.size(); ++i)
            {
                best_batch.targets[i].save(ep_dir + "/img_best_" + to_string(best_batch.image_ids[i]) + "_target" +
                                           params->train_params.output_file_type);
                best_batch.outputs[i].save(ep_dir + "/img_best_" + to_string(best_batch.image_ids[i]) + "_output" +
                                           params->train_params.output_file_type);
            }
            for (int i = 0; i < worst_batch.targets.size(); ++i)
            {
                worst_batch.targets[i].save(ep_dir + "/img_worst_" + to_string(worst_batch.image_ids[i]) + "_target" +
                                            params->train_params.output_file_type);
                worst_batch.outputs[i].save(ep_dir + "/img_worst_" + to_string(worst_batch.image_ids[i]) + "_output" +
                                            params->train_params.output_file_type);
            }
        }

        for (auto& sd : train_scenes->data)
        {
            console << "Loss: " << std::setw(20) << sd.scene->scene->scene_name << " ";
            sd.epoch_loss.Average().Print();
        }
    }
};

int main(int argc, char* argv[])
{
    std::cout << "Git ref: " << GIT_SHA1 << std::endl;

    std::string config_file;
    CLI::App app{"Train ADOP on your Scenes", "adop_train"};
    app.add_option("--config", config_file)->required();
    CLI11_PARSE(app, argc, argv);

    if (argc <= 1)
    {
        std::cout << "usage: ./Train <train_config_file>" << std::endl;
        return 0;
    }


    console << "Train Config: " << config_file << std::endl;


    params = std::make_shared<CombinedParams>(config_file);
    if (params->train_params.random_seed == 0)
    {
        std::cout << "generating random seed..." << std::endl;
        params->train_params.random_seed = Random::generateTimeBasedSeed();
    }

    {
        std::cout << "Using Random Seed: " << params->train_params.random_seed << std::endl;
        Random::setSeed(params->train_params.random_seed);
        torch::manual_seed(params->train_params.random_seed * 937545);
    }


    params->Check();
    console << "torch::cuda::cudnn_is_available() " << torch::cuda::cudnn_is_available() << std::endl;
    std::filesystem::create_directories("experiments/");

    {
        NeuralTrainer trainer;
    }

    return 0;
}
