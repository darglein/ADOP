/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Dataset.h"

#include "saiga/vision/torch/RandomCrop.h"


TemplatedImage<vec2> InitialUVImage(int h, int w)
{
    TemplatedImage<vec2> uv_target;
    uv_target.create(h, w);


    for (int i : uv_target.rowRange())
    {
        for (int j : uv_target.colRange())
        {
            // int y = h - i - 1;
            vec2 texel(j, i);
            texel.x() /= (w - 1);
            texel.y() /= (h - 1);
            vec2 centered_uv = (vec2(texel) - vec2(0.5, 0.5)) * 2;

            uv_target(i, j) = centered_uv;
        }
    }

    SAIGA_ASSERT(uv_target.getImageView().isFinite());
    return uv_target;
}


SceneDataTrainSampler::SceneDataTrainSampler(std::shared_ptr<SceneData> dataset, std::vector<int> indices,
                                             bool down_scale, ivec2 crop_size, int inner_batch_size,
                                             bool use_image_mask)
    : inner_batch_size(inner_batch_size),
      scene(dataset),
      indices(indices),
      down_scale(down_scale),
      use_image_mask(use_image_mask)
{
    image_size_input = ivec2(dataset->scene_cameras.front().w, dataset->scene_cameras.front().h);
    for (auto& cam : dataset->scene_cameras)
    {
        // all cameras must have the same image size
        SAIGA_ASSERT(cam.w == image_size_input(0));
        SAIGA_ASSERT(cam.h == image_size_input(1));
    }

    if (down_scale)
    {
        SAIGA_ASSERT(crop_size(0) > 0);
        image_size_crop = crop_size;
    }
    else
    {
        image_size_crop = image_size_input;
    }

    uv_target = InitialUVImage(image_size_input(1), image_size_input(0));
}


template <typename T>
void warpPerspective(ImageView<T> src, ImageView<T> dst, IntrinsicsPinholef dst_2_src)
{
    for (auto y : dst.rowRange())
    {
        for (auto x : dst.colRange())
        {
            vec2 p(x, y);
            vec2 ip  = dst_2_src.normalizedToImage(p);
            float dx = ip(0);
            float dy = ip(1);
            if (src.template inImage(dy, dx))
            {
                dst(y, x) = src.inter(dy, dx);
            }
        }
    }
}


template <typename T>
void warpPerspectiveNearest(ImageView<T> src, ImageView<T> dst, IntrinsicsPinholef dst_2_src)
{
    for (auto y : dst.rowRange())
    {
        for (auto x : dst.colRange())
        {
            vec2 p(x, y);
            vec2 ip = dst_2_src.normalizedToImage(p);
            int dx  = round(ip(0));
            int dy  = round(ip(1));
            if (src.template inImage(dy, dx))
            {
                dst(y, x) = src(dy, dx);
            }
        }
    }
}

std::vector<NeuralTrainData> SceneDataTrainSampler::Get(int __index)
{
    static thread_local bool init = false;
    static std::atomic_int count  = 2346;
    if (!init)
    {
        int x = count.fetch_add(142);
        Saiga::Random::setSeed(98769264 * x);
    }

    long actual_index = indices[__index];
    const auto fd     = scene->Frame(actual_index);

    SAIGA_ASSERT(std::filesystem::exists(scene->dataset_params.image_dir + "/" + fd.target_file));
    Saiga::TemplatedImage<ucvec3> img_gt_large(scene->dataset_params.image_dir + "/" + fd.target_file);

    Saiga::TemplatedImage<unsigned char> img_mask_large;
    if (use_image_mask)
    {
        auto f = scene->dataset_params.mask_dir + "/" + fd.mask_file;
        SAIGA_ASSERT(std::filesystem::exists(f));
        bool ret = img_mask_large.load(f);
        if (!ret)
        {
            SAIGA_EXIT_ERROR("could not load mask image " + f);
        }
    }

    std::vector<NeuralTrainData> result;
    result.reserve(inner_batch_size);

    auto crops = RandomImageCrop(inner_batch_size, inner_sample_size, image_size_input, image_size_crop, prefere_border,
                                 random_translation, min_max_zoom);
    SAIGA_ASSERT(crops.size() == inner_batch_size);

    for (int i = 0; i < inner_batch_size; ++i)
    {
        NeuralTrainData pd = std::make_shared<TorchFrameData>();

        pd->img.crop_transform = IntrinsicsPinholef();

        pd->img.h = image_size_crop(1);
        pd->img.w = image_size_crop(0);

        float zoom = 1;
        if (down_scale)
        {
            pd->img.crop_transform = crops[i];
            zoom                   = pd->img.crop_transform.fx;

            Saiga::TemplatedImage<ucvec3> gt_crop(image_size_crop.y(), image_size_crop.x());
            Saiga::TemplatedImage<vec2> uv_crop(image_size_crop.y(), image_size_crop.x());

            gt_crop.makeZero();
            uv_crop.makeZero();

            auto dst_2_src = pd->img.crop_transform.inverse();

            warpPerspective(img_gt_large.getImageView(), gt_crop.getImageView(), dst_2_src);
            warpPerspective(uv_target.getImageView(), uv_crop.getImageView(), dst_2_src);

            pd->target = ImageViewToTensor(gt_crop.getImageView());
            pd->uv     = ImageViewToTensor(uv_crop.getImageView());

            TemplatedImage<unsigned char> mask(pd->img.h, pd->img.w);
            for (int i : mask.rowRange())
            {
                for (int j : mask.colRange())
                {
                    if (uv_crop(i, j).isZero() && gt_crop(i, j).isZero())
                    {
                        mask(i, j) = 0;
                    }
                    else
                    {
                        mask(i, j) = 255;
                    }
                }
            }
            pd->target_mask = ImageViewToTensor(mask.getImageView());

            if (use_image_mask)
            {
                Saiga::TemplatedImage<unsigned char> mask_crop(image_size_crop.y(), image_size_crop.x());
                warpPerspective(img_mask_large.getImageView(), mask_crop.getImageView(), dst_2_src);
                pd->target_mask = pd->target_mask * ImageViewToTensor(mask_crop.getImageView());
            }

            SAIGA_ASSERT(uv_crop.getImageView().isFinite());
        }
        else
        {
            pd->target = ImageViewToTensor(img_gt_large.getImageView());
            pd->uv     = ImageViewToTensor(uv_target.getImageView());

            TemplatedImage<unsigned char> mask(pd->img.h, pd->img.w);
            mask.getImageView().set(255);
            pd->target_mask = ImageViewToTensor(mask.getImageView());

            if (use_image_mask)
            {
                pd->target_mask = pd->target_mask * ImageViewToTensor(img_mask_large.getImageView());
            }
        }


        float min_mask_value = 0.1;
        pd->target_mask      = pd->target_mask.clamp_min(min_mask_value);

        long camera_index         = actual_index;
        pd->img.camera_model_type = scene->dataset_params.camera_model;
        pd->img.image_index       = actual_index;
        pd->img.camera_index      = fd.camera_index;
        pd->camera_index = torch::from_blob(&camera_index, {1}, torch::TensorOptions().dtype(torch::kLong)).cuda();
        pd->scale        = torch::from_blob(&zoom, {1, 1, 1}, torch::TensorOptions().dtype(torch::kFloat32)).cuda();
        pd->to(torch::kCUDA);
        result.push_back(std::move(pd));
    }


    // sample map debug image
    // Test if the random intrinsics provide a good cover of the large input image
#if 0
    std::cout << "create sample map debug image" << std::endl;

    TemplatedImage<ucvec3> input(img_gt_large);
    input.makeZero();

    ucvec3 color(255, 0, 0);
    int i = 0;
    for (auto td : result)
    {
        std::cout << "crop transform: " << td->img.crop_transform << std::endl;
        std::vector<vec2> corners = {vec2(0, 0), vec2(image_size_crop(0) - 1, 0),
                                     vec2(image_size_crop(0) - 1, image_size_crop(1) - 1),
                                     vec2(0, image_size_crop(1) - 1)};
        std::cout << "Image size      " << image_size_input.transpose() << std::endl;
        std::cout << "Image size crop " << image_size_crop.transpose() << std::endl;
        for (auto& c : corners)
        {
            auto o = c;
            c      = td->img.crop_transform.inverse().normalizedToImage(c);
            c      = c.array().round();
            ImageDraw::drawCircle(input.getImageView(), c, 10, color);
            std::cout << "Corner " << c.transpose() << std::endl;
        }

        vec2 center = image_size_crop.cast<float>() * 0.5f;
        vec2 c      = td->img.crop_transform.inverse().normalizedToImage(center);
        c           = c.array().round();
        ImageDraw::drawCircle(input.getImageView(), c, 5, color);
        std::cout << "Center " << c.transpose() << std::endl;

        ImageDraw::drawLineBresenham(input.getImageView(), corners[0], corners[1], color);
        ImageDraw::drawLineBresenham(input.getImageView(), corners[1], corners[2], color);
        ImageDraw::drawLineBresenham(input.getImageView(), corners[2], corners[3], color);
        ImageDraw::drawLineBresenham(input.getImageView(), corners[3], corners[0], color);

        TensorToImage<ucvec3>(td->target).save("debug/crop_" + std::to_string(i) + "_target.jpg");
        TensorToImage<unsigned char>(td->target_mask).save("debug/crop_" + std::to_string(i) + "_mask.png");
        // TensorToImage<unsigned char>(td->classes).save("debug/crop_" + std::to_string(i) + "_classes.png");

        i++;
    }

    input.save("debug/sample_map.png");
    img_gt_large.save("debug/gt.jpg");
    exit(0);
#endif
    return result;
}

torch::MultiDatasetSampler::MultiDatasetSampler(std::vector<uint64_t> _sizes, int batch_size, bool shuffle)
    : sizes(_sizes), batch_size(batch_size)
{
    int total_batches = 0;
    std::vector<std::vector<size_t>> indices;
    for (int i = 0; i < sizes.size(); ++i)
    {
        auto& s = sizes[i];
        std::vector<size_t> ind(s);
        std::iota(ind.begin(), ind.end(), 0);

        if (shuffle)
        {
            std::shuffle(ind.begin(), ind.end(), Random::generator());
        }

        s = Saiga::iAlignDown(s, batch_size);
        ind.resize(s);
        SAIGA_ASSERT(ind.size() % batch_size == 0);
        total_batches += ind.size() / batch_size;

        for (auto j : ind)
        {
            combined_indices.emplace_back(i, j);
        }
    }

    batch_offsets.resize(total_batches);
    std::iota(batch_offsets.begin(), batch_offsets.end(), 0);
    if (shuffle)
    {
        std::shuffle(batch_offsets.begin(), batch_offsets.end(), Random::generator());
    }
}

torch::optional<std::vector<size_t>> torch::MultiDatasetSampler::next(size_t batch_size)
{
    SAIGA_ASSERT(this->batch_size == batch_size);

    if (current_index >= batch_offsets.size()) return {};

    auto bo = batch_offsets[current_index++];

    std::vector<size_t> result;

    for (int i = 0; i < batch_size; ++i)
    {
        auto [scene, image] = combined_indices[i + bo * batch_size];

        size_t comb = (size_t(scene) << 32) | size_t(image);
        result.push_back(comb);
    }


    return result;
}



TorchSingleSceneDataset::TorchSingleSceneDataset(std::vector<SceneDataTrainSampler> sampler) : sampler(sampler) {}

std::vector<NeuralTrainData> TorchSingleSceneDataset::get2(size_t index)
{
    int scene = index >> 32UL;
    int image = index & ((1UL << 32UL) - 1UL);

    auto data = sampler[scene].Get(image);
    for (auto& l : data) l->scene_id = scene;
    return data;
}
std::vector<NeuralTrainData> TorchSingleSceneDataset::get_batch(torch::ArrayRef<size_t> indices)
{
    std::vector<NeuralTrainData> batch;
    batch.reserve(indices.size());
    for (const auto i : indices)
    {
        auto e = get2(i);
        for (auto& l : e)
        {
            batch.push_back(std::move(l));
        }
    }
    return batch;
}
