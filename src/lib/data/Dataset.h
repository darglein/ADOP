/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/Core.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/torch/ImageTensor.h"

#include "SceneData.h"
#include "config.h"
#include "rendering/RenderInfo.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <vector>

TemplatedImage<vec2> InitialUVImage(int h, int w);


using NeuralTrainData = std::shared_ptr<TorchFrameData>;

class SceneDataTrainSampler
{
   public:
    SceneDataTrainSampler() {}
    SceneDataTrainSampler(std::shared_ptr<SceneData> dataset, std::vector<int> indices, bool down_scale,
                          ivec2 crop_size, int inner_batch_size, bool use_image_mask);

    std::vector<NeuralTrainData> Get(int index);

    int Size() const { return indices.size(); }

   public:
    int inner_batch_size = 1;
    ivec2 image_size_crop;
    ivec2 image_size_input;

    TemplatedImage<vec2> uv_target;

    std::shared_ptr<SceneData> scene;
    std::vector<int> indices;

    int num_classes = -1;

    bool down_scale         = false;
    bool random_translation = true;
    bool random_zoom        = true;
    bool prefere_border     = true;
    bool use_image_mask     = false;
    int inner_sample_size   = 1;

    vec2 min_max_zoom = vec2(0.75, 1.5);
};

namespace torch
{
class MultiDatasetSampler : public torch::data::samplers::Sampler<>
{
   public:
    MultiDatasetSampler(std::vector<uint64_t> sizes, int batch_size, bool shuffle);

    void reset(torch::optional<size_t> new_size = torch::nullopt) override {}

    /// Returns the next batch of indices.
    optional<std::vector<size_t>> next(size_t batch_size) override;

    /// Serializes the `RandomSampler` to the `archive`.
    void save(serialize::OutputArchive& archive) const override {}

    /// Deserializes the `RandomSampler` from the `archive`.
    void load(serialize::InputArchive& archive) override {}

    /// Returns the current index of the `RandomSampler`.
    size_t index() const noexcept { return current_index; }

    int Size() { return batch_offsets.size(); }

    size_t current_index = 0;
    std::vector<std::pair<int, int>> combined_indices;
    std::vector<int> batch_offsets;
    std::vector<uint64_t> sizes;
    int batch_size;
};
}  // namespace torch


class TorchSingleSceneDataset : public torch::data::Dataset<TorchSingleSceneDataset, NeuralTrainData>
{
   public:
    TorchSingleSceneDataset(std::vector<SceneDataTrainSampler> sampler);
    virtual torch::optional<size_t> size() const override { return sampler.front().Size(); }
    virtual std::vector<NeuralTrainData> get2(size_t index);


    virtual NeuralTrainData get(size_t index) override { return {}; }

    std::vector<NeuralTrainData> get_batch(torch::ArrayRef<size_t> indices) override;

   private:
    std::vector<SceneDataTrainSampler> sampler;
};
