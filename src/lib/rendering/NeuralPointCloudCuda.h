/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/cuda.h"
#include "saiga/cuda/thrust_helper.h"
#include "saiga/normal_packing.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "NeuralPointCloud.h"

#include <torch/torch.h>

#include "cuda_fp16.h"

class NeuralPointCloudCudaImpl : public NeuralPointCloud, public torch::nn::Module
{
   public:
    NeuralPointCloudCudaImpl(const Saiga::UnifiedMesh& model);



    void MakeOutlier(int max_index);

    std::vector<int> Indices();
    void SetIndices(std::vector<int>& indices);

    int Size();
    Saiga::UnifiedMesh Mesh();

    // [n, 4]
    torch::Tensor t_position;
    // [n, 4]
    torch::Tensor t_normal;
    // torch::Tensor t_normal_test;

    // [n, 1] (index1)
    torch::Tensor t_index;

    using PointType  = vec4;
    using NormalType = vec4;
};


TORCH_MODULE(NeuralPointCloudCuda);


// A simple helper class to make the kernels more compact.
struct DevicePointCloud
{
    int4* position;
    // int4* normal_test;
    // half2* normal;
    int* normal;
    int* index;

    int n;

    DevicePointCloud() = default;

    DevicePointCloud(NeuralPointCloudCuda pc)
    {
        SAIGA_ASSERT(pc->t_position.size(0) == pc->t_normal.size(0));
        SAIGA_ASSERT(pc->t_position.size(0) == pc->t_index.size(0));
        SAIGA_ASSERT(pc->t_position.size(0) == pc->Size());

        position = (int4*)pc->t_position.data_ptr<float>();
        // normal   = (half2*)pc->t_normal.data_ptr<torch::Half>();
        normal = pc->t_normal.data_ptr<int>();
        // normal_test = (int4*)pc->t_normal_test.data_ptr<float>();
        index = (int*)pc->t_index.data_ptr();

        n = pc->Size();
    }
    HD inline thrust::tuple<vec3, vec3, float> GetPoint(int point_index)
    {
        vec4 p;
        vec4 n_test;



        // float4 global memory loads are vectorized!
        reinterpret_cast<int4*>(&p)[0] = position[point_index];
        // reinterpret_cast<int4*>(&n_test)[0] = normal_test[point_index];

        // half2 n_half = normal[point_index];
        // vec2 enc(n_half.x, n_half.y);

        auto enc = normal[point_index];

        vec3 n = UnpackNormal10Bit(enc);



        // if (point_index % 100000 == 0)
        // {
        //     printf("normal check: %d | %f %f %f | %f %f %f\n", point_index, n(0), n(1), n(2), n_test(0), n_test(1),
        //            n_test(2));
        // }

        float drop_out_radius = p(3);

        return {p.head<3>(), n.head<3>(), drop_out_radius};
    }

    HD inline int GetIndex(int tid) { return index[tid]; }
    HD inline int Size() { return n; }
};