/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "NeuralPointCloudCuda.h"

#include "saiga/normal_packing.h"

NeuralPointCloudCudaImpl::NeuralPointCloudCudaImpl(const UnifiedMesh& model) : NeuralPointCloud(model)
{
    std::vector<vec4> data_position;
    std::vector<vec4> data_normal;
    std::vector<int> data_normal_compressed;

    std::vector<int> indices;
    for (int i = 0; i < points.size(); ++i)
    {
        indices.push_back(points[i].index);

        float drop_out_radius = 0;
        if (data.size() == points.size())
        {
            drop_out_radius = data[i](3);
        }

        vec3 n = normal[i].head<3>();
        SAIGA_ASSERT(n.allFinite());
        auto n_enc = PackNormal10Bit(n);

        data_normal_compressed.push_back(n_enc);

        data_position.push_back(make_vec4(points[i].position, drop_out_radius));
        data_normal.push_back(normal[i]);
    }

    t_position = torch::from_blob(data_position.data(), {(long)data_position.size(), 4},
                                  torch::TensorOptions().dtype(torch::kFloat32))
                     .contiguous()
                     .cuda()
                     .clone();

    //    t_normal = torch::from_blob(data_normal_compressed.data(), {(long)data_normal_compressed.size(), 2},
    //                                torch::TensorOptions().dtype(torch::kFloat32))
    //                   .contiguous()
    //                   .cuda()
    //                   .clone();

    t_normal = torch::from_blob(data_normal_compressed.data(), {(long)data_normal_compressed.size(), 1},
                                torch::TensorOptions().dtype(torch::kInt32))
                   .contiguous()
                   .cuda()
                   .clone();

    //    t_normal_test = torch::from_blob(data_normal.data(), {(long)data_normal.size(), 4},
    //                                     torch::TensorOptions().dtype(torch::kFloat32))
    //                        .contiguous()
    //                        .cuda()
    //                        .clone();

    //    PrintTensorInfo(t_normal);
    // t_normal = t_normal.to(torch::kFloat16);
    //    PrintTensorInfo(t_normal);

    t_index = torch::from_blob(indices.data(), {(long)indices.size(), 1}, torch::TensorOptions().dtype(torch::kInt32))
                  .contiguous()
                  .cuda()
                  .clone();


    register_parameter("t_position", t_position);
    register_buffer("t_normal", t_normal);
    // register_buffer("t_normal_test", t_normal_test);
    register_buffer("t_index", t_index);

    SAIGA_ASSERT(t_position.isfinite().all().item().toBool());
    // SAIGA_ASSERT(t_normal.isfinite().all().item().toBool());

    std::cout << "GPU memory - Point Cloud: "
              << (t_position.nbytes() + t_normal.nbytes() + t_index.nbytes()) / 1000000.0 << "MB" << std::endl;
}
Saiga::UnifiedMesh NeuralPointCloudCudaImpl::Mesh()
{
    std::cout << "Extracing Point Cloud from device data" << std::endl;

    // Position
    PrintTensorInfo(t_position);
    std::vector<vec4> data_position(t_position.size(0), vec4(-1, -1, -1, -1));
    torch::Tensor cp_position = t_position.contiguous().cpu();
    memcpy(data_position[0].data(), cp_position.data_ptr(), sizeof(vec4) * data_position.size());

    // Normal
    PrintTensorInfo(t_normal);
    std::vector<int> data_normal(t_normal.size(0));
    torch::Tensor cp_normal = t_normal.contiguous().cpu();
    memcpy(data_normal.data(), cp_normal.data_ptr(), sizeof(int) * data_normal.size());


    Saiga::UnifiedMesh mesh;
    for (auto p : data_position)
    {
        mesh.position.push_back(p.head<3>());
        mesh.data.push_back(vec4(0, 0, 0, p(3)));
    }
    for (auto n : data_normal)
    {
        vec3 n_dec = UnpackNormal10Bit(n);
        mesh.normal.push_back(n_dec);
    }
    return mesh;
}
void NeuralPointCloudCudaImpl::MakeOutlier(int max_index)
{
    SAIGA_ASSERT(0);
    torch::NoGradGuard ngg;
    t_index.uniform_(0, max_index);
}
std::vector<int> NeuralPointCloudCudaImpl::Indices()
{
    std::vector<int> indices(t_index.size(0));

    torch::Tensor cp_index = t_index.contiguous().cpu();

    memcpy(indices.data(), cp_index.data_ptr(), sizeof(int) * indices.size());

    return indices;
}
void NeuralPointCloudCudaImpl::SetIndices(std::vector<int>& indices)
{
#if 1
    // torch::NoGradGuard  ngg;
    t_index.set_data(
        torch::from_blob(indices.data(), {(long)indices.size(), 1}, torch::TensorOptions().dtype(torch::kFloat32))
            .contiguous()
            .cuda()
            .clone());
#else
    t_index = torch::from_blob(indices.data(), {(long)indices.size(), 1}, torch::TensorOptions().dtype(torch::kFloat32))
                  .contiguous()
                  .cuda()
                  .clone();
#endif
}
int NeuralPointCloudCudaImpl::Size()
{
    SAIGA_ASSERT(t_position.defined());
    return t_position.size(0);
}
