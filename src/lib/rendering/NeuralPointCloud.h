/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/math.h"
#include "saiga/core/sophus/Sophus.h"

#include "config.h"
#include "data/SceneData.h"

using namespace Saiga;

struct SAIGA_ALIGN(16) PositionIndex
{
    vec3 position;
    int index;
};

class NeuralPointCloud : public Saiga::Object3D
{
   public:
    NeuralPointCloud(const Saiga::UnifiedMesh& model)
    {
        SAIGA_ASSERT(model.NumVertices() == model.normal.size());
        SAIGA_ASSERT(model.NumVertices() == model.color.size());
        // SAIGA_ASSERT(model.NumVertices() == model.data.size());

        for (int i = 0; i < model.NumVertices(); ++i)
        {
            PositionIndex npv;
            npv.position = model.position[i];
            npv.index = i;
            points.push_back(npv);

            normal.push_back(make_vec4(model.normal[i].normalized(), 0));
            color.push_back(model.color[i]);

            if (model.NumVertices() == model.data.size())
            {
                data.push_back(model.data[i]);
            }
        }
    }

    std::vector<PositionIndex> points;
    std::vector<vec4> normal;
    std::vector<vec4> color;
    std::vector<vec4> data;

};
