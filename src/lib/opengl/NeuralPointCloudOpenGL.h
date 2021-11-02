/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/all.h"

#include "rendering/NeuralPointCloud.h"


class NeuralPointCloudOpenGL : public NeuralPointCloud
{
   public:
    NeuralPointCloudOpenGL(const Saiga::UnifiedMesh& model);



    void render(const FrameData& fd, float scale);

    void imgui();

    std::shared_ptr<Saiga::Shader> shader;

    Saiga::VertexBuffer<PositionIndex> gl_points;
    Saiga::TemplatedBuffer<vec4> gl_normal = {GL_ARRAY_BUFFER};
    Saiga::TemplatedBuffer<vec4> gl_color  = {GL_ARRAY_BUFFER};
    Saiga::TemplatedBuffer<vec4> gl_data   = {GL_ARRAY_BUFFER};

    // render settings
    float max_value = 10;
    int render_mode = 0;
};
