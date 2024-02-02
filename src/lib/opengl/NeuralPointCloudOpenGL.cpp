/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */


#include "NeuralPointCloudOpenGL.h"
#include "saiga/core/imgui/imgui.h"

using namespace Saiga;
template <>
void Saiga::VertexBuffer<PositionIndex>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    //    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    //    glEnableVertexAttribArray(3);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PositionIndex), NULL);
    //    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(NeuralPointVertex), (void*)(4 * sizeof(GLfloat)));
    glVertexAttribIPointer(2, 1, GL_INT, sizeof(PositionIndex), (void*)(3 * sizeof(GLfloat)));
    //    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(NeuralPointVertex), (void*)(9 * sizeof(GLfloat)));
}


NeuralPointCloudOpenGL::NeuralPointCloudOpenGL(const UnifiedMesh& model) : NeuralPointCloud(model)
{
    shader = Saiga::shaderLoader.load<Saiga::Shader>("point_render.glsl");
    gl_points.setDrawMode(GL_POINTS);
    gl_points.set(points, GL_STATIC_DRAW);

    if(color.size() == points.size())
    {
        gl_color.create(color, GL_STATIC_DRAW);
        gl_points.addExternalBuffer(gl_color, 1, 4, GL_FLOAT, GL_FALSE, sizeof(vec4), 0);
    }


    if(normal.size() == points.size())
    {
        gl_normal.create(normal, GL_STATIC_DRAW);
        gl_points.addExternalBuffer(gl_normal, 3, 4, GL_FLOAT, GL_FALSE, sizeof(vec4), 0);
    }

    if (data.size() == points.size())
    {
        gl_data.create(data, GL_STATIC_DRAW);
        gl_points.addExternalBuffer(gl_data, 4, 4, GL_FLOAT, GL_FALSE, sizeof(vec4), 0);
    }
}


void NeuralPointCloudOpenGL::render(const FrameData& fd, float scale)
{
    if (shader->bind())
    {
        glPointSize(1);
        glEnable(GL_PROGRAM_POINT_SIZE);
        auto cam = fd.GLCamera();

        shader->upload(0, mat4(mat4::Identity()));
        shader->upload(1, cam.view);
        shader->upload(2, cam.proj);

        mat4 v = fd.pose.inverse().matrix().cast<float>().eval();
        mat3 k = fd.K.matrix();
        vec2 s(fd.w, fd.h);

        shader->upload(3, v);
        shader->upload(4, k);
        shader->upload(5, s);
        shader->upload(6, scale);

        vec4 dis1 = fd.distortion.Coeffs().head<4>();
        vec4 dis2 = fd.distortion.Coeffs().tail<4>();

        shader->upload(7, dis1);
        shader->upload(8, dis2);
        shader->upload(9, exp2(fd.exposure_value));

        // render settings
        shader->upload(10, render_mode);
        shader->upload(11, max_value);

        gl_points.bindAndDraw();

        glDisable(GL_PROGRAM_POINT_SIZE);
        shader->unbind();
    }
}
void NeuralPointCloudOpenGL::imgui()
{
    ImGui::InputFloat("render_max_value", &max_value);
    ImGui::SliderInt("Point Render Mode", &render_mode, 0, 4);
}
