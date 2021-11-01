/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER
#version 430

layout(location=0) in vec4 in_position;
layout(location=1) in vec4 in_color;
layout(location=2) in int in_index;
layout(location=3) in vec3 in_normal;
layout(location=4) in vec4 in_data;


layout(location=0) uniform mat4 model;
layout(location=1) uniform mat4 view;
layout(location=2) uniform mat4 proj;


layout(location=3) uniform mat4 cv_view;
layout(location=4) uniform mat3 cv_proj;
layout(location=5) uniform vec2 cv_viewport;
layout(location=6) uniform float cv_scale;
layout(location=7) uniform vec4 cv_dis1;
layout(location=8) uniform vec4 cv_dis2;

#include "vision/distortion.glsl"


out vec4 v_color;
out vec4 v_data;
flat out int v_index;

void main() {
    v_color = in_color;
    v_data = in_data;
    v_index = in_index;
    vec4 world_p = vec4(in_position.xyz, 1);
    vec4 world_n = vec4(in_normal.xyz, 0);

    vec3 view_p = vec3(cv_view  * model *  world_p);
    vec3 view_n = normalize(vec3(cv_view * model *  world_n));

    float z = view_p.z;

    if (z <= 0)
    {
        gl_Position = vec4(0, 0, -100, 0);
        return;
    }

    if (dot(normalize(view_p), view_n) > 0){
        gl_Position = vec4(0, 0, -100, 0);
        return;
    }

    vec2 image_p = view_p.xy / z;
    image_p = distortNormalizedPoint(image_p, cv_dis1, cv_dis2);
    image_p = vec2(cv_proj * vec3(image_p, 1));

    float znear = 0.1;
    float zfar = 500;
    float f = zfar;
    float n = znear;
    float d = z * (f + n) / (f - n) -  (2 * f * n) / (f - n);
    float d2 = z;

    // remove viewport transform
    image_p = 2 * image_p  / cv_viewport - vec2(1);
    image_p.y *= -1;

    gl_Position = vec4(image_p * d2, d, d2);

    gl_PointSize = 1;
}


##GL_FRAGMENT_SHADER
#version 430

#include "include/saiga/colorize.h"
in vec4 v_color;
in vec4 v_data;
flat in int v_index;
layout(location=9) uniform float color_scale = 1;
layout(location=10) uniform int render_mode = 0;
layout(location=11) uniform float max_value = 1;


layout(location=0) out vec4 out_color;
layout(location=1) out int out_index;

void main() {
    vec4 color = vec4(1,0,0,0);
    if(render_mode == 0) {color = v_color;}

    if(render_mode >= 1) {
        float v = v_data[render_mode-1] / max_value;
        color = vec4(colorizeFusion(v), 1);
    }


    out_color = color_scale * color;
    out_index = v_index;
}


