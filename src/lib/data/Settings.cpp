/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Settings.h"
#include "saiga/core/imgui/imgui.h"


void CombinedParams::Check()
{
    if (net_params.conv_block == "partial" || net_params.conv_block == "partial_multi" ||
        pipeline_params.enable_environment_map || pipeline_params.cat_masks_to_color)
    {
        render_params.output_background_mask = true;
    }

    if (pipeline_params.skip_neural_render_network)
    {
        pipeline_params.num_texture_channels = 3;
        net_params.num_input_layers          = 1;
    }

    net_params.num_input_channels      = pipeline_params.num_texture_channels;
    render_params.num_texture_channels = pipeline_params.num_texture_channels;


    SAIGA_ASSERT(!train_params.texture_color_init || pipeline_params.num_texture_channels == 3);

    if (pipeline_params.cat_env_to_color)
    {
        net_params.num_input_channels += pipeline_params.env_map_channels;
    }
    else
    {
        pipeline_params.env_map_channels = pipeline_params.num_texture_channels;
    }

    if (pipeline_params.cat_masks_to_color)
    {
        net_params.num_input_channels += 1;
    }
}
void CombinedParams::imgui()
{
    ImGui::Checkbox("render_points", &render_params.render_points);
    ImGui::Checkbox("render_outliers", &render_params.render_outliers);
    ImGui::Checkbox("drop_out_points_by_radius", &render_params.drop_out_points_by_radius);
    ImGui::SliderFloat("drop_out_radius_threshold", &render_params.drop_out_radius_threshold, 0, 5);
    ImGui::Checkbox("super_sampling", &render_params.super_sampling);

    ImGui::Checkbox("check_normal", &render_params.check_normal);


    ImGui::Checkbox("debug_weight_color", &render_params.debug_weight_color);
    ImGui::Checkbox("debug_depth_color", &render_params.debug_depth_color);
    ImGui::SliderFloat("debug_max_weight", &render_params.debug_max_weight, 0, 100);
    ImGui::Checkbox("debug_print_num_rendered_points", &render_params.debug_print_num_rendered_points);

    ImGui::SliderFloat("dropout", &render_params.dropout, 0, 1);
    ImGui::SliderFloat("depth_accept", &render_params.depth_accept, 0, 0.1);

    ImGui::SliderFloat("dist_cutoff", &render_params.dist_cutoff, 0, 1);

    ImGui::Separator();
}
