/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "config.h"
#include "opengl/RealTimeRenderer.h"
#include "opengl/SceneViewer.h"

#include <stack>
using namespace Saiga;
enum class ViewMode
{
    // Only the model
    MODEL,
    // Only neural render
    NEURAL,
    // 4x4 Split with neural, model, and point cloud
    SPLIT_NEURAL,
};


class ViewerBase
{
   public:
    ViewerBase()
    {
        {
            // ignore performance warning when downloading auto exposure data
            std::vector<GLuint> ids;
            ids.push_back(131186);
            Error::ignoreGLError(ids);
        }

        // Define GUI layout
        auto editor_layout = std::make_unique<EditorLayoutLSplit2x2>();
        editor_layout->RegisterImguiWindow("Video Recording", EditorLayoutLSplit2x2::WINDOW_POSITION_LEFT);
        editor_layout->RegisterImguiWindow("Model Viewer", EditorLayoutLSplit2x2::WINDOW_POSITION_LEFT);
        editor_layout->RegisterImguiWindow("Capture View", EditorLayoutLSplit2x2::WINDOW_POSITION_LEFT_BOTTOM);
        editor_layout->RegisterImguiWindow("Neural View", EditorLayoutLSplit2x2::WINDOW_POSITION_MAIN_12);
        editor_layout->RegisterImguiWindow("Debug View", EditorLayoutLSplit2x2::WINDOW_POSITION_MAIN_21);
        editor_layout->RegisterImguiWindow("Closest Ground Truth", EditorLayoutLSplit2x2::WINDOW_POSITION_MAIN_22);
        editor_layout->RegisterImguiWindow("Neural Renderer", EditorLayoutLSplit2x2::WINDOW_POSITION_LEFT_BOTTOM);
        editor_gui.SetLayout(std::move(editor_layout));
    }

    void LoadScene(std::string scene_dir)
    {
        neural_renderer = nullptr;
        scene           = std::make_shared<SceneViewer>(std::make_shared<SceneData>(scene_dir));
        current_scene   = next_scene;
        object_tex      = {};
        object_col      = {};
    }


    void imgui()
    {
        ImGui::Begin("Model Viewer");


        ImGui::Checkbox("renderPoints", &render_points);
        ImGui::Checkbox("renderWireframe", &renderWireframe);
        ImGui::Checkbox("renderObject", &renderObject);
        ImGui::Checkbox("renderTexture", &renderTexture);

        if (ImGui::Button("reset objects"))
        {
            object_tex = {};
            object_col = {};
        }

        if (ImGui::Button("RemoveInvalidPoints"))
        {
            scene->RemoveInvalidPoints();
        }



        scene->imgui();

        ImGui::End();

        if (neural_renderer)
        {
            neural_renderer->imgui();
        }
    }

   protected:
    int current_scene = -1;
    int next_scene    = 0;
    std::shared_ptr<SceneViewer> scene;

    std::shared_ptr<TexturedAsset> object_tex;
    std::shared_ptr<ColoredAsset> object_col;

    std::unique_ptr<RealTimeRenderer> neural_renderer;

    bool renderObject    = true;
    bool renderTexture   = true;
    bool renderWireframe = false;
    bool render_points   = true;
    bool render_debug    = true;
};
