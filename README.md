# ADOP: Approximate Differentiable One-Pixel Point Rendering

<div style="text-align: center;">Darius Rückert, Linus Franke, Marc Stamminger</div>

![](images/adop_overview.png)


**Abstract:** We present a novel point-based, differentiable neural rendering pipeline for
scene refinement and novel view synthesis. The input are an initial estimate of
the point cloud and the camera parameters. The output are synthesized images
from arbitrary camera poses. The point cloud rendering is performed by a
differentiable renderer using multi-resolution one-pixel point rasterization.
Spatial gradients of the discrete rasterization are approximated by the novel
concept of ghost geometry. After rendering, the neural image pyramid is passed
through a deep neural network for shading calculations and hole-filling. A
differentiable, physically-based tonemapper then converts the intermediate
output to the target image. Since all stages of the pipeline are
differentiable, we optimize all of the scene's parameters i.e. camera model,
camera pose, point position, point color, environment map, rendering network
weights, vignetting, camera response function, per image exposure, and per
image white balance. We show that our system is able to synthesize sharper and
more consistent novel views than existing approaches because the initial
reconstruction is refined during training. The efficient one-pixel point
rasterization allows us to use arbitrary camera models and display scenes with
well over 100M points in real time.

* The source code will be published after the paper has been accepted to a conference.

[[Full Paper]](https://arxiv.org/abs/2110.06635)

### Video

  <a href="https://www.youtube.com/watch?v=WJRyu1JUtVw"><img  width="300" src="https://img.youtube.com/vi/WJRyu1JUtVw/hqdefault.jpg"> </a>


### Compile Instructions

 * ADOP is implemented in C++/CUDA using libTorch.
 * A python wrapper for pyTorch is currently not available. Feel free to submit a pull-request on that issue.
 * The detailed compile instructions can be found here: [src/README.md](src/README.md)

### Running ADOP on pretrained models

After a successful compilation, the best way to get started is to run `adop_viewer` on the *tanks and temples* scenes using our pretrained models.
First, download the [scenes](todo) and extract them into `ADOP/scenes`. 
Now, download the [model checkpoints](todo) and extract them into `ADOP/experiments`.
Your folder structure should look like this:
```shell
ADOP/
    build/
        ...
    scenes/
        tt_train/
        tt_playground/
        ...
    experiments/
        2021-10-15_08:26:49_multi_scene/
        ...
```


### ADOP Viewer

The `adop_viewer` can now be run by passing a scene and the experiment directory. 
For example:
```shell
cd ADOP
./build/bin/adop_viewer scenes/tt_playground experiments/
```

 * The working dir of `adop_viewer` must be the ADOP root directory.
 * Pass the parent experiment directory and not a specific experiment. You can switch between experiments inside the viewer.
 * The most important keyboard shortcuts are:
    * F1: Switch to 3DView
    * F2: Switch to neural view
    * F3: Switch to split view (default)
    * WASD: Move camera
    * Center Mouse + Drag: Rotate around camera center
    * Left Mouse + Drag: Rotate around world center
    * Right click in 3DView: Select camera
    * Q: Move camera to selected camera

<img  width="400"  src="images/adop_viewer.png"> <img width="400"  src="images/adop_viewer_demo.gif">

## HDR Scenes

ADOP supports HDR scenes due to the physically-based tone mapper.
The input images can therefore have different exposure settings.
The dynamic range of a scene is the difference between the smallest and largest EV of all images.
For example, our boat scene (see below) has a dynamic range of ~10 stops.
If you want to fit ADOP to your own HDR scene consider the following:

 * For small dynamic ranges (<4) you can use the default pipeline.
 * For scenes with a large dynamic range, change to the log texture format and reduce the texture learning rate. Use the train config of our boat scene as reference.
 * Check if an initial EV guess is available. Many cameras store the exposure settings in the EXIF data.
 * Set the scene EV in the dataset.ini to the mean EV of all frames. This keeps the weights in a reasonable range.

When viewing HDR scenes in the `adop_viewer` you can press [F7] to open the tone mapper tab.
Here you can change the exposure value of the virtual camera.
In the render settings you find an option to use OpenGL based tone mapping instead of the learned on.



https://user-images.githubusercontent.com/16142878/138316754-ef8b2a8a-d421-4542-9b7b-4ee86bd15e97.mp4



### Scene Description
 * ADOP uses a simple, text-based scene description format.
 * To run ADOP on your scenes you have to convert them into this format.
 * After that you run adop_scene_preprocess to precompute various parameters.
 * If you have created your scene with COLMAP (like us) you can use the colmap2adop converter.
 * More infos on this topic can be found here: [scenes/README.md](scenes/README.md)

