# ADOP: Approximate Differentiable One-Pixel Point Rendering

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

[![Watch the video](https://img.youtube.com/vi/WJRyu1JUtVw/hqdefault.jpg)](https://www.youtube.com/watch?v=WJRyu1JUtVw)
