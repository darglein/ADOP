/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/kdtree.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/CoordinateSystems.h"
#include "saiga/core/math/random.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/vision/VisionTypes.h"

#include "build_config.h"

#ifdef _WIN32
#    ifdef BUILD_NEURAL_POINTS_DLL
#        define _NP_API __declspec(dllexport)
#        define _NP_INLINE_API inline
#    else
#        define _NP_API __declspec(dllimport)
#        define _NP_INLINE_API
#    endif

#else
// UNIX, ignore _API
#    define _NP_API
#    define _NP_INLINE_API inline
#endif


namespace Saiga
{
class Camera;
}
_NP_INLINE_API _NP_API Saiga::Camera* camera;

using Saiga::mat3;
using Saiga::mat4;
using Saiga::ucvec3;
using Saiga::vec3;
using Saiga::vec4;
