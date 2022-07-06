// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch {
namespace jit {
// Moved from shape_analysis.cpp

// Requirements:
//   dims           : preserved from the first argument
//   scalar type    : preserved from the first argument (doesn't have to
//                    match other arguments)
//   device         : always matching and preserved
//   tensor inputs  : *
//   tensor outputs : 1
// NB: those ops (with slight adjustments) are good candidates for restarts.
//     Knowing the type and device of weights or biases is usually enough to
//     infer the output type.
std::shared_ptr<OperatorSet> nn_ops_first_input_preserving();

// Requirements:
//   dims           : Changed from first argument
//   scalar type    : preserved from the first argument
//   device         : always matching and preserved
//   tensor inputs  : 1
//   tensor outputs : 1
std::shared_ptr<OperatorSet> ops_one_tensor_in_shape_transform();
} // namespace jit
} // namespace torch
