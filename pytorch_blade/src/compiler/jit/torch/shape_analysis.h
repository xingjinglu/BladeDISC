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

#include <memory>

namespace torch {
namespace jit {
struct Graph;
struct Block;
struct Node;
struct Value;
} // namespace jit
} // namespace torch

namespace torch {
namespace blade {
using namespace torch::jit;
struct propagation_error : std::exception {};

class PropertyPropBase {
  // Used for both Shape Propagation and Dtype/Device Propagation
 public:
  explicit PropertyPropBase(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}
  virtual ~PropertyPropBase() = default;

  void propagateBlock(Block* block, bool insert_expands = true);
  // insert_expands is used for shape inference

  void processIf(Node* node);
  void processLoop(Node* node);

 protected:
  virtual void propagateNode(Node* node, bool insert_expands = true) = 0;
  void setUnshapedType(Value* o);
  void setUnshapedType(Node* node);
  std::shared_ptr<Graph> graph_;
};

void EraseShapeInformation(const std::shared_ptr<Graph>& graph);
void PropagateInputShapes(const std::shared_ptr<Graph>& graph);

bool mergeTypes(
    c10::ArrayRef<Value*> lhs,
    c10::ArrayRef<Value*> rhs,
    c10::ArrayRef<Value*> outputs);

} // namespace blade
} // namespace torch
