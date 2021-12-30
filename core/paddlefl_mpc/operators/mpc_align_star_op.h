/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "mpc_op.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MpcAlignStarKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *x = ctx.Input<Tensor>("X");
        auto *out = ctx.Output<Tensor>("Out");
        auto *lod = ctx.Input<Tensor>("Lod");

        size_t size = lod->numel();

        int64_t max_len = *(std::max_element(lod->data<int64_t>(), lod->data<int64_t>() + size));

        std::vector<int64_t> output_dims{2, size, max_len};

        out->mutable_data<T>(framework::make_ddim(output_dims), ctx.GetPlace());

        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->align_star(
            x, lod, out); 
    }
};

}  // namespace operators
}  // namespace paddle

