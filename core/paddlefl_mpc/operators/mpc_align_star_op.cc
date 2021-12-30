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

#include "paddle/fluid/framework/op_registry.h"
#include "mpc_align_star_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcAlignStar : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("X"), true,
            platform::errors::NotFound("Input(X) of Mpc AlignStar should not be null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("Lod"), true,
            platform::errors::NotFound("The Lod of X of Mpc AlignStar should not be null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasOutput("Out"), true,
            platform::errors::NotFound("Output(Out) of MpcAlignStar should not be null."));

        auto x_dims = ctx->GetInputDim("X");
        auto lod_dims = ctx->GetInputDim("Lod");
        PADDLE_ENFORCE_EQ(
            lod_dims[0], 1,
            platform::errors::NotFound("The lod level of Input(X) of Mpc AlignStar should be 1."));
        
        VLOG(3) << "mpc pdistance operator x.shape=" << x_dims;
        
        //ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
        //ctx->ShareLoD("X", /*->*/ "Out");
    }
};

class MpcAlignStarMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("X", "(LodTensor), The first input tensor of mpc AlignStar op.");
        AddInput("Lod", "(LodTensor), The lod of first input tensor of mpc AlignStar op.");
        AddOutput("Out", "(Tensor), The output tensor of mpc AlignStar op.");
        
        AddComment(R"DOC(
MPC AlignStar Operator.
)DOC");
    }
};

class MpcAlignStarInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
protected:
    std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
            const override {
        static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
        return m;
    }
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mpc_align_star, ops::MpcAlignStar,
                 ops::MpcAlignStarMaker,
                 ops::MpcAlignStarInferVarType);

REGISTER_OP_CPU_KERNEL(
    mpc_align_star,
    ops::MpcAlignStarKernel<paddle::platform::CPUDeviceContext, int64_t>);