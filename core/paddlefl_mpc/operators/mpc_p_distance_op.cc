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
#include "mpc_p_distance_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcPDistance : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("X"), true,
            platform::errors::NotFound("Input(X) of Mpc PDistance should not be null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("Miss"), true,
            platform::errors::NotFound("Input(Miss) of MpcPDistance should not be null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasOutput("Out"), true,
            platform::errors::NotFound("Output(Out) of MpcPDistance should not be null."));

        auto x_dims = ctx->GetInputDim("X");
        auto miss_dims = ctx->GetInputDim("Miss");

        VLOG(3) << "mpc pdistance operator x.shape=" << x_dims ;
        /*
        PADDLE_ENFORCE_EQ(
            x_dims.size(), miss_dims.size(),
            platform::errors::InvalidArgument(
                "The input tensor X's dimensions of MpcPDistance "
                "should be equal to Y's dimensions. But received X's "
                "dimensions = %d, Y's dimensions = %d",
                x_dims.size(), miss_dims.size()));

        PADDLE_ENFORCE_EQ(x_dims[1], miss_dims[1],
                          platform::errors::InvalidArgument(
                        "The input tensor X's shape of MpcPDistance "
                        "should be equal to Y's shape. But received X's "
                        "shape = [%s], Y's shape = [%s]",
                        x_dims, miss_dims));
        */
        std::vector<int64_t> output_dims{2, x_dims[1], x_dims[1]};

        ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
        ctx->ShareLoD("X", /*->*/ "Out");
    }
};

class MpcPDistanceMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("X", "(Tensor), The first input tensor of mpc PDistance op.");
        AddInput("Miss", "(Tensor), The third input tensor of mpc PDistance op.");
        AddOutput("Out", "(Tensor), The output tensor of mpc PDistance op.");
        
        AddComment(R"DOC(
MPC PDistance Operator.
)DOC");
    }
};

class MpcPDistanceInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
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
REGISTER_OPERATOR(mpc_p_distance, ops::MpcPDistance,
                 ops::MpcPDistanceMaker,
                 ops::MpcPDistanceInferVarType);

REGISTER_OP_CPU_KERNEL(
    mpc_p_distance,
    ops::MpcPDistanceKernel<paddle::platform::CPUDeviceContext, int64_t>);