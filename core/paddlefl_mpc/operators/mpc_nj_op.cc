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
#include "mpc_nj_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcNJ : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("X"), true,
            platform::errors::NotFound("Input(X) of Mpc NJ should not be null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasOutput("Out"), true,
            platform::errors::NotFound("Output(Out) of Mpc NJ should not be null."));
        
        std::vector<std::string> ids = ctx->Attrs().Get<std::vector<std::string>>("ids");
        auto x_dims = ctx->GetInputDim("X");
        
        VLOG(3) << "mpc NJ operator x.shape=" << x_dims;
        
        ctx->ShareLoD("X", /*->*/ "Out");
    }
    framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext& ctx) const override {
            return framework::OpKernelType(
                OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
    }
};

class MpcNJMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("X", "(Tensor), The first input tensor of mpc NJ op.");
        AddOutput("Out", "(Tensor), The output tensor of mpc NJ op.");
        AddAttr<std::vector<std::string>>("ids",
            "(vector<string>) The ids of input tensor");
        AddComment(R"DOC(
MPC NJ Operator.
)DOC");
    }
};

class MpcNJInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
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
REGISTER_OPERATOR(mpc_nj, ops::MpcNJ,
                 ops::MpcNJMaker,
                 ops::MpcNJInferVarType);

REGISTER_OP_CPU_KERNEL(
    mpc_nj,
    ops::MpcNJKernel<paddle::platform::CPUDeviceContext, int64_t>);