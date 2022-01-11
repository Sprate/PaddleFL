// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <algorithm>

#include "paddle/fluid/platform/enforce.h"

namespace aby3 {
template<typename T, size_t N>
FixedPointTensor<T, N>::FixedPointTensor(TensorAdapter<T>* share_tensor[2]) {
    // TODO: check tensors' shapes
    _share[0] = share_tensor[0];
    _share[1] = share_tensor[1];
}

template<typename T, size_t N>
FixedPointTensor<T, N>::FixedPointTensor(TensorAdapter<T>* share_tensor_0,
                                         TensorAdapter<T>* share_tensor_1) {
    // TODO: check tensors' shapes
    _share[0] = share_tensor_0;
    _share[1] = share_tensor_1;
}

template<typename T, size_t N>
TensorAdapter<T>* FixedPointTensor<T, N>::mutable_share(size_t idx) {
    PADDLE_ENFORCE_LT(idx, 2, "Input should be less than 2.");
    return _share[idx];
}

template<typename T, size_t N>
const TensorAdapter<T>* FixedPointTensor<T, N>::share(size_t idx) const {
    PADDLE_ENFORCE_LT(idx, 2, "Input should be less than 2.");
    return _share[idx];
}

// reveal fixedpointtensor to one party
template<typename T, size_t N>
void FixedPointTensor<T, N>::reveal_to_one(size_t party,
                                           TensorAdapter<T>* ret) const {

    if (party == this->party()) {
        // TODO: check if tensor shape equal

        auto buffer = tensor_factory()->template create<T>(ret->shape());
        aby3_ctx()->network()->template recv(pre_party(), *buffer);

        share(0)->add(buffer.get(), ret);
        share(1)->add(ret, ret);
        ret->scaling_factor() = N;

    } else if (party == next_party()) {

        aby3_ctx()->network()->template send(party, *share(0));
    }
}

// reveal fixedpointtensor to all parties
template<typename T, size_t N>
void FixedPointTensor<T, N>::reveal(TensorAdapter<T>* ret) const {
    for (size_t i = 0; i < 3; ++i) {
        reveal_to_one(i, ret);
    }
}

template<typename T, size_t N>
const std::vector<size_t> FixedPointTensor<T, N>::shape() const {
    return _share[0]->shape();
}

//convert TensorAdapter to shares
template<typename T, size_t N>
void FixedPointTensor<T, N>::share(const TensorAdapter<T>* input,
                                    TensorAdapter<T>* output_shares[3],
                                    block seed) {

    if (equals(seed, g_zero_block)) {
        seed = block_from_dev_urandom();
    }
    //set seed of prng[2]
    aby3_ctx()->set_random_seed(seed, 2);

    aby3_ctx()->template gen_random_private(*output_shares[0]);
    aby3_ctx()->template gen_random_private(*output_shares[1]);

    auto temp = tensor_factory()->template create<T>(input->shape());
    output_shares[0]->add(output_shares[1], temp.get());
    input->sub(temp.get(), output_shares[2]);
    for (int i = 0; i < 3; ++i) {
        output_shares[i]->scaling_factor() = input->scaling_factor();
    }
}

//convert TensorAdapter to shares and distribute to all parties
template<typename T, size_t N>
void FixedPointTensor<T, N>::online_share(const size_t party,
                                          const TensorAdapter<T>* input,
                                          FixedPointTensor<T, N>* ret) {
    // create a tensor which contains two shares to send/recv
    auto shape = input->shape();
    std::vector<size_t> shape_ = shape;
    shape_.insert(shape_.begin(), 2);
    auto one_party_shares = tensor_factory()->template create<T>(shape_);

    if (party == FixedPointTensor::party()) {
        // this party has original data:
        // encrypt input into 3 shares
        auto temp = tensor_factory()->template malloc_tensor<T>(3, input->shape());
        TensorAdapter<T>* shares[3]{temp[0].get(), temp[1].get(), temp[2].get()};
        share(input, shares);

        // share 0&1
        shares[0]->copy(ret->_share[0]);
        shares[1]->copy(ret->_share[1]);

#ifdef __NVCC__
        // send share 1&2 to next_party
        auto one_party_shares_ = tensor_factory()->template create<T>(shape_);

        cudaMemcpy(one_party_shares.get()->data(), shares[1]->data(),
                   shares[1]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);
        cudaMemcpy(one_party_shares.get()->data() + shares[1]->numel(), shares[2]->data(),
                   shares[2]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);

        // send share 2&0 to pre_party
        cudaMemcpy(one_party_shares_.get()->data(), shares[2]->data(),
                   shares[2]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);
        cudaMemcpy(one_party_shares_.get()->data() + shares[2]->numel(), shares[0]->data(),
                   shares[0]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);

        NCCL_GROUP_START
        aby3_ctx()->network()->template send(next_party(), *one_party_shares);
        aby3_ctx()->network()->template send(pre_party(), *one_party_shares_);
        NCCL_GROUP_START
#else // __NVCC__
        // send share 1&2 to next_party
        std::copy(shares[1]->data(), shares[1]->data() + shares[1]->numel(),
                  one_party_shares.get()->data());
        std::copy(shares[2]->data(), shares[2]->data() + shares[2]->numel(),
                  one_party_shares.get()->data() + shares[1]->numel());
        aby3_ctx()->network()->template send(next_party(), *one_party_shares);
        // send share 2&0 to pre_party
        std::copy(shares[2]->data(), shares[2]->data() + shares[2]->numel(),
                  one_party_shares.get()->data());
        std::copy(shares[0]->data(), shares[0]->data() + shares[0]->numel(),
                  one_party_shares.get()->data() + shares[2]->numel());
        aby3_ctx()->network()->template send(pre_party(), *one_party_shares);
#endif // __NVCC__
    } else {
        // recv share from 'party' who has original data
        aby3_ctx()->network()->template recv(party, *(one_party_shares));
#ifdef __NVCC__
        cudaMemcpy(ret->_share[0]->data(), one_party_shares.get()->data(),
                   ret->_share[0]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);
        cudaMemcpy(ret->_share[1]->data(), one_party_shares.get()->data() + ret->_share[0]->numel(),
                   ret->_share[1]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);
#else // __NVCC__
        std::copy(one_party_shares->data(), one_party_shares->data() + one_party_shares->numel() / 2,
                  ret->_share[0]->data());
        std::copy(one_party_shares->data() + one_party_shares->numel() / 2,
                  one_party_shares->data() + one_party_shares->numel(),
                  ret->_share[1]->data());
#endif // __NVCC__
    }
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::add(const FixedPointTensor<T, N>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    _share[0]->add(rhs->_share[0], ret->_share[0]);
    _share[1]->add(rhs->_share[1], ret->_share[1]);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::add(const TensorAdapter<T>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
                        "no match scaling factor");
    if (party() == 0) {
        _share[0]->add(rhs, ret->_share[0]);
        _share[1]->copy(ret->_share[1]);
    } else if (party() == 1) {
        _share[0]->copy(ret->_share[0]);
        _share[1]->copy(ret->_share[1]);
    } else {
        _share[0]->copy(ret->_share[0]);
        _share[1]->add(rhs, ret->_share[1]);
    }
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::sub(const FixedPointTensor<T, N>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    _share[0]->sub(rhs->_share[0], ret->_share[0]);
    _share[1]->sub(rhs->_share[1], ret->_share[1]);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::sub(const TensorAdapter<T>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
                        "no match scaling factor");
    if (party() == 0) {
        _share[0]->sub(rhs, ret->_share[0]);
        _share[1]->copy(ret->_share[1]);
    } else if (party() == 1) {
        _share[0]->copy(ret->_share[0]);
        _share[1]->copy(ret->_share[1]);
    } else {
        _share[0]->copy(ret->_share[0]);
        _share[1]->sub(rhs, ret->_share[1]);
    }
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::negative(FixedPointTensor<T, N>* ret) const {
    _share[0]->negative(ret->_share[0]);
    _share[1]->negative(ret->_share[1]);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::mul(const FixedPointTensor<T, N>* rhs,
                                 FixedPointTensor<T, N>* ret) const {
    mul_trunc(this, rhs, ret, [](
            const TensorAdapter<T>* lhs,
            const TensorAdapter<T>* rhs,
            TensorAdapter<T>* ret) {
        lhs->mul(rhs, ret);
        });
}

#ifdef USE_ABY3_TRUNC1 //use aby3 trunc1
template<typename T, size_t N>
void FixedPointTensor<T, N>::truncate(const FixedPointTensor<T, N>* op,
                                       FixedPointTensor<T, N>* ret,
                                       size_t scaling_factor) {
    if (scaling_factor == 0) {
        op->share(0)->copy(ret->mutable_share(0));
        op->share(1)->copy(ret->mutable_share(1));
    }
    // implement ABY3's truncate1 algorithm
    if (party() == 0) {
        // party0
        op->_share[0]->rshift(scaling_factor, ret->_share[0]);
        aby3_ctx()->network()->template recv(1, *(ret->_share[1]));

    } else if (party() == 1) {
        // party1
        auto r_12 = tensor_factory()->template create<T>(op->shape());
        aby3_ctx()->template gen_random(*r_12.get(), true);

        op->_share[0]->add(op->_share[1], ret->_share[0]);
        // trunc from [SecureML, Thm.1]
        ret->_share[0]->negative(ret->_share[0]);
        ret->_share[0]->rshift(scaling_factor, ret->_share[0]);
        ret->_share[0]->negative(ret->_share[0]);
        ret->_share[0]->sub(r_12.get(), ret->_share[0]);

        aby3_ctx()->network()->template send(0, *(ret->_share[0]));
        r_12->copy(ret->_share[1]);

    } else {
        // party2
        op->_share[1]->rshift(scaling_factor, ret->_share[1]);

        auto r_21 = tensor_factory()->template create<T>(op->shape());
        aby3_ctx()->template gen_random(*r_21.get(), false);

        r_21->copy(ret->_share[0]);
    }

    return;
}

#else // use truncate3

// Protocol. `truncate3` (illustrated for data type T = int64_t)
// motivation:
// truncates in aby3 may cause msb error with small probability
// the reason is that before rishft op, its masked value e.g., x' - r' may overflow in int64_t
// so that, in `truncate3`, we limit r' in (-2^62, 2^62) to avoid the problem.

// notice:
// when r' is contrainted in (-2^62, 2^62),
// the SD (statistical distance) of x' - r' between this
// and r' in Z_{2^64} is equal to |X| / (2^63 + |X|)

// detail protocol:
// Input: P0 (x0', x1'), P1 (x1', x2'), P2 (x2', x0')
// P2: 1. gen r' randomly from [-2^(l-2), 2^(l-2)]
//     2. gen r0 using preshared seed with P0
//     3. gen r1 randomly
//     4. compute r2=r'/2^N - r0 - r1
//     5. x2 := r1 + r2, x0 := r0
// P2->>P0: x2' - r'
// P2->>P1: x0' - r', x2
// P0: 1. x0 := r0
//     2. x1 := (x2' - r' + x0' + x1')/2^N
// P1: 1. x1 := (x0' - r' + x1' + x2')/2^N
//     2. x2:= x2
template<typename T, size_t N>
void FixedPointTensor<T, N>::truncate(const FixedPointTensor<T, N>* op,
                                       FixedPointTensor<T, N>* ret,
                                       size_t scaling_factor) {
    if (scaling_factor == 0) {
        op->share(0)->copy(ret->mutable_share(0));
        op->share(1)->copy(ret->mutable_share(1));
        return;
    }
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    if (party() == 2) {
        for (int i = 0; i < 6; ++i) {
            temp.emplace_back(
                tensor_factory()->template create<T>(op->shape()));
        }

        // r'
        aby3_ctx()->template gen_random_private(*temp[0]);
        temp[0]->rshift(1, temp[0].get());

        // r
        temp[0]->rshift(scaling_factor, temp[1].get());

        // r_0
        aby3_ctx()->template gen_random(*temp[2], true);

        // r_1
        aby3_ctx()->template gen_random_private(*temp[3]);

        // r_2
        temp[1]->sub(temp[2].get(), temp[1].get());
        temp[1]->sub(temp[3].get(), temp[1].get());

        // x0' - r'
        op->share(1)->sub(temp[0].get(), temp[4].get());

        // x2' - r'
        op->share(0)->sub(temp[0].get(), temp[0].get());

        // x_2 = r_2 + r_1
        temp[1]->add(temp[3].get(), temp[1].get());

        auto shape_ = op->shape();
        shape_.insert(shape_.begin(), 2);
        temp[5]->reshape(shape_);
        // merge msg to save send
#ifdef __NVCC__
        cudaMemcpy(temp[5]->data(), temp[1]->data(),
                   temp[1]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);
        cudaMemcpy(temp[5]->data() + temp[1]->numel(), temp[4]->data(),
                   temp[4]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);
#else // __NVCC__
        std::copy(temp[1]->data(), temp[1]->data() + temp[1]->numel(),
                  temp[5]->data());
        std::copy(temp[4]->data(), temp[4]->data() + temp[4]->numel(),
                  temp[5]->data() + temp[1]->numel());
#endif // __NVCC__
        NCCL_GROUP_START
        // send x_2, x0' - r' to P1
        aby3_ctx()->network()->template send(1, *temp[5]);
        // send x2' - r' to P0
        aby3_ctx()->network()->template send(0, *temp[0]);
        NCCL_GROUP_END

        temp[1]->copy(ret->mutable_share(0));
        temp[2]->copy(ret->mutable_share(1));

    } else if (party() == 1) {
        for (int i = 0; i < 4; ++i) {
            temp.emplace_back(
                tensor_factory()->template create<T>(op->shape()));
        }
        auto shape_ = op->shape();
        shape_.insert(shape_.begin(), 2);
        temp[3]->reshape(shape_);
        // recv x_2, x'_0 - r'
        NCCL_GROUP_START
        aby3_ctx()->network()->template recv(2, *temp[3]);
        NCCL_GROUP_END
#ifdef __NVCC__
        cudaMemcpy(temp[0]->data(), temp[3]->data(),
                   temp[0]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);
        cudaMemcpy(temp[1]->data(), temp[3]->data() + temp[0]->numel(),
                   temp[1]->numel() * sizeof(T),cudaMemcpyDeviceToDevice);
#else // __NVCC__
        std::copy(temp[3]->data(), temp[3]->data() + temp[0]->numel(),
                  temp[0]->data());
        std::copy(temp[3]->data() + temp[0]->numel(),
                  temp[3]->data() + temp[0]->numel() + temp[1]->numel(),
                  temp[1]->data());
#endif // __NVCC__

        // P1 reveals x' - r'
        op->share(0)->add(op->share(1), temp[2].get());
        temp[2]->add(temp[1].get(), temp[2].get());
        // truncate x'-r'
        temp[2]->rshift(scaling_factor, temp[2].get());

        temp[2]->copy(ret->mutable_share(0));
        temp[0]->copy(ret->mutable_share(1));
    } else { // party == 0
        for (int i = 0; i < 2; ++i) {
            temp.emplace_back(
                tensor_factory()->template create<T>(op->shape()));
        }
        // recv x'_2 - r'
        NCCL_GROUP_START
        aby3_ctx()->network()->template recv(2, *temp[0]);
        NCCL_GROUP_END

        // P0 reveals x' - r'
        op->share(0)->add(op->share(1), temp[1].get());
        temp[1]->add(temp[0].get(), temp[0].get());
        // truncate x'-r'
        temp[0]->rshift(scaling_factor, temp[0].get());

        // x_1
        temp[0]->copy(ret->mutable_share(1));

        // x_0 = r_0
        aby3_ctx()->template gen_random(*ret->mutable_share(0), false);

    }

    // compensation for carry in
    auto tensor_carry_in = tensor_factory()->template create<T>(ret->shape());
    assign_to_tensor(tensor_carry_in.get(), (T)1);
    tensor_carry_in->scaling_factor() = N;
    ret->add(tensor_carry_in.get(), ret);
}
#endif //USE_ABY3_TRUNC1

template<typename T, size_t N>
template<typename MulFunc>
void FixedPointTensor<T, N>::mul_trunc(const FixedPointTensor<T, N>* lhs,
                                        const FixedPointTensor<T, N>* rhs,
                                        FixedPointTensor<T, N>* ret,
                                        MulFunc mul_func) {

    auto r_zero = tensor_factory()->template create<T>(ret->shape());
    aby3_ctx()->gen_zero_sharing_arithmetic(*r_zero.get());

    // temp = _share[0]->mul(rhs->_share[0]) +
    //        _share[0]->mul(rhs->_share[1]) +
    //        _share[1]->mul(rhs->_share[0]) +
    //        r_zero
    auto temp = tensor_factory()->template create<T>(ret->shape());
    auto temp1 = tensor_factory()->template create<T>(ret->shape());

    // use mul_func to fit both element_wise mul and mat mul
    mul_func(lhs->share(0), rhs->share(0), temp.get());
    mul_func(lhs->share(0), rhs->share(1), temp1.get());
    temp1->add(temp.get(), temp1.get());

    mul_func(lhs->share(1), rhs->share(0), temp.get());
    temp1->add(r_zero.get(), temp1.get());
    temp->add(temp1.get(), temp.get());

    auto temp2 = tensor_factory()->template create<T>(ret->shape());
    auto temp3 = tensor_factory()->template create<T>(ret->shape());

    TensorAdapter<int64_t>* temp_array[2] = {temp2.get(), temp3.get()};

    std::shared_ptr<FixedPointTensor<T, N>> ret_no_trunc =
            std::make_shared<FixedPointTensor<T, N>>(temp_array);

    temp->copy(ret_no_trunc->_share[0]);
    reshare(temp.get(), ret_no_trunc->_share[1]);

    truncate(ret_no_trunc.get(), ret, N);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::mul(const TensorAdapter<T>* rhs,
                                 FixedPointTensor<T, N>* ret) const {
    // PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
    //                   "no match scaling factor");
    auto temp0 = tensor_factory()->template create<T>(this->shape());
    auto temp1 = tensor_factory()->template create<T>(this->shape());
    std::shared_ptr<FixedPointTensor<T, N>> temp =
        std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());

    _share[0]->mul(rhs, temp->_share[0]);
    _share[1]->mul(rhs, temp->_share[1]);
    truncate(temp.get(), ret, rhs->scaling_factor());
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::sum(FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(ret->numel(), 1, "output size should be 1.");
    _share[0]->sum(ret->mutable_share(0));
    _share[1]->sum(ret->mutable_share(1));
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::dot_mul(const CTensor<T, N1...>* rhs,
                                     FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(ret->numel(), 1, "output size should be 1.");

    auto temp0 = tensor_factory()->template create<T>(this->shape());
    auto temp1 = tensor_factory()->template create<T>(this->shape());
    std::shared_ptr<FixedPointTensor<T, N>> temp =
            std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
    this->mul(rhs, temp.get());
    temp->sum(ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::dot_mul(const TensorAdapter<T>* rhs, FixedPointTensor* ret) const {
    PADDLE_ENFORCE_EQ(ret->numel(), 1, "output size should be 1.");

    auto temp0 = tensor_factory()->template create<T>(this->shape());
    auto temp1 = tensor_factory()->template create<T>(this->shape());
    std::shared_ptr<FixedPointTensor<T, N>> temp =
            std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
    this->mul(rhs, temp.get());
    temp->sum(ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::mat_mul(const FixedPointTensor<T, N>* rhs,
                                     FixedPointTensor<T, N>* ret,
                                     bool trans_lhs,
                                     bool trans_rhs,
                                     bool sum_reduce_batch) const {
    mul_trunc(this, rhs, ret, [trans_lhs, trans_rhs, sum_reduce_batch](
            const TensorAdapter<T>* lhs,
            const TensorAdapter<T>* rhs,
            TensorAdapter<T>* ret) {
        lhs->mat_mul(rhs, ret, trans_lhs, trans_rhs, sum_reduce_batch);
        });
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::mat_mul(const TensorAdapter<T>* rhs,
                                     FixedPointTensor<T, N>* ret,
                                     bool trans_lhs,
                                     bool trans_rhs,
                                     bool sum_reduce_batch) const {
    _share[0]->mat_mul(rhs, ret->_share[0], trans_lhs, trans_rhs, sum_reduce_batch);
    _share[1]->mat_mul(rhs, ret->_share[1], trans_lhs, trans_rhs, sum_reduce_batch);
    truncate(ret, ret, rhs->scaling_factor());
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::div(const TensorAdapter<T>* rhs,
                                 FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
                        "no match scaling factor");

    double scale = std::pow(2, rhs->scaling_factor());
    auto temp = tensor_factory()->template create<T>(this->shape());
    auto temp2 = tensor_factory()->template create<T>(this->shape());
    assign_to_tensor(temp.get(), (T)(scale * scale));

    temp->scaling_factor() = rhs->scaling_factor();
    temp2->scaling_factor() = rhs->scaling_factor();

    temp->div(rhs, temp2.get());

    this->mul(temp2.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::div(const FixedPointTensor<T, N>* rhs,
                                 FixedPointTensor<T, N>* ret,
                                 size_t iter, double x0) const {
    auto temp0 = tensor_factory()->template create<T>(ret->shape());
    auto temp1 = tensor_factory()->template create<T>(ret->shape());
    std::shared_ptr<FixedPointTensor<T, N>> temp =
        std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
    reciprocal(rhs, temp.get(), iter, x0);
    this->mul(temp.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::exp(FixedPointTensor<T, N>* ret,
                                 size_t iter) const {
    // exp approximate: exp(x) = \lim_{n->inf} (1+x/n)^n
    // where n = 2^ite
    auto pow_iter = tensor_factory()->template create<T>(this->shape());
    assign_to_tensor(pow_iter.get(), (T) (pow(2, N -iter)));
    pow_iter->scaling_factor() = N;

    auto tensor_one = tensor_factory()->template create<T>(this->shape());
    assign_to_tensor(tensor_one.get(), (T) 1 << N);
    tensor_one->scaling_factor() = N;

    this->mul(pow_iter.get(), ret);

    ret->add(tensor_one.get(), ret);

    for (int i = 0; i < iter; ++i) {
        ret->mul(ret, ret);
    }
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::relu(FixedPointTensor<T, N>* ret) const {
    //utilize polynomial_piecewise
    // break_point = {0}, coeff[0] = {0, 0}, coeff[1] = {0, 1}
    // break_point.shape = {1, this->shape}, coeff.shape = {2, 2, this->shape}

    auto shape_ = shape();
    //construct break_point
    auto b_shape = shape_;
    b_shape.insert(b_shape.begin(), 1);

    auto break_point = tensor_factory()->template create<T>(b_shape);

    assign_to_tensor(break_point.get(), (T)0);
    break_point->scaling_factor() = N;

    //contruct coeff
    std::vector<size_t> c_shape = {4, 1};
    c_shape.insert(c_shape.end(), shape_.begin(), shape_.end());

    auto coeff = tensor_factory()->template create<T>(c_shape);

    auto slice = tensor_factory()->template create<T>();
    coeff->slice(0, 3, slice.get());

    assign_to_tensor(slice.get(), (T)0);

    coeff->slice(3, 4, slice.get());

    assign_to_tensor(slice.get(), (T) 1 << N);

    c_shape[0] = 2;
    c_shape[1] = 2;
    coeff->reshape(c_shape);
    coeff->scaling_factor() = N;

    this->polynomial_piecewise(coeff.get(), break_point.get(), ret);
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::relu_with_derivative(
    FixedPointTensor<T, N>* ret, BooleanTensor<T>* derivative) const {

    auto shape_ = shape();
    auto zero = tensor_factory()->template create<T>(shape_);

    assign_to_tensor(zero.get(), (T)0);
    zero->scaling_factor() = N;

    auto tmp0 = tensor_factory()->template create<T>(shape_);
    auto tmp1 = tensor_factory()->template create<T>(shape_);

    BooleanTensor<T> der(tmp0.get(), tmp1.get());

    gt(zero.get(), &der);

    der.mul(this, ret);

    if (derivative) {
        der.share(0)->copy(derivative->share(0));
        der.share(1)->copy(derivative->share(1));
    }
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::sigmoid_chebyshev(FixedPointTensor<T, N>* ret) const {
    //utilize Chebyshev polynomial approximation
    // more accurate in small range, such as [-4, 4]
    auto shape = ret->shape();
    std::vector<size_t> shape_ = shape;
    shape_.insert(shape_.begin(), 10);
    auto numel = ret->numel();
    auto coeff = tensor_factory()->template create<T>(shape_);
    std::vector<double> w;
    w.resize(10, 0.0f);
    w[0] = 0.5;
    w[1] = 0.2159198015;
    w[3] = -0.0082176259;
    w[5] = 0.0001825597;
    w[7] = -0.0000018848;
    w[9] = 0.0000000072;
    double scale = pow(2, N);
    auto slice = tensor_factory()->template create<T>();
    for (int i = 0; i < 10; ++i) {
        coeff->slice(i, i + 1, slice.get());
        assign_to_tensor(slice.get(), (T) (w[i] * scale));
    }
    coeff->scaling_factor() = N;
    polynomial(coeff.get(), ret);
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::sigmoid(FixedPointTensor<T, N>* ret) const {
    //utilize polynomial_piecewise
    // break_point = {-2.5, 2.5}
    // coeff[0] = {10^-4, 0}, coeff[1] = {0.5, 0.17}
    // coeff[2] = {1 - 10^-4, 0}
    // break_point.shape = {2, this->shape}, coeff.shape = {3, 2, this->shape}

    //construct break_point
    auto shape_ = shape();
    //construct break_point
    auto b_shape = shape_;
    b_shape.insert(b_shape.begin(), 2);

    auto break_point = tensor_factory()->template create<T>(b_shape);

    auto slice = tensor_factory()->template create<T>();
    break_point->slice(0, 1, slice.get());
    assign_to_tensor(slice.get(), (T) (-2.5 * pow(2, N)));

    break_point->slice(1, 2, slice.get());
    assign_to_tensor(slice.get(), (T) (2.5 * pow(2, N)));

    break_point->scaling_factor() = N;

    //contruct coeff
    std::vector<size_t> c_shape = {6, 1};
    c_shape.insert(c_shape.end(), shape_.begin(), shape_.end());

    auto coeff = tensor_factory()->template create<T>(c_shape);

    double scale = std::pow(2, N);

    coeff->slice(0, 1, slice.get());
    assign_to_tensor(slice.get(), (T) (0.0001 * scale));
    coeff->slice(1, 2, slice.get());
    assign_to_tensor(slice.get(), (T) (0));
    coeff->slice(2, 3, slice.get());
    assign_to_tensor(slice.get(), (T) (0.5 * scale));
    coeff->slice(3, 4, slice.get());
    assign_to_tensor(slice.get(), (T) (0.17 * scale));
    coeff->slice(4, 5, slice.get());
    assign_to_tensor(slice.get(), (T) ((1 - 0.0001) * scale));
    coeff->slice(5, 6, slice.get());
    assign_to_tensor(slice.get(), (T) (0));

    c_shape[0] = 3;
    c_shape[1] = 2;

    coeff->reshape(c_shape);

    coeff->scaling_factor() = N;

    this->polynomial_piecewise(coeff.get(), break_point.get(), ret);
}

// sigmoid(x) = 1 / (1 + exp(-x))
template< typename T, size_t N>
void FixedPointTensor<T, N>::sigmoid_high_precision(FixedPointTensor<T, N>* ret) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 2; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(ret->shape()));
    }
    auto tensor_one_share0 = tensor_factory()->template create<T>(shape());
    auto tensor_one_share1 = tensor_factory()->template create<T>(shape());
    auto tensor_one = tensor_factory()->template create<T>(shape());
    assign_to_tensor(tensor_one.get(), (T) (1.0 * pow(2, N)));
    tensor_one->scaling_factor() = N;
    assign_to_tensor(tensor_one_share0.get(), (T) (1.0 * pow(2, N) / 3.0));
    assign_to_tensor(tensor_one_share1.get(), (T) (1.0 * pow(2, N) / 3.0));

    FixedPointTensor tensor_one_ft(tensor_one_share0.get(), tensor_one_share1.get());
    FixedPointTensor out(temp[0].get(), temp[1].get());
    this->negative(&out);
    out.exp(&out);
    out.add(tensor_one.get(), &out);
    tensor_one_ft.long_div(&out, ret);
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::sigmoid_enhanced(FixedPointTensor<T, N>* ret) const {
    //utilize polynomial_piecewise
    // break_point = {-5, -2.5, 2.5, 5}
    // coeff[0] = {10^-4, 0}, coeff[1] = {0.145, 0.02776}
    // coeff[2] = {0.5, 0.17}, coeff[3] = {0.85498, 0.02776}, coeff[4] = {0.9999, 0}
    // break_point.shape = {4, this->shape}, coeff.shape = {5, 2, this->shape}

    //construct break_point
    auto shape_ = shape();
    //construct break_point
    auto b_shape = shape_;
    b_shape.insert(b_shape.begin(), 4);

    auto break_point = tensor_factory()->template create<T>(b_shape);

    double scale = pow(2, N);
    double bp_[4] = {-5, -2.5, 2.5, 5};
    auto slice = tensor_factory()->template create<T>();
    for (int i = 0; i < 4; ++i) {
        break_point->slice(i, i + 1, slice.get());
        assign_to_tensor(slice.get(), (T) (bp_[i] * scale));
    }

    break_point->scaling_factor() = N;

    double w_[10] = {0.0001, 0, 0.145, 0.02776, 0.5, 0.17, 0.85498, 0.02776 ,0.9999, 0};
    //contruct coeff
    std::vector<size_t> c_shape = {10, 1};
    c_shape.insert(c_shape.end(), shape_.begin(), shape_.end());
    auto coeff = tensor_factory()->template create<T>(c_shape);
    for (int i = 0; i < 10; ++i) {
        coeff->slice(i, i + 1, slice.get());
        assign_to_tensor(slice.get(), (T) (w_[i] * scale));
    }
    c_shape[0] = 5;
    c_shape[1] = 2;
    coeff->reshape(c_shape);
    coeff->scaling_factor() = N;

    this->polynomial_piecewise(coeff.get(), break_point.get(), ret);
}

#ifndef USE_CUDA
template< typename T, size_t N>
void FixedPointTensor<T, N>::softmax(FixedPointTensor<T, N>* ret,
                                     bool use_relu, bool use_long_div) const {
    // softmax axis = -1
    const size_t col = *(shape().end() - 1);
    const size_t row = numel() / col;

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    // 11 for allocating temp tensor
    for (size_t i = 0; i < 11; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }

    temp[0]->reshape({row, col});
    temp[1]->reshape({row, col});
    FixedPointTensor<T, N> x(temp[0].get(), temp[1].get());

    if (!use_relu) {
        temp[2]->reshape({col, row});
        temp[3]->reshape({col, row});

        temp[4]->reshape({1, row});
        temp[5]->reshape({1, row});
    }
    FixedPointTensor<T, N> x_t(temp[2].get(), temp[3].get());
    FixedPointTensor<T, N> max_x_t(temp[4].get(), temp[5].get());

    temp[6]->reshape({row, 1});
    temp[7]->reshape({row, 1});
    FixedPointTensor<T, N> max_x(temp[6].get(), temp[7].get());

    temp[8]->reshape({row, col});
    temp[9]->reshape({row, col});
    FixedPointTensor<T, N> max_x_broadcast(temp[8].get(), temp[9].get());

    temp[10]->reshape({row, col});
    auto exp_lower_bound = temp[10].get();

    auto transpose = [](const TensorAdapter<T>* in, TensorAdapter<T>* out) {
        std::vector<int> axis{ 1, 0 };
        // suppose input dims = 2
        dynamic_cast<const common::PaddleTensor<T>*>(in)->template Transpose<2>(axis, out);
    };

    auto broadcast = [](const TensorAdapter<T>* in, TensorAdapter<T>* out) {
        // suppose input dims = 2
        const size_t col = out->shape()[1];
        std::vector<int> axis{ 1, col };
        dynamic_cast<const common::PaddleTensor<T>*>(in)->template Broadcast<2>(axis, out);
    };

    share(0)->copy(x.mutable_share(0));
    share(1)->copy(x.mutable_share(1));

    if (use_relu) {

        x.relu(&x);

    } else { // use exp
        transpose(x.share(0), x_t.mutable_share(0));
        transpose(x.share(1), x_t.mutable_share(1));

        // x = max(input - max(input), exp_lower_bound)
        x_t.max_pooling(&max_x_t);

        transpose(max_x_t.share(0), max_x.mutable_share(0));
        transpose(max_x_t.share(1), max_x.mutable_share(1));

        broadcast(max_x.share(0), max_x_broadcast.mutable_share(0));
        broadcast(max_x.share(1), max_x_broadcast.mutable_share(1));

        x.sub(&max_x_broadcast, &x);

        // n = 64, see exp
        assign_to_tensor(exp_lower_bound, (T)(-64 * (1 << N)));
        exp_lower_bound->scaling_factor() = N;

        x.sub(exp_lower_bound, &x);
        x.relu(&x);
        x.add(exp_lower_bound, &x);

        x.exp(&x);
    }

    // reuse max_x as sum
    reduce(&x, &max_x);

    if (!use_long_div) { // invert sum by Newton's method
    // divisor range = [1/col, 1.0]
    // TODO: find better iter num & init val
        reciprocal(&max_x, &max_x, 16, 0.5 / col);
    }

    broadcast(max_x.share(0), max_x_broadcast.mutable_share(0));
    broadcast(max_x.share(1), max_x_broadcast.mutable_share(1));

    if (use_long_div) {
        x.long_div(&max_x_broadcast, &x, 1);
    } else {
        x.mul(&max_x_broadcast, &x);
    }

    x.share(0)->copy(ret->mutable_share(0));
    x.share(1)->copy(ret->mutable_share(1));
}
#endif // USE_CUDA

template<typename T, size_t N>
void FixedPointTensor<T, N>::long_div(const FixedPointTensor<T, N>* rhs,
                                 FixedPointTensor<T, N>* ret,
                                 size_t int_len) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 16; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(ret->shape()));
    }

    BooleanTensor<T> sign_lhs(temp[0].get(), temp[1].get());
    BooleanTensor<T> sign_rhs(temp[2].get(), temp[3].get());
    BooleanTensor<T> sign_ret(temp[4].get(), temp[5].get());
    FixedPointTensor<T, N> abs_lhs(temp[6].get(), temp[7].get());
    FixedPointTensor<T, N> abs_rhs(temp[8].get(), temp[9].get());
    FixedPointTensor<T, N> sub_rhs(temp[10].get(), temp[11].get());
    BooleanTensor<T> cmp_res(temp[12].get(), temp[13].get());
    BooleanTensor<T> cmp_res_all(temp[14].get(), temp[15].get());

    assign_to_tensor(cmp_res_all.share(0), (T)0);
    assign_to_tensor(cmp_res_all.share(1), (T)0);

    const size_t msb = sizeof(T) * 8 - 1;
    sign_lhs.bit_extract(msb, this);
    sign_rhs.bit_extract(msb, rhs);
    sign_lhs.bitwise_xor(&sign_rhs, &sign_ret);

    auto lshift = []  (const FixedPointTensor<T, N>* in,
                       size_t rhs,
                       FixedPointTensor<T, N>* out) {
        in->share(0)->lshift(rhs, out->mutable_share(0));
        in->share(1)->lshift(rhs, out->mutable_share(1));
    };

    // abs = val - 2 * sign * val
    auto abs = [lshift] (const FixedPointTensor<T, N>* in,
                   const BooleanTensor<T>* sign,
                   FixedPointTensor<T, N>* out) {
        lshift(in, 1, out);
        sign->mul(out, out);
        in->sub(out, out);
    };

    auto out0 = tensor_factory()->template create<T>(ret->shape());

    abs(this, &sign_lhs, &abs_lhs);

    abs(rhs, &sign_rhs, &abs_rhs);


    for (ssize_t i = int_len - 1; i >= 0; --i) {
        lshift(&abs_rhs, i, &sub_rhs);


        abs_lhs.gt(&sub_rhs, &cmp_res);


        cmp_res.mul(&sub_rhs, &sub_rhs);
        cmp_res.lshift(N + i, &cmp_res);
        abs_lhs.sub(&sub_rhs, &abs_lhs);
        cmp_res.bitwise_xor(&cmp_res_all, &cmp_res_all);

    }

    for (size_t i = 1; i <= N; ++i) {
        truncate(&abs_rhs, &sub_rhs, i);
        abs_lhs.gt(&sub_rhs, &cmp_res);
        cmp_res.mul(&sub_rhs, &sub_rhs);
        cmp_res.lshift(N - i, &cmp_res);
        abs_lhs.sub(&sub_rhs, &abs_lhs);
        cmp_res.bitwise_xor(&cmp_res_all, &cmp_res_all);
    }

    // use abs_lhs as buffer
    cmp_res_all.b2a(&abs_lhs);

    abs(&abs_lhs, &sign_ret, ret);
}

#ifndef USE_CUDA
// reduce last dim
template <typename T, size_t N>
void FixedPointTensor<T, N>::reduce(const FixedPointTensor<T, N>* input,
                                    FixedPointTensor<T, N>* ret) {
    //enfoce shape: input->shape[0 ... (n-2)] == ret shape
    dynamic_cast<const common::PaddleTensor<T>*>(input->_share[0])->sum_reduce_last_dim(ret->_share[0]);
    dynamic_cast<const common::PaddleTensor<T>*>(input->_share[1])->sum_reduce_last_dim(ret->_share[1]);
}
#endif // USE_CUDA

template< typename T, size_t N>
void FixedPointTensor<T, N>::polynomial(const TensorAdapter<T>* coeff,
                                        FixedPointTensor<T, N>* ret) const {

    // e.g., x.shape = {2, 3}, coeff.shape = {n, 2, 3} (n: polynomial power)

    //TODO: improve performance: [ABY3]
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 7; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> x_pow_i =
            std::make_shared<FixedPointTensor<T, N>>(
                                temp[0].get(), temp[1].get());
    std::shared_ptr<FixedPointTensor<T, N>> temp_fixed =
            std::make_shared<FixedPointTensor<T, N>>(
                                temp[2].get(), temp[3].get());
    std::shared_ptr<FixedPointTensor<T, N>> result =
            std::make_shared<FixedPointTensor<T, N>>(
                                temp[5].get(), temp[6].get());
    assign_to_tensor(result->_share[0], (T) 0);
    assign_to_tensor(result->_share[1], (T) 0);

    //x_pow_i.get() = 1;
    assign_to_tensor(x_pow_i.get()->_share[0], (T) 0);
    assign_to_tensor(x_pow_i.get()->_share[1], (T) 0);
    assign_to_tensor(temp[4].get(), (T) 1 << N);
    temp[4]->scaling_factor() = N;
    x_pow_i->add(temp[4].get(), x_pow_i.get());

    for (int i = 0; i < coeff->shape()[0]; ++i) {
        auto t = tensor_factory()->template create<T>();
        coeff->slice(i, i + 1, t.get());
        auto t_shape = t->shape();
        // remove leading 1
        t_shape.erase(t_shape.begin());
        t->reshape(t_shape);
        x_pow_i->mul(t.get(), temp_fixed.get());
        result->add(temp_fixed.get(), result.get());
        x_pow_i->mul(this, x_pow_i.get());
    }
    result->share(0)->copy(ret->mutable_share(0));
    result->share(1)->copy(ret->mutable_share(1));
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::polynomial_piecewise(
                    const TensorAdapter<T>* coeff,
                    const TensorAdapter<T>* break_point,
                    FixedPointTensor<T, N>* ret) const {

    // e.g., x.shape = {2, 3},
    // break_point.shape = {k, 2, 3} (k: num of break point)
    //       coeff.shape = {k + 1, n, 2, 3} (n: poly power)

    // copy ret
    auto ret_cpy_s0 = tensor_factory()->create_int64_t(ret->share(0)->shape());
    ret->share(0)->copy(ret_cpy_s0.get());
    auto ret_cpy_s1 = tensor_factory()->create_int64_t(ret->share(1)->shape());
    ret->share(1)->copy(ret_cpy_s1.get());
    std::shared_ptr<FixedPointTensor<T, N>> ret_cpy{new FixedPointTensor<T, N>(ret_cpy_s0.get(), ret_cpy_s1.get())};

    std::vector<std::shared_ptr<BooleanTensor<T>>> msb;

    int len_break_point = break_point->shape()[0];
    int len_coeff = coeff->shape()[0];

    //number of temp tensor used
    int temp_total = 4 * len_break_point + 2 +
                     2 * (len_break_point - 1) + 2 + 4 * len_coeff;
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < temp_total; ++i) {
        temp.emplace_back(tensor_factory()->
                          template create<T>(this->shape()));
    }
    int temp_index = 0;

    // std::vector<std::shared_ptr<TensorAdapter<T>>> paddle_t_break;
    std::vector<std::shared_ptr<FixedPointTensor<T, N>>> temp1;

    for (int i = 0; i < break_point->shape()[0]; ++i) {
        // msb[i] = msb(x - break_point[i])
        auto t_break = tensor_factory()->template create<T>();
        break_point->slice(i, i + 1, t_break.get());

        auto t_shape = t_break->shape();
        // remove leading 1
        t_shape.erase(t_shape.begin());
        t_break->reshape(t_shape);

        temp1.emplace_back(
                    std::make_shared<FixedPointTensor<T, N>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));
        this->sub(t_break.get(), temp1[i].get());
        msb.emplace_back(std::make_shared<BooleanTensor<T>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));
        msb[i]->bit_extract(sizeof(T) * 8 - 1, temp1[i].get());
    }

    // b[0] = msb[0], b[i + 1] = ~ msb[i] & msb[i + 1]
    std::vector<std::shared_ptr<BooleanTensor<T>>> b;
    b.emplace_back(std::make_shared<BooleanTensor<T>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));
    b[0] = msb[0];

    for (int i = 0; i < len_break_point - 1; ++i) {
        b.emplace_back(std::make_shared<BooleanTensor<T>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));

        msb[i]->bitwise_not(b[i + 1].get());
        b[i + 1]->bitwise_and(msb[i + 1].get(), b[i + 1].get());
    }

    b.emplace_back(std::make_shared<BooleanTensor<T>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));
    msb[len_break_point - 1]->bitwise_not(b[len_break_point].get());

    // ret += b[i].mul(polynomial(coeff[i]))
    std::vector<std::shared_ptr<FixedPointTensor<T, N>>> temp_fixed;
    std::vector<std::shared_ptr<FixedPointTensor<T, N>>> temp_fixed1;

    assign_to_tensor(ret_cpy->_share[0], (T) 0);
    assign_to_tensor(ret_cpy->_share[1], (T) 0);

    for (int i = 0; i < len_coeff; ++i) {
        temp_fixed.emplace_back(
                    std::make_shared<FixedPointTensor<T, N>>(
                                                temp[temp_index++].get(),
                                                temp[temp_index++].get()));
        temp_fixed1.emplace_back(
                    std::make_shared<FixedPointTensor<T, N>>(
                                                temp[temp_index++].get(),
                                                temp[temp_index++].get()));
        auto t = tensor_factory()->template create<T>();
        coeff->slice(i, i + 1, t.get());
        auto t_shape = t->shape();
        // remove leading 1
        t_shape.erase(t_shape.begin());
        t->reshape(t_shape);;
        this->polynomial(t.get(), temp_fixed[i].get());
        b[i]->bit_extract(0, b[i].get());
        b[i]->mul(temp_fixed[i].get(), temp_fixed1[i].get());
        ret_cpy->add(temp_fixed1[i].get(), ret_cpy.get());
    }
    ret_cpy->share(0)->copy(ret->mutable_share(0));
    ret_cpy->share(1)->copy(ret->mutable_share(1));
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::lt(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 2; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> sub_result =
        std::make_shared<FixedPointTensor<T, N>>(
                                temp[0].get(), temp[1].get());
    this->sub(rhs, sub_result.get());
    ret->bit_extract(sizeof(T) * 8 - 1, sub_result.get());
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::lt(const TensorAdapter<T>* rhs, BooleanTensor<T>* ret) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 2; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> sub_result =
        std::make_shared<FixedPointTensor<T, N>>(
                                temp[0].get(), temp[1].get());
    this->sub(rhs, sub_result.get());
    ret->bit_extract(sizeof(T) * 8 - 1, sub_result.get());
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::leq(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    this->gt(rhs, ret);
    auto tensor_one = tensor_factory()->
                            template create<T>(this->shape());

    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::leq(const TensorAdapter<T>* rhs, BooleanTensor<T>* ret) const {
    this->gt(rhs, ret);
    auto tensor_one = tensor_factory()->
                            template create<T>(this->shape());

    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::gt(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 2; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> sub_result =
        std::make_shared<FixedPointTensor<T, N>>(
                                    temp[0].get(), temp[1].get());
    this->sub(rhs, sub_result.get());
    sub_result->negative(sub_result.get());
    ret->template bit_extract(sizeof(T) * 8 - 1, sub_result.get());
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::gt(const TensorAdapter<T>* rhs, BooleanTensor<T>* ret) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 2; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> sub_result =
        std::make_shared<FixedPointTensor<T, N>>(
                                    temp[0].get(), temp[1].get());
    this->sub(rhs, sub_result.get());
    sub_result->negative(sub_result.get());
    ret->template bit_extract(sizeof(T) * 8 - 1, sub_result.get());
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::geq(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    this->lt(rhs, ret);
    auto tensor_one = tensor_factory()->
                            template create<T>(this->shape());

    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::geq(const TensorAdapter<T>* rhs, BooleanTensor<T>* ret) const {
    this->lt(rhs, ret);
    auto tensor_one = tensor_factory()->
                            template create<T>(this->shape());

    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::eq(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    this->neq(rhs, ret);
    auto tensor_one = tensor_factory()->template create<T>(this->shape());
    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::eq(const TensorAdapter<T>* rhs, BooleanTensor<T>* ret) const {
    this->neq(rhs, ret);
    auto tensor_one = tensor_factory()->template create<T>(this->shape());
    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::neq(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 4; i ++) {
        temp.emplace_back(tensor_factory()->
                                template create<T>(this->shape()));
    }
    std::shared_ptr<BooleanTensor<T>> lt =
            std::make_shared<BooleanTensor<T>>(
                                temp[0].get(), temp[1].get());
    std::shared_ptr<BooleanTensor<T>> gt =
            std::make_shared<BooleanTensor<T>>(
                                temp[2].get(), temp[3].get());

    this->lt(rhs, lt.get());
    this->gt(rhs, gt.get());
    lt->bitwise_or(gt.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::neq(const TensorAdapter<T>* rhs, BooleanTensor<T>* ret) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 4; i ++) {
        temp.emplace_back(tensor_factory()->
                                template create<T>(this->shape()));
    }
    std::shared_ptr<BooleanTensor<T>> lt =
            std::make_shared<BooleanTensor<T>>(
                                temp[0].get(), temp[1].get());
    std::shared_ptr<BooleanTensor<T>> gt =
            std::make_shared<BooleanTensor<T>>(
                                temp[2].get(), temp[3].get());

    this->lt(rhs, lt.get());
    this->gt(rhs, gt.get());
    lt->bitwise_or(gt.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::reciprocal(const FixedPointTensor<T, N>* op, FixedPointTensor<T, N>* ret,
                                        size_t iter, double x0) {
    auto temp0 = tensor_factory()->template create<T>(ret->shape());
    auto temp1 = tensor_factory()->template create<T>(ret->shape());
    auto temp2 = tensor_factory()->template create<T>(ret->shape());
    auto temp3 = tensor_factory()->template create<T>(ret->shape());
    std::shared_ptr<FixedPointTensor<T, N>> result =
        std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
    std::shared_ptr<FixedPointTensor<T, N>> x_copy =
        std::make_shared<FixedPointTensor<T, N>>(temp2.get(), temp3.get());
    assign_to_tensor(result->mutable_share(0), (T) 0);
    assign_to_tensor(result->mutable_share(1), (T) 0);
    auto tensor_x0 = tensor_factory()->template create<T>(op->shape());
    assign_to_tensor(tensor_x0.get(), (T)(x0 * pow(2, N)));
    tensor_x0->scaling_factor() = N;
    result->add(tensor_x0.get(), result.get());
    auto tensor_2 = tensor_factory()->template create<T>(op->shape());
    tensor_2->scaling_factor() = N;
    assign_to_tensor(tensor_2.get(), (T)(2 << N));
    for (int i = 0; i < iter; ++i) {
        result->share(0)->copy(x_copy->mutable_share(0));
        result->share(1)->copy(x_copy->mutable_share(1));
        auto res_ptr = result.get();
        op->mul(res_ptr, res_ptr);
        result->negative(res_ptr);
        result->add(tensor_2.get(), res_ptr);
        x_copy->mul(res_ptr, res_ptr);
    }
    result->share(0)->copy(ret->mutable_share(0));
    result->share(1)->copy(ret->mutable_share(1));
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::inverse_square_root(FixedPointTensor* ret,
                                                 size_t iter,
                                                 double x0) const {
    auto temp = tensor_factory()->template malloc_tensor<T>(4, this->shape());

    std::shared_ptr<FixedPointTensor<T, N> > sqrt =
        std::make_shared<FixedPointTensor<T, N> >(temp[0].get(), temp[1].get());

    std::shared_ptr<FixedPointTensor<T, N> > one =
        std::make_shared<FixedPointTensor<T, N> >(temp[2].get(), temp[3].get());

    this->square_root(sqrt.get(), iter, x0);

    float one_share = 1. / 3;
    assign_to_tensor(one->mutable_share(0), (T)(one_share * pow(2, N)));
    assign_to_tensor(one->mutable_share(1), (T)(one_share * pow(2, N)));

    one->long_div(sqrt.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::square_root(FixedPointTensor* ret,
                                         size_t iter,
                                         double x0) const {
    auto temp = tensor_factory()->template malloc_tensor<T>(4, this->shape());

    std::shared_ptr<FixedPointTensor<T, N> > x_n =
        std::make_shared<FixedPointTensor<T, N> >(temp[0].get(), temp[1].get());
    std::shared_ptr<FixedPointTensor<T, N> > tmp =
        std::make_shared<FixedPointTensor<T, N> >(temp[2].get(), temp[3].get());

    // split to 3 shares
    x0 /= 3;

    assign_to_tensor(x_n->mutable_share(0), (T)(x0 * pow(2, N)));
    assign_to_tensor(x_n->mutable_share(1), (T)(x0 * pow(2, N)));

    for (int i = 0; i < iter; ++i) {
        this->long_div(x_n.get(), tmp.get());
        x_n->add(tmp.get(), x_n.get());
        truncate(x_n.get(), x_n.get(), 1);
    }

    x_n->share(0)->copy(ret->mutable_share(0));
    x_n->share(1)->copy(ret->mutable_share(1));

}
// Newton's method, var naming from Quake III Arena: Q_rsqrt
// float threehalfs = 1.5F;
// x2 = number * 0.5F;
// y  = x0; // since 0x5f3759df does not fit fixed-point arithmetic
// y  = y * ( threehalfs - ( x2 * y * y ) ); // iteration of Newton's method
template<typename T, size_t N>
void FixedPointTensor<T, N>::inverse_square_root(const FixedPointTensor* op,
                                                 FixedPointTensor* ret,
                                                 size_t iter,
                                                 double x0) {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 7; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(op->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> y =
        std::make_shared<FixedPointTensor<T, N>>(temp[0].get(), temp[1].get());
    std::shared_ptr<FixedPointTensor<T, N>> x2 =
        std::make_shared<FixedPointTensor<T, N>>(temp[2].get(), temp[3].get());
    // x2 = 0.5 * op
    truncate(op, x2.get(), 1);

    // split to 3 shares
    x0 /= 3;
    assign_to_tensor(y->mutable_share(0), (T)(x0 * pow(2, N)));
    assign_to_tensor(y->mutable_share(1), (T)(x0 * pow(2, N)));

    // threehalfs
    temp[4]->scaling_factor() = N;
    assign_to_tensor(temp[4].get(), T(1.5 * pow(2, N)));

    std::shared_ptr<FixedPointTensor<T, N>> y_copy =
        std::make_shared<FixedPointTensor<T, N>>(temp[5].get(), temp[6].get());

    for (int i = 0; i < iter; ++i) {
        y->share(0)->copy(y_copy->mutable_share(0));
        y->share(1)->copy(y_copy->mutable_share(1));
        y->mul(y.get(), y.get());
        y->mul(x2.get(), y.get());
        y->negative(y.get());
        y->add(temp[4].get(), y.get());
        y_copy->mul(y.get(), y.get());
    }
    y->share(0)->copy(ret->mutable_share(0));
    y->share(1)->copy(ret->mutable_share(1));
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::max(const CTensor<T, N1...>* rhs,
                                 FixedPointTensor* ret,
                                 BooleanTensor<T>* cmp) const {
    // max = lhs + (rhs - lhs) if rhs > lhs else lhs
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    bool output_cmp = cmp != nullptr;
    // if cmp is not null, store cmp results in cmp
    // else, store them in tmp tensors
    for (int i = 0; i < 2 + 2 * (!output_cmp); ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    FixedPointTensor<T, N> delta(temp[0].get(), temp[1].get());
    sub(rhs, &delta);
    BooleanTensor<T> sign;
    if (output_cmp) {
        sign = *cmp;
    } else {
        sign = BooleanTensor<T>(temp[2].get(), temp[3].get());
    }
    sign.template bit_extract(sizeof(T) * 8 - 1, &delta);
    delta.negative(&delta);
    sign.mul(&delta, &delta);
    add(&delta, ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::max(const TensorAdapter<T>* rhs,
                                 FixedPointTensor* ret,
                                 BooleanTensor<T>* cmp) const {
    // max = lhs + (rhs - lhs) if rhs > lhs else lhs
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    bool output_cmp = cmp != nullptr;
    // if cmp is not null, store cmp results in cmp
    // else, store them in tmp tensors
    for (int i = 0; i < 2 + 2 * (!output_cmp); ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    FixedPointTensor<T, N> delta(temp[0].get(), temp[1].get());
    sub(rhs, &delta);
    BooleanTensor<T> sign;
    if (output_cmp) {
        sign = *cmp;
    } else {
        sign = BooleanTensor<T>(temp[2].get(), temp[3].get());
    }
    sign.template bit_extract(sizeof(T) * 8 - 1, &delta);
    delta.negative(&delta);
    sign.mul(&delta, &delta);
    add(&delta, ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::max_pooling(FixedPointTensor* ret,
                                         BooleanTensor<T>* pos) const {
    size_t k = shape()[0];
    std::vector<std::shared_ptr<TensorAdapter<T>>> tmp;
    for (int i = 0; i < 4; ++i) {
        tmp.emplace_back(
            tensor_factory()->template create<T>());
    }

    FixedPointTensor now(tmp[0].get(), tmp[1].get());
    BooleanTensor<T> cmp(tmp[2].get(), tmp[3].get());
    auto cmp_ptr = pos ? &cmp : nullptr;

    share(0)->slice(0, 1, tmp[0].get());
    share(1)->slice(0, 1, tmp[1].get());

    tmp[0]->copy(ret->mutable_share(0));
    tmp[1]->copy(ret->mutable_share(1));

    if (pos) {
        pos->share(0)->slice(0, 1, tmp[2].get());
        pos->share(1)->slice(0, 1, tmp[3].get());

        // set init 1, slice_0 is larger than null
        if (party() == 0 || party() == 2) {
            size_t idx = 2 + (party() == 2);
            assign_to_tensor(tmp[idx].get(), T(1));
            assign_to_tensor(tmp[5 - idx].get(), T(0));
        } else {
            assign_to_tensor(tmp[2].get(), T(0));
            assign_to_tensor(tmp[3].get(), T(0));
        }

    }

    for (size_t i = 1; i < k; ++i) {
        share(0)->slice(i, i + 1, tmp[0].get());
        share(1)->slice(i, i + 1, tmp[1].get());

        if (pos) {
            pos->share(0)->slice(i, i + 1, tmp[2].get());
            pos->share(1)->slice(i, i + 1, tmp[3].get());
        }

        ret->max(&now, ret, cmp_ptr);

    }

    if (pos) {
        pos->onehot_from_cmp();
    }

}

template<typename T, size_t N>
void FixedPointTensor<T, N>::avg_pooling(FixedPointTensor* ret) const {
    size_t k = shape()[0];
    std::vector<std::shared_ptr<TensorAdapter<T>>> tmp;
    for (int i = 0; i < 3; ++i) {
        tmp.emplace_back(
            tensor_factory()->template create<T>());
    }

    assign_to_tensor(ret->mutable_share(0), (T)0);
    assign_to_tensor(ret->mutable_share(1), (T)0);

    FixedPointTensor now(tmp[0].get(), tmp[1].get());

    for (size_t i = 0; i < k; ++i) {
        share(0)->slice(i, i + 1, tmp[0].get());
        share(1)->slice(i, i + 1, tmp[1].get());

        ret->add(&now, ret);
    }

    tmp[2]->reshape(ret->shape());

    tmp[2]->scaling_factor() = N;
    assign_to_tensor(tmp[2].get(), (T)((1 << N) / k));

    ret->mul(tmp[2].get(), ret);

}

template<typename T, size_t N>
void FixedPointTensor<T, N>::preds_to_indices(const FixedPointTensor* preds,
                                              FixedPointTensor* indices,
                                              float threshold) {
    // 3 for allocating temp tensor
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 3; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }

    auto shape_ = preds->shape();

    // plaintext tensor for threshold
    temp[0]->reshape(shape_);
    temp[0]->scaling_factor() = N;
    assign_to_tensor(temp[0].get(), T(threshold * (T(1) << N)));

    temp[1]->reshape(shape_);
    temp[2]->reshape(shape_);
    BooleanTensor<T> cmp_res(temp[1].get(), temp[2].get());

    preds->gt(temp[0].get(), &cmp_res);

    cmp_res.lshift(N, &cmp_res);

    cmp_res.b2a(indices);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::calc_tp_fp_fn(
    const FixedPointTensor* indices,
    const FixedPointTensor* labels,
    FixedPointTensor* tp_fp_fn) {

    PADDLE_ENFORCE_EQ(indices->shape().size(), 1,
                      "multi-classification not support yet");

    PADDLE_ENFORCE_EQ(tp_fp_fn->shape().size(), 1,
                      "multi-classification not support yet");

    PADDLE_ENFORCE_EQ(tp_fp_fn->shape()[0], 3,
                      "store tp fp fn for binary-classification only");

    // 4 for allocating temp tensor
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 4; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }

    auto shape_ = indices->shape();
    std::vector<size_t> shape_one = {1};
    std::vector<size_t> shape_3 = {3};

    temp[0]->reshape(shape_);
    temp[1]->reshape(shape_);

    FixedPointTensor true_positive(temp[0].get(), temp[1].get());

    indices->mul(labels, &true_positive);

    temp[2]->reshape(shape_one);
    temp[3]->reshape(shape_one);

    FixedPointTensor scalar(temp[2].get(), temp[3].get());

    // tp
    reduce(&true_positive, &scalar);

    auto slice00 = tensor_factory()->template create<T>();
    auto slice01 = tensor_factory()->template create<T>();

    auto slice10 = tensor_factory()->template create<T>();
    auto slice11 = tensor_factory()->template create<T>();

    auto slice20 = tensor_factory()->template create<T>();
    auto slice21 = tensor_factory()->template create<T>();

    tp_fp_fn->mutable_share(0)->slice(0, 1, slice00.get());
    tp_fp_fn->mutable_share(1)->slice(0, 1, slice01.get());

    tp_fp_fn->mutable_share(0)->slice(1, 2, slice10.get());
    tp_fp_fn->mutable_share(1)->slice(1, 2, slice11.get());

    tp_fp_fn->mutable_share(0)->slice(2, 3, slice20.get());
    tp_fp_fn->mutable_share(1)->slice(2, 3, slice21.get());

    FixedPointTensor res0(slice00.get(), slice01.get());
    FixedPointTensor res1(slice10.get(), slice11.get());
    FixedPointTensor res2(slice20.get(), slice21.get());

    scalar.share(0)->copy(slice00.get());
    scalar.share(1)->copy(slice01.get());

    // tp + fp
    reduce(indices, &scalar);

    scalar.sub(&res0, &res1);

    // tp + fn
    reduce(labels, &scalar);

    scalar.sub(&res0, &res2);

}

#ifndef USE_CUDA
template<typename T, size_t N>
void FixedPointTensor<T, N>::calc_precision_recall(
    const FixedPointTensor* tp_fp_fn,
    TensorAdapter<T>* ret) {
    PADDLE_ENFORCE_EQ(tp_fp_fn->shape().size(), 1,
                      "multi-classification not support yet");

    PADDLE_ENFORCE_EQ(tp_fp_fn->shape()[0], 3,
                      "store tp fp fn for binary-classification only");

    PADDLE_ENFORCE_EQ(ret->shape().size(), 1,
                      "multi-classification not support yet");

    PADDLE_ENFORCE_EQ(ret->shape()[0], 3,
                      "store precision recall f1-score"
                      "for binary-classification only");
    // 5 for allocating temp tensor
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 7; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }
    std::vector<size_t> shape_ = {3};

    std::vector<size_t> shape_one = {1};

    temp[0]->reshape(shape_one);
    temp[1]->reshape(shape_one);
    FixedPointTensor scalar(temp[0].get(), temp[1].get());

    temp[2]->reshape(shape_one);
    temp[3]->reshape(shape_one);
    FixedPointTensor scalar2(temp[2].get(), temp[3].get());

    temp[4]->reshape(shape_one);
    temp[5]->reshape(shape_one);
    FixedPointTensor tmp_(temp[4].get(), temp[5].get());

    auto get = [&tp_fp_fn](size_t idx, FixedPointTensor* dest) {
        tp_fp_fn->share(0)->slice(idx, idx + 1, dest->mutable_share(0));
        tp_fp_fn->share(1)->slice(idx, idx + 1, dest->mutable_share(1));
    };

    get(0, &scalar);
    get(1, &scalar2);

    // tp + fp
    scalar.add(&scalar2, &tmp_);

    scalar.long_div(&tmp_, &tmp_);

    temp[6]->reshape(shape_one);

    tmp_.reveal(temp[6].get());

    ret->scaling_factor() = N;
    ret->data()[0] = temp[6]->data()[0];

    get(2, &scalar2);

    // tp + fn
    scalar.add(&scalar2, &tmp_);

    scalar.long_div(&tmp_, &tmp_);
    tmp_.reveal(temp[6].get());

    ret->data()[1] = temp[6]->data()[0];

    float precision = 1.0 * ret->data()[0] / (T(1) << N);
    float recall = 1.0 * ret->data()[1] / (T(1) << N);
    float f1_score = 0.0;
    if (precision + recall > 0) {
        f1_score = 2 * precision * recall / (precision + recall);
    }

    ret->data()[2] = T(f1_score * (T(1) << N));
}
#endif // USE_CUDA

template<typename T, size_t N>
void FixedPointTensor<T, N>::calc_p_distance(const FixedPointTensor* lhs,
                                             const FixedPointTensor* rhs,
                                             const TensorAdapter<T>* miss,
                                             FixedPointTensor* ret) {
    
    size_t len = lhs->shape()[1];
    auto shape = lhs->shape();
    auto miss_len = miss->shape()[0];
    std::vector<size_t> shape_one = {1};
    std::vector<size_t> distance_shape = {len};
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 8; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(lhs->shape()));
    }
    for (size_t i = 8; i < 12; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }
    FixedPointTensor<T, N> cmp_result(temp[0].get(), temp[1].get());
    FixedPointTensor<T, N> cmp0(temp[2].get(), temp[3].get());
    BooleanTensor<T> cmp_bool(temp[4].get(), temp[5].get());
    lhs->sub(rhs, &cmp0);
    cmp0.share(0)->copy(cmp_result.mutable_share(0));
    cmp0.share(1)->copy(cmp_result.mutable_share(1));
    
    auto _nN = tensor_factory()->template create<T>();
    _nN->scaling_factor() = N;
    for (size_t idx = 0; idx < miss_len; ++idx) {
        miss->slice(idx, idx + 1, _nN.get());
        _nN->reshape(shape);
        lhs->sub(_nN.get(), &cmp0);
        cmp_result.mul(&cmp0, &cmp_result);
        rhs->sub(_nN.get(), &cmp0);
        cmp_result.mul(&cmp0, &cmp_result);
    }
    auto zero_tensor = tensor_factory()->template create<T>(lhs->shape());
    zero_tensor->scaling_factor() = N;
    assign_to_tensor(zero_tensor.get(), T(0));
    cmp_result.neq(zero_tensor.get(), &cmp_bool);
    
    FixedPointTensor<T, N> distance(temp[6].get(), temp[7].get());
    FixedPointTensor<T, N> temp_distance(temp[8].get(), temp[9].get());
    FixedPointTensor<T, N> temp_sum(temp[10].get(), temp[11].get());
    temp_sum.mutable_share(0)->reshape(shape_one);
    temp_sum.mutable_share(1)->reshape(shape_one);
    
    cmp_bool.lshift(N, &cmp_bool);
    cmp_bool.b2a(&distance);
    
    auto scale = tensor_factory()->template create<T>(ret->shape());
    scale->scaling_factor() = N;
    assign_to_tensor(scale.get(), T(len * (T(1) << N)));
    
    for (size_t i = 0; i < distance.shape()[0]; ++i) {
        distance.share(0)->slice(i, i + 1, temp_distance.mutable_share(0));
        temp_distance.mutable_share(0)->reshape(distance_shape);
        distance.share(1)->slice(i, i + 1, temp_distance.mutable_share(1));
        temp_distance.mutable_share(1)->reshape(distance_shape);
        temp_distance.sum(&temp_sum);
        ret->mutable_share(0)->data()[i] = temp_sum.share(0)->data()[0];
        ret->mutable_share(1)->data()[i] = temp_sum.share(1)->data()[0];
    }
    //distance.sum(ret);
    ret->div(scale.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::calc_multi_p_distance(const FixedPointTensor* lhs,
                                                   const TensorAdapter<T>* miss,
                                                   FixedPointTensor* ret) {
    
    auto shape = ret->shape();
    auto shape_ = lhs->shape();
    //shape_.erase(shape_.begin());
    std::vector<size_t> shape_one_ret = {shape_[0]};
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 7; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }
    
    temp[4]->reshape(shape_one_ret);
    temp[5]->reshape(shape_one_ret);
    FixedPointTensor<T, N> temp_ret(temp[4].get(), temp[5].get());
    
    std::vector<size_t> shape_miss = {miss->shape()[0], lhs->shape()[0], lhs->shape()[1]};
    auto miss_ = tensor_factory()->template create<T>(shape_miss);
    miss_->scaling_factor() = N;
    for (size_t i = 0; i < miss->shape()[0]; ++i) {
        miss_->slice(i, i + 1, temp[6].get());
        assign_to_tensor(temp[6].get(), T(miss->data()[i]));
    }

    for (size_t i = 0; i < shape[0]; ++i) {
        lhs->share(0)->slice(i, i + 1, temp[0].get());
        lhs->share(1)->slice(i, i + 1, temp[1].get());

        temp[2]->reshape(shape_);
        temp[3]->reshape(shape_);
        FixedPointTensor<T, N> temp_l(temp[2].get(), temp[3].get());
        for(size_t idx = 0; idx < shape_[0]; ++idx) {
            temp[0]->copy(temp_l.mutable_share(0)->operator[](idx).get());
            temp[1]->copy(temp_l.mutable_share(1)->operator[](idx).get());
        }
        calc_p_distance(&temp_l, lhs, miss_.get(), &temp_ret);
        temp_ret.share(0)->copy(ret->mutable_share(0)->operator[](i).get());
        temp_ret.share(1)->copy(ret->mutable_share(1)->operator[](i).get());
        std::cout << "distance " << i << std::endl;
    }
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::align_nw_two(const FixedPointTensor* lhs,
                                          const FixedPointTensor* rhs,
                                          FixedPointTensor* ret,
                                          FixedPointTensor* score_bottom_right,
                                          FixedPointTensor* paths,
                                          std::vector<std::shared_ptr<FixedPointTensor<T, N>>> &aligned,
                                          std::vector<size_t> &l) {
    
    auto n1 = lhs->shape()[0];
    auto n2 = rhs->shape()[0];
    auto shape_o = ret->shape();
    std::vector<size_t> shape_one = {1};
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 26; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(shape_one));
        temp[i]->scaling_factor() = N;
    }
    
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp_bool;
    for (size_t i = 0; i < 2; ++i) {
        temp_bool.emplace_back(
            tensor_factory()->template create<T>(shape_one));
    }
    
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp_paths;
    for (size_t i = 0; i < 2; ++i) {
        temp_paths.emplace_back(
            tensor_factory()->template create<T>(shape_o));
    }
    
    std::shared_ptr<FixedPointTensor<T, N> > paths_o =
        std::make_shared<FixedPointTensor<T, N> >(temp_paths[0].get(), temp_paths[1].get());
    
    // 2 -1 1 3
    std::shared_ptr<FixedPointTensor<T, N> > two =
        std::make_shared<FixedPointTensor<T, N> >(temp[16].get(), temp[17].get());
    std::shared_ptr<FixedPointTensor<T, N> > neg_one =
        std::make_shared<FixedPointTensor<T, N> >(temp[18].get(), temp[19].get());
    std::shared_ptr<FixedPointTensor<T, N> > one =
        std::make_shared<FixedPointTensor<T, N> >(temp[20].get(), temp[21].get());
    std::shared_ptr<FixedPointTensor<T, N> > fixed_three =
        std::make_shared<FixedPointTensor<T, N> >(temp[22].get(), temp[23].get());
    std::shared_ptr<FixedPointTensor<T, N> > fixed_underline =
        std::make_shared<FixedPointTensor<T, N> >(temp[24].get(), temp[25].get());
    auto two_share = tensor_factory()->template create<T>(shape_one);
    auto neg_one_share = tensor_factory()->template create<T>(shape_one);
    auto scalar_three = tensor_factory()->template create<T>(shape_one);
    auto underline = tensor_factory()->template create<T>(shape_one);
    neg_one_share->scaling_factor() = N;
    two_share->scaling_factor() = N;
    scalar_three->scaling_factor() = N;
    underline->scaling_factor() = N;
    assign_to_tensor(two_share.get(), (T)(2 * pow(2, N)));
    assign_to_tensor(neg_one_share.get(), (T)(-1 * pow(2, N)));
    assign_to_tensor(scalar_three.get(), T(3 * (T(1) << N)));
    assign_to_tensor(underline.get(), T(95 * (T(1) << N)));
    online_share(0, two_share.get(), two.get());
    online_share(0, neg_one_share.get(), neg_one.get());
    online_share(0, scalar_three.get(), fixed_three.get());
    online_share(0, underline.get(), fixed_underline.get());
    neg_one->negative(one.get());
    
    
    //ret[0][0] = 0
    ret->mutable_share(0)->data()[0] = T(0);
    ret->mutable_share(1)->data()[0] = T(0);
    paths_o->mutable_share(0)->data()[0] = T(0);
    paths_o->mutable_share(1)->data()[0] = T(0);
    for(size_t idx = 1; idx < n1 + 1; ++idx) {
        auto value = T(-1 * idx) * (T(1) << N);
        ret->mutable_share(0)->data()[idx * (n2 + 1)] = value;
        ret->mutable_share(1)->data()[idx * (n2 + 1)] = value;
        paths_o->mutable_share(0)->data()[idx * (n2 + 1)] = fixed_three->share(0)->data()[0];
        paths_o->mutable_share(1)->data()[idx * (n2 + 1)] = fixed_three->share(1)->data()[0];

    }
    for(size_t idx = 1; idx < n2 + 1; ++idx) {
        auto value = T(-1 * idx) * (T(1) << N);
        ret->mutable_share(0)->data()[idx] = value;
        ret->mutable_share(1)->data()[idx] = value;
        paths_o->mutable_share(0)->data()[idx] = one->share(0)->data()[0];
        paths_o->mutable_share(1)->data()[idx] = one->share(1)->data()[0];
    }
    
    FixedPointTensor<T, N> c1(temp[0].get(), temp[1].get());
    FixedPointTensor<T, N> c2(temp[2].get(), temp[3].get());
    FixedPointTensor<T, N> c3(temp[4].get(), temp[5].get());
    FixedPointTensor<T, N> seq1(temp[6].get(), temp[7].get());
    FixedPointTensor<T, N> seq2(temp[8].get(), temp[9].get());
    FixedPointTensor<T, N> cost(temp[10].get(), temp[11].get());
    FixedPointTensor<T, N> cost_temp(temp[12].get(), temp[13].get());
    FixedPointTensor<T, N> score(temp[14].get(), temp[15].get());
    BooleanTensor<T> cmp(temp_bool[0].get(), temp_bool[1].get());
    
    for (size_t i = 1; i < n1 + 1; ++i) {
        for (size_t j = 1; j < n2 + 1; ++j) {
            //cost(seq1[i - 1], seq2[j - 1])
            seq1.mutable_share(0)->data()[0] = lhs->share(0)->data()[i - 1]; 
            seq1.mutable_share(1)->data()[0] = lhs->share(1)->data()[i - 1];
            seq2.mutable_share(0)->data()[0] = rhs->share(0)->data()[j - 1]; 
            seq2.mutable_share(1)->data()[0] = rhs->share(1)->data()[j - 1];
            seq1.eq(&seq2, &cmp);
            // equal 2  not equal -1
            cmp.mul(two.get(), &cost);
            cmp.bitwise_not(&cmp);
            cmp.mul(neg_one.get(), &cost_temp);
            cost.add(&cost_temp, &cost);

            // c1, c2, c3
            c1.mutable_share(0)->data()[0] = ret->share(0)->data()[(i - 1) * (n2 + 1) + j - 1];
            c1.mutable_share(1)->data()[0] = ret->share(1)->data()[(i - 1) * (n2 + 1) + j - 1];
            c1.add(&cost, &c1);

            c2.mutable_share(0)->data()[0] = ret->share(0)->data()[(i - 1) * (n2 + 1) + j];
            c2.mutable_share(1)->data()[0] = ret->share(1)->data()[(i - 1) * (n2 + 1) + j];
            c2.sub(scalar_three.get(), &c2);
 
            c3.mutable_share(0)->data()[0] = ret->share(0)->data()[i * (n2 + 1) + j - 1];
            c3.mutable_share(1)->data()[0] = ret->share(1)->data()[i * (n2 + 1) + j - 1];
            c3.sub(scalar_three.get(), &c3);

            c1.max(&c2, &score, &cmp);
            cmp.mul(fixed_three.get(), &cost);
            cmp.bitwise_not(&cmp);
            cmp.mul(two.get(), &cost_temp);
            cost.add(&cost_temp, &cost);

            score.max(&c3, &score, &cmp);
            cmp.mul(one.get(), &cost_temp);
            cmp.bitwise_not(&cmp);
            cmp.mul(&cost,&cost);
            cost.add(&cost_temp, &cost);

            ret->mutable_share(0)->data()[ i * (n2 + 1) + j] = score.share(0)->data()[0];
            ret->mutable_share(1)->data()[ i * (n2 + 1) + j] = score.share(1)->data()[0];

            paths_o->mutable_share(0)->data()[ i * (n2 + 1) + j] = cost.share(0)->data()[0];
            paths_o->mutable_share(1)->data()[ i * (n2 + 1) + j] = cost.share(1)->data()[0];
        }
    }
    score_bottom_right->mutable_share(0)->data()[0] = ret->share(0)->data()[n1 * (n2 + 1) + n2];
    score_bottom_right->mutable_share(1)->data()[0] = ret->share(1)->data()[n1 * (n2 + 1) + n2];
    paths_o->mutable_share(0)->copy(paths->mutable_share(0));
    paths_o->mutable_share(1)->copy(paths->mutable_share(1));

    size_t i = n1;
    size_t j = n2;
    std::vector<size_t> shape_max = {n1 + n2};
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp_path_i_j;
    for (size_t i = 0; i < 4; ++i) {
        temp_path_i_j.emplace_back(
            tensor_factory()->template create<T>(shape_max));
    }
    
    FixedPointTensor<T, N> aligned0(temp_path_i_j[0].get(), temp_path_i_j[1].get());
    FixedPointTensor<T, N> aligned1(temp_path_i_j[2].get(), temp_path_i_j[3].get());
    FixedPointTensor<T, N> path_i_j(temp[0].get(), temp[1].get());
    auto path_i_j_reveal = tensor_factory()->template create<T>(shape_one);
    //two->mul(scalar_three.get(), two.get());
    size_t idx = 0;
    while(i!=0 || j!=0) {
        path_i_j.mutable_share(0)->data()[0] = paths_o->share(0)->data()[i * (n2 + 1) + j];
        path_i_j.mutable_share(1)->data()[0] = paths_o->share(1)->data()[i * (n2 + 1) + j];
        path_i_j.reveal(path_i_j_reveal.get());
        size_t path_value = path_i_j_reveal->data()[0] >> N;
        if (path_value == 1) {
            aligned0.mutable_share(0)->data()[idx] = fixed_underline->share(0)->data()[0];
            aligned0.mutable_share(1)->data()[idx] = fixed_underline->share(1)->data()[0];
            aligned1.mutable_share(0)->data()[idx] = rhs->share(0)->data()[j - 1];
            aligned1.mutable_share(1)->data()[idx] = rhs->share(1)->data()[j - 1];
            ++idx;
            --j;
        } else if (path_value == 2) {
            aligned0.mutable_share(0)->data()[idx] = lhs->share(0)->data()[i - 1];
            aligned0.mutable_share(1)->data()[idx] = lhs->share(1)->data()[i - 1];
            aligned1.mutable_share(0)->data()[idx] = rhs->share(0)->data()[j - 1];
            aligned1.mutable_share(1)->data()[idx] = rhs->share(1)->data()[j - 1];
            ++idx;
            --j;
            --i;
        } else if (path_value == 3) {
            aligned0.mutable_share(0)->data()[idx] = lhs->share(0)->data()[i - 1];
            aligned0.mutable_share(1)->data()[idx] = lhs->share(1)->data()[i - 1];
            aligned1.mutable_share(0)->data()[idx] = fixed_underline->share(0)->data()[0];
            aligned1.mutable_share(1)->data()[idx] = fixed_underline->share(1)->data()[0];
            ++idx;
            --i;
            l.emplace_back(j);
        }  
    }
    
    std::vector<size_t> shape_real = {idx};
    aligned0.mutable_share(0)->reshape(shape_real);
    aligned0.mutable_share(1)->reshape(shape_real); 
    aligned1.mutable_share(0)->reshape(shape_real);
    aligned1.mutable_share(1)->reshape(shape_real);
    
    std::reverse(aligned0.mutable_share(0)->data(), aligned0.mutable_share(0)->data() + shape_real[0]);
    std::reverse(aligned0.mutable_share(1)->data(), aligned0.mutable_share(1)->data() + shape_real[0]);
    std::reverse(aligned1.mutable_share(0)->data(), aligned1.mutable_share(0)->data() + shape_real[0]);
    std::reverse(aligned1.mutable_share(1)->data(), aligned1.mutable_share(1)->data() + shape_real[0]);
    
    for (size_t i = 0; i < 2; ++i) {
        aligned[i]->mutable_share(0)->reshape(shape_real);
        aligned[i]->mutable_share(1)->reshape(shape_real);
    }

    aligned0.mutable_share(0)->copy(aligned[0]->mutable_share(0));
    aligned0.mutable_share(1)->copy(aligned[0]->mutable_share(1)); 
    aligned1.mutable_share(0)->copy(aligned[1]->mutable_share(0));
    aligned1.mutable_share(1)->copy(aligned[1]->mutable_share(1));
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::align_nw_two_socre(const FixedPointTensor* lhs,
                                                const FixedPointTensor* rhs,
                                                FixedPointTensor* ret,
                                                FixedPointTensor* score_bottom_right) {
    
    auto n1 = lhs->shape()[0];
    auto n2 = rhs->shape()[0];
    auto shape_o = ret->shape();
    std::vector<size_t> shape_one = {1};
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 20; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(shape_one));
        temp[i]->scaling_factor() = N;
    }

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp_bool;
    for (size_t i = 0; i < 2; ++i) {
        temp_bool.emplace_back(
            tensor_factory()->template create<T>(shape_one));
    }
    
    // 2 -1 1 3
    std::shared_ptr<FixedPointTensor<T, N> > two =
        std::make_shared<FixedPointTensor<T, N> >(temp[16].get(), temp[17].get());
    std::shared_ptr<FixedPointTensor<T, N> > neg_one =
        std::make_shared<FixedPointTensor<T, N> >(temp[18].get(), temp[19].get());
    auto two_share = tensor_factory()->template create<T>(shape_one);
    auto neg_one_share = tensor_factory()->template create<T>(shape_one);
    auto scalar_three = tensor_factory()->template create<T>(shape_one);
    neg_one_share->scaling_factor() = N;
    two_share->scaling_factor() = N;
    scalar_three->scaling_factor() = N;
    assign_to_tensor(two_share.get(), (T)(2 * pow(2, N)));
    assign_to_tensor(neg_one_share.get(), (T)(-1 * pow(2, N)));
    assign_to_tensor(scalar_three.get(), T(3 * (T(1) << N)));
    online_share(0, two_share.get(), two.get());
    online_share(0, neg_one_share.get(), neg_one.get());
    
    //ret[0][0] = 0
    ret->mutable_share(0)->data()[0] = T(0);
    ret->mutable_share(1)->data()[0] = T(0);
    for(size_t idx = 1; idx < n1 + 1; ++idx) {
        auto value = T(-1 * idx) * (T(1) << N);
        ret->mutable_share(0)->data()[idx * (n2 + 1)] = value;
        ret->mutable_share(1)->data()[idx * (n2 + 1)] = value;
    }
    for(size_t idx = 1; idx < n2 + 1; ++idx) {
        auto value = T(-1 * idx) * (T(1) << N);
        ret->mutable_share(0)->data()[idx] = value;
        ret->mutable_share(1)->data()[idx] = value;
    }
    
    FixedPointTensor<T, N> c1(temp[0].get(), temp[1].get());
    FixedPointTensor<T, N> c2(temp[2].get(), temp[3].get());
    FixedPointTensor<T, N> c3(temp[4].get(), temp[5].get());
    FixedPointTensor<T, N> seq1(temp[6].get(), temp[7].get());
    FixedPointTensor<T, N> seq2(temp[8].get(), temp[9].get());
    FixedPointTensor<T, N> cost(temp[10].get(), temp[11].get());
    FixedPointTensor<T, N> cost_temp(temp[12].get(), temp[13].get());
    FixedPointTensor<T, N> score(temp[14].get(), temp[15].get());
    BooleanTensor<T> cmp(temp_bool[0].get(), temp_bool[1].get());
    
    for (size_t i = 1; i < n1 + 1; ++i) {
        for (size_t j = 1; j < n2 + 1; ++j) {
            //cost(seq1[i - 1], seq2[j - 1])
            seq1.mutable_share(0)->data()[0] = lhs->share(0)->data()[i - 1]; 
            seq1.mutable_share(1)->data()[0] = lhs->share(1)->data()[i - 1];
            seq2.mutable_share(0)->data()[0] = rhs->share(0)->data()[j - 1]; 
            seq2.mutable_share(1)->data()[0] = rhs->share(1)->data()[j - 1];
            seq1.eq(&seq2, &cmp);
            // equal 2  not equal -1
            cmp.mul(two.get(), &cost);
            cmp.bitwise_not(&cmp);
            cmp.mul(neg_one.get(), &cost_temp);
            cost.add(&cost_temp, &cost);

            // c1, c2, c3
            c1.mutable_share(0)->data()[0] = ret->share(0)->data()[(i - 1) * (n2 + 1) + j - 1];
            c1.mutable_share(1)->data()[0] = ret->share(1)->data()[(i - 1) * (n2 + 1) + j - 1];
            c1.add(&cost, &c1);

            c2.mutable_share(0)->data()[0] = ret->share(0)->data()[(i - 1) * (n2 + 1) + j];
            c2.mutable_share(1)->data()[0] = ret->share(1)->data()[(i - 1) * (n2 + 1) + j];
            c2.sub(scalar_three.get(), &c2);
 
            c3.mutable_share(0)->data()[0] = ret->share(0)->data()[i * (n2 + 1) + j - 1];
            c3.mutable_share(1)->data()[0] = ret->share(1)->data()[i * (n2 + 1) + j - 1];
            c3.sub(scalar_three.get(), &c3);

            c1.max(&c2, &score, &cmp);
            score.max(&c3, &score, &cmp);

            ret->mutable_share(0)->data()[ i * (n2 + 1) + j] = score.share(0)->data()[0];
            ret->mutable_share(1)->data()[ i * (n2 + 1) + j] = score.share(1)->data()[0];
            //std::cout << i * (n2 + 1) + j << std::endl;
        }
    }
    score_bottom_right->mutable_share(0)->data()[0] = ret->share(0)->data()[n1 * (n2 + 1) + n2];
    score_bottom_right->mutable_share(1)->data()[0] = ret->share(1)->data()[n1 * (n2 + 1) + n2];
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::align_star_multiple(std::vector<std::shared_ptr<FixedPointTensor<T, N>>> &seqs,
                                                 FixedPointTensor* ret) {
    
    
    std::vector<std::shared_ptr<FixedPointTensor<T, N>>> mutable_seqs;
    std::vector<std::shared_ptr<TensorAdapter<T>>> seqs_temp;

    for (size_t i = 0; i < seqs.size(); ++i) {
        seqs_temp.emplace_back(
            tensor_factory()->template create<T>(seqs[i]->shape()));
        seqs_temp.emplace_back(
            tensor_factory()->template create<T>(seqs[i]->shape()));
    }

    for (size_t i = 0; i < seqs.size(); ++i) {

        seqs[i]->share(0)->copy(seqs_temp[2 * i].get());
        seqs[i]->share(1)->copy(seqs_temp[2 * i + 1].get());
        mutable_seqs.emplace_back(
            new FixedPointTensor<T, N>(seqs_temp[2 * i].get(), seqs_temp[2 * i + 1].get())); 
    }

    std::vector<size_t> shape_one = {1};
    std::vector<std::pair<size_t, size_t>> pairs;
    size_t seqs_len = seqs.size();
    for (size_t i = 0; i < seqs_len; ++i) {
        for (size_t j = i + 1; j < seqs_len; ++j) {
            pairs.emplace_back(i, j);
        }
    }
    size_t pairs_len = pairs.size();
    
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 4; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(shape_one));
    }
    for (size_t i = 4; i < 20; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }

    FixedPointTensor<T, N> score_bottom_right (temp[0].get(), temp[1].get());
    std::vector<std::shared_ptr<FixedPointTensor<T, N>>> aligned;
    aligned.emplace_back(new FixedPointTensor<T, N>(temp[4].get(), temp[5].get()));
    aligned.emplace_back(new FixedPointTensor<T, N>(temp[6].get(), temp[7].get()));
    
    std::vector<size_t> shape_scores2a2 = {pairs_len};
    temp[8]->reshape(shape_scores2a2);
    temp[9]->reshape(shape_scores2a2);
    FixedPointTensor<T, N> scores2a2(temp[8].get(), temp[9].get());
    std::cout << "time 0" << std::endl;
    for (size_t i = 0; i < pairs_len; ++i) {
        size_t lhs_idx = pairs[i].first;
        size_t rhs_idx = pairs[i].second;
        std::vector<std::shared_ptr<TensorAdapter<T>>> ret_path;
        
        std::vector<size_t> shape_ret_path = {mutable_seqs[lhs_idx]->shape()[0] + 1,
                                              mutable_seqs[rhs_idx]->shape()[0] + 1};
        for (size_t iter = 0; iter < 2; ++iter) {
            ret_path.emplace_back(
                tensor_factory()->template create<T>(shape_ret_path));
        }

        FixedPointTensor<T, N> score_matrix(ret_path[0].get(), ret_path[1].get());

        align_nw_two_socre(mutable_seqs[lhs_idx].get(), mutable_seqs[rhs_idx].get(), &score_matrix, &score_bottom_right);
        scores2a2.mutable_share(0)->data()[i] = score_bottom_right.share(0)->data()[0];
        scores2a2.mutable_share(1)->data()[i] = score_bottom_right.share(1)->data()[0];
    }
    std::cout << "time 1" << std::endl;
    std::vector<size_t> shape_seqs = {seqs_len, 1};
    std::vector<size_t> shape_global_scores_temp = {seqs_len - 1};
    temp[10]->reshape(shape_seqs);
    temp[11]->reshape(shape_seqs);
    FixedPointTensor<T, N> global_scores(temp[10].get(), temp[11].get());

    temp[12]->reshape(shape_global_scores_temp);
    temp[13]->reshape(shape_global_scores_temp);
    FixedPointTensor<T, N> global_scores_temp(temp[12].get(), temp[13].get());
    
    for (size_t i = 0; i < seqs_len; ++i) {
        std::vector<size_t> scores_idx;
        for (size_t iter = 0; iter < pairs_len; ++iter) {
            if(pairs[iter].first == i || pairs[iter].second == i) {
                scores_idx.emplace_back(iter);
            }
        }
        for (size_t j = 0; j < seqs_len - 1; ++j) {
            global_scores_temp.mutable_share(0)->data()[j] = scores2a2.share(0)->data()[scores_idx[j]];
            global_scores_temp.mutable_share(1)->data()[j] = scores2a2.share(1)->data()[scores_idx[j]];
        }
        global_scores_temp.sum(&score_bottom_right);
        global_scores.mutable_share(0)->data()[i] = score_bottom_right.share(0)->data()[0];
        global_scores.mutable_share(1)->data()[i] = score_bottom_right.share(1)->data()[0];
    }
    
    std::vector<size_t> shape_pooling = {seqs_len, 1};
    std::vector<size_t> shape_one_one = {1, 1};
    global_scores.mutable_share(0)->reshape(shape_pooling);
    global_scores.mutable_share(1)->reshape(shape_pooling);
    
    temp[14]->reshape(shape_pooling);
    temp[15]->reshape(shape_pooling);
    BooleanTensor<T> max_idx_one_hot(temp[14].get(), temp[15].get());
    std::cout << "time 2" << std::endl;
    score_bottom_right.mutable_share(0)->reshape(shape_one_one);
    score_bottom_right.mutable_share(1)->reshape(shape_one_one);
    global_scores.max_pooling(&score_bottom_right, &max_idx_one_hot);
    std::cout << "time 3" << std::endl;
    auto max_tensor = tensor_factory()->template create<T>(shape_pooling);
    max_idx_one_hot.reveal(max_tensor.get());
    size_t max_idx = 0;
    for (size_t i = 0; i < seqs_len; ++i) {
        if(max_tensor->data()[i] == 1) {
            max_idx = i;
        }
    }
    
    temp[16]->reshape(mutable_seqs[max_idx]->shape());
    temp[17]->reshape(mutable_seqs[max_idx]->shape());
    FixedPointTensor<T, N> pivot(temp[16].get(), temp[17].get());
    mutable_seqs[max_idx]->share(0)->copy(pivot.mutable_share(0));
    mutable_seqs[max_idx]->share(1)->copy(pivot.mutable_share(1));

    FixedPointTensor<T, N> gap(temp[18].get(), temp[19].get());
    score_bottom_right.mutable_share(0)->reshape(shape_one);
    score_bottom_right.mutable_share(1)->reshape(shape_one);

    FixedPointTensor<T, N> underline(temp[2].get(), temp[3].get());
    auto underline_share = tensor_factory()->template create<T>(shape_one);
    underline_share->scaling_factor() = N;
    assign_to_tensor(underline_share.get(), (T)(95 * pow(2, N)));
    online_share(0, underline_share.get(), &underline);

    for (size_t i = 0; i < seqs_len; ++i) {
        std::vector<std::shared_ptr<TensorAdapter<T>>> ret_path;
        std::vector<size_t> shape_ret_path = {mutable_seqs[i]->shape()[0] + 1,
                                            pivot.shape()[0] + 1};
        for (size_t iter = 0; iter < 4; ++iter) {
            ret_path.emplace_back(
                tensor_factory()->template create<T>(shape_ret_path));
        }
        FixedPointTensor<T, N> score_matrix(ret_path[0].get(), ret_path[1].get());
        FixedPointTensor<T, N> paths(ret_path[2].get(), ret_path[3].get());
        std::vector<size_t> l;
        align_nw_two(mutable_seqs[i].get(), &pivot, &score_matrix, &score_bottom_right,
                    &paths, aligned, l);
        //std::cout << "align_nw_two " << i << std::endl;
        mutable_seqs[i]->mutable_share(0)->reshape(aligned[0]->shape());
        aligned[0]->share(0)->copy(mutable_seqs[i]->mutable_share(0));
        mutable_seqs[i]->mutable_share(1)->reshape(aligned[0]->shape());
        aligned[0]->share(1)->copy(mutable_seqs[i]->mutable_share(1));
        if (!l.empty()) {
            mutable_seqs[max_idx]->mutable_share(0)->reshape(aligned[1]->shape());
            aligned[1]->share(0)->copy(mutable_seqs[max_idx]->mutable_share(0));
            mutable_seqs[max_idx]->mutable_share(1)->reshape(aligned[1]->shape());
            aligned[1]->share(1)->copy(mutable_seqs[max_idx]->mutable_share(1));
            pivot.mutable_share(0)->reshape(mutable_seqs[max_idx]->shape());
            pivot.mutable_share(1)->reshape(mutable_seqs[max_idx]->shape());
            mutable_seqs[max_idx]->share(0)->copy(pivot.mutable_share(0));
            mutable_seqs[max_idx]->share(1)->copy(pivot.mutable_share(1));
            for(size_t j = 0; j < i; ++i) {
                if (j != max_idx) {
                    gap.mutable_share(0)->reshape(pivot.shape());
                    gap.mutable_share(1)->reshape(pivot.shape());
                    std::vector<int64_t> gap_data_0;
                    std::vector<int64_t> gap_data_1;
                    std::copy(mutable_seqs[j]->mutable_share(0)->data(), 
                              mutable_seqs[j]->mutable_share(0)->data() + mutable_seqs[j]->shape()[0],
                              gap_data_0.begin());
                    std::copy(mutable_seqs[j]->mutable_share(1)->data(), 
                              mutable_seqs[j]->mutable_share(1)->data() + mutable_seqs[j]->shape()[0],
                              gap_data_1.begin());
                    for (size_t l_idx = 0; l_idx < l.size(); ++l_idx) {
                        size_t insert_idx = l[l.size() - 1 - l_idx];
                        gap_data_0.insert(gap_data_0.begin() + insert_idx, underline.share(0)->data()[0]);
                        gap_data_1.insert(gap_data_1.begin() + insert_idx, underline.share(1)->data()[0]);
                    }
                    std::copy(gap_data_0.begin(), gap_data_0.end(), gap.mutable_share(0)->data());
                    std::copy(gap_data_1.begin(), gap_data_1.end(), gap.mutable_share(1)->data());
                    mutable_seqs[j]->mutable_share(0)->reshape(pivot.shape());
                    mutable_seqs[j]->mutable_share(1)->reshape(pivot.shape());
                    gap.share(0)->copy(mutable_seqs[j]->mutable_share(0));
                    gap.share(1)->copy(mutable_seqs[j]->mutable_share(1));
                }
            }
        }
    }
    std::cout << "time 4" << std::endl;
    std::vector<size_t> result_shape = {seqs_len, pivot.shape()[0]};
    ret->mutable_share(0)->reshape(result_shape);
    ret->mutable_share(1)->reshape(result_shape);
    for (size_t i = 0; i < seqs_len; ++i) {
        mutable_seqs[i]->share(0)->copy(ret->mutable_share(0)->operator[](i).get());
        mutable_seqs[i]->share(1)->copy(ret->mutable_share(1)->operator[](i).get());
    }
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::nj(const FixedPointTensor* dm, const std::vector<std::string> &ids,
                                std::string & tree) {
    PADDLE_ENFORCE_GE(dm->shape()[0], 3,
                      "Distance matrix must be at least 3x3 ");
    
    auto dm0 = tensor_factory()->template create<T>(dm->shape());
    auto dm1 = tensor_factory()->template create<T>(dm->shape());
    std::shared_ptr<FixedPointTensor<T, N> > mutable_dm =
            std::make_shared<FixedPointTensor<T, N> >(dm0.get(), dm1.get());
    dm->share(0)->copy(mutable_dm->mutable_share(0));
    dm->share(1)->copy(mutable_dm->mutable_share(1));

    std::vector<std::string> mutable_ids;
    for(auto id: ids) {
        mutable_ids.emplace_back(id);
    }

    std::string node_definition = "";
    auto shape_dm = dm->shape();
    std::vector<size_t> shape_one = {1};
    std::vector<size_t> shape_one_one = {1, 1};
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 16; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }
    for (size_t i = 16; i < 22; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(shape_one));
    }

    FixedPointTensor<T, N> q(temp[0].get(), temp[1].get());
    auto n_minus_two = tensor_factory()->template create<T>();
    n_minus_two->scaling_factor() = N;
    FixedPointTensor<T, N> dm_i(temp[2].get(), temp[3].get());
    FixedPointTensor<T, N> dm_j(temp[4].get(), temp[5].get());
    FixedPointTensor<T, N> temp_sum_tensor(temp[6].get(), temp[7].get());
    FixedPointTensor<T, N> temp_sum_result(temp[8].get(), temp[9].get());
    BooleanTensor<T> cmp_pooling(temp[10].get(), temp[11].get());
    BooleanTensor<T> cmp(temp[12].get(), temp[13].get());
    FixedPointTensor dm_temp(temp[14].get(), temp[15].get());
    
    //dm[i, k] + dm[j, k] - dm[i, j]
    FixedPointTensor dm_i_k(temp[16].get(), temp[17].get());
    FixedPointTensor k_to_u(temp[18].get(), temp[19].get());
    FixedPointTensor temp_pooling(temp[20].get(), temp[21].get());
    
    //0  0.5
    auto zero_tensor = tensor_factory()->template create<T>(shape_one);
    zero_tensor->scaling_factor() = N;
    assign_to_tensor(zero_tensor.get(), (T)(0));
    auto half_tensor = tensor_factory()->template create<T>(shape_one);
    half_tensor->scaling_factor() = N;
    assign_to_tensor(half_tensor.get(), (T)(0.5 * pow(2, N)));

    auto bool_reveal_tensor = tensor_factory()->template create<T>();
    auto fixed_reveal_tensor = tensor_factory()->template create<T>();
    fixed_reveal_tensor->scaling_factor() = N;
    
    while(mutable_dm->shape()[0] > 3) {
        //q = _compute_q(dm)
        std::cout << " nj " << mutable_dm->shape()[0] << std::endl;
        std::vector<size_t> shape_sum = {mutable_dm->shape()[0]};
        q.mutable_share(0)->reshape(mutable_dm->shape());
        q.mutable_share(1)->reshape(mutable_dm->shape());
        mutable_dm->share(0)->copy(q.mutable_share(0));
        mutable_dm->share(1)->copy(q.mutable_share(1));
        // n-2
        n_minus_two->reshape(mutable_dm->shape());
        assign_to_tensor(n_minus_two.get(), (T)((mutable_dm->shape()[0] - 2) * pow(2, N)));

        dm_i.mutable_share(0)->reshape(mutable_dm->shape());
        dm_i.mutable_share(1)->reshape(mutable_dm->shape());
        dm_j.mutable_share(0)->reshape(mutable_dm->shape());
        dm_j.mutable_share(1)->reshape(mutable_dm->shape());
        temp_sum_tensor.mutable_share(0)->reshape(shape_sum);
        temp_sum_tensor.mutable_share(1)->reshape(shape_sum);
        size_t len = mutable_dm->shape()[0];
        temp_sum_result.mutable_share(0)->reshape(shape_one);
        temp_sum_result.mutable_share(1)->reshape(shape_one);
        for (size_t i = 0; i < len; ++i) {
            // dm[i].sum
            std::copy(mutable_dm->share(0)->data() + i * len, mutable_dm->share(0)->data() + (i + 1) * len,
                      temp_sum_tensor.mutable_share(0)->data());
            std::copy(mutable_dm->share(1)->data() + i * len, mutable_dm->share(1)->data() + (i + 1) * len,
                      temp_sum_tensor.mutable_share(1)->data());
            
            temp_sum_tensor.sum(&temp_sum_result);
            for (size_t j = 0; j < len; ++j) {
                dm_i.mutable_share(0)->data()[i * len + j] = 
                                             temp_sum_result.share(0)->data()[0];
                dm_i.mutable_share(1)->data()[i * len + j] = 
                                             temp_sum_result.share(1)->data()[0];
                if (j == i) {
                    dm_j.mutable_share(0)->data()[j * len + i] = 
                                                 -temp_sum_result.share(0)->data()[0];
                    dm_j.mutable_share(1)->data()[j * len + i] = 
                                                 -temp_sum_result.share(1)->data()[0];
                } else {
                    dm_j.mutable_share(0)->data()[j * len + i] = 
                                                 temp_sum_result.share(0)->data()[0];
                    dm_j.mutable_share(1)->data()[j * len + i] = 
                                                 temp_sum_result.share(1)->data()[0];
                }
            }
        }
        q.mul(n_minus_two.get(), &q);

        q.sub(&dm_i, &q);
        q.sub(&dm_j, &q);

        /* debug
        fixed_reveal_tensor->reshape(q.shape());
        q.reveal(fixed_reveal_tensor.get());
        
        if (party() == 0) {
            std::cout << " q is " << std::endl;
            std::cout << fixed_reveal_tensor << std::endl;
        }
        */

        q.negative(&q);
        //idx1, idx2 = _lowest_index(q)

        size_t cmp_size = len * (len - 1) / 2; 
        size_t cmp_row = (len % 2)? (len - 1) / 2 : len / 2 ;
        size_t cmp_col = cmp_size / cmp_row ;
        std::vector<size_t> shape_cmp_pooling = {cmp_row, cmp_col};
        std::vector<size_t> shape_pooling_result_0 = {1, cmp_col};
        std::vector<size_t> shape_pooling_result_1 = {cmp_col, 1};
        std::vector<size_t> shape_pooling_result_2 = {cmp_row, 1};
        cmp_pooling.share(0)->reshape(shape_cmp_pooling);
        cmp_pooling.share(1)->reshape(shape_cmp_pooling);

        temp_sum_tensor.mutable_share(0)->reshape(shape_cmp_pooling);
        temp_sum_tensor.mutable_share(1)->reshape(shape_cmp_pooling);
        std::map<size_t, std::pair<size_t, size_t>> lowest_index;
        size_t count = 0;
        for (size_t i = 1; i < len; ++i) {
            for (size_t j = 0; j < i; ++j) {

                temp_sum_tensor.mutable_share(0)->data()[(count % cmp_row) * cmp_col + count / cmp_row] =
                                q.share(0)->data()[i * len + j];
                temp_sum_tensor.mutable_share(1)->data()[(count % cmp_row) * cmp_col + count / cmp_row] =
                                q.share(1)->data()[i * len + j];
                
                lowest_index[count++] = std::make_pair(i, j);
 
            }
        }

        temp_sum_result.mutable_share(0)->reshape(shape_pooling_result_0);
        temp_sum_result.mutable_share(1)->reshape(shape_pooling_result_0);
        temp_sum_tensor.max_pooling(&temp_sum_result, &cmp_pooling);

        temp_sum_result.mutable_share(0)->reshape(shape_pooling_result_1);
        temp_sum_result.mutable_share(1)->reshape(shape_pooling_result_1);
        temp_pooling.mutable_share(0)->reshape(shape_one_one);
        temp_pooling.mutable_share(1)->reshape(shape_one_one);

        cmp.share(0)->reshape(shape_pooling_result_1);
        cmp.share(1)->reshape(shape_pooling_result_1);
        temp_sum_result.max_pooling(&temp_pooling, &cmp);

        //std::cout << "max pooling end" << std::endl;

        bool_reveal_tensor->reshape(cmp.shape());
        cmp.reveal(bool_reveal_tensor.get());
        
        size_t index_row = 0;
        size_t index_col = 0;
        for (size_t i = 0; i < cmp_col ; ++i) {
            if(bool_reveal_tensor->data()[i] == 1) {
                index_col = i;
            }
        }
        cmp.share(0)->reshape(shape_pooling_result_2);
        cmp.share(1)->reshape(shape_pooling_result_2);
        for (size_t i = 0; i < cmp_row; ++i) {
            cmp.share(0)->data()[i] = cmp_pooling.share(0)->data()[ i * cmp_col + index_col];
            cmp.share(1)->data()[i] = cmp_pooling.share(1)->data()[ i * cmp_col + index_col];
        }
        bool_reveal_tensor->reshape(cmp.shape());
        cmp.reveal(bool_reveal_tensor.get());
        for (size_t i = 0; i < cmp_row; ++i) {
            if(bool_reveal_tensor->data()[i] == 1) {
                index_row = i;
            }
        }
        size_t idx = index_col * cmp_row + index_row;
        size_t idx1 = lowest_index[idx].first;
        size_t idx2 = lowest_index[idx].second;
        
        //std::cout << "idx is " << idx1 << " " << idx2 << std::endl;
        auto pair_member_1 = mutable_ids[idx1];
        auto pair_member_2 = mutable_ids[idx2];
        node_definition = "(" + pair_member_1 + ", " + pair_member_2 + ")";

        //_compute_collapsed_dm
        std::vector<size_t> shape_out = {len - 1, len - 1};
        std::vector<std::pair<size_t, std::string>> out_ids;
        out_ids.reserve(mutable_ids.size() - 1);
        out_ids.emplace_back(0, node_definition);
        for (size_t i = 0; i < mutable_ids.size(); ++i) {
            if (i != idx1 && i != idx2) {
                out_ids.emplace_back(i, mutable_ids[i]);
            }
        }
        dm_temp.mutable_share(0)->reshape(shape_out);
        dm_temp.mutable_share(1)->reshape(shape_out);
        assign_to_tensor(dm_temp.mutable_share(0), (T)(0));
        assign_to_tensor(dm_temp.mutable_share(1), (T)(0));
        //std::cout << " time1 " << std::endl;
        for (size_t i = 1; i < out_ids.size(); ++i) {

            size_t k = out_ids[i].first;
            dm_i_k.mutable_share(0)->data()[0] = 
                                mutable_dm->share(0)->data()[idx1 * len + k];
            dm_i_k.mutable_share(1)->data()[0] = 
                                mutable_dm->share(1)->data()[idx1 * len + k];
            k_to_u.mutable_share(0)->data()[0] = 
                                mutable_dm->share(0)->data()[idx2 * len + k];
            k_to_u.mutable_share(1)->data()[0] = 
                                mutable_dm->share(1)->data()[idx2 * len + k];
            k_to_u.add(&dm_i_k, &k_to_u);

            dm_i_k.mutable_share(0)->data()[0] = 
                                mutable_dm->share(0)->data()[idx1 * len + idx2];
            dm_i_k.mutable_share(1)->data()[0] = 
                                mutable_dm->share(1)->data()[idx1 * len + idx2];
            k_to_u.sub(&dm_i_k, &k_to_u);
            k_to_u.mul(half_tensor.get(), &k_to_u);
            cmp.share(0)->reshape(shape_one);
            cmp.share(1)->reshape(shape_one);
            k_to_u.geq(zero_tensor.get(), &cmp);
            cmp.mul(&k_to_u, &k_to_u);

            dm_temp.mutable_share(0)->data()[i] = k_to_u.share(0)->data()[0];
            dm_temp.mutable_share(1)->data()[i] = k_to_u.share(1)->data()[0];
            dm_temp.mutable_share(0)->data()[i * (len - 1)] = k_to_u.share(0)->data()[0];
            dm_temp.mutable_share(1)->data()[i * (len - 1)] = k_to_u.share(1)->data()[0];

            for (size_t j = 1; j < i ; ++j) {
                size_t out_id2 = out_ids[j].first;
                dm_temp.mutable_share(0)->data()[i * (len - 1) + j] =
                        mutable_dm->share(0)->data()[k * len + out_id2];
                dm_temp.mutable_share(1)->data()[i * (len - 1) + j] =
                        mutable_dm->share(1)->data()[k * len + out_id2];
                dm_temp.mutable_share(0)->data()[j * (len - 1) + i] =
                        mutable_dm->share(0)->data()[k * len + out_id2];
                dm_temp.mutable_share(1)->data()[j * (len - 1) + i] =
                        mutable_dm->share(1)->data()[k * len + out_id2];                    
            }
        }
        mutable_dm->mutable_share(0)->reshape(dm_temp.shape());
        mutable_dm->mutable_share(1)->reshape(dm_temp.shape());
        dm_temp.share(0)->copy(mutable_dm->mutable_share(0));
        dm_temp.share(1)->copy(mutable_dm->mutable_share(1));
        //std::cout << " time2 " << std::endl;
        /*
        fixed_reveal_tensor->reshape(mutable_dm->shape());
        mutable_dm->reveal(fixed_reveal_tensor.get());
        
        if (party() == 0) {
            std::cout << "dm is " << std::endl;
            std::cout << fixed_reveal_tensor << std::endl;
        }
        */
        

        mutable_ids.resize(out_ids.size());
        for(size_t i = 0; i < out_ids.size(); ++i) {
            mutable_ids[i] = out_ids[i].second;
        }
    }
    if(node_definition.empty()) {
        node_definition = mutable_ids[0];
    }
    std::string pair_member_1 = mutable_ids[1];
    std::string pair_member_2 = mutable_ids[2];
    tree = "(" + pair_member_1 + ", " + node_definition + ", " + pair_member_2 + ");";
    std::cout << "total comm is " << aby3_ctx()->network()->bytes << std::endl;
}

} // namespace aby3

#ifdef USE_CUDA
#include "./fixedpoint_tensor_imp.cu.h"
#endif // USE_CUDA
