# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
metrcis client implementation
client-side(alice)
"""

import math
import random
import numpy as np
import grpc

import he_utils as hu
from ..proto import metrics_pb2_grpc
from ..proto import metrics_pb2

__all__ = [
    'get_mpc_postive_ratio_alice',
    'get_mpc_woe_alice',
    'get_mpc_iv_alice',
    'get_mpc_ks_alice',
    'get_mpc_auc_alice'
]

def get_mpc_postive_ratio_alice(channel, labels, paillier):
    """
    reutrn mpc postive ratio to alice
    params:
        channel: grpc client-side channel with server
        labels: a list in the shape of (sample_size, 1)
                labels[i] is either 0 or 1, represents negative and positive resp.
                e.g. [[1], [0], [1],...,[1]]
        paillier: paillier instance
    return:
        a positive ratio list including feature_size dicts
        each dict represents the positive ratio (float) of each feature value
        e.g. [{0: 0.2, 1: 0.090909}, {1: 0.090909, 0: 0.2, 2: 0.02439}...]   
    """
    stub = metrics_pb2_grpc.MpcPositiveRatioStub(channel)

    request = metrics_pb2.Sample(sample_size = len(labels), feature_size = 0)
    sample = stub.SyncSampleSize(request)
    if (sample.sample_size == -1):
        raise ValueError("Sample size of Alice and Bob not equal")
    
    pubkey = paillier.export_pk_bytes()
    status = stub.SendPubkey(metrics_pb2.Pubkey(pk = pubkey))
    if (status.code != 1):
        raise ValueError("Bob receive pubkey failed")
    
    # encrypt labels and get labels sum
    labels = [item for sublist in labels for item in sublist] 
    labels_cipher = paillier.batch_encrypt_int64_t(labels)
    encode_labels_cipher = paillier.batch_encode_cipher_bytes(labels_cipher)
    enc_samples_labels = metrics_pb2.EncSampleLabels(sample_size = len(encode_labels_cipher),
                                                        labels = encode_labels_cipher)
    feature_labels_sum = stub.GetLablesSum(enc_samples_labels)
    
    # cal pos ratio
    all_pos_ratio_list = []

    for feature_idx in range(feature_labels_sum.feature_size):
        pos_sum = feature_labels_sum.labels[feature_idx].positive_sum
        neg_sum = feature_labels_sum.labels[feature_idx].negative_sum
        pos_ratio_dict = {}
        for key in pos_sum.keys():
            blind_pos_sum_cipher = paillier.decode(pos_sum[key])
            blind_neg_sum_cipher = paillier.decode(neg_sum[key])
            blind_pos_sum = paillier.decrypt(blind_pos_sum_cipher)
            blind_neg_sum = paillier.decrypt(blind_neg_sum_cipher)
            pos_ratio_dict[key] = round(hu.cal_pos_ratio(blind_pos_sum, blind_neg_sum), 6)
        all_pos_ratio_list.append(pos_ratio_dict)
    
    return all_pos_ratio_list


def get_mpc_woe_alice(channel, labels, paillier):
    """
    reutrn mpc woe to alice
    params:
        channel: grpc client-side channel with server
        labels: a list in the shape of (sample_size, 1)
                labels[i] is either 0 or 1, represents negative and positive resp.
                e.g. [[1], [0], [1],...,[1]]
        paillier: paillier instance
    return:
        a woe list including feature_size dicts
        each dict represents the woe (float) of each feature value
        e.g. [{1: 0.0, 0: 0.916291}, {2: -1.386294, 1: 0.0, 0: 0.916291}]  
    """
    stub = metrics_pb2_grpc.MpcWOEStub(channel)

    request = metrics_pb2.Sample(sample_size = len(labels), feature_size = 0)
    sample = stub.SyncSampleSize(request)
    if (sample.sample_size == -1):
        raise ValueError("Sample size of Alice and Bob not equal")
    
    pubkey = paillier.export_pk_bytes()
    status = stub.SendPubkey(metrics_pb2.Pubkey(pk = pubkey))
    if (status.code != 1): 
        raise ValueError("Bob receive pubkey failed")
    
    # encrypt labels and get labels sum
    labels = [item for sublist in labels for item in sublist] 
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0:
        raise ValueError("labels are all negative")
    if total_neg == 0:
        raise ValueError("labels are all positive")

    labels_cipher = paillier.batch_encrypt_int64_t(labels)
    encode_labels_cipher = paillier.batch_encode_cipher_bytes(labels_cipher)
    enc_samples_labels = metrics_pb2.EncSampleLabels(sample_size = len(encode_labels_cipher),
                                                        labels = encode_labels_cipher)
    feature_labels_sum = stub.GetLablesSum(enc_samples_labels)
    
    _feature_size = feature_labels_sum.feature_size

    # cal woe
    all_woe_list = []

    for feature_idx in range(_feature_size):
        pos_sum = feature_labels_sum.labels[feature_idx].positive_sum
        neg_sum = feature_labels_sum.labels[feature_idx].negative_sum
        woe_dict = {}
        for key in pos_sum.keys():
            blind_pos_sum_cipher = paillier.decode(pos_sum[key])
            blind_neg_sum_cipher = paillier.decode(neg_sum[key])
            blind_pos_sum = paillier.decrypt(blind_pos_sum_cipher)
            blind_neg_sum = paillier.decrypt(blind_neg_sum_cipher)
            res = hu.cal_woe(blind_pos_sum, blind_neg_sum, total_pos, total_neg)
            woe_dict[key] = round(res, 6)
            
        all_woe_list.append(woe_dict)
    
    woe_metric_dict = metrics_pb2.FeatureMetricDict(feature_size = len(all_woe_list),
                                                    values = [])
    for feature_idx in range(_feature_size):
        bin_metric = metrics_pb2.BinMetric(bins_size = len(all_woe_list[feature_idx]),
                                            value_dict = all_woe_list[feature_idx])
        woe_metric_dict.values.append(bin_metric)
    
    # send woe
    status = stub.SendWOE(woe_metric_dict)
    if (status.code != 1):
        raise ValueError("Bob receive woe failed")
    
    return all_woe_list


def get_mpc_iv_alice(channel, labels, paillier, get_woe=False):
    """
    reutrn mpc iv to alice
    params:
        channel: grpc client-side channel with server
        labels: a list in the shape of (sample_size, 1)
                labels[i] is either 0 or 1, represents negative and positive resp.
                e.g. [[1], [0], [1],...,[1]]
        paillier: paillier instance
        get_woe: whether return woe
    return:
        get_woe = False:
        a list corresponding to the iv of each feature
        e.g. [0.56653, 0.56653]
        get_woe = True:
        an tuple of woe and iv
    """
    stub = metrics_pb2_grpc.MpcIVStub(channel)
    request = metrics_pb2.Sample(sample_size = len(labels), feature_size = 0)
    sample = stub.SyncSampleSize(request)
    if (sample.sample_size == -1):
        raise ValueError("Sample size of Alice and Bob not equal")
        
    pubkey = paillier.export_pk_bytes()
    status = stub.SendPubkey(metrics_pb2.Pubkey(pk = pubkey))
    if (status.code != 1): 
        raise ValueError("Bob receive pubkey failed")
        
    # encrypt labels and get labels sum
    labels = [item for sublist in labels for item in sublist]

    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0:
        raise ValueError("labels are all negative")
    if total_neg == 0:
        raise ValueError("labels are all positive")

    labels_cipher = paillier.batch_encrypt_int64_t(labels)
    encode_labels_cipher = paillier.batch_encode_cipher_bytes(labels_cipher)
    enc_samples_labels = metrics_pb2.EncSampleLabels(sample_size = len(encode_labels_cipher),
                                                         labels = encode_labels_cipher)
    feature_labels_sum = stub.GetLablesSum(enc_samples_labels)
        
    _feature_size = feature_labels_sum.feature_size
        
    # cal iv
    all_blind_iv_list = []
    all_woe_list = []

    for feature_idx in range(_feature_size):
        pos_sum = feature_labels_sum.labels[feature_idx].positive_sum
        neg_sum = feature_labels_sum.labels[feature_idx].negative_sum
        blind_iv_dict = {}
        woe_dict = {}
        for key in pos_sum.keys():
            blind_pos_sum_cipher = paillier.decode(pos_sum[key])
            blind_neg_sum_cipher = paillier.decode(neg_sum[key])
            blind_pos_sum = paillier.decrypt(blind_pos_sum_cipher)
            blind_neg_sum = paillier.decrypt(blind_neg_sum_cipher)
            res = hu.cal_woe(blind_pos_sum, blind_neg_sum, total_pos, total_neg)
            woe_dict[key] = round(res, 6)
            blind_iv = hu.cal_blind_iv(blind_pos_sum, blind_neg_sum, total_pos, total_neg)
            blind_iv_cipher = paillier.encode_cipher_bytes(paillier.encrypt(blind_iv))
            blind_iv_dict[key] = blind_iv_cipher

        all_woe_list.append(woe_dict)
        all_blind_iv_list.append(blind_iv_dict)
    
    if (get_woe):
        woe_metric_dict = metrics_pb2.FeatureMetricDict(feature_size = len(all_woe_list),
                                                        values = [])
        for feature_idx in range(_feature_size):
            bin_metric = metrics_pb2.BinMetric(bins_size = len(all_woe_list[feature_idx]),
                                                value_dict = all_woe_list[feature_idx])
            woe_metric_dict.values.append(bin_metric)
    
        # send woe
        status = stub.SendWOE(woe_metric_dict)
        if (status.code != 1):
            raise ValueError("Bob receive woe failed")

    enc_iv_metric_dict = metrics_pb2.EncFeatureMetricDict(feature_size = len(all_blind_iv_list),
                                                            values = [])
    for feature_idx in range(_feature_size):
        enc_bin_metric = metrics_pb2.EncBinMetric(bins_size = len(all_blind_iv_list[feature_idx]),
                                                    value_dict = all_blind_iv_list[feature_idx])
        enc_iv_metric_dict.values.append(enc_bin_metric)
    
    enc_iv_metric = stub.GetEncIV(enc_iv_metric_dict)
    plain_iv = paillier.batch_decode(enc_iv_metric.values)
    
    for feature_idx in range(len(plain_iv)):
        plain_iv[feature_idx] = hu.cal_unblind_iv(paillier.decrypt(plain_iv[feature_idx]))
        plain_iv[feature_idx] = round(plain_iv[feature_idx], 6)

    # send iv
    status = stub.SendIV(metrics_pb2.FeatureMetric(feature_size = len(plain_iv),
                                                    values = plain_iv))
    if (status.code != 1):
        raise ValueError("Bob receive iv failed")
    if (get_woe):
        return all_woe_list, plain_iv
    else:
        return plain_iv


def get_mpc_ks_alice(channel, labels, paillier):
    """
    reutrn mpc ks to alice
    params:
        channel: grpc client-side channel with server
        labels: a list in the shape of (sample_size, 1)
                labels[i] is either 0 or 1, represents negative and positive resp.
                e.g. [[1], [0], [1],...,[1]]
        paillier: paillier instance
    return:
        a list corresponding to the ks of each feature
        e.g. [0.3, 0.3]
    """
    stub = metrics_pb2_grpc.MpcKSStub(channel)
    request = metrics_pb2.Sample(sample_size = len(labels), feature_size = 0)
    sample = stub.SyncSampleSize(request)
    if (sample.sample_size == -1):
        raise ValueError("Sample size of Alice and Bob not equal")
        
    pubkey = paillier.export_pk_bytes()
    status = stub.SendPubkey(metrics_pb2.Pubkey(pk = pubkey))
    if (status.code != 1): 
        raise ValueError("Bob receive pubkey failed")
        
    # encrypt labels and get labels sum
    labels = [item for sublist in labels for item in sublist]

    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0:
        raise ValueError("labels are all negative")
    if total_neg == 0:
        raise ValueError("labels are all positive")

    labels_cipher = paillier.batch_encrypt_int64_t(labels)
    encode_labels_cipher = paillier.batch_encode_cipher_bytes(labels_cipher)
    enc_samples_labels = metrics_pb2.EncSampleLabels(sample_size = len(encode_labels_cipher),
                                                         labels = encode_labels_cipher)
    cum_labels_sum = stub.GetCumLablesSum(enc_samples_labels)
        
    _feature_size = cum_labels_sum.feature_size
        
    # cal ks
    all_blind_ks_list = []

    for feature_idx in range(_feature_size):
        pos_sum = cum_labels_sum.labels[feature_idx].positive_sum
        neg_sum = cum_labels_sum.labels[feature_idx].negative_sum
        blind_ks_dict = {}
        for key in pos_sum.keys():
            blind_pos_sum_cipher = paillier.decode(pos_sum[key])
            blind_neg_sum_cipher = paillier.decode(neg_sum[key])
            blind_pos_sum = paillier.decrypt(blind_pos_sum_cipher)
            blind_neg_sum = paillier.decrypt(blind_neg_sum_cipher)
            blind_ks = hu.cal_blind_ks(blind_pos_sum, blind_neg_sum, total_pos, total_neg)
            blind_ks_cipher = paillier.encode_cipher_bytes(paillier.encrypt(blind_ks))
            blind_ks_dict[key] = blind_ks_cipher

        all_blind_ks_list.append(blind_ks_dict)
    
    enc_ks_metric_dict = metrics_pb2.EncFeatureMetricDict(feature_size = len(all_blind_ks_list),
                                                            values = [])
    for feature_idx in range(_feature_size):
        enc_bin_metric = metrics_pb2.EncBinMetric(bins_size = len(all_blind_ks_list[feature_idx]),
                                                    value_dict = all_blind_ks_list[feature_idx])
        enc_ks_metric_dict.values.append(enc_bin_metric)
    
    enc_ks_metric_list = stub.GetEncKS(enc_ks_metric_dict)
    plain_ks = []
    for feature_idx in range(_feature_size):
        plain_bin_ks = paillier.batch_decode(enc_ks_metric_list.values[feature_idx].value)
        for idx in range(len(plain_bin_ks)):
            plain_bin_ks[idx] = paillier.decrypt(plain_bin_ks[idx])
        plain_ks.append(round(hu.cal_max_ks(plain_bin_ks), 6))

    # send ks
    status = stub.SendKS(metrics_pb2.FeatureMetric(feature_size = len(plain_ks),
                                                    values = plain_ks))
    if (status.code != 1):
        raise ValueError("Bob receive ks failed")

    return plain_ks


def get_mpc_auc_alice(channel, labels, paillier):
    """
    reutrn mpc auc to alice
    params:
        channel: grpc client-side channel with server
        labels: a list in the shape of (sample_size, 1)
                labels[i] is either 0 or 1, represents negative and positive resp.
                e.g. [[1], [0], [1],...,[1]]
        paillier: paillier instance
    return:
        a list corresponding to the auc of each feature
        e.g. [0.33, 0.33]
    """
    stub = metrics_pb2_grpc.MpcAUCStub(channel)
    request = metrics_pb2.Sample(sample_size = len(labels), feature_size = 0)
    sample = stub.SyncSampleSize(request)
    if (sample.sample_size == -1):
        raise ValueError("Sample size of Alice and Bob not equal")
        
    pubkey = paillier.export_pk_bytes()
    status = stub.SendPubkey(metrics_pb2.Pubkey(pk = pubkey))
    if (status.code != 1): 
        raise ValueError("Bob receive pubkey failed")
        
    # encrypt labels and get labels sum
    labels = [item for sublist in labels for item in sublist]

    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0:
        raise ValueError("labels are all negative")
    if total_neg == 0:
        raise ValueError("labels are all positive")

    labels_cipher = paillier.batch_encrypt_int64_t(labels)
    encode_labels_cipher = paillier.batch_encode_cipher_bytes(labels_cipher)
    enc_samples_labels = metrics_pb2.EncSampleLabels(sample_size = len(encode_labels_cipher),
                                                         labels = encode_labels_cipher)
    auc_labels_sum = stub.GetLablesSum(enc_samples_labels)
        
    _feature_size = auc_labels_sum.feature_size
        
    all_blind_auc = []
    for feature_idx in range(_feature_size):
        stat_pos_sum = auc_labels_sum.labels[feature_idx].positive_sum
        stat_neg_sum = auc_labels_sum.labels[feature_idx].negative_sum
        blind_pos_sum = []
        blind_neg_sum = []
        for key in stat_pos_sum.keys():
            blind_pos_sum_cipher = paillier.decode(stat_pos_sum[key])
            blind_neg_sum_cipher = paillier.decode(stat_neg_sum[key])
            blind_pos_sum.append(paillier.decrypt(blind_pos_sum_cipher))
            blind_neg_sum.append(paillier.decrypt(blind_neg_sum_cipher))
        
        blind_auc = hu.cal_blind_auc(blind_pos_sum, blind_neg_sum)
        blind_auc_bytes = paillier.encode_cipher_bytes(paillier.encrypt(blind_auc))
        all_blind_auc.append(blind_auc_bytes)
    
    enc_auc_metric = metrics_pb2.EncFeatureMetric(feature_size = len(all_blind_auc),
                                                  values = all_blind_auc)
                                                          
    all_enc_auc = stub.GetEncAUC(enc_auc_metric)
    plain_auc = paillier.batch_decode(all_enc_auc.values)
    for feature_idx in range(len(plain_auc)):
        plain_auc[feature_idx] = paillier.decrypt(plain_auc[feature_idx])
    plain_auc = hu.cal_unblind_auc(plain_auc, total_pos, total_neg)

    plain_auc = [round(val, 6) for val in plain_auc]

    # send auc
    status = stub.SendAUC(metrics_pb2.FeatureMetric(feature_size = len(plain_auc),
                                                    values = plain_auc))
    if (status.code != 1):
        raise ValueError("Bob receive auc failed")

    return plain_auc