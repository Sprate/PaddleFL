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
metrics servicer implementation
server-side(bob) 
"""

from concurrent import futures
import random
import numpy as np
import grpc

import he_utils as hu
from ..proto import metrics_pb2
from ..proto import metrics_pb2_grpc


class MpcPositiveRatioServicer(metrics_pb2_grpc.MpcPositiveRatioServicer):
    """
    Positive ratio servicer implementation
    """
    def __init__(self, features, stop_event):
        """
        load feature to server
        prams:
            features: a feature list in the shape of (sample_size, features_size)
                      e.g. [[4, 3, 1], [1, 2, 5],...,[2, 3 ,2]] (feature_size = 3)
            stop_event: control the server shutdown when the server does not 
                        participate in the protocol
        """
        self._sample_size = len(features)
        self._features = features
        self._feature_size = len(features[0])
        self._stop_event = stop_event

    def SyncSampleSize(self, request, context):
        """
        client sync sample size and feature size with server
        """
        if request.sample_size == self._sample_size:
            return metrics_pb2.Sample(sample_size = self._sample_size,
                                      feature_size = self._feature_size)
        else:
            return metrics_pb2.Sample(sample_size = -1, feature_size = 0)

    def SendPubkey(self, request, context):
        """
        client send pubkey to server
        """
        paillier = hu.Paillier()
        paillier.import_pk(request.pk)
        self._paillier = paillier
        return metrics_pb2.Status(code = 1)
    
    def GetLablesSum(self, request, context):
        """
        client get labels sum from server
        """
        batch_size = request.sample_size 
        if (self._sample_size != batch_size):
            raise ValueError("sample size not equal")
        self._enc_labels = self._paillier.batch_decode(request.labels)
        
        all_pos_sum = [] 
        all_neg_sum = []
        for feature_idx in range(self._feature_size):
            pos_sum = {}
            neg_sum = {}
            feature_bin = {}
            for sample_index in range(self._sample_size):
                feature_value = self._features[sample_index][feature_idx]
                if(feature_value in feature_bin):
                    pos_sum[feature_value] = self._paillier.homm_add(pos_sum[feature_value], 
                                                             self._enc_labels[sample_index])
                    feature_bin[feature_value] += 1
                else:
                    pos_sum[feature_value] = self._enc_labels[sample_index]
                    feature_bin[feature_value] = 1
            
            # cal neg sum
            for key, value in pos_sum.items():
                bin_size_cipher = self._paillier.encrypt_int64_t(feature_bin[key])
                neg_sum[key] = self._paillier.homm_minus(bin_size_cipher, value) 
            
            #Blind using random numbers  N/8 bits
            for key in pos_sum.keys():
                blind_r = self._paillier.get_random_bits(self._paillier.byte_len(0))
                pos_sum[key] = self._paillier.homm_mult(pos_sum[key], blind_r)
                pos_sum[key] = self._paillier.encode_cipher_bytes(pos_sum[key])
                neg_sum[key] = self._paillier.homm_mult(neg_sum[key], blind_r)
                neg_sum[key] = self._paillier.encode_cipher_bytes(neg_sum[key])
            
            all_pos_sum.append(pos_sum)
            all_neg_sum.append(neg_sum)
        
        feature_labels_sum = metrics_pb2.FeatureLabelsSum(feature_size = self._feature_size,
                                                          labels = [])
        for feature_idx in range(self._feature_size):
            bin_labels_sum = metrics_pb2.BinLabelsSum(bins_size = len(all_pos_sum[feature_idx]),
                                                      positive_sum = all_pos_sum[feature_idx],
                                                      negative_sum = all_neg_sum[feature_idx])
            feature_labels_sum.labels.append(bin_labels_sum)
        self._stop_event.set()
        return feature_labels_sum


class MpcWOEServicer(metrics_pb2_grpc.MpcWOEServicer):
    """
    woe servicer implementation
    """
    def __init__(self, features, stop_event, woe_list):
        """
        load feature to server
        prams:
            features: a feature list in the shape of (sample_size, features_size)
                      e.g. [[4, 3, 1], [1, 2, 5],...,[2, 3 ,2]] (feature_size = 3)
            stop_event: control the server shutdown when the server does not 
                        participate in the protocol
            woe_list: server store the result in woe_list
        """
        self._sample_size = len(features)
        self._features = features
        self._feature_size = len(features[0])
        self._stop_event = stop_event
        self._woe_list = woe_list

    def SyncSampleSize(self, request, context):
        """
        client sync sample size and feature size with server
        """
        if request.sample_size == self._sample_size:
            return metrics_pb2.Sample(sample_size = self._sample_size,
                                      feature_size = self._feature_size)
        else:
            return metrics_pb2.Sample(sample_size = -1, feature_size = 0)

    def SendPubkey(self, request, context):
        """
        client send pubkey to server
        """
        paillier = hu.Paillier()
        paillier.import_pk(request.pk)
        self._paillier = paillier
        return metrics_pb2.Status(code = 1)
    
    def GetLablesSum(self, request, context):
        """
        client get labels sum from server
        """
        batch_size = request.sample_size 
        if (self._sample_size != batch_size):
            raise ValueError("sample size not equal")
        self._enc_labels = self._paillier.batch_decode(request.labels)
        
        all_pos_sum = [] 
        all_neg_sum = []
        for feature_idx in range(self._feature_size):
            pos_sum = {}
            neg_sum = {}
            feature_bin = {}
            for sample_index in range(self._sample_size):
                feature_value = self._features[sample_index][feature_idx]
                if(feature_value in feature_bin):
                    pos_sum[feature_value] = self._paillier.homm_add(pos_sum[feature_value], 
                                                             self._enc_labels[sample_index])
                    feature_bin[feature_value] += 1
                else:
                    pos_sum[feature_value] = self._enc_labels[sample_index]
                    feature_bin[feature_value] = 1
            
            # cal neg sum
            for key, value in pos_sum.items():
                bin_size_cipher = self._paillier.encrypt_int64_t(feature_bin[key])
                neg_sum[key] = self._paillier.homm_minus(bin_size_cipher, value) 
            
            #Blind using random numbers  N/8 bits
            for key in pos_sum.keys():
                blind_r = self._paillier.get_random_bits(self._paillier.byte_len(0))
                pos_sum[key] = self._paillier.homm_mult(pos_sum[key], blind_r)
                pos_sum[key] = self._paillier.encode_cipher_bytes(pos_sum[key])
                neg_sum[key] = self._paillier.homm_mult(neg_sum[key], blind_r)
                neg_sum[key] = self._paillier.encode_cipher_bytes(neg_sum[key])
            
            all_pos_sum.append(pos_sum)
            all_neg_sum.append(neg_sum)
        
        feature_labels_sum = metrics_pb2.FeatureLabelsSum(feature_size = self._feature_size,
                                                          labels = [])
        for feature_idx in range(self._feature_size):
            bin_labels_sum = metrics_pb2.BinLabelsSum(bins_size = len(all_pos_sum[feature_idx]),
                                                      positive_sum = all_pos_sum[feature_idx],
                                                      negative_sum = all_neg_sum[feature_idx])
            feature_labels_sum.labels.append(bin_labels_sum)

        return feature_labels_sum
    
    def SendWOE(self, request, context):
        """
        client send woe to server
        """
        for feature_idx in range(request.feature_size):
            woe_dict = request.values[feature_idx].value_dict
            woe_dict_ = {}
            for key in woe_dict.keys():
                woe_dict_[key] = round(woe_dict[key], 6)
            self._woe_list.append(woe_dict_)
        self._stop_event.set()
        return metrics_pb2.Status(code = 1)


class MpcIVServicer(metrics_pb2_grpc.MpcIVServicer):
    """
    iv servicer implementation
    """
    def __init__(self, features, stop_event, iv_list, woe_list=[]):
        """
        load feature to server
        prams:
            features: a feature list in the shape of (sample_size, features_size)
                      e.g. [[4, 3, 1], [1, 2, 5],...,[2, 3 ,2]] (feature_size = 3)
            stop_event: control the server shutdown when the server does not 
                        participate in the protocol
            iv_list: server store the result in iv_list
            woe_list: server store the woe in woe_list
        """
        self._sample_size = len(features)
        self._features = features
        self._feature_size = len(features[0])
        self._stop_event = stop_event
        self._iv_list = iv_list
        self._woe_list = woe_list

    def SyncSampleSize(self, request, context):
        """
        client sync sample size and feature size with server
        """
        if request.sample_size == self._sample_size:
            return metrics_pb2.Sample(sample_size = self._sample_size,
                                      feature_size = self._feature_size)
        else:
            return metrics_pb2.Sample(sample_size = -1, feature_size = 0)

    def SendPubkey(self, request, context):
        """
        client send pubkey to server
        """
        paillier = hu.Paillier()
        paillier.import_pk(request.pk)
        self._paillier = paillier
        return metrics_pb2.Status(code = 1)
    
    def GetLablesSum(self, request, context):
        """
        client get labels sum from server
        """
        batch_size = request.sample_size 
        if (self._sample_size != batch_size):
            raise ValueError("sample size not equal")
        self._enc_labels = self._paillier.batch_decode(request.labels)
        
        all_pos_sum = [] 
        all_neg_sum = []
        self._all_bind_r_inv = []
        for feature_idx in range(self._feature_size):
            pos_sum = {}
            neg_sum = {}
            blind_r_inv_dict = {}
            feature_bin = {}
            for sample_index in range(self._sample_size):
                feature_value = self._features[sample_index][feature_idx]
                if(feature_value in feature_bin):
                    pos_sum[feature_value] = self._paillier.homm_add(pos_sum[feature_value], 
                                                             self._enc_labels[sample_index])
                    feature_bin[feature_value] += 1
                else:
                    pos_sum[feature_value] = self._enc_labels[sample_index]
                    feature_bin[feature_value] = 1
            
            # cal neg sum
            for key, value in pos_sum.items():
                bin_size_cipher = self._paillier.encrypt_int64_t(feature_bin[key])
                neg_sum[key] = self._paillier.homm_minus(bin_size_cipher, value) 
            
            #Blind using random numbers  N/8 bits
            for key in pos_sum.keys():
                blind_r = self._paillier.get_random_bits(self._paillier.byte_len(0))
                blind_r_inv_dict[key] = hu.mod_inv(blind_r, self._paillier.n())
                pos_sum[key] = self._paillier.homm_mult(pos_sum[key], blind_r)
                pos_sum[key] = self._paillier.encode_cipher_bytes(pos_sum[key])
                neg_sum[key] = self._paillier.homm_mult(neg_sum[key], blind_r)
                neg_sum[key] = self._paillier.encode_cipher_bytes(neg_sum[key])
            
            all_pos_sum.append(pos_sum)
            all_neg_sum.append(neg_sum)
            self._all_bind_r_inv.append(blind_r_inv_dict)
        
        feature_labels_sum = metrics_pb2.FeatureLabelsSum(feature_size = self._feature_size,
                                                          labels = [])
        for feature_idx in range(self._feature_size):
            bin_labels_sum = metrics_pb2.BinLabelsSum(bins_size = len(all_pos_sum[feature_idx]),
                                                      positive_sum = all_pos_sum[feature_idx],
                                                      negative_sum = all_neg_sum[feature_idx])
            feature_labels_sum.labels.append(bin_labels_sum)
        return feature_labels_sum
    
    def GetEncIV(self, request, context):
        """
        client get enc iv 
        """
        all_iv = []

        for feature_idx in range(request.feature_size):
            blind_iv_dict = {}
            recved_dict = request.values[feature_idx].value_dict
            iv = self._paillier.encrypt_int64_t(0)
            for key in recved_dict:
                blind_iv_dict[key] = self._paillier.decode(recved_dict[key])
                blind_iv_dict[key] = self._paillier.homm_mult(blind_iv_dict[key],
                                         self._all_bind_r_inv[feature_idx][key])
                iv = self._paillier.homm_add(iv, blind_iv_dict[key])

            all_iv.append(self._paillier.encode_cipher_bytes(iv))

        return metrics_pb2.EncFeatureMetric(feature_size = len(all_iv),
                                            values = all_iv)
        
    def SendIV(self, request, context):
        """
        client send iv to server
        """
        iv_list = request.values
        for feature_idx in range(len(iv_list)):
            self._iv_list.append(round(iv_list[feature_idx], 6))
        self._stop_event.set()
        return metrics_pb2.Status(code = 1)
    
    def SendWOE(self, request, context):
        """
        client send woe to server
        """
        for feature_idx in range(request.feature_size):
            woe_dict = request.values[feature_idx].value_dict
            woe_dict_ = {}
            for key in woe_dict.keys():
                woe_dict_[key] = round(woe_dict[key], 6)
            self._woe_list.append(woe_dict_)
        return metrics_pb2.Status(code = 1)


class MpcKSServicer(metrics_pb2_grpc.MpcKSServicer):
    """
    ks servicer implementation
    """
    def __init__(self, features, stop_event, ks_list):
        """
        load feature to server
        prams:
            features: a feature list in the shape of (sample_size, features_size)
                      e.g. [[4, 3, 1], [1, 2, 5],...,[2, 3 ,2]] (feature_size = 3)
            stop_event: control the server shutdown when the server does not 
                        participate in the protocol
            ks_list: server store the result in ks_list
        """
        self._sample_size = len(features)
        self._features = features
        self._feature_size = len(features[0])
        self._stop_event = stop_event
        self._ks_list = ks_list

    def SyncSampleSize(self, request, context):
        """
        client sync sample size and feature size with server
        """
        if request.sample_size == self._sample_size:
            return metrics_pb2.Sample(sample_size = self._sample_size,
                                      feature_size = self._feature_size)
        else:
            return metrics_pb2.Sample(sample_size = -1, feature_size = 0)

    def SendPubkey(self, request, context):
        """
        client send pubkey to server
        """
        paillier = hu.Paillier()
        paillier.import_pk(request.pk)
        self._paillier = paillier
        return metrics_pb2.Status(code = 1)
    
    def GetCumLablesSum(self, request, context):
        """
        client get cum labels sum from server
        """
        batch_size = request.sample_size 
        if (self._sample_size != batch_size):
            raise ValueError("sample size not equal")
        self._enc_labels = self._paillier.batch_decode(request.labels)
        
        all_pos_cum_sum = [] 
        all_neg_cum_sum = []
        self._all_bind_r_inv = []
        for feature_idx in range(self._feature_size):
            pos_sum = {}
            neg_sum = {}
            blind_r_inv_dict = {}
            feature_bin = {}
            for sample_index in range(self._sample_size):
                feature_value = self._features[sample_index][feature_idx]
                if(feature_value in feature_bin):
                    pos_sum[feature_value] = self._paillier.homm_add(pos_sum[feature_value], 
                                                             self._enc_labels[sample_index])
                    feature_bin[feature_value] += 1
                else:
                    pos_sum[feature_value] = self._enc_labels[sample_index]
                    feature_bin[feature_value] = 1
            
            # sort pos sum
            pos_sum = dict(sorted(pos_sum.items(), key = lambda item:item[0]))

            # cal neg sum
            for key, value in pos_sum.items():
                bin_size_cipher = self._paillier.encrypt_int64_t(feature_bin[key])
                neg_sum[key] = self._paillier.homm_minus(bin_size_cipher, value) 
            
            # cum pos and neg 
            pos_temp = self._paillier.encrypt_int64_t(0)
            neg_temp = self._paillier.encrypt_int64_t(0)
            for key in pos_sum.keys():
                pos_sum[key] = self._paillier.homm_add(pos_sum[key], pos_temp)
                pos_temp = pos_sum[key]
                neg_sum[key] = self._paillier.homm_add(neg_sum[key], neg_temp)
                neg_temp = neg_sum[key]
            
            #Blind using random numbers  N/8 bits
            for key in pos_sum.keys():
                blind_r = self._paillier.get_random_bits(self._paillier.byte_len(0))
                blind_r_inv_dict[key] = hu.mod_inv(blind_r, self._paillier.n())
                pos_sum[key] = self._paillier.homm_mult(pos_sum[key], blind_r)
                pos_sum[key] = self._paillier.encode_cipher_bytes(pos_sum[key])
                neg_sum[key] = self._paillier.homm_mult(neg_sum[key], blind_r)
                neg_sum[key] = self._paillier.encode_cipher_bytes(neg_sum[key])
            
            all_pos_cum_sum.append(pos_sum)
            all_neg_cum_sum.append(neg_sum)
            self._all_bind_r_inv.append(blind_r_inv_dict)
        
        feature_labels_sum = metrics_pb2.FeatureLabelsSum(feature_size = self._feature_size,
                                                          labels = [])
        for feature_idx in range(self._feature_size):
            bin_labels_sum = metrics_pb2.BinLabelsSum(bins_size = len(all_pos_cum_sum[feature_idx]),
                                                      positive_sum = all_pos_cum_sum[feature_idx],
                                                      negative_sum = all_neg_cum_sum[feature_idx])
            feature_labels_sum.labels.append(bin_labels_sum)
        return feature_labels_sum
    
    def GetEncKS(self, request, context):
        """
        client get enc ks 
        """
        all_ks = metrics_pb2.EncFeatureMetricList(feature_size = request.feature_size, 
                                                  values = [])
        for feature_idx in range(request.feature_size):
            blind_ks_list = []
            recved_dict = request.values[feature_idx].value_dict
            for key in recved_dict.keys():
                blind_ks = self._paillier.decode(recved_dict[key])
                blind_ks = self._paillier.homm_mult(blind_ks, 
                                                    self._all_bind_r_inv[feature_idx][key])
                blind_ks = self._paillier.encode_cipher_bytes(blind_ks)
                blind_ks_list.append(blind_ks)
            random.shuffle(blind_ks_list)
            enc_bin_metric_list = metrics_pb2.EncBinMetricList(bins_size = len(blind_ks_list),
                                                               value = blind_ks_list)
            all_ks.values.append(enc_bin_metric_list)

        return all_ks
        
    def SendKS(self, request, context):
        """
        client send ks to server
        """
        ks_list = request.values
        for feature_idx in range(len(ks_list)):
            self._ks_list.append(round(ks_list[feature_idx], 6))
        self._stop_event.set()
        return metrics_pb2.Status(code = 1)


class MpcAUCServicer(metrics_pb2_grpc.MpcAUCServicer):
    """
    auc servicer implementation
    """
    def __init__(self, features, stop_event, auc_list, num_thresholds=4095):
        """
        load feature to server
        prams:
            features: a feature list in the shape of (sample_size, features_size)
                      e.g. [[4, 3, 1], [1, 2, 5],...,[2, 3 ,2]] (feature_size = 3)
            stop_event: control the server shutdown when the server does not 
                        participate in the protocol
            auc_list: server store the result in auc_list
        """
        self._sample_size = len(features)
        self._features = features
        self._feature_size = len(features[0])
        self._stop_event = stop_event
        self._auc_list = auc_list
        self._num_thresholds = num_thresholds

    def SyncSampleSize(self, request, context):
        """
        client sync sample size and feature size with server
        """
        if request.sample_size == self._sample_size:
            return metrics_pb2.Sample(sample_size = self._sample_size,
                                      feature_size = self._feature_size)
        else:
            return metrics_pb2.Sample(sample_size = -1, feature_size = 0)

    def SendPubkey(self, request, context):
        """
        client send pubkey to server
        """
        paillier = hu.Paillier()
        paillier.import_pk(request.pk)
        self._paillier = paillier
        return metrics_pb2.Status(code = 1)
    
    def GetLablesSum(self, request, context):
        """
        client get blind auc from server
        """
        batch_size = request.sample_size 
        if (self._sample_size != batch_size):
            raise ValueError("sample size not equal")
        self._enc_labels = self._paillier.batch_decode(request.labels)
        
        all_pos_sum = [] 
        all_neg_sum = []
        self._all_auc_blind = []
        for feature_idx in range(self._feature_size):
            stat_pos_sum = {}
            stat_neg_sum = {}
            feature_bin = {}
            feature_values = [val[feature_idx] for val in self._features]
            Max = np.max(feature_values)
            Min = np.min(feature_values)
            feature_values = (feature_values - Min) / (Max - Min)
            for sample_index in range(self._sample_size):
                bin_idx = int(feature_values[sample_index] * self._num_thresholds)
                if(bin_idx in feature_bin):
                    stat_pos_sum[bin_idx] = self._paillier.homm_add(stat_pos_sum[bin_idx], 
                                                                    self._enc_labels[sample_index])
                    feature_bin[bin_idx] += 1
                else:
                    stat_pos_sum[bin_idx] = self._enc_labels[sample_index]
                    feature_bin[bin_idx] = 1
            
            # sort pos sum
            stat_pos_sum = dict(sorted(stat_pos_sum.items(), key = lambda item:item[0], reverse=True))

            # cal neg sum
            for key, value in stat_pos_sum.items():
                bin_size_cipher = self._paillier.encrypt_int64_t(feature_bin[key])
                stat_neg_sum[key] = self._paillier.homm_minus(bin_size_cipher, value) 
            
            # cal blind_auc and blind res
            tot_pos = self._paillier.encrypt_int64_t(0)
            tot_neg = self._paillier.encrypt_int64_t(0)
            tot_blind = self._paillier.encrypt_int64_t(0)
            for key in stat_pos_sum.keys():
                tot_pos_prev = tot_pos
                tot_neg_prev = tot_neg
                tot_pos = self._paillier.homm_add(stat_pos_sum[key], tot_pos)
                tot_neg = self._paillier.homm_add(stat_neg_sum[key], tot_neg)
        
                neg_temp = self._paillier.homm_minus(tot_neg, tot_neg_prev)
                pos_temp = self._paillier.homm_add(tot_pos, tot_pos_prev)
                neg_blind_r = self._paillier.get_random_bits(self._paillier.byte_len(0))
                pos_blind_r = self._paillier.get_random_bits(self._paillier.byte_len(0))

                cipher_temp = self._paillier.homm_mult(neg_temp, pos_blind_r)
                tot_blind = self._paillier.homm_add(tot_blind, cipher_temp)
                cipher_temp = self._paillier.homm_mult(pos_temp, neg_blind_r)
                tot_blind = self._paillier.homm_add(tot_blind, cipher_temp)
                cipher_temp = self._paillier.homm_mult(self._paillier.encrypt(neg_blind_r), pos_blind_r)
                tot_blind = self._paillier.homm_add(tot_blind, cipher_temp)
                
                neg_temp = self._paillier.homm_add(neg_temp, self._paillier.encrypt(neg_blind_r))
                pos_temp = self._paillier.homm_add(pos_temp, self._paillier.encrypt(pos_blind_r))

                stat_pos_sum[key] = self._paillier.encode_cipher_bytes(pos_temp)
                stat_neg_sum[key] = self._paillier.encode_cipher_bytes(neg_temp)

            all_pos_sum.append(stat_pos_sum)
            all_neg_sum.append(stat_neg_sum)
            self._all_auc_blind.append(tot_blind)
        
        feature_labels_sum = metrics_pb2.FeatureLabelsSum(feature_size = self._feature_size,
                                                          labels = [])
        for feature_idx in range(self._feature_size):
            bin_labels_sum = metrics_pb2.BinLabelsSum(bins_size = len(all_pos_sum[feature_idx]),
                                                      positive_sum = all_pos_sum[feature_idx],
                                                      negative_sum = all_neg_sum[feature_idx])
            feature_labels_sum.labels.append(bin_labels_sum)
        return feature_labels_sum
    
    def GetEncAUC(self, request, context):
        """
        client get enc auc 
        """
        all_auc_ = []
        for feature_idx in range(request.feature_size):
            enc_auc = self._paillier.decode(request.values[feature_idx])
            enc_auc = self._paillier.homm_minus(enc_auc, self._all_auc_blind[feature_idx])
            all_auc_.append(self._paillier.encode_cipher_bytes(enc_auc))

        all_auc = metrics_pb2.EncFeatureMetric(feature_size = request.feature_size, 
                                               values = all_auc_)
        return all_auc
        
    def SendAUC(self, request, context):
        """
        client send auc to server
        """
        auc_list = request.values
        for feature_idx in range(len(auc_list)):
            self._auc_list.append(round(auc_list[feature_idx], 6))
        self._stop_event.set()
        return metrics_pb2.Status(code = 1)