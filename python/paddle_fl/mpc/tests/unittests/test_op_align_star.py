#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.
"""
This module test add op.

"""
import unittest
from multiprocessing import Manager

import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')


class TestOpAlignStar(test_op_base.TestOpBase):
    """R
    """

    def align_star(self, **kwargs):
        """
        Add two variables with one dimension.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        lod = kwargs['lod']
        return_result = kwargs['expect_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[sum(lod[0])], dtype='int64')
        lod_t = fluid.data(name='lod', shape=lod.shape, dtype='int64')
        out = pfl_mpc.layers.align_star(x=x, lod=lod_t)
        
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'lod': lod}, fetch_list=[out])

        return_result.append(results[0])

        #self.assertTrue(np.allclose(results[0], results[1]))
        #self.assertEqual(results[0].shape, (5, 5))
        #self.assertTrue(np.allclose(results[0], expected_out))


    def test_align_star(self):
        """R
        """
        data_1 = [[1, 5, 3, 1, 3, 1, 5],
                [1, 3, 3, 1, 3, 1, 3, 5],
                [3, 3, 1, 3, 3],
                [1, 3, 3, 3, 1, 3, 5],
                [1, 3, 5, 1, 2]]

        data_1_lod = np.array([[len(seq) for seq in data_1]]).astype("int64")
        data_1_flatten = [item for sublist in data_1 for item in sublist]
        data_1 = np.array(data_1_flatten).astype("int64")
        data_1 = data_1.astype("int64")
        expect_results = np.array([[1, 5, 3, 1, 3, 1, 6, 5],
                                    [1, 3, 3, 1, 3, 1, 3, 5],
                                    [6, 3, 3, 1, 3, 6, 3, 6],
                                    [1, 3, 3, 6, 3, 1, 3, 5],
                                    [1, 6, 3, 6, 5, 1, 6, 2]]).astype('int64')

        data_1_shares = aby3.make_shares(data_1)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
         
        return_results = Manager().list()

        ret = self.multi_party_run(target=self.align_star,
                                   data_1=data_1_all3shares,
                                   lod=data_1_lod,
                                   expect_results=return_results)
        print(ret[0])
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        self.assertEqual(revealed.shape, expect_results.shape)
        self.assertTrue(np.allclose(revealed, expected_out))

if __name__ == '__main__':
    unittest.main()
