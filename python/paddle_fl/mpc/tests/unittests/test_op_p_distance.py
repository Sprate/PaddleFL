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


class TestOpPDistance(test_op_base.TestOpBase):
    """R
    """

    def p_distance(self, **kwargs):
        """
        Add two variables with one dimension.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        d_3 = kwargs['data_3'][role]
        return_result = kwargs['expect_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        print("1")
        x = pfl_mpc.data(name='x', shape=[5, 8], dtype='int64')
        print("1")
        y = pfl_mpc.data(name='y', shape=[5, 8], dtype='int64')
        print("1")
        miss = pfl_mpc.data(name='miss', shape=[3, 8], dtype='int64')
        print("1")
        out = pfl_mpc.layers.p_distance(x=x, y=y, miss=miss)
        print("1")
        
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2, 'miss': d_3}, fetch_list=[out])

        return_result.append(results[0])

        #self.assertTrue(np.allclose(results[0], results[1]))
        #self.assertEqual(results[0].shape, (5, 5))
        #self.assertTrue(np.allclose(results[0], expected_out))


    def test_p_distance(self):
        """R
        """
        data_1 = np.array([[1, 5, 3, 1, 3, 1, 6, 5],
                            [1, 3, 3, 1, 3, 1, 3, 5],
                            [6, 3, 3, 1, 3, 6, 3, 6],
                            [1, 3, 3, 6, 3, 1, 3, 5],
                            [1, 6, 3, 6, 5, 1, 6, 2]]).astype('int64')
        data_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [4, 4, 4, 4, 4, 4, 4, 4],
                            [7, 7, 7, 7, 7, 7, 7, 7]]).astype('int64')
        expect_results = np.array([[0., 0.25, 0.625, 0.375, 0.5],
                                    [0.25, 0., 0.375, 0.125, 0.625],
                                    [0.625, 0.375, 0., 0.5, 0.875],
                                    [0.375, 0.125, 0.5, 0., 0.5],
                                    [0.5, 0.625, 0.875, 0.5, 0.]])

        data_1_shares = aby3.make_shares(data_1)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])

        data_2_shares = aby3.make_shares(data_1)
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])

        data_3_shares = aby3.make_shares(data_2)
        data_3_all3shares = np.array([aby3.get_shares(data_3_shares, i) for i in range(3)])
         
        return_results = Manager().list()

        ret = self.multi_party_run(target=self.p_distance,
                                   data_1=data_1_all3shares,
                                   data_2=data_2_all3shares,
                                   data_3=data_3_all3shares,
                                   expect_results=return_results)
        print(ret[0])
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        self.assertEqual(revealed.shape, (5, 5))
        self.assertTrue(np.allclose(revealed, expected_out))

if __name__ == '__main__':
    unittest.main()
