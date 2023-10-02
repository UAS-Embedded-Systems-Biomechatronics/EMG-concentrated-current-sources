# Copyright 2023 Malte Mechtenberg
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
# ==============================================================================

import unittest
import pickle
from emg_hom_iso_unbound import MUAP_trains, model_config

import numpy as np


class test_MUAP_trains(unittest.TestCase):

    def test_gen_time_vec(self):
        t_events = np.array([0.2, 1.2])
        t_template = np.arange(0, 0.1, 0.001)

        full_time, dt = MUAP_trains.gen_time_vec(
            t_events, t_template)

        self.assertEqual(dt, 0.001)
        self.assertEqual(full_time[-1], 1.2+t_template[-1])

    def test_time_shift_template(self):
        data = np.zeros(shape =(2,1,1,5))
        data[0,0,0,:] = [ 1, 2, 3, 4, 5]
        data[1,0,0,:] = [10,20,30,40,50]

        dt      = 0.001
        t_event = 0.2

        muap_array = np.zeros(shape=(2,1,1, 400))

        muap_array = MUAP_trains.time_shift_template(
            data , t_event , dt ,muap_array)

        #import pdb; pdb.set_trace()
        n_orig_shift = int(t_event/dt)

        self.assertEqual(muap_array[0,0,0, n_orig_shift],   1)
        self.assertEqual(muap_array[0,0,0, n_orig_shift+4], 5)

        self.assertEqual(muap_array[1,0,0, n_orig_shift],   10)
        self.assertEqual(muap_array[1,0,0, n_orig_shift+4], 50)


        t_event = t_event + dt
        #import pdb; pdb.set_trace()
        muap_array = MUAP_trains.time_shift_template(
            data , t_event , dt ,muap_array)

        self.assertEqual(muap_array[0,0,0, n_orig_shift],   1)
        self.assertEqual(muap_array[0,0,0, n_orig_shift+1], 3)
        self.assertEqual(muap_array[0,0,0, n_orig_shift+4], 9)
        self.assertEqual(muap_array[0,0,0, n_orig_shift+5], 5)

        self.assertEqual(muap_array[1,0,0, n_orig_shift],   10)
        self.assertEqual(muap_array[1,0,0, n_orig_shift+2],   50)
        self.assertEqual(muap_array[1,0,0, n_orig_shift+5], 50)

    def test_gen_muap_for_firing_train(self):
        dt      = 0.001
        t_event = 0.2
        n_orig_shift = int(t_event/dt)

        mu_config = model_config.motorUnit(
            firing_pattern = [t_event, t_event+dt])

        data = np.zeros(shape =(2,1,1,5))
        data[0,0,0,:] = [ 1, 2, 3, 4, 5]
        data[1,0,0,:] = [10,20,30,40,50]

        time_template = np.arange(0, 5*dt, dt)

        (time_array, muap_array)  = MUAP_trains.gen_muap_for_firing_train( 
            mu_config, data, time_template,)

        self.assertEqual(muap_array[0,0,0, n_orig_shift],   1)
        self.assertEqual(muap_array[0,0,0, n_orig_shift+1], 3)
        self.assertEqual(muap_array[0,0,0, n_orig_shift+4], 9)
        self.assertEqual(muap_array[0,0,0, n_orig_shift+5], 5)

        self.assertEqual(muap_array[1,0,0, n_orig_shift],   10)
        self.assertEqual(muap_array[1,0,0, n_orig_shift+2],   50)
        self.assertEqual(muap_array[1,0,0, n_orig_shift+5], 50)







if __name__ == "__main__":
    unittest.main()


