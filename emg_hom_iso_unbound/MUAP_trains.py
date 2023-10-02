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

from emg_hom_iso_unbound import model_config
import numpy as np

from typing import Tuple

def gen_muap_for_firing_train(
          motor_unit_config    : model_config.motorUnit
        , motor_unit_potential : np.ndarray
        , time_motor_unit_potential : np.ndarray):
    """
    Paramters:
        - motor_unit_potential:
            must be an numpy array with the 
            shape of (N_elec_x, N_elec_y, N_elec_z, N_time)
        - time_motor_unit_potential:
            must be an numpy array 
            with the shape of (N_time)
    """

    t_array, dt = gen_time_vec(
          np.array(motor_unit_config.firing_pattern)
        , time_motor_unit_potential)

    muap_array = np.zeros(
        shape=motor_unit_potential.shape[0:3] + (len(t_array),)
    )


    for event in motor_unit_config.firing_pattern:
        muap_array = time_shift_template(
            motor_unit_potential, event, dt, muap_array)

    return (t_array, muap_array)





def time_shift_template(
          template : np.ndarray
        , t_event : float 
        , dt : float
        , muap_array : np.ndarray) -> np.ndarray:

    n_shift = int(t_event / dt)

    muap_array[:,:,:,n_shift:n_shift+template.shape[3]] += template

    return muap_array

def gen_time_vec(
              t_events: np.ndarray
            , t_template : np.ndarray) -> Tuple[np.ndarray, float]:

    t_event_last    = t_events[-1]
    t_template_last = t_template[-1]

    t_last = t_event_last + t_template_last
    dt     = t_template[1] - t_template[0]

    t_array = np.arange(0,t_last+dt, dt)

    return (t_array, dt)
