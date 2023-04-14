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

from emg_hom_iso_unbound import sim_infrastructure, model_config

import numpy as np

import os

import pickle

class demoOneFiber(object):

    def __init__(self):
        conf_m = model_config.muscle()

        motorUnit1 = model_config.motorUnit()
        motorUnit1.fibers.append(model_config.muscleFiber())
        motorUnit1.fibers[0].L = 14e-3*2
        motorUnit1.fibers[0].IP = motorUnit1.fibers[0].L / 3

        conf_m.motor_units.append(motorUnit1)

        max_time = (motorUnit1.fibers[0].L / 2 + 14e-3) / motorUnit1.fibers[0].v
        time = np.linspace(0, max_time, 10000)

        electrode_matrix = model_config.electrodes(   x = np.arange( -motorUnit1.fibers[0].L*0.05 
                                                           , motorUnit1.fibers[0].L*1.05 
                                                           , 0.0005)
                                          , y = np.arange(-0.005, 0.005,   0.0005)
                                          , z = np.arange(-0.0015, 0.0015, 0.0005))

        root_conf = model_config.root(  muscle     = conf_m
                , time       = time
                , electrodes = electrode_matrix 
		, export_vtk = False
		, export_npz = True)

        path = os.path.join('.', 'demo_one_Fiber')
        
        try:
            os.mkdir(path)
        except :
            pass

        simJobs = sim_infrastructure. \
                simJobsByModelConfig(  
                   config=root_conf
                 , project_dir=path)

        sim_infrastructure. \
                execute_sim_jobs(jobs=simJobs)

if __name__ == "__main__":
    dF = demoOneFiber()
