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


from singleton_decorator import singleton

from . import model_config as mc
from . import model as model

import os

import re

import shutil

from tqdm.auto import tqdm

import numpy as np

from typing import List, Dict, TypedDict

       

def loadRootConfigFile(strConfigPath : str) -> mc.root:
    with open(strConfigPath, 'r') as f:
        str_imported = f.read()
    
    return mc.JSON_config_factory(str_imported)
        

class simJobsByModelConfig(object):
    """
    """

    def __init__(self, config, project_dir = "./",
                 create_project_dir = False, force_project_overwrite = False):
        """
        Creates an sEMG sim job object.

        Based on a json configuration file or python model_config.root 
        configuration object a simulation Job is created.

        Parameters
        ----------
            config:
                - (str) string file path to json sim configuration file relative to project_dir
                - (model_config.root) a configuration object
            project_dir (str): path to a simulation project. It is assumed that this path exists.
                               all paths are relative to the project directory location.

        """

        self.project_dir = project_dir
        self.prepare_project_dir(force_project_overwrite, create_project_dir)

        self.config = self.load_config(config)

        self._calc_max_iter()

        self._lastModel = None

    def prepare_project_dir(
            self, force_project_overwrite : bool, create_project_dir : bool):
        if force_project_overwrite and os.path.isdir(self.project_dir):
            shutil.rmtree(self.project_dir)

        if create_project_dir:
            os.mkdir(self.project_dir)


    def load_config(self, config):
        if isinstance(config, str):
            return loadRootConfigFile(config)
        elif isinstance(config, mc.root):
            return config

        raise TypeError("argument config has to be a string" \
            " config file path or model_config.root")


    def save_config2projectDir(self):
        project_dir_list = self.project_dir.split("/")
        with open(os.path.join(*(project_dir_list + ['config.json'])), 'x') as f:
            f.write(self.config.as_json_s)
            

    def createSimTask(  self
                      , idx_motor_unit : int
                      , idx_muscle_fiber : int) -> 'model.simTask':

        description_str = "idx_motor_unit_{} idx_muscle_fiber_{}". \
                format( idx_motor_unit
                , idx_muscle_fiber)

        simTask = model.simTask(
                  time       = self.config.time
                , mf_config  = self.config \
                    .muscle \
                    .motor_units[idx_motor_unit] \
                    .fibers[idx_muscle_fiber]
                , electrodes  = self.config.electrodes
                , is_batch = self.config.electrodes.isGridConfig
                
                , description = description_str
                , export_dir  = self.project_dir

                , vtk_export  = self.config.export_vtk
                , npz_export  = self.config.export_pyz

                , tf_model_in = self._lastModel

                , idx_motor_unit    = idx_motor_unit
                , idx_muscle_fiber  = idx_muscle_fiber)


        self._lastModel = simTask.tf_model

        return simTask

    def _calc_max_iter(self):
        iter_count = 0
        for idx_motor_unit in range(len(self.config.muscle.motor_units)):
            iter_count += len(self.config.muscle.motor_units[idx_motor_unit].fibers)

        self._max_iter_count_allJobs = iter_count

        return iter_count

    def __iter__(self) -> 'model.simTask':
        for idx_motor_unit in range(len(self.config.muscle.motor_units)):
            n_mf = len( self.config.muscle.motor_units[idx_motor_unit].fibers)
            for idx_muscle_fiber in range(n_mf):
                yield lambda : self.createSimTask(idx_motor_unit, idx_muscle_fiber)

    def execute_all(self, showProgressbar=True, export_config=True):
        if export_config:
            self.save_config2projectDir()

        if showProgressbar:
            task_iter = tqdm(self, total=self._max_iter_count_allJobs)
        else:
            task_iter = self

        for task in task_iter:
            task_obj = task()
            task_obj.compute_task()
            task_obj.export_results()
            if self.config.calc_sum_potential:
                mus = motor_unit_store(self.config.muscle)
                mus.add_mu_result(task_obj)



class motor_unit_store_dict(TypedDict):
    n_mf : int
    n_mf_processed : int
    sum_pot : 'np.ndarray'

    exported : bool


@singleton
class motor_unit_store:
    """
    this class keeps track of the muscle fibers
    that are part of motor unit and their simulation result.
    It automatically calculates the electrode sum potential
    for each motor unit. When all Mus are simulated it stores the 
    sum potential.

    TODO
    ----
        - add capability to read sum_pot from disc optionally
          such that the RAM does not have to hold all motor unit 
          simulation results

        - Test what happens when multiple muscles are simualted
    """
    def __init__(self, muscle_config) -> None:
        self.muscle_config = muscle_config
        self.mu_s : Dict[int, motor_unit_store_dict] = {}
        pass

    def load_from_files(self, str_project_dir) -> None:

        for strFileName in os.listdir(str_project_dir):
            m = re.fullmatch(r"e_sum_pot_idx_motor_unit_(\d+).npz", strFileName)
            if m is None:
                continue

            file_path = os.path.join(str_project_dir, strFileName)

            self.create_mu( motor_unit_id=int(m.groups()[0])
                , e_pot= np.load(file_path)['e_sumPot']
                , exported= True)


    def create_mu(self, motor_unit_id, e_pot, exported = False) -> None:
        n_mf = len( self.muscle_config.motor_units[motor_unit_id].fibers)

        n_mf_processed :int = 1
        if exported:
            n_mf_processed = n_mf

        self.mu_s[motor_unit_id] = {
              'n_mf'           : n_mf
            , 'n_mf_processed' : n_mf_processed
            , 'sum_pot'  : e_pot
            , 'exported' : False
        }


    def add_mu_result(self, task_obj : model.simTask) -> None :
        motor_unit_id :int = task_obj.idx_motor_unit
        e_pot : 'np.ndarray' = task_obj.res

        if motor_unit_id not in self.mu_s:
            self.create_mu(motor_unit_id, e_pot)
            return

        assert self.mu_s[motor_unit_id]['exported'] == False , \
            "motor unit result is already exported"

        self.mu_s[motor_unit_id]['sum_pot'] += e_pot
        self.mu_s[motor_unit_id]['n_mf_processed'] += 1

        if self.mu_s[motor_unit_id]['n_mf_processed'] >= self.mu_s[motor_unit_id]['n_mf']:
            file_name = "e_sum_pot_idx_motor_unit_{}.npz".format(motor_unit_id)
            file_path = os.path.join(task_obj.export_dir, file_name)

            np.savez(file_path, e_sumPot = self.mu_s[motor_unit_id]['sum_pot'])
            self.mu_s[motor_unit_id]['exported'] = True
