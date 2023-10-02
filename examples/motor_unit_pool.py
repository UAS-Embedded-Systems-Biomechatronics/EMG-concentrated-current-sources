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
import os.path

import numba

import pickle

import tqdm

import ray

_m  = 1
_s  = 1
_mps = _m / _s

# MF_pdfMU
@numba.jit
def y(i, R=188.6, a=21,  n = 774):
    return int(a * np.exp( np.log(R) / n  * i) + 0.5)

def gen_velocity_vector(
          mu_ids : 'np.ndarray'
        , n_mf_per_id : 'np.ndarray'
        , min_cv : float = 2.5 * _mps # Andreassen1986
        , max_cv :float  = 5.4 * _mps # Andreassen1986

        , std_cv_mf : float = 0.22    # Farina2000

        , subset_mu_ids : 'np.ndarray' = None
        ):

    b = min_cv
    m = (max_cv - min_cv) / (mu_ids[-1])
    line_cv = lambda n : m * n + b

    #
    d_cvs = {}

    if subset_mu_ids is None:
        subset_mu_ids = mu_ids
    
    for idx in subset_mu_ids:
        d_cvs[idx] = {}

        d_cvs[idx]['cv_m_target'] = line_cv(idx)
        d_cvs[idx]['cv_mf']       = np.random.normal(
                  loc   = d_cvs[idx]['cv_m_target']
                , scale = 0.22
                , size  = int(n_mf_per_id[idx] + 0.5)
                )
        d_cvs[idx]['cv_m_real']   = np.mean(d_cvs[idx]['cv_mf'])
        d_cvs[idx]['cv_N']        = int(n_mf_per_id[idx])

    return d_cvs

def gen_commondrive_start_vector_Fuglevasnd_1993(
         mu_ids : 'np.ndarray' # shape == (N_dix,)
        , N_mu   : int
        , common_drive_full : float
    ) -> 'np.ndarray':
    """
    based on Fuglevand et al. (1993) as cited in eq (1) of:
        A Comprehensive Mathematical Model of Motor Unit Pool Organization, 
        Surface Electromyography, and Force Generation 
        doi: 10.3389/fphys.2019.00176
    """
    a = np.log(100* common_drive_full) / N_mu # scalar
    CD_start_vec = np.exp(a* mu_ids) / 100
    return CD_start_vec


def gen_commondrive_start_vector_deLuca2012(
         mu_ids : 'np.ndarray' # shape == (N_dix,)
        , N_mu   : int
        , 
    ):
    """
    based on De Luca and Contaeesa (2012) as cited in eq (2)
        A Comprehensive Mathematical Model of Motor Unit Pool Organization, 
        Surface Electromyography, and Force Generation 
        doi: 10.3389/fphys.2019.00176
    """
    raise NotImplemented()
    pass


_mm  = 1e-3
_cm  = 1e-2
_cm2 = _cm**2
_m   = 1


Muscle_gen_MF = []
#Winters1988

A_R = 10 * _cm2
L_f = 15 * _cm

Muscle_gen_MF.append(
    lambda : model_config.MF_Merletti1999(
      W_I = 2*_cm
    , L_L = L_f * 0.5
    , L_R = L_f * 0.5

    , W_TL = 0.5*_cm
    , W_TR = 0.5*_cm

    , R   = np.sqrt((A_R) / np.pi))
 )

N_MU   = 774
mu_ids = np.arange(0, N_MU, 1)

subset_mu_ids = np.arange(400, N_MU, 50)
N_mf_per_id   = np.array([ y(idx, n=N_MU) for idx in mu_ids])

D_cvs = gen_velocity_vector(mu_ids, N_mf_per_id, subset_mu_ids = subset_mu_ids)

assumed_full_commonDrive = 0.8
common_drive_start_vec = gen_commondrive_start_vector_Fuglevasnd_1993(
    subset_mu_ids, N_MU, assumed_full_commonDrive)

max_L = 0.0

conf_m_list = [] 

# cunstruct all the muscles
print("#"*5 + "  generate config  " + "#"*5)
for muscle_gen_MF in tqdm.tqdm(Muscle_gen_MF, desc = "muscle"):

    conf_m = model_config.muscle()

    for mu_idx in tqdm.tqdm(subset_mu_ids, desc = "motor_unit", leave=False):
        motorUnit   = model_config.motorUnit()
        motorUnit.firing_behavior = model_config.firingBehavior()

        motorUnit.firing_behavior.firing_frequenzy.C1 = 20.0
        motorUnit.firing_behavior.firing_frequenzy.C2 =  1.5 
        motorUnit.firing_behavior.firing_frequenzy.C3 = 30
        motorUnit.firing_behavior.firing_frequenzy.C4 = 13
        motorUnit.firing_behavior.firing_frequenzy.C5 =  8
        motorUnit.firing_behavior.firing_frequenzy.C6 =  8
        motorUnit.firing_behavior.firing_frequenzy.C7 =  0.05

        motorUnit.firing_behavior.start_common_drive = common_drive_start_vec[mu_idx]




        for idx_mf in tqdm.tqdm(range(D_cvs[mu_idx]['cv_N']), desc="muscle_fiber", leave=False):
            mf = muscle_gen_MF()

            mf.v = D_cvs[mu_idx]['cv_mf'][idx_mf]
            if mf.L > max_L:
                max_L = mf.L

            motorUnit.fibers.append(mf)

        conf_m.motor_units.append(motorUnit)

    conf_m_list.append(conf_m)

max_time = ( max_L ) / motorUnit.v

_kHz = 1e3

fs = 5 * _kHz
time = np.arange(0, max_time, 1/fs)

electrode_matrix = model_config.electrodes(
          x =  np.arange(-17*_cm, 17*_cm, 5*_mm)
        , y =  np.array([0])
        , z =  np.array([2*_cm])
        )

m_id = 0
root_conf_dict = {}

for muscle in tqdm.tqdm(conf_m_list, desc="sim_muscle", total=len(conf_m_list)):
    root_conf_dict[m_id] =  model_config.root(
              muscle = muscle
            , time = time
            , electrodes = electrode_matrix
            , export_vtk = False
            , export_pyz = True
            )

    m_id += 1

ray.init()
id_root_conf_dict = ray.put(root_conf_dict)

@ray.remote
def remote_function(root_conf_dict, m_id):
    root_conf = root_conf_dict[m_id]
    path = os.path.join('.', f'MUAP_center_tracking_v1_muscle_{m_id}')

    try:
        os.mkdir(path)
    except :
        pass

    with open(path + "/d_cvs.pkl", 'wb') as f:
        pickle.dump( D_cvs, f )

    simJobs = sim_infrastructure. \
            simJobsByModelConfig(
                      config      = root_conf
                    , project_dir = path
                    )

    print("#"*5 + "  execute sim  " + "#"*5)
    sim_infrastructure. \
            execute_sim_jobs(jobs=simJobs)

ray.get([ remote_function.remote(id_root_conf_dict, idx) for idx in root_conf_dict ])
