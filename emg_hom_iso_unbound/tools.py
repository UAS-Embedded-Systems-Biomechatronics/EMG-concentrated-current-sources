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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pickle
import json

import numba 
import scipy.signal

from tqdm.auto import tqdm

import os

import re



def generate_df_sim_res_muscles(glob_list, multibleMuscels=False):
    """
    this function goes through a list of file pahts *glob_list* 
    and looks for spcific strings in order to derive
    the 
        - muscle id                     : "idx_muscle"
        - motor unit id                 : "idx_motor_unit"
        - and the muscle fiber id       : "idx_muscle_fiber"
        - the id of the simulation run  : "idx_run"
    a dataframe containing the ids file path and a unique motor unit
    id is returned. The unique motor unit id is generated based on 
    the respective motor unit id and the muscle id. This whay motor units
    with the same id that are from different muscles can be differentiated.

    Each id is the integer number behind the respective fields in the file path.
    That are
    "_muscle_<idx_muscle>" ... "_motor_unit_<id>" ... "_muscle_fiber_<id>_<idx_run>"

    Example:
    the file path

        "./_muscle_1/_motor_unit_3_muscle_fiber_900_1.npz"
    results in 
        idx_muscle       = 1
        idx_motor_unit   = 3
        idx_muscle_fiber = 900
        idx_run          = 1

    Prameters
    --------
    glob_list : list
        is a list of file paths

    multibleMuscels : bool
        if set to true the muscel id is derived from the given file paths
        if set to flase the muscel id is set to 0
    """

    def get_id(str_regex : str, str_in : str) -> int:
        findall = re.findall(str_regex, str_in)
        len(findall) == 0 , "more than one object matched for {} in\n\t".format(str_regex, str_in)
        return int(findall[0])

    muscle_id_list = []
    mu_id_list     = []
    mf_id_list     = []
    run_id_list    = []

    entry_list     = []

    for entry in tqdm(glob_list):
        if multibleMuscels:
            muscle_id = get_id("\_muscle\_(\d+)", entry)
        else:
            muscle_id = 0 

        muscle_id_list.append(muscle_id)

        mu_id     = get_id("\_motor\_unit\_(\d+) ", entry)
        mu_id_list.append(mu_id)

        mf_id     = get_id("\_muscle\_fiber\_(\d+)", entry)
        mf_id_list.append(mf_id)

        run_id    = get_id("\_muscle\_fiber\_\d+\_(\d+)", entry)
        run_id_list.append(run_id)

        entry_list.append(entry)
        
    df_ids2 = pd.DataFrame({
          "idx_muscle" : muscle_id_list
        , "idx_motor_unit" : mu_id_list
        , "idx_muscle_fiber" : mf_id_list
        , "idx_run": run_id_list
        , "file_path" : entry_list})

    df_ids2["unique_idx_motor_unit"], _ = pd.factorize(pd.MultiIndex.from_frame(df_ids2[['idx_muscle', 'idx_motor_unit']]))

    return df_ids2


class sEMG_sim_result(object):

    def _gen_SD_DD_from_sum_df_Mono(self):
        self.sum_df_SD = self._sum_df2diff(self.sum_df_Mono)
        self.sum_df_DD = self._sum_df2diff(self.sum_df_SD, isDD=True) 

    def _sum_df2diff(self, sum_df, isDD=False):
        sum_df_SD = pd.DataFrame()
        
        end = -1
        if isDD:
            end = -2
        
        for elec_id in self.electrode_ids[0:end]: # -2 due to the field 'time_s'
            sum_df_SD[elec_id] = sum_df[elec_id] - sum_df[elec_id+1]

        sum_df_SD['time_s'] = sum_df['time_s']

        return sum_df_SD

    def _load_hom_iso_unbound_sim_pd_dir_config(self, projectDir :str):
        config_file_path = os.path.join(projectDir, "config.json")
        with open(config_file_path, 'rb') as f:
            self.config = json.load(f)
    
    def get_elec_SD_iterator(self):
        return iter(self.electrode_ids[:-1])

    def get_elec_DD_iterator(self):
        return iter(self.electrode_ids[:-2])

    def calc_white_gaussian_sigma_dB_power_forDD(self, dbSNR):
        self.sum_df_DD_addGausian = self.sum_df_DD.copy()

        sigma_vec = []
        for e_id in self.get_elec_DD_iterator():
            x = self.sum_df_DD[e_id].to_numpy()
            sigma_vec.append(self._get_white_gaussian_sigma_dB_Power(x, dbSNR))

        sigma = np.median(sigma_vec)

        for e_id in self.get_elec_DD_iterator():
            x = self.sum_df_DD[e_id].to_numpy()

            e = np.random.normal(loc = 0, scale = sigma, size=x.shape[0])

            self.sum_df_DD_addGausian[e_id] = x + e

    
    def calc_white_gaussian_sigma_dB_power_forSD(self, dbSNR):
        self.sum_df_SD_addGausian = self.sum_df_SD.copy()

        sigma_vec = []
        for e_id in self.get_elec_SD_iterator():
            x = self.sum_df_SD[e_id].to_numpy()
            sigma_vec.append(self._get_white_gaussian_sigma_dB_Power(x, dbSNR))

        sigma = np.median(sigma_vec)

        for e_id in self.get_elec_SD_iterator():
            x = self.sum_df_SD[e_id].to_numpy()

            e = np.random.normal(loc = 0, scale = sigma, size=x.shape[0])

            self.sum_df_SD_addGausian[e_id] = x + e

    def _get_white_gaussian_sigma_dB_Power(self, x, dB):
        sigma = np.sqrt(( np.sum(np.abs(x)**2) / len(x)) / (
            10 ** (dB/10)))

        return sigma

    def sos_filter(self, sos_function, N, Wn, btype):
        """
        sos_function:
            has to be a function object generating iir filter coefficiants 
            in the form as defined in scypi.signal as for example scipy.signal.butter
        N:
            is the amount of second order sections therefor the filter order will be 2*N
        Wn: 
            array like cutoff frequencies
        btype : 
            must be "bandpass" or "low" or "high" and defines the type of filter
        """
        T  = self.config['time'][1] - self.config['time'][0]
        Fs = 1 / T
        return sos_function(N=N, Wn=Wn, btype=btype, analog=False, output='sos', fs=Fs)

    def sos_filterOn_DD(self, sos_function, N, Wn, btype):
        self.sum_df_DD_filtered = apply_sos_filt_on_elecDF(
                  self.sos_filter(sos_function, N=N, Wn=Wn, btype=btype)
                , on = self.sum_df_DD
                , iterator= self.get_elec_DD_iterator()
        )

        self.sum_df_DD_addGausian_filtered = apply_sos_filt_on_elecDF(
                  self.sos_filter(sos_function, N=N, Wn=Wn, btype=btype)
                , on = self.sum_df_DD_addGausian
                , iterator= self.get_elec_DD_iterator()
        )


    def sos_filterOn_SD(self, sos_function, N, Wn, btype):
        self.sum_df_SD_filtered = apply_sos_filt_on_elecDF(
                  self.sos_filter(sos_function, N=N, Wn=Wn, btype=btype)
                , on = self.sum_df_SD
                , iterator= self.get_elec_SD_iterator()
        )

        self.sum_df_SD_addGausian_filtered = apply_sos_filt_on_elecDF(
                  self.sos_filter(sos_function, N=N, Wn=Wn, btype=btype)
                , on = self.sum_df_SD_addGausian
                , iterator= self.get_elec_SD_iterator()
        )


class npz_sEMG_sim_results(sEMG_sim_result):
    def __init__(self
                 , sim_project_dir : str
                 , npz_file_df : 'pd.DataFrame'
                 , electrode_slice_tuple = None
                 , idx_motor_unit = 0
                 , idx_muscle = 0
                 , flat_electrodes = False
                 , npz_potential_key = "electrode_potentials"):
        """
        File df must contain a column called "file_path", "unique_idx_motor_unit"
        """
        self._flat_electrodes = flat_electrodes

        self._electrode_slice_tuple = electrode_slice_tuple

        self._load_hom_iso_unbound_sim_pd_dir(sim_project_dir, npz_file_df, npz_potential_key)
        
        self.idx_motor_unit = idx_motor_unit
        self.idx_muscle     = idx_muscle

    def _load_hom_iso_unbound_sim_pd_dir(self, projectDir, _npz_file_df, npz_potential_key):
        self._load_hom_iso_unbound_sim_pd_dir_config(projectDir)

        idx_unique_mu_list = _npz_file_df["unique_idx_motor_unit"].unique()
        assert len(idx_unique_mu_list) == 1 , "exactly one motor unit needet"
        self.idx_unique_mu = idx_unique_mu_list[0]

        _electrode_pot = {}
        for idx in _npz_file_df.index:
            npz_file_disc            = np.load(_npz_file_df.loc[idx]["file_path"])
            if self._electrode_slice_tuple is None:
                _electrode_pot[idx] = npz_file_disc[npz_potential_key]
            else:
                _electrode_pot[idx] = npz_file_disc[npz_potential_key][self._electrode_slice_tuple]

            
            npz_file_disc.close()

        self.res_mu = {}

        res = np.zeros_like(_electrode_pot[next(iter(_electrode_pot.keys()))])

        for idx_data in _npz_file_df[ _npz_file_df["unique_idx_motor_unit"] == self.idx_unique_mu].index:
            res += _electrode_pot[idx_data]
                
            self.res_mu[int(self.idx_unique_mu)] = res

        self.electrode_meta = pd.DataFrame({ key : self.config['electrodes'][key] for key in ['x', 'y', 'z']})
        if self._electrode_slice_tuple is None:
            self.electrode_ids = self.electrode_meta.index.to_numpy(dtype=int)
        else:
            if self._flat_electrodes:
                self.electrode_ids = self.electrode_meta.index.to_numpy(dtype=int)[self._electrode_slice_tuple]
            else:
                self.electrode_ids = self.electrode_meta.index.to_numpy(dtype=int)[self._electrode_slice_tuple[2]]
        
        self._gen_sum_df_Mono_for_mu()
        self._gen_SD_DD_from_sum_df_Mono()

    def _mean_p_IP(self):
        p_IP_sum = np.zeros(shape=(3,1))
        if not (self.config_dom is None):
            fibers = self.config_dom.\
                muscle.motor_units[self.idx_motor_unit].\
                fibers
        else:
            raise NotImplmentedError()
            
        for fiber in fibers:
            p_IP_sum += fiber.origin_location + np.array([[fiber.IP], [0], [0]])
            
        self.p_IP_mean = p_IP_sum / len(fibers)
        return self.p_IP_mean
        
        
    def _gen_sum_df_Mono_for_mu(self):
        if self._flat_electrodes:
            self.sum_df_Mono = pd.DataFrame({
                self.electrode_ids[e_id_local] : self.res_mu[self.idx_unique_mu][e_id_local] for e_id_local in range(0,len(self.electrode_ids))
            })
        else:            
            self.sum_df_Mono = pd.DataFrame({
                self.electrode_ids[e_id_local] : self.res_mu[self.idx_unique_mu][e_id_local, 0, 0] for e_id_local in range(0,len(self.electrode_ids))
            })
        
        self.sum_df_Mono['time_s'] = self.config['time']


def apply_sos_filt_on_elecDF(sos, on, iterator):
    result_df = on.copy()

    for e_id in iterator:
        result_df[e_id] = scipy.signal.sosfilt(sos=sos, x = on[e_id])

    return result_df

def sim_data2electrode_dfs(sim_data, time_vec):
    """
    input: the simulation data as exported by the script 
                    core_hom/export_as_pandas.py

    return:
        electrode_meta, sum_df, elec_df
        where:
            -  electrode_meta is a dataframe containing the position of the electrodes 
            - sum_df is a data frame containing the sum over all muscle fibers per electrode
            - where elec_df is the sim data split up in columns of electrodes containing the individual muscle
              fiber surface potentials phi
    """
    electrode_meta = pd.DataFrame({'electrode_id' : [], 'p_x' : [], 'p_y' : [],'p_z' : [], 'row' : [], 'col' : []})
    electrode_meta['electrode_id'] = np.sort(sim_data['electrode_id'].astype(int).unique())

    elec_df = None

    for elec_id in sim_data['electrode_id'].unique():
        sel = sim_data['electrode_id'] == elec_id

        local_elec_df = sim_data[sim_data['electrode_id'] == elec_id][['timepoint', 'phi', 'muscle_id']]
        local_elec_df.index = pd.MultiIndex.from_frame(local_elec_df[['timepoint', 'muscle_id']])
        local_elec_df[int(elec_id)] = local_elec_df['phi']
        local_elec_df.drop('phi', axis=1, inplace=True)

        if elec_df is None:
            elec_df= local_elec_df
        else:
            elec_df = elec_df.join(local_elec_df[elec_id])



        electrode_meta.at[int(elec_id), 'p_x'] = sim_data[sel]['p_electrode_x'].iloc[0].item()
        electrode_meta.at[int(elec_id), 'p_y'] = sim_data[sel]['p_electrode_y'].iloc[0].item()
        electrode_meta.at[int(elec_id), 'p_z'] = sim_data[sel]['p_electrode_z'].iloc[0].item()

    elec_df.index = list(range(len(elec_df)))
    

    sum_df = None
    for m_id in elec_df['muscle_id'].unique():
        selected_df = elec_df[elec_df['muscle_id'] == m_id].copy()
        selected_df.index = selected_df['timepoint']
        selected_df.drop('timepoint', axis=1, inplace=True)
        selected_df.drop('muscle_id', axis=1, inplace=True)

        if sum_df is None:
            sum_df = selected_df
        else:
            sum_df += selected_df
            
            
    elec_df['time_s'] = sim_data['timepoint'].map(lambda x: time_vec[int(x)])
    sum_df['time_s']  = time_vec
    
    return electrode_meta, sum_df, elec_df

def sum_df2diff(sum_df):
    sum_df_SD = pd.DataFrame()

    for elec_id in range(len(sum_df.keys()) - 2): # -2 due to the field 'time_s'
        sum_df_SD[elec_id] = sum_df[elec_id] - sum_df[elec_id+1]

    sum_df_SD['time_s'] = sum_df['time_s']

    return sum_df_SD

def plotSum_df(   sum_df
                , str_title   = 'single differential electrode array (SD)' 
                , str_y_label = 'SD'
                , delta = -0.10
                , ax = None
                , every_nth_y_tick = None
                , plot_args = {}
              ):
    if ax is None:
        fig = plt.figure()
        ax  = fig.gca()
        fig.suptitle( str_title, fontsize=16)
    else:
        ax.set_title(str_title)
        fig = None

    style_list  = ["k", "tab:gray"]

    y_ticks = []

    e_ids = sum_df.keys()[~(sum_df.keys() == 'time_s')]

    pair_counter = 0
    for idx_ax in e_ids:
        ax.plot(sum_df['time_s'], sum_df[idx_ax] + (idx_ax * delta), style_list[pair_counter % 2 ] , **plot_args)
        y_ticks += [(idx_ax * delta)]
        pair_counter += 1

    _ = ax.set_yticks(y_ticks)
    _ = ax.set_yticklabels([ "{}{:>2}".format(str_y_label, int(n)) for n in e_ids])

    if not (every_nth_y_tick is None):
        _ = ax.set_yticks(  ax.get_yticks()[::every_nth_y_tick]
                          , minor=False)

    return fig, ax
