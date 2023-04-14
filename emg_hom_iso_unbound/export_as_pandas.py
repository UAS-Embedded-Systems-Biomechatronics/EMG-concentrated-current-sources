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

import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from glob import glob


def load_vtu(str_file):
    r = vtk.vtkXMLUnstructuredGridReader()

    r.SetFileName(str_file)
    r.Update()

    points = r.GetOutput().GetPoints().GetData()
    np_points = vtk_to_numpy(points)


    phi = r.GetOutput().GetPointData().GetArray(0)
    np_phi = vtk_to_numpy(phi)

    return {'points':np_points, 'phi' : np_phi}


file_list = glob("./*_motor_unit_* idx_muscle_fiber_*_1/t_*.vtu")

df = pd.DataFrame({'file_name' : file_list})

df['muscle_fiber_id'] = df.file_name.map(lambda x: int( x.split(' ')[-1].split('_')[3] ))
df['electrodes']      = df.file_name.map(lambda x: x.split('/')[-1].split('_')[0] == 't')
df['sources']         = df.file_name.map(lambda x: x.split('/')[-1].split('_')[0] == 'sources')
df['time']            = df.file_name.map(lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))

muscle_ids = df.muscle_fiber_id.unique()

muscle_fiber_phi = {}

def import_muscle_data(muscle_id : '[int]', df : 'pd.DataFrame' ) -> 'dict':
    sel = (df.muscle_fiber_id == muscle_id)

    df_local = df[sel & df.electrodes]

    muscle_fiber_phi = {}
    muscle_fiber_phi[muscle_id] = {}

    for t in df_local.time:
        file_selected = df_local.file_name[df_local.time == t]
        assert len(file_selected) == 1 , 'more than one file selected'

        try:
            data_dict = load_vtu( file_selected.item() )
        except :
            import pdb; pdb.set_trace()

        phi    = data_dict['phi']

        muscle_fiber_phi[muscle_id][t] = phi

    return muscle_fiber_phi

def import_electrode_positions(df : 'pd.DataFrame') -> 'nd.array' :
        data_dict = load_vtu( df.file_name[0])
        
        return data_dict['points']

def sort_time(d):
    time_idx = np.sort([t for t in d])
    return np.array( [ d[t] for t in time_idx ])

iterator = tqdm(map(lambda x : import_muscle_data(x, df), muscle_ids), total = len(muscle_ids))

muscle_fiber_phi = { k: sort_time(v) for d in iterator for k , v in d.items()  }

electtrode_df = pd.DataFrame({'timepoint' : [], 'phi': [], 'electrode_id': [], 'muscle_id' : []})

electrode_positions = import_electrode_positions(df)

for e_id in tqdm( range(len(muscle_fiber_phi[0][0,:])) ):
    for m_id in muscle_fiber_phi:

        df_length = len(muscle_fiber_phi[m_id][:,e_id])

        electtrode_df_m = pd.DataFrame({
            'timepoint':    [t for t in range(len(muscle_fiber_phi[m_id][:,e_id]))]
            , 'phi':          muscle_fiber_phi[m_id][:,e_id]
            , 'muscle_id':    [m_id] * df_length
            , 'electrode_id': [e_id] * df_length
            , 'p_electrode_x' : [electrode_positions[e_id,0]] * df_length
            , 'p_electrode_y' : [electrode_positions[e_id,1]] * df_length
            , 'p_electrode_z' : [electrode_positions[e_id,2]] * df_length
            })

        electtrode_df = pd.concat([electtrode_df, electtrode_df_m], ignore_index = True )

pd.to_pickle(electtrode_df, "./pandas_full_data.pkl")


