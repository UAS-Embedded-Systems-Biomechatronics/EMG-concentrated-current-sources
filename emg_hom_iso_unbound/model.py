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

import tensorflow as tf
import numpy as np

from . import model_config

import os
import evtk
import evtk.hl
from datetime import datetime

import ray
import psutil

tracing_notice = False

class tf_model(tf.Module):
    def __init__(  self
                 , time : np.ndarray
                 , mf_config  : model_config.muscleFiber
                 , electrodes : model_config.electrodes
                 ):
        super(tf_model,self).__init__()


        self._init_parameters(time, mf_config, electrodes)

        self._is_init_batch = False
        self._is_init_flattend_batch = False

    def _init_parameters(self,
                         time : np.ndarray,
                         mf_config  : model_config.muscleFiber,
                         electrodes : model_config.electrodes):
        """ !!! closely coulpled to self.set_parameters(....) """
        # TODO only init type??

        self._time = tf.Variable(time, dtype=tf.float64)

        self._paraGeom_IP   = tf.Variable(mf_config.IP,  dtype=tf.float64)
        self._paraGeom_L    = tf.Variable(mf_config.L,   dtype=tf.float64)
        self._paraGeom_Sart = tf.constant(0,             dtype=tf.float64)

        self._param_v  = tf.Variable(mf_config.v, dtype=tf.float64)

        self._param_cs_a = tf.Variable(mf_config.currentSource.a, dtype=tf.float64)
        self._param_cs_l = tf.Variable(mf_config.currentSource.l, dtype=tf.float64)
        self._param_cs_b = tf.Variable(mf_config.currentSource.b, dtype=tf.float64)

        self._param_Re = tf.Variable(mf_config.Re, dtype=tf.float64)
        self._param_Ri = tf.Variable(mf_config.Ri, dtype=tf.float64)


        self._rot_z = tf.Variable(mf_config.rotation.z, dtype = tf.float64)
        self._rot_y = tf.Variable(mf_config.rotation.y, dtype = tf.float64)

        self._m_origin_x = tf.Variable(mf_config.origin_location[0].item(), dtype= tf.float64)
        self._m_origin_y = tf.Variable(mf_config.origin_location[1].item(), dtype= tf.float64)
        self._m_origin_z = tf.Variable(mf_config.origin_location[2].item(), dtype= tf.float64)

        # TODO currently the only allowd vector for electrode Positions
        self._tf_x = tf.Variable(electrodes.x, dtype=tf.float64)

        # 2x3 -- alpha,beta x S1, S2, S3
        self._borders = tf.Variable(mf_config.currentSource.sections.T , dtype=tf.float64)

    def set_electrodes(self, electrodes : 'model_config.electrodes'):
        # TODO currently the only allowed vector for electrode Positions 
        self._tf_x.assign(electrodes.x)

    def set_parameters(self, mf_config : model_config.muscleFiber):
        """ !!! closely coulpled to self._init_parameters(....) """

        self._paraGeom_IP.assign(mf_config.IP)
        self._paraGeom_L.assign(mf_config.L)

        self._param_v.assign(mf_config.v)

        self._param_cs_a.assign(mf_config.currentSource.a)
        self._param_cs_l.assign(mf_config.currentSource.l)
        self._param_cs_b.assign(mf_config.currentSource.b)

        self._param_Re.assign(mf_config.Re)
        self._param_Ri.assign(mf_config.Ri)

        self._rot_z.assign(mf_config.rotation.z)
        self._rot_y.assign(mf_config.rotation.y)

        self._m_origin_x.assign(mf_config.origin_location[0].item())
        self._m_origin_y.assign(mf_config.origin_location[1].item())
        self._m_origin_z.assign(mf_config.origin_location[2].item())


        # 2x3 -- alpha,beta x S1, S2, S3
        self._borders.assign(mf_config.currentSource.sections.T)

    @tf.function
    def c_local_to_c_global(self, func_c_m_local):
        # https://en.wikipedia.org/wiki/Rotation_matrix
        # as the muscle fiber is linear [x,0,0] only the orientaiton around z,y is relevant
        if tracing_notice:
            print("tracing c_local_to_c_global")

        func_c_m_global_x =   tf.math.cos(self._rot_z) * tf.math.cos(self._rot_y) * func_c_m_local + self._m_origin_x
        func_c_m_global_y =   tf.math.sin(self._rot_z) * tf.math.cos(self._rot_y) * func_c_m_local + self._m_origin_y
        func_c_m_global_z = - tf.math.sin(self._rot_y)                            * func_c_m_local + self._m_origin_z

        return tf.stack([ func_c_m_global_x, func_c_m_global_y, func_c_m_global_z ])


    @tf.function
    def phy_sum_over_x_Array_and_Time(self
                                      , func_c_l_m_global
                                      , func_i_l_fin
                                      , func_c_r_m_global
                                      , func_i_r_fin
                                      , tf_y, tf_z):
        if tracing_notice:
            print("tracing phy_sum_over_x_Array_and_Time")

        func_c_l_m_global_x = func_c_l_m_global[0,:,:]
        func_c_l_m_global_y = func_c_l_m_global[1,:,:]
        func_c_l_m_global_z = func_c_l_m_global[2,:,:]

        func_c_r_m_global_x = func_c_r_m_global[0,:,:]
        func_c_r_m_global_y = func_c_r_m_global[1,:,:]
        func_c_r_m_global_z = func_c_r_m_global[2,:,:]

        # calculate the range tensor
        tf_x_3Dim = tf.expand_dims( tf.expand_dims(self._tf_x, -1) , -1 )     # px1x1

        r_fin_l_nan = tf.sqrt(
              tf.pow(func_c_l_m_global_x - tf_x_3Dim, 2)
            + tf.pow(func_c_l_m_global_y - tf_y,2)
            + tf.pow(func_c_l_m_global_z - tf_z,2)
        )

        r_fin_r_nan = tf.sqrt(
              tf.pow(func_c_r_m_global_x - tf_x_3Dim, 2)
            + tf.pow(func_c_r_m_global_y - tf_y,2)
            + tf.pow(func_c_r_m_global_z - tf_z,2)
        )

        r_fin_l = tf.where(tf.math.is_nan(r_fin_l_nan) , tf.ones_like(r_fin_l_nan) , r_fin_l_nan)
        r_fin_r = tf.where(tf.math.is_nan(r_fin_r_nan) , tf.ones_like(r_fin_r_nan) , r_fin_r_nan)

        inSum_Big_l = func_i_l_fin / r_fin_l
        inSum_Big_r = func_i_r_fin / r_fin_r

        tf_Sum_l = tf.reduce_sum(inSum_Big_l, 1)
        tf_Sum_r = tf.reduce_sum(inSum_Big_r, 1)

        Sum = tf_Sum_l + tf_Sum_r

        return Sum # p_x x t
    
    @tf.function
    def phy_sum_over_xyz_Array_and_Time(self
                                      , func_c_l_m_global
                                      , func_i_l_fin
                                      , func_c_r_m_global
                                      , func_i_r_fin
                                      , tf_y, tf_z):
        if tracing_notice:
            print("tracing phy_sum_over_xyz_Array_and_Time")


        func_c_l_m_global_x = func_c_l_m_global[0,:,:]
        func_c_l_m_global_y = func_c_l_m_global[1,:,:]
        func_c_l_m_global_z = func_c_l_m_global[2,:,:]

        func_c_r_m_global_x = func_c_r_m_global[0,:,:]
        func_c_r_m_global_y = func_c_r_m_global[1,:,:]
        func_c_r_m_global_z = func_c_r_m_global[2,:,:]

        # calculate the range tensor
        tf_x_3Dim = tf.expand_dims( tf.expand_dims(self._tf_x, -1) , -1 )     # px1x1
        tf_y_3Dim = tf.expand_dims( tf.expand_dims(tf_y, -1) , -1 )     # px1x1
        tf_z_3Dim = tf.expand_dims( tf.expand_dims(tf_z, -1) , -1 )     # px1x1

        r_fin_l_nan = tf.sqrt(
              tf.pow(func_c_l_m_global_x - tf_x_3Dim, 2)
            + tf.pow(func_c_l_m_global_y - tf_y_3Dim,2)
            + tf.pow(func_c_l_m_global_z - tf_z_3Dim,2)
        )

        r_fin_r_nan = tf.sqrt(
              tf.pow(func_c_r_m_global_x - tf_x_3Dim, 2)
            + tf.pow(func_c_r_m_global_y - tf_y_3Dim,2)
            + tf.pow(func_c_r_m_global_z - tf_z_3Dim,2)
        )

        r_fin_l = tf.where(tf.math.is_nan(r_fin_l_nan) , tf.ones_like(r_fin_l_nan) , r_fin_l_nan)
        r_fin_r = tf.where(tf.math.is_nan(r_fin_r_nan) , tf.ones_like(r_fin_r_nan) , r_fin_r_nan)

        inSum_Big_l = func_i_l_fin / r_fin_l
        inSum_Big_r = func_i_r_fin / r_fin_r

        tf_Sum_l = tf.reduce_sum(inSum_Big_l, 1)
        tf_Sum_r = tf.reduce_sum(inSum_Big_r, 1)

        Sum = tf_Sum_l + tf_Sum_r

        return Sum # p_x x t


    def set_time(self, time: np.ndarray):
        self._time.assign(self.time)

    @tf.function
    def compute_current_sources(self):

        if tracing_notice:
            print("tracing compute_current_sources")

        borders_extendet = tf.expand_dims(self._borders, -1) # 2 x 3 x 1
        time_extendet    = tf.expand_dims(tf.expand_dims(self._time, 0) , 0 ) # 1x1xt

        # [sectionStart, sectionEnd] x currentSources x time
        # 2x3xt
        tf_borders_l =    borders_extendet + self._paraGeom_IP - self._param_v * time_extendet
        tf_borders_r =  - borders_extendet + self._paraGeom_IP + self._param_v * time_extendet

        ### left side rstrictions
        tf_ip_Acceeding_borders_l     = tf.math.greater_equal(tf_borders_l, self._paraGeom_IP)    # c_conc_r_t >= self.param.geometry.IP
        tf_tendon_Acceeding_borders_l = tf.math.less_equal(tf_borders_l,    self._paraGeom_Sart)  # c_conc_r_t <= 0


        tf_borders_l_pre_fin = tf.where( tf_ip_Acceeding_borders_l,
                                        tf.fill(tf_borders_l.shape, self._paraGeom_IP),
                                        tf_borders_l)

        borders_l_fin = tf.where( tf_tendon_Acceeding_borders_l,
                                    tf.fill(tf_borders_l_pre_fin.shape, self._paraGeom_Sart),
                                    tf_borders_l_pre_fin)

        ### Right side restrictions
        tf_ip_Acceeding_borders_r     = tf.math.less_equal(tf_borders_r,    self._paraGeom_IP) # c_conc_r_t <= self.param.geometry.IP
        tf_tendon_Acceeding_borders_r = tf.math.greater_equal(tf_borders_r, self._paraGeom_L)  # c_conc_r_t >= self.param.geometry.L

        tf_borders_r_pre_fin = tf.where( tf_ip_Acceeding_borders_r,
                                        tf.fill(tf_borders_r.shape, self._paraGeom_IP),
                                        tf_borders_r)

        borders_r_fin = tf.where( tf_tendon_Acceeding_borders_r,
                                    tf.fill(tf_borders_r_pre_fin.shape, self._paraGeom_L),
                                    tf_borders_r_pre_fin )


        # transform into actio potential local coordinate system
        # 2x3xt
        borders_l_fin_xAP =  borders_l_fin - self._paraGeom_IP + self._param_v * time_extendet
        borders_r_fin_xAP = -borders_r_fin + self._paraGeom_IP + self._param_v * time_extendet

        # currentSources x time
        # 3 x t
        func_c_l_fin_xAP, func_i_l_fin = self.computationBranch_conc_im_zAP(borders_l_fin_xAP)
        func_c_r_fin_xAP, func_i_r_fin = self.computationBranch_conc_im_zAP(borders_r_fin_xAP)

        # transform into fiber coordinate system
        # x=0       x=IP            x=L
        #  |---------x---------------|
        #
        func_c_l_m_local  =   func_c_l_fin_xAP + self._paraGeom_IP - self._param_v * time_extendet #1x3xt
        func_c_r_m_local  = - func_c_r_fin_xAP + self._paraGeom_IP + self._param_v * time_extendet #1x3xt

        func_c_l_m_global = self.c_local_to_c_global( func_c_l_m_local)
        func_c_r_m_global = self.c_local_to_c_global( func_c_r_m_local)

        return (  func_c_l_m_global
                , func_i_l_fin   ,  func_c_r_m_global, func_i_r_fin
                , func_c_l_fin_xAP, func_c_r_fin_xAP,  borders_l_fin_xAP, borders_r_fin_xAP)

    @tf.function
    def computationBranch_conc_im_zAP(self, tf_borders_fin):
        if tracing_notice:
            print("tracing computationBranch_conc_im_zAP")

        tf_gamma_fin_zAP   = tf_borders_fin[0] # 3xt
        tf_beta_fin_zAP    = tf_borders_fin[1] # 3xt

        tf_i_pre   = self._param_cs_a/( self._param_Re + self._param_Ri )

        # todo geminsamen term extra in zwischenergebnis
        # sinvolle variablennamen

        # TODO handle vanishing current source
        tf_i_direction_pre = (
                        tf.pow(tf_gamma_fin_zAP, 2) * ( tf_gamma_fin_zAP * self._param_cs_l
                                                        - tf.constant(3, dtype=tf.float64)
                                                    ) * tf.exp(- self._param_cs_l * tf_gamma_fin_zAP)
                        - tf.pow(tf_beta_fin_zAP, 2) * ( tf_beta_fin_zAP  * self._param_cs_l
                                                        - tf.constant(3, dtype=tf.float64)
                                                        ) * tf.exp(- self._param_cs_l * tf_beta_fin_zAP)
        )

        tf_i_fin = tf_i_pre * tf_i_direction_pre

        # TODO quellfunktionenn ueberpruefen und anzahl der operationen minimieren

        tf_c_fin_zAP = (
            (
                (
                    tf.pow(tf_beta_fin_zAP, 4) * self._param_cs_l
                    - 2 * tf.pow(tf_beta_fin_zAP, 3)
                ) * tf.exp(tf_gamma_fin_zAP * self._param_cs_l)
                + (
                    2 * tf.pow(tf_gamma_fin_zAP, 3)
                    - tf.pow(tf_gamma_fin_zAP, 4) * self._param_cs_l
                ) * tf.exp(tf_beta_fin_zAP * self._param_cs_l)
            )
            / (
                (
                    tf.pow(tf_beta_fin_zAP, 3) * self._param_cs_l
                  - 3 * tf.pow(tf_beta_fin_zAP, 2)
                ) * tf.exp(tf_gamma_fin_zAP * self._param_cs_l)
                +(
                    3 * tf.pow(tf_gamma_fin_zAP, 2) - tf.pow(tf_gamma_fin_zAP, 3) * self._param_cs_l
                ) * tf.exp(tf_beta_fin_zAP * self._param_cs_l)
            )
        )

        return tf_c_fin_zAP, tf_i_fin
    def init_flattend_batch(self
                   , y_vec : tf.Tensor
                   , z_vec : tf.Tensor):
        
        tf.debugging.assert_shapes([
             ( self._tf_x, ('X'))
            ,( y_vec, ('X'))
            ,( z_vec, ('X'))
        ])

        self.y_vec = y_vec
        self.z_vec = z_vec
        
        if self._is_init_flattend_batch:
            self.res.assign(
                tf.zeros(
                    [ self._tf_x.shape[0]
                    , self._time.shape[0]
                    ] , dtype=tf.float64
                )
            )
        else:
            self.res = tf.Variable(
                tf.zeros(
                    [ self._tf_x.shape[0]
                    , self._time.shape[0]
                    ] , dtype=tf.float64
                )
            )
            self._is_init_flattend_batch = True
            
    @tf.function
    def electrode_flattend_batch(self):
        (  func_c_l_m_global
         , func_i_l_fin
         , func_c_r_m_global
         , func_i_r_fin
         , func_c_l_m_local, func_c_r_m_local
         , borders_l_fin_xAP, borders_r_fin_xAP) = self.compute_current_sources()

        # TODO investigate side effects of self.res usage!!
        if tracing_notice:
            print("tracing electrode_flattend_batch")


        self.res[:, :].assign(
            self.phy_sum_over_xyz_Array_and_Time(
                  func_c_l_m_global , func_i_l_fin
                , func_c_r_m_global , func_i_r_fin
                , self.y_vec, self.z_vec
            )
        )

        return (
              self.res, tf.stack([func_c_l_m_global, func_c_r_m_global])
            , tf.stack([func_i_l_fin, func_i_r_fin])
            , tf.stack([func_c_l_m_local, func_c_r_m_local])
            , tf.stack([borders_l_fin_xAP, borders_r_fin_xAP])
        )
    
    def init_batch(self
                   , y_vec : tf.Tensor
                   , z_vec : tf.Tensor):

        tf.debugging.assert_shapes([
             ( self._tf_x, ('X'))
            ,( y_vec, ('Y'))
            ,( z_vec, ('Z'))
        ])

        self.y_vec = y_vec
        self.z_vec = z_vec

        if self._is_init_batch:
            self.res.assign(
                tf.zeros(
                    [
                      self._tf_x.shape[0]
                    , y_vec.shape[0]
                    , z_vec.shape[0]
                    , self._time.shape[0]
                    ] , dtype=tf.float64
                )
            )

        else:
            self.res = tf.Variable(
                tf.zeros(
                    [                      
                      self._tf_x.shape[0]
                    , y_vec.shape[0]
                    , z_vec.shape[0]
                    , self._time.shape[0]
                    ] , dtype=tf.float64
                )
            )

            self._is_init_batch = True


    @tf.function
    def electrode_batch(self):
        if tracing_notice:
            print("tracing electrode_batch")

        (  func_c_l_m_global
         , func_i_l_fin
         , func_c_r_m_global
         , func_i_r_fin
         , func_c_l_m_local, func_c_r_m_local
         , borders_l_fin_xAP, borders_r_fin_xAP) = self.compute_current_sources()

        # TODO investigate side effects of self.res usage!!

        for z_idx in tf.range(0, self.z_vec.shape[0], dtype = tf.int32):
            for y_idx in tf.range(0, self.y_vec.shape[0], dtype = tf.int32):
                self.res[:, y_idx, z_idx, :].assign(
                    self.phy_sum_over_x_Array_and_Time(
                          func_c_l_m_global , func_i_l_fin
                        , func_c_r_m_global , func_i_r_fin
                        , self.y_vec[y_idx], self.z_vec[z_idx]
                    )
                )

        return (
              self.res, tf.stack([func_c_l_m_global, func_c_r_m_global])
            , tf.stack([func_i_l_fin, func_i_r_fin])
            , tf.stack([func_c_l_m_local, func_c_r_m_local])
            , tf.stack([borders_l_fin_xAP, borders_r_fin_xAP])
        )
        #p_z x p_y x p_x x t


class simTask:
    def __init__(self
                 , time:        np.ndarray
                 , mf_config  : model_config.muscleFiber
                 , electrodes : model_config.electrodes
                 
                 , is_batch : bool

                 , vtk_export : bool = True
                 , npz_export : bool = False

                 , description: str = "basic_simTask"
                 , export_dir='.'
                 , tf_model_in = None):

        self.mf_config  = mf_config
        self.electrodes = electrodes
        self.time       = time

        self._is_batch = is_batch
        
        self._flag_vtk_export = vtk_export
        self._flag_npz_export = npz_export

        if tf_model_in is None:
            self.tf_model = tf_model(
                time
                , mf_config = mf_config
                , electrodes=electrodes
            )
        else:
            self.tf_model = tf_model_in
            self.tf_model.set_parameters(mf_config)

        self.tf_profile = False

        self.description    = description
        self.run_id         = 0
        self.str_time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.export_dir     = export_dir

    def _compute_electrode_batch_task(self):
        if isinstance(self.electrodes.y, np.ndarray):
            y = tf.Variable(self.electrodes.y, dtype=tf.float64)
        elif isinstance(self.electrodes.y, float):
            y = tf.Variable(np.array([ self.electrodes.y ]), dtype=tf.float64)
        else:
            raise TypeError("electrode cooridates have either"
                            + "to be an float or an numpy nndarray of floats")

        if isinstance(self.electrodes.z, np.ndarray):
            z = tf.Variable(self.electrodes.z, dtype=tf.float64)
        elif isinstance(self.electrodes.z, float):
            z = tf.Variable(np.array([ self.electrodes.z ]), dtype=tf.float64)
        else:
            raise TypeError("electrode cooridates have either"
                            + "to be an float or an numpy nndarray of floats")

        self.tf_model.init_batch(y, z)

        res, func_c_m_global, func_i_fin , self.c_local, self.borders \
                = self.tf_model.electrode_batch()



        return res.numpy(), func_c_m_global.numpy(), func_i_fin.numpy()


    def _compute_task(self):
        if isinstance(self.electrodes.y, np.ndarray):
            y = tf.Variable(self.electrodes.y, dtype=tf.float64)
        elif isinstance(self.electrodes.y, float):
            y = tf.Variable(np.array([ self.electrodes.y ]), dtype=tf.float64)
        else:
            raise TypeError("electrode cooridates have either"
                            + "to be an float or an numpy nndarray of floats")

        if isinstance(self.electrodes.z, np.ndarray):
            z = tf.Variable(self.electrodes.z, dtype=tf.float64)
        elif isinstance(self.electrodes.z, float):
            z = tf.Variable(np.array([ self.electrodes.z ]), dtype=tf.float64)
        else:
            raise TypeError("electrode cooridates have either"
                            + "to be an float or an numpy nndarray of floats")

        self.tf_model.init_flattend_batch(y, z)

        res, func_c_m_global, func_i_fin , self.c_local, self.borders \
                = self.tf_model.electrode_flattend_batch()

        return res.numpy(), func_c_m_global.numpy(), func_i_fin.numpy()
    

    def compute_task(self):
        elec_y_is_ndArray = isinstance( self.electrodes.y, np.ndarray )
        elec_z_is_ndArray = isinstance( self.electrodes.z, np.ndarray )

        if self.tf_profile:
            tf.summary.trace_on(graph=True, profiler=self.tf_profile)

        if self._is_batch:
            self.res, self.func_c_m_global, self.func_i_fin \
                = self._compute_electrode_batch_task()
        else:
            self.res, self.func_c_m_global, self.func_i_fin \
                = self._compute_task()

        """
        if elec_y_is_ndArray or elec_z_is_ndArray:
            self.res, self.func_c_m_global, self.func_i_fin \
                = self._compute_electrode_batch_task()
        else:
            self.res, self.func_c_m_global, self.func_i_fin \
                    = self._compute_task()
        """
        self.run_id += 1


    @ray.remote
    def _ray_save_res_to_vtk(m_x_flat, m_y_flat, m_z_flat, res, dir_name_list, t_idx):
        f_path = dir_name_list + ["t_{}".format(t_idx)]

        f_str = os.path.join(*f_path)

        evtk.hl.pointsToVTK(  f_str
                            , x = m_x_flat
                            , y = m_y_flat
                            , z = m_z_flat
                            , data = { 'phi' : res[:,:,:,t_idx].flatten()}
            )
        return 0


    @property
    def base_export_dir_list(self):
        return [self.export_dir
                , self.str_time_stamp + self.description + "_" + str(self.run_id)]


    def export_results(self, nProc = 12):
        if not os.path.exists(os.path.join(* self.base_export_dir_list)):
            os.makedirs(os.path.join(* self.base_export_dir_list))

        if self.tf_profile:
            writer = tf.summary.create_file_writer(
                        os.path.join(*(self.base_export_dir_list + ["tf_trace"]))
                    )
            with writer.as_default():
                tf.summary.trace_export(
                    name="my_func_trace",
                    step=0,
                    profiler_outdir= os.path.join(
                        *(self.base_export_dir_list + ["tf_trace"])
                    )
                )

        if self._flag_vtk_export:
            self._export_as_vtk(nProc)
        
        if self._flag_npz_export:
            self._export_as_numpy()

    def _export_as_numpy(self, compressed : bool = False):
        str_export_file = os.path.join( *(self.base_export_dir_list + ["electrode_potentials.npz"]) )

        export_dict = {  "electrode_potentials" : self.res
                       , "source_location" : self.func_c_m_global
                       , "source_current"  : self.func_i_fin }

        np.savez(str_export_file, ** export_dict)


    def _export_as_vtk(self, nProc):
        # TODO in case of large results safe in tempfile
        m_y, m_z, m_x = np.meshgrid(
              self.electrodes.y
            , self.electrodes.z
            , self.electrodes.x
        )

        m_x = m_x.flatten()
        m_y = m_y.flatten()
        m_z = m_z.flatten()

        ray._private.utils.get_system_memory = lambda: psutil.virtual_memory().total
        ray.shutdown()
        ray.init(num_cpus=nProc, include_dashboard = False)

        m_x_flat_id = ray.put(m_x)
        m_y_flat_id = ray.put(m_y)
        m_z_flat_id = ray.put(m_z)

        res_id      = ray.put(self.res)

        l_ray_save_res_to_vtk = lambda t_idx : self._ray_save_res_to_vtk.remote(
                m_x_flat_id, m_y_flat_id,
                m_z_flat_id, res_id, 
                self.base_export_dir_list, t_idx
            )

        ray.get([ l_ray_save_res_to_vtk(t_idx) for t_idx in range(0, len(self.time)) ])

        func_c_m_global_id = ray.put(self.func_c_m_global)
        func_i_fin_id      = ray.put(self.func_i_fin)

        l_ray_save_source_to_vtk = lambda t_idx : self._ray_save_source_to_vtk.remote(
                func_c_m_global_id
                ,func_i_fin_id
                , self.base_export_dir_list
                , t_idx)

        ray.get([ l_ray_save_source_to_vtk(t_idx) for t_idx in range(0, len(self.time)) ])


    @ray.remote
    def _ray_save_source_to_vtk(func_c_m_global, func_i_fin, dir_name_list, t_idx):
        func_c_x = func_c_m_global[:, 0, 0, :, t_idx].flatten()
        func_c_y = func_c_m_global[:, 1, 0, :, t_idx].flatten()
        func_c_z = func_c_m_global[:, 2, 0, :, t_idx].flatten()
        func_i   = func_i_fin[:, :, t_idx].flatten()

        f_path = dir_name_list + ["sources_t_{}".format(t_idx)]
        f_str = os.path.join(*f_path)

        evtk.hl.pointsToVTK(  f_str
                            , x = func_c_x
                            , y = func_c_y
                            , z = func_c_z
                            , data = { 'i' : func_i}
            )
        return 0

    def get_c_m(self, lr , xyz, c_id, t):
        return self.func_c_m_global[lr,xyz,0,c_id,t]
