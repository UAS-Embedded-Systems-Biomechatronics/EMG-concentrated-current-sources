import unittest
import pickle
from emg_hom_iso_unbound import MUAP_firing_frequenzy, model_config

import numpy as np

import matplotlib.pyplot as plt

class test_MUAP_firing_frequenzy(unittest.TestCase):

    def test_basic_config(self):
        mu_c = model_config.motorUnit()
        mu_c.firing_behavior = model_config.firingBehavior()

        ftg = MUAP_firing_frequenzy.FiringTrain_generator(mu_c)

        self.assertEqual(ftg.get_firing_rate(0), 0.0)

    def test_recti_cofnig(self):
        mu_c = model_config.motorUnit()
        mu_c.firing_behavior = model_config.firingBehavior()
        
        mu_c.firing_behavior.firing_frequenzy.C1 = 20.0
        mu_c.firing_behavior.firing_frequenzy.C2 =  1.5 
        mu_c.firing_behavior.firing_frequenzy.C3 = 30
        mu_c.firing_behavior.firing_frequenzy.C4 = 13
        mu_c.firing_behavior.firing_frequenzy.C5 =  8
        mu_c.firing_behavior.firing_frequenzy.C6 =  8
        mu_c.firing_behavior.firing_frequenzy.C7 =  0.05


        lams = []
        scds = [0, 0.1, 0.15, 0.25, 0.3]
        cd_vec = np.arange(0,1, 1e-3)

        for scd in scds:
            mu_c.firing_behavior.start_common_drive = scd
            ftg = MUAP_firing_frequenzy.FiringTrain_generator(mu_c)

            lam = [ ftg.get_firing_rate(cd) for cd in cd_vec]
            lams.append(lam)
            plt.plot(cd_vec, lam, label="start_common_drive = {}".format(scd))

        plt.ylim((0,45))
        plt.legend()
        plt.show()

    def test_gen_firing_instances_Petersen2019(self):
        mu_c = model_config.motorUnit()
        def fn_common_drive(t):
            if t<1:
                return 0.01
            elif 1 <= t <= 2:
                return 0.8
            else:
                return 0.01

        mu_c.firing_behavior = model_config.firingBehavior()
        mu_c.firing_behavior.firing_frequenzy.C1 = 20.0
        mu_c.firing_behavior.firing_frequenzy.C2 =  1.5 
        mu_c.firing_behavior.firing_frequenzy.C3 = 30
        mu_c.firing_behavior.firing_frequenzy.C4 = 13
        mu_c.firing_behavior.firing_frequenzy.C5 =  8
        mu_c.firing_behavior.firing_frequenzy.C6 =  8
        mu_c.firing_behavior.firing_frequenzy.C7 =  0.05
        
        mu_c.firing_behavior.start_common_drive = 0

        ftg = MUAP_firing_frequenzy.FiringTrain_generator(mu_c
                                                    , fn_common_drive
                                                    , (0,3))

        firing_vec = np.array(ftg.generate_firing_instances())

        a = np.ones_like(firing_vec)
        a = np.linspace(0, 1, len(firing_vec))
        plt.plot(firing_vec,a , '-*')
        plt.show()

    def test_gen_firing_instances_Petersen2019_scd0(self):
        mu_c = model_config.motorUnit()

        param_dict = { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4 }

        fn_common_drive = lambda t: \
            MUAP_firing_frequenzy.trapezoid_function(t, **param_dict)


        mu_c.firing_behavior = model_config.firingBehavior()
        mu_c.firing_behavior.firing_frequenzy.C1 = 20.0
        mu_c.firing_behavior.firing_frequenzy.C2 =  1.5 
        mu_c.firing_behavior.firing_frequenzy.C3 = 30
        mu_c.firing_behavior.firing_frequenzy.C4 = 13
        mu_c.firing_behavior.firing_frequenzy.C5 =  8
        mu_c.firing_behavior.firing_frequenzy.C6 =  8
        mu_c.firing_behavior.firing_frequenzy.C7 =  0.05
        
        mu_c.firing_behavior.start_common_drive = 0

        time_span = (0,5)
        ftg = MUAP_firing_frequenzy.FiringTrain_generator(mu_c
                                                    , fn_common_drive
                                                    , time_span)

        firing_vec = np.array(ftg.generate_firing_instances())

        t_vec = np.linspace(time_span[0],time_span[1], int(1e3))
        y_vec = [MUAP_firing_frequenzy.trapezoid_function(t, **param_dict) for t in t_vec]

        a = np.linspace(0, 1, len(firing_vec))
        a = np.ones_like(firing_vec)
        a = np.zeros_like(firing_vec)
        plt.plot(t_vec, y_vec, '-k')
        plt.plot(firing_vec,a , '*', alpha=0.4)
        plt.title("start_common_drive (scd) = {}".format(
            mu_c.firing_behavior.start_common_drive))
        plt.show()

    def test_gen_firing_instances_Petersen2019_scd0_2(self):
        mu_c = model_config.motorUnit()

        param_dict = { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4 }

        fn_common_drive = lambda t: \
            MUAP_firing_frequenzy.trapezoid_function(t, **param_dict)


        mu_c.firing_behavior = model_config.firingBehavior()
        mu_c.firing_behavior.firing_frequenzy.C1 = 20.0
        mu_c.firing_behavior.firing_frequenzy.C2 =  1.5 
        mu_c.firing_behavior.firing_frequenzy.C3 = 30
        mu_c.firing_behavior.firing_frequenzy.C4 = 13
        mu_c.firing_behavior.firing_frequenzy.C5 =  8
        mu_c.firing_behavior.firing_frequenzy.C6 =  8
        mu_c.firing_behavior.firing_frequenzy.C7 =  0.05
        
        mu_c.firing_behavior.start_common_drive = 0.2

        time_span = (0,5)
        ftg = MUAP_firing_frequenzy.FiringTrain_generator(mu_c
                                                    , fn_common_drive
                                                    , time_span)

        firing_vec = np.array(ftg.generate_firing_instances())

        t_vec = np.linspace(time_span[0],time_span[1], int(1e3))
        y_vec = [MUAP_firing_frequenzy.trapezoid_function(t, **param_dict) for t in t_vec]

        import pdb; pdb.set_trace()
        a = np.linspace(0, 1, len(firing_vec))
        a = np.ones_like(firing_vec)
        a = np.zeros_like(firing_vec)
        plt.plot(t_vec, y_vec, '-k')
        plt.plot(firing_vec,a , '*', alpha=0.4)
        plt.title("start_common_drive (scd) = {}".format(
            mu_c.firing_behavior.start_common_drive))
        plt.show()


    def test_trapez(self):
        param_dict = { 'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3 }

        #t_vec = np.linspace(0,3, int(1e3))
        #y_vec = [MUAP_firing_frequenzy.trapezoid_function(t, **param_dict) for t in t_vec]
        #plt.plot(t_vec, y_vec)
        #plt.show()

        self.assertEqual(MUAP_firing_frequenzy.trapezoid_function(0, **param_dict), 0)
        self.assertEqual(MUAP_firing_frequenzy.trapezoid_function(1, **param_dict), 1)
        self.assertEqual(MUAP_firing_frequenzy.trapezoid_function(2, **param_dict), 1)
        self.assertEqual(MUAP_firing_frequenzy.trapezoid_function(3, **param_dict), 0)
