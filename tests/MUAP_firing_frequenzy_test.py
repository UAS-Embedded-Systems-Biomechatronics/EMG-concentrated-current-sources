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
