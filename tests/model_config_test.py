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
import emg_hom_iso_unbound.model_config as mc

import numpy as np



class test_BaseConfig(unittest.TestCase):

    def test_rosCurrentSource(self):
        rcs = mc.rosenfalck_current_source()
        sections = np.array(
            [
                  [0                            , - (np.sqrt(3) - 3) / (rcs.l)]
                , [- (np.sqrt(3) - 3) / (rcs.l) ,   (np.sqrt(3) + 3) / (rcs.l)]
                , [(np.sqrt(3) + 3) / (rcs.l)   ,   14e-3                      ]
            ]
            , dtype=np.float64
        )
        self.assertTrue( (rcs.sections == sections).all() )
        self.assertTrue((sections[:,0] == rcs.integration_boundaries[0]).all())
        self.assertTrue((sections[:,1] == rcs.integration_boundaries[1]).all())

    def testConfigTree(self):
        mu : 'mc.motorUnit' = mc.motorUnit()

        mf1 : 'mc.muscleFiber' = mc.muscleFiber()
        mf2 : 'mc.muscleFiber' = mc.muscleFiber()

        mu.fibers.append(mf1)
        mu.fibers.append(mf2)

        self.assertEqual(mu.Re, mf1.Re)
        self.assertEqual(mf2.Re, mf1.Re)

        mf1.Re = 10

        self.assertEqual(mf1.Re,    10)
        self.assertEqual(mf1.Re, mu.fibers[0].Re)
        self.assertEqual(mf2.Re, mu.fibers[1].Re)

        self.assertNotEqual(mf1.Re, mu.Re)

        self.assertEqual(mf2.Re,    mu.Re)

        mu_json     = mu.as_json_s
        mu_imported = mc.JSON_config_factory(mu_json)

        self.assertEqual(mu_imported.fibers[0].Re, mu.fibers[0].Re)
        self.assertEqual(mu_imported.fibers[1].Re, mu.fibers[1].Re)

        self.assertNotEqual(mu_imported.fibers[0].Re, mu_imported.fibers[1].Re)

        self.assertEqual(mu_imported.Ri, mu.Ri)

        self.assertEqual(len(mu_imported.fibers), len(mu.fibers))






if __name__ == '__main__':
    unittest.main()
