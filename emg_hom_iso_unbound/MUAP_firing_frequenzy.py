from numba.core.types import common
from emg_hom_iso_unbound import model_config
import numpy as np

from numba import njit

class FiringTrain_generator():
    def get_firing_rate(self, common_drive):
        raise NotImplemented("get_firing_rate is not implemented")

    def __init__(self, motor_unit_config : model_config.motorUnit) -> None:
        self.mu_config = motor_unit_config

        if isinstance(
            self.mu_config.firing_behavior.firing_frequenzy
            , model_config.firingFreq_Petersen2019):

            self.firingFreq_Petersen = self.mu_config\
                .firing_behavior\
                .firing_frequenzy

            self.get_firing_rate = lambda common_drive: \
                get_firing_rate_Petersen2019(
                  common_drive
                , start_common_drive=self.mu_config\
                    .firing_behavior \
                    .start_common_drive
                , C1 = self.firingFreq_Petersen.C1
                , C2 = self.firingFreq_Petersen.C2
                , C3 = self.firingFreq_Petersen.C3
                , C4 = self.firingFreq_Petersen.C4
                , C5 = self.firingFreq_Petersen.C5
                , C6 = self.firingFreq_Petersen.C6
                , C7 = self.firingFreq_Petersen.C7
            )

@njit
def get_firing_rate_Petersen2019(common_drive : float
                                 , start_common_drive: float
                                 , C1, C2, C3, C4, C5, C6, C7):
    if common_drive < start_common_drive:
        return np.nan

    e_exponent = - (
        (common_drive - start_common_drive)
        / (C7)
    )

    firing_rate = (
        - C1 * ( C2 - common_drive ) * start_common_drive
        + C3 * common_drive
        + C4
        - (C5 - C6 * common_drive) * np.exp(e_exponent)
    )

    return firing_rate
