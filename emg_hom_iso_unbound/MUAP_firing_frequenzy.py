from collections.abc import Callable
from typing import Tuple

from emg_hom_iso_unbound import model_config
import numpy as np

from numba import njit

import scipy.optimize

class FiringTrain_generator():
    def get_firing_rate(self, common_drive):
        raise NotImplemented("get_firing_rate is not implemented")

    def generate_firing_instances(self):
        raise NotImplemented("generate_firing_instances is not implemented")


    def __init__(self
                 , motor_unit_config : model_config.motorUnit
                 , fn_common_drive = None
                 , time_span = None
                 , firing_rage = None) -> None:

        print("init")
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

            if fn_common_drive is not None:
                self.generate_firing_instances = lambda : \
                    generate_firing_instances_Petersen2019(
                          CD_t = fn_common_drive
                        , time_span   = time_span
                        , firing_rate = self.get_firing_rate
                        , start_common_drive= self.mu_config\
                                .firing_behavior \
                                .start_common_drive
                    )


def find_next_activation(
      CD_t #: Callable[[float], float]
    , current_t          : float
    , start_common_drive : float
    , time_span          : Tuple[float,float]
    ):

    print ("stuff")

    def t_error_func(x):
        #x = x[0]
        d = CD_t(x) - start_common_drive
        gain = 1e4
        punish = 1
        if d < 0:
            punish = 2

        error =  d**2 * punish * gain
        print("CD {} sCD {}".format(CD_t(x), start_common_drive))
        print(f"error, {error}")
        return error


    maxiter = 1000
    iter    = 0

    while iter < maxiter:

        res = scipy.optimize.least_squares(t_error_func
                                , x0       = current_t
                                , method   ="dogbox"
                               , xtol=1e-10
                               , ftol=1e-10
                               , gtol=1e-10
                                , verbose  = 2)


        if CD_t(res.x) < start_common_drive:
            iter += 1
            current_t += 1e-4

        if res.x[0] < time_span[0] or res.x[0] > time_span[1]:
            return np.nan

    import pdb; pdb.set_trace()
    return res.x

def generate_firing_instances_Petersen2019(
          CD_t #: Callable[[float], float]
        , time_span #: Tuple[float, float]
        , firing_rate #: Callable[[float], float]
        , start_common_drive #: float
    ):
    """
    currently time_span[0] is not part of the firing instances
             - ( CD - CD_rec) 
        a = -----------------
                    2.5

                           (a)
        c(CD) = 10 + 20 * e

                     *                *
        ISI  ~ N (ISI (CD),c(CD) * ISI  )
           j         j                j
    """

    t = time_span[0]
    firing_instances = []

    #import pdb; pdb.set_trace()
    while t<= time_span[1]:
        CD = CD_t(t)
        if CD < start_common_drive:
            t = t + 1e-5
            continue

        ISI_star_j = 1 / firing_rate(CD)

        a = (
            (- (CD - start_common_drive))
            / (2.5)
        )
        c = (10 + 20 * np.exp(a)) / 100
        sd = c * ISI_star_j
        ISI_j = np.random.normal(ISI_star_j # scale is the standart deviation
                     , scale=sd)            #                 *
                                            # here c(CD) * ISI
                                            #                 j

        t = t + ISI_j
        firing_instances.append(t)


    return firing_instances

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


def combinational_ridge_function(
        t : float, a: float, b: float, c:float, d:float) -> float:
    """
        [source](https://www.researchgate.net/journal/Advances-in-Mechanical-Engineering-1687-8140/publication/275066324_Nonintrusive-Polynomial-Chaos-Based_Kinematic_Reliability_Analysis_for_Mechanisms_with_Mixed_Uncertainty/links/618cf542d7d1af224bd576c6/Nonintrusive-Polynomial-Chaos-Based-Kinematic-Reliability-Analysis-for-Mechanisms-with-Mixed-Uncertainty.pdf?origin=publicationDetail&_sg%5B0%5D=L1XTeWMqh0remyJw2J-4-i729ZF0AuvBNB21F9VKF6DVgPRPMeuByPQW-v3JO5m8UV5LsVsJqJuwGLA6g9QRGw.zvSLgomw89G0pf_MhuhCWriJ8iPL14Jel2dgllde1F6fhYCiP84IoOHZYUfe9FNI-UH-TdrsdbCJPFdXswOrDw&_sg%5B1%5D=sulgKJRyK4AezqsML21KUPvZdrm2BJ9qbMbpiEttVEe_KS_9wnQq1HsmXK0ZdcnXaBp7upgKpmfOvNgGu5S0b3UqB9gtCSE88XfhD5BvcrEn.zvSLgomw89G0pf_MhuhCWriJ8iPL14Jel2dgllde1F6fhYCiP84IoOHZYUfe9FNI-UH-TdrsdbCJPFdXswOrDw&_sg%5B2%5D=GzQnvN9Ivgw5xyDqi48BEHLVxRtbcLzNZveL3FtssKq-g2_xwAxaqU0fMG2ZJh7fEqXh6VaEg_WCDfo.LleS53e6GQNIC9eUKW39BQf2yok0bZGSmkAQTU0mIdkCZ42qJvLT1rKsJ0Srk2ldjAYeLnV7q0OPpw7olIVg9A&_iepl=&_rtd=eyJjb250ZW50SW50ZW50IjoibWFpbkl0ZW0ifQ%3D%3D)
    """

    if a <= t <b:
        return 0.5 + 0.5 * np.sin(np.pi/(b-a) * (t-(a+b)/2))
    if b<=t<= c:
        return 1
    if c < t <= d:
        return 0.5 + 0.5 * np.sin(np.pi/(c-d) * (t- (c+d)/2))

    return 0.0


def combinational_normal_function(t:float, a:float, b:float, k:float):
    if t< a:
        return np.exp(-k * t**2)
    if a<= t <= b:
        return 1
    else: #if t>b:
        return np.exp(-k*t**2)


def trapezoid_function(t:float , a:float, b:float, c:float, d:float)->float:
    if a<= t < b:
        return (t-a)/(b-a)
    if b<= t<=c:
        return 1 
    if c < t <= d:
        return (d - t) / (d-c)
    else:
        return 0

