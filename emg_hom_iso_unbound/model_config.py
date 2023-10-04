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

import traits.api
import traits
import traits.has_traits

import typing

import json
import numpy as np

_cm = 0.01
_mm = 0.001


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, base_config):
            return dict(obj)

        return json.JSONEncoder.default(self, obj)

def JSON_config_factory(sj : str) -> 'base_config':
    """
    JSON_config_factory(sj : str) -> 'base_config'

    Whit this function the model config objects can
    be vuild from a json string.

    This function is usually used to load 
    exported model configurations (<project_dir>/config.json)
    that is created during the simulation.
    """
    return dict_config_factory(
        json.loads(sj)
    )

def dict_config_factory(d : dict) -> 'base_config':
    """
    dict_config_factory(d: dict) -> 'base_config'

    Whith this function the model config objects can 
    be defined by a json dict.

    Usually this function is not used directly.

    Note that you can load the exported json configuration
    with the function JSON_config_factory.
    """
    assert isinstance(d, dict), "arguemnt d is not a dict"

    for label, item in d.items():
        if isinstance(item, dict):
            d[label] = dict_config_factory(item)
        if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
            d[label] = [ dict_config_factory(list_entry) for list_entry in item]

    return globals()[d['sType']](**d)

class base_config(traits.api.HasTraits):
    def __init__(self, *args, **kwargs) -> None:
        self.add_trait('sType', traits.api.Enum(self.__class__.__name__))
        super(base_config, self).__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return self.as_json_s

    def __iter__(self) -> typing.Generator:
        """
        __iter__ and __dict__ overwrite is nedded to explicitly display traits in
        __dict__ property as of traits==6.3.2 this behavior is not implemented
        """
        for trait_name in self.trait_names(type=traits.has_traits.not_event):
            trait_object = getattr(self, trait_name)

            if self._filter_trait(trait_object, trait_name):
                continue

            if trait_object is None:
                continue

            if isinstance(trait_object, traits.api.HasTraits):
                yield (trait_name, dict(trait_object))
            else:
                yield (trait_name, trait_object)

    def _filter_trait(self, trait_object : traits.api.TraitType, trait_name : str) -> bool:
        return False

    @property
    def as_json_s(self) -> str:
        """
        Adopted from [1]
        [1] https://pynative.com/python-serialize-numpy-ndarray-into-json/
        """
        return json.dumps(dict(self), cls=NumpyArrayEncoder, indent=2)

# model configs


class CurrentSource(base_config):
    @property
    def sections(self) -> 'np.ndarray':
        raise NotImplementedError("this is a base class. A current Source should overwrite this property")
        

    @property
    def integration_boundaries(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("this is a base class. A current Source should overwrite this property")


class rosenfalck_current_source(CurrentSource):
    a = traits.api.Float(96)
    l = traits.api.Float(1000)
    b = traits.api.Float(-0.080)

    @property
    def sections(self) -> 'np.ndarray':
        return np.array(
            [
                [0                               , - (np.sqrt(3) - 3) / (self.l)]
                , [- (np.sqrt(3) - 3) / (self.l) ,   (np.sqrt(3) + 3) / (self.l)]
                , [(np.sqrt(3) + 3) / (self.l)   ,   14e-3                      ]
            ]
            , dtype=np.float64
        )

    @property
    def integration_boundaries(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        S = self.sections
        return S[:,0], S[:,1]

electrode_coordinate_type = traits.api.Union(
    traits.api.Float() , traits.api.Array(dtype = np.float64 , shape = (None))
    , desc  = "descirption"
    , label = "electrode_coordinate")

class electrodes(base_config):
    """
    electrodes(x = 0, y=0 ,z = 0, isGridConfig = True)

    Electrode locations within the global coordinate system.
    These are the locations where the potential field is evluated.

    Attributes
    ----------
    x : electrode_coordinate_type
    y : electrode_coordinate_type
    z : electrode_coordinate_type
        
        where the electrode_coordinate_type is either a *float* or a 
        *numpy.ndarray(dtype=np.float64)*
    
    isGridConfig : bool
        is an optional attribute. By default it is set to True.
        If it is True  the electrode positions used during simulation are
        are np.meshgrid(x,y,z, indexing='ij').
        
        If set to False
        the electrode positions x,y,z are interpreted
        as flattened coordinates. The electrode location p_n
        whould be
        
        p_n = np.array([  [x[n]]
                        , [y[n]]
                        , [z[n]]])
    """        
    x = electrode_coordinate_type
    y = electrode_coordinate_type
    z = electrode_coordinate_type
    
    isGridConfig = traits.api.Bool(True)

position_vector_type = traits.api.Array(
      dtype = np.float64
    , shape = (3,1)
    , value = [[0],[0],[0]]
)

class muscleFiberRotation(base_config):
    """
    muscleFiberRotation(z = 0, y = 0)

    Attributes
    ----------

    z : float
        rotation of the muscle fiber around the muscle
        fiber local z coordinate.

    y : float
        rotation of the muscle fiber around the muscle
        fiber local y coordinate.
    """

    z = traits.api.Float(0)
    y = traits.api.Float(0)


class MF_Merletti1999_hyperparameters(base_config):
    W_I = traits.api.Float(1*_cm)
    L_L = traits.api.Float(5*_cm)
    L_R = traits.api.Float(5*_cm)

    W_TL = traits.api.Float(0.5*_cm)
    W_TR = traits.api.Float(0.5*_cm)

    R   = traits.api.Float(1*_cm)
    
    P_IZ = traits.api.Array(shape=(3,), value=[0,0,0])

    def generate_muscle_config(self):
        return MF_Merletti1999(
            W_I= self.W_I,
            R  = self.R,

            L_L=self.L_L,
            W_TL=self.W_TL,

            L_R = self.L_R,
            W_TR= self.W_TR,

            p_IZ = self.P_IZ
        )



def MF_Merletti1999(W_I, R, L_L, W_TL, L_R, W_TR, p_IZ = np.array([0,0,0])):
    r"""                                                         x x x x x    <- electrodes
                                                   --------------------------------------------------------------------
    musclefiber zone in z,y plane                                    .                             ^ h
           XXXXXXX   --------                          W_TL          |            W_TR             |
         XX z ^   XX      |                         o-------o    o-------o    o---------o          |
       XX    /|\    XX    |                         |       |    |   |   |    |         |          |
      X       |    \  X   |                         |    o--+----+---+o--+----+-----o   |   z ^    |
      X       o-----> X   |2 * R                    |     o-+----+---+--o+----+o        |     |    |
      X          y /  X   |                         |  o----+----+---+-o-+----+--o      |     |    |
       XX           XX    |                         |      o+----+-o-+---+----+-----o   |     x------->
         XX       XX      |                         |o------+----+o--+---+----+--------o|           x
           XXXXXXX   --------                       |       |    |   |   |    |         |
                                                    o-------o    o-------o    o---------o
                                                        |--------|---|---|----------|
                                                       L_L       |   '   |         L_R
                                                                 |  x=0  |
                                                                 |-------|
                                                                     W_I

    """

    rand_vec = np.random.rand(5)
    r   = rand_vec[1] * R
    phi = rand_vec[2] * 2*np.pi

    dx_I_P = (rand_vec[0] - 0.5) * W_I
    I_P = np.array([
          [  dx_I_P + p_IZ[0] ] # x
        , [ r * np.cos(phi) + p_IZ[1] ]           # y
        , [ r * np.sin(phi) + p_IZ[2] ]           # z
        ])

    l_l =  np.abs(- L_L + (rand_vec[3]-0.5) * W_TL - dx_I_P) # actual distance between IP and lenft muscle fiber end
    l_r =  np.abs(  L_R + (rand_vec[4]-0.5) * W_TR - dx_I_P) # actual distance between IP and right muscle fiber end

    return _muscleFiber_factory(ip_location = I_P, left_length = l_l, right_length = l_r)


def _muscleFiber_factory(
          ip_location  : 'np.ndarray'
        , left_length  : 'float'
        , right_length : 'float'
        , orientation = None) -> 'muscleFiber':
    """
    A function that returns a muscle fiber configuration.
    Only the geometry parameters are set wiht this function.
    Rotation of the muscle fiber is not supported by this factory, 
    eaven thogh beeing supported by the simulation it self.

    ip_location : 'np.array'
        the location of the muscle fibers innervation point 
        in global coordinates

    left_length : 'float'
        distance from the muscle fiber innervation point to the 
        left myotendinous junction

    right_length : 'float'
        distance from the muscle fiber innervation point to the 
        right myotendinous junction
    """

    if not ( orientation == None ):
        raise NotImplementedError(
                "orientation is not implemented in this factory")

    return muscleFiber(
              IP = left_length 
            , L  = left_length + right_length
            , origin_location = ip_location - np.array([[left_length], [0], [0] ]))

class muscleFiber(base_config):
    """
    Configuration class describing the 
    muscle fiber parameters.

    Attributes
    ----------

    IP : Float
    L  : Float

    origin_location : position_vector_type
    rotation        : either muscleFiberRotation or None

    Prottotype Attributes
    ---------------------
    Prototype Attributes are types defined by the traits packge.
    The protptype Attributes listed here are linked to the motor unit
    a muscle fiber is part of. If these are not explicitly set 
    the muscle fiber uses the configuration from the motor unit.
    but they can be overwritten to individulize the muscle fiber.

    currentSource : CurrentSource
        The current source used to derive the concentrated current sources.
        It is recommended to use the *rosenfalck_current_source*.
        The rosenfalck_current_source is used by default.

    Re : float
        extracellular resistence of the core conducter
        model used to describe the trans membrane current source

    Ri : float
        intracelluar resistance of the core conducter
        model used to describe the trans membrane current source

    v  : float
        Conduction velocity in meters per second (m/s). This
        is the velocity at which the current soureces travel 
        along the muscle fiber.
    """
    IP = traits.api.Union(traits.api.Float(0.05), None )
    L  = traits.api.Union(traits.api.Float(0.10), None )

    origin_location = position_vector_type
    rotation        = traits.api.Instance(muscleFiberRotation, ())

    # electrical Parameters
    # overwrites parameters given by motorUnit
    currentSource = traits.api.Instance(rosenfalck_current_source)

    Re = traits.api.Union(None, traits.api.Float())
    Ri = traits.api.Union(None, traits.api.Float())
    v  = traits.api.Union(None, traits.api.Float())

    def _filter_trait(self
                      , trait_object : traits.api.TraitType
                      , trait_name : str) -> bool:
        if trait_name == "sType":
            return False

        if ('motor_unit' in self.trait_names(type=traits.has_traits.not_event)
            and not (self.motor_unit is None)
            and trait_name in self.motor_unit.trait_names(
                type=traits.has_traits.not_event)
            ):
            trait_object_parent = getattr(self.motor_unit, trait_name)
            return trait_object == trait_object_parent

        return trait_name == "motor_unit"

    def __init__(self, *args, **kwargs):
        self.add_trait('motor_unit', traits.api.Instance(motorUnit))

        super(muscleFiber, self).__init__(*args, **kwargs)

    def init_prototypes(self, mu : 'motorUnit') -> None:
        self.motor_unit = mu

        if self.Re is None:
            self.remove_trait('Re')
            self.add_trait('Re', traits.api.PrototypedFrom('motor_unit'))

        if self.Ri is None:
            self.remove_trait('Ri')
            self.add_trait('Ri', traits.api.PrototypedFrom('motor_unit'))

        if self.v is None:
            self.remove_trait('v')
            self.add_trait('v' , traits.api.PrototypedFrom('motor_unit'))

        if self.currentSource is None:
            self.remove_trait('currentSource')
            self.add_trait('currentSource' , traits.api.PrototypedFrom('motor_unit'))

class firingFreq_config(base_config):
    pass

class firingFreq_Petersen2019(firingFreq_config):
    C1 = traits.api.Float(0)
    C2 = traits.api.Float(0)
    C3 = traits.api.Float(0)
    C4 = traits.api.Float(0)
    C5 = traits.api.Float(0)
    C6 = traits.api.Float(0)
    C7 = traits.api.Float(1)


class firingBehavior(base_config):
    firing_frequenzy   = traits.api.Instance(firingFreq_config, firingFreq_Petersen2019)
    start_common_drive = traits.api.Float(0)

    def __init__(self, *args, **kwargs) -> None:
        super(firingBehavior, self).__init__(*args, **kwargs)



class motorUnit(base_config):
    """
    motorUnit(fibers : list, currentSource : CurrentSource, Re : float, Ri : float, v : float)

    Attributes
    ----------

    fibers : list
        List of fibers that are part of the motor
        unit

    currentSource : CurrentSource
        Current source that is used as prototype 
        for the muscle fibers that are part of this motor unit.

    Re : float
        extracellular resistence of the core conducter
        model used to describe the trans membrane current source.
        This is a prototype parameter for all fibers that are part 
        of this motor unit. Can be overwritten if set explicitly 
        for each muscle fiber.

    Ri : float
        intracelluar resistance of the core conducter
        model used to describe the trans membrane current source
        This is a prototype parameter for all fibers that are part 
        of this motor unit. Can be overwritten if set explicitly 
        for each muscle fiber.

    v  : float
        Conduction velocity in meters per second (m/s). This
        is the velocity at which the current soureces travel 
        along the muscle fibers.
        This is a prototype parameter for all fibers that are part 
        of this motor unit. Can be overwritten if set explicitly 
        for each muscle fiber.
    """
    fibers : 'traits.api.List' = traits.api.List(muscleFiber)

    @traits.api.observe("fibers.items")
    def _fibers_update(self, event):
        #print(type(event))
        init_on = []
        if isinstance(event,traits.observation._trait_change_event.TraitChangeEvent):
            init_on = event.new
        elif isinstance(event, traits.observation._list_change_event.ListChangeEvent):
            init_on = event.added

            if event.removed:
                raise NotImplementedError("removing muscle fibers from the motor unit")


        for fiber in init_on:
            fiber.init_prototypes(self)


    currentSource = traits.api.Instance(rosenfalck_current_source, ())

    Re = traits.api.Union(traits.api.Float(0.5), None)
    Ri = traits.api.Union(traits.api.Float(0.5), None)
    v  = traits.api.Union(traits.api.Float(5),   None) # m/s

    firing_behavior : firingBehavior = traits.api.Instance(firingBehavior)

    firing_pattern : 'traits.api.List' = traits.api.List(
        trait=traits.api.Float, value=[0.0], minlen=0)

class muscle(base_config):
    """
    Muscle configuration.

    Attributes
    ----------

    motor_units : list
        List of motor units that are part of a muscle.
    """
    motor_units : 'traits.api.List' = traits.api.List(motorUnit)

class root(base_config):
    """
    The root model configuration which is used 
    to fully define the simulation parameters.

    Attributes
    ----------

    time : numpy.ndarray
        One dimensional array defining the time points at which the
        the electrode potentials are calculated.

    electrodes : electrodes
        Electrodes object wich defines the electrode positions.

    muscle : muscle
        The muscle configuration.

    export_vtk : bool
        If True the simulation results per muscle fiber are exported
        in a Visualization Toolkit (by Kitware, VTK) compatible
        file format. This export method may be used to display
        simulation results in Paraview.

    export_pyz : bool
        If true export the simulation results per muscle fiber in the
        numpy file format (.npz)

        simulation results are accesable by
        
        ```python3
            npz_res = np.load(str_result_file, mmap_mode='r')

            npz_res['electrode_potentials']
            npz_res['source_location']
            npz_res['source_current']
        ```
        electrode_potentials is an array containing all electrode 
        potentials over time.

        sourec_location is an array containing the concentrated current sourece
        locations over time

        source_current is an array containing the respective concentrated current
        source strength

    For an interactive documentation please look into the jupyter notebook introduction.ipynb.
    """

    time = traits.api.Array(dtype=np.float64, shape = (None), casting='safe')

    calc_sum_potential = traits.api.Bool(False)

    electrodes = traits.api.Instance(electrodes)
    muscle     = traits.api.Instance(muscle)

    export_vtk = traits.api.Bool(False)
    export_pyz = traits.api.Bool(True)
