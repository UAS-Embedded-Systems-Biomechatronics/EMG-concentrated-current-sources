// Copyright 2023 Malte Mechtenberg
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use numpy::{
    ndarray::{Array, Array1, Array2, ArrayD, ArrayViewD, ArrayViewMutD, Dim},
    IntoPyArray, Ix1, IxDyn, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray, PyReadonlyArrayDyn,
    ToPyArray,
};
use pyo3::prelude::*;
use pyo3::types::*;

use rand::prelude::*;
use rand_distr::StandardNormal;

use rand::Rng;

use std::time::{Duration, Instant};

struct FiringRatePeterson2019 {
    start_common_drive: f64,
    c: [f64; 7],
}

impl FiringRatePeterson2019 {
    fn new(motor_unit_config: &PyAny) -> FiringRatePeterson2019 {
        let fb = motor_unit_config.getattr("firing_behavior").unwrap();

        let fc = fb.getattr("firing_frequenzy").unwrap();

        let mut c: [f64; 7] = [0.0; 7];

        let start_common_drive: f64 = fb
            .getattr("start_common_drive")
            .unwrap()
            .downcast::<PyFloat>()
            .unwrap()
            .extract()
            .unwrap();

        for (pos, key) in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
            .iter()
            .enumerate()
        {
            c[pos] = fc
                .getattr(*key)
                .unwrap()
                .downcast::<PyFloat>()
                .unwrap()
                .extract()
                .unwrap();
        }

        FiringRatePeterson2019 {
            start_common_drive,
            c,
        }
    }

    fn get(&self, common_drive: f64) -> Option<f64> {
        if common_drive < self.start_common_drive {
            return None;
        }

        let e_exponent = -(common_drive - self.start_common_drive) / (self.c[6]);
        let firing_rate = -self.c[0] * (self.c[1] - common_drive) * self.start_common_drive
            + self.c[2] * common_drive
            + self.c[3]
            - (self.c[4] - self.c[5] * common_drive) * e_exponent.exp();

        return Some(firing_rate);
    }
}

trait CdT {
    fn cd_t(&self, t: f64) -> f64;
}

struct PyCdT<'a> {
    py: Python<'a>,
    py_function: &'a pyo3::types::PyFunction,
}

impl PyCdT<'_> {
    fn new<'a>(py: Python<'a>, py_function: &'a pyo3::types::PyFunction) -> PyCdT<'a> {
        PyCdT { py, py_function }
    }
}

impl CdT for PyCdT<'_> {
    fn cd_t(&self, t: f64) -> f64 {
        let args = (t,);

        self.py_function
            .call(args, None)
            .unwrap()
            .downcast::<PyFloat>()
            .unwrap()
            .extract()
            .unwrap()
    }
}

enum GenericCdFunctions {
    Trapez,
}
struct RCdT {
    kind: GenericCdFunctions,
    arg_list: Vec<f64>,
}

impl RCdT {
    fn new(kind: GenericCdFunctions, arg_list: &Vec<f64>) -> RCdT {
        RCdT {
            kind,
            arg_list: arg_list.to_vec(),
        }
    }
}

impl CdT for RCdT {
    fn cd_t(&self, t: f64) -> f64 {
        match self.kind {
            GenericCdFunctions::Trapez => {
                assert!(self.arg_list.len() == 4);

                trapezoid_function(
                    t,
                    self.arg_list[0],
                    self.arg_list[1],
                    self.arg_list[2],
                    self.arg_list[3],
                )
            }
        }
    }
}

struct TimeSpan {
    start: f64,
    end: f64,
}

impl TimeSpan {
    fn from_py_tuple(pt: &PyTuple) -> TimeSpan {
        assert!(pt.len() == 2, "time span tuple has to have size 2");
        let start: f64 = pt[0].extract().unwrap();
        let end: f64 = pt[1].extract().unwrap();

        TimeSpan { start, end }
    }
}

fn generate_firing_instances_peterson_2019<T>(
    cd_t: &Box<T>,
    time_span: TimeSpan,
    firing_rate: FiringRatePeterson2019,
) -> Array1<f64>
where
    T: CdT + ?Sized,
{
    let t_basic_stepsize = 1e-5;

    let mut rng = thread_rng();
    //let mut rng = SmallRng::from_entropy();

    let mut firing_instances: Vec<f64> = Vec::with_capacity(1_000);
    let mut fr: f64;

    let mut t = time_span.start;

    while t <= time_span.end {
        let cd = cd_t.cd_t(t);

        match firing_rate.get(cd) {
            Some(value) => fr = value,
            None => {
                t += t_basic_stepsize;
                continue;
            }
        };

        let isi_star_j = 1.0 / fr;

        let a = (-(cd - firing_rate.start_common_drive)) / (2.5);
        let c = (10.0 + 20.0 * a.exp()) / 100.0;

        let sd = c * isi_star_j;
        let normal_rand: f64 = rng.sample(StandardNormal);
        let isi_j: f64 = normal_rand * sd + isi_star_j;

        t += isi_j;
        firing_instances.push(t);
    }

    Array1::from(firing_instances)
}

// 'wrap_generate_firing_instances_peterson_2019' is the function that is called from python
//
// the arguement py is consumed and contains a reference to the python ineterpreter that is calling
// this function.
//
// kind encodes the commondrive function kind as a string valid values arguement
//      - "trapez": in this case args is a list of parameters that is used to configure the trapez
//      function
//      - "PyFn": in this case args is a reference to a python function (t: f64) -> f64
//      that takes one skalar argument that is the time and reuturns the common drive a time t.
//
#[pyfunction]
#[pyo3(name = "batch_generate_firing_instances_peterson_2019")]
fn wrap_batch_generate_firing_instances_peterson_2019<'a>(
    py: Python<'a>,
    motor_unit_configs: &'a PyList,
    time_span: &'a PyTuple,
    commondrive_kind: &str,
    args: &PyAny,
) -> &'a PyList {
    //let t_start = Instant::now();
    let cd_t: Option<Box<dyn CdT>> = match commondrive_kind {
        "trapez" => {
            let arg_list: Vec<f64> = args.downcast::<PyList>().unwrap().extract().unwrap();
            Some(Box::new(RCdT::new(GenericCdFunctions::Trapez, &arg_list)))
        }
        "PyFn" => Some(Box::new(PyCdT::new(
            py,
            args.downcast::<PyFunction>().unwrap(),
        ))),
        _ => None,
    };

    let cd_t = cd_t.unwrap();

    let mut results: Vec<&PyArray1<f64>> = Vec::with_capacity(1000);

    for motor_unit_config in motor_unit_configs {
        let frp2019 = FiringRatePeterson2019::new(motor_unit_config);
        let ar1 = generate_firing_instances_peterson_2019(
            &cd_t,
            TimeSpan::from_py_tuple(time_span),
            frp2019,
        );

        results.push(ar1.to_pyarray(py))
    }

    //let duration = t_start.elapsed();
    //eprintln!("Time elapsed {:?}", duration);

    PyList::new(py, results)
}

// 'wrap_generate_firing_instances_peterson_2019' is the function that is called from python
//
// the arguement py is consumed and contains a reference to the python ineterpreter that is calling
// this function.
//
// kind encodes the commondrive function kind as a string valid values arguement
//      - "trapez": in this case args is a list of parameters that is used to configure the trapez
//      function
//      - "PyFn": in this case args is a reference to a python function (t: f64) -> f64
//      that takes one skalar argument that is the time and reuturns the common drive a time t.
//
#[pyfunction]
#[pyo3(name = "generate_firing_instances_peterson_2019")]
fn wrap_generate_firing_instances_peterson_2019<'a>(
    py: Python<'a>,
    motor_unit_config: &'a PyAny,
    time_span: &'a PyTuple,
    commondrive_kind: &str,
    args: &PyAny,
) -> &'a PyArray1<f64> {
    //let t_start = Instant::now();
    let a: Option<Box<dyn CdT>> = match commondrive_kind {
        "trapez" => {
            let arg_list: Vec<f64> = args.downcast::<PyList>().unwrap().extract().unwrap();
            Some(Box::new(RCdT::new(GenericCdFunctions::Trapez, &arg_list)))
        }
        "PyFn" => Some(Box::new(PyCdT::new(
            py,
            args.downcast::<PyFunction>().unwrap(),
        ))),
        _ => None,
    };

    let a = a.unwrap();

    let frp2019 = FiringRatePeterson2019::new(motor_unit_config);
    let ar1 =
        generate_firing_instances_peterson_2019(&a, TimeSpan::from_py_tuple(time_span), frp2019);

    //let duration = t_start.elapsed();
    //eprintln!("Time elapsed {:?}", duration);

    ar1.to_pyarray(py)
}

#[pyfunction()]
fn get_firing_rate<'a>(
    py: Python<'a>,
    motor_unit_config: &'a PyAny,
    common_drive: PyReadonlyArrayDyn<'a, f64>,
) -> &'a PyArray1<f64> {
    let common_drive = common_drive.as_array().into_owned();
    let shape = common_drive.shape();

    let mut firing_rate = Array1::zeros(shape[0]);

    let frp2019 = FiringRatePeterson2019::new(motor_unit_config);

    for (pos, c_t) in common_drive.iter().enumerate() {
        let a = match frp2019.get(*c_t) {
            Some(value) => value,
            None => f64::NAN,
        };
        firing_rate[pos] = a;
    }

    firing_rate.to_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn lib<'py>(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        wrap_generate_firing_instances_peterson_2019,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        wrap_batch_generate_firing_instances_peterson_2019,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(get_firing_rate, m)?)?;

    Ok(())
}

fn trapezoid_function(t: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    if (a <= t) && (t < b) {
        return (t - a) / (b - a);
    }
    if (b <= t) && (t <= c) {
        return 1.0;
    }
    if (c < t) && (t <= d) {
        return (d - t) / (d - c);
    } else {
        return 0.0;
    }
}
