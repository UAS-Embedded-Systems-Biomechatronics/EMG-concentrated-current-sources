use numpy::{
    ndarray::{Array, Array1, ArrayD, ArrayViewD, ArrayViewMutD, Dim},
    IntoPyArray, Ix1, IxDyn, PyArray1, PyArrayDyn, PyReadonlyArray, PyReadonlyArrayDyn, ToPyArray,
};
use pyo3::prelude::*;
use pyo3::types::*;

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

    fn get(&self, common_drive: f64) -> f64 {
        if common_drive < self.start_common_drive {
            return f64::NAN;
        }

        let e_exponent = -(common_drive - self.start_common_drive) / (self.c[6]);
        let firing_rate = -self.c[0] * (self.c[1] - common_drive) * self.start_common_drive
            + self.c[2] * common_drive
            + self.c[3]
            - (self.c[4] - self.c[5] * common_drive) * e_exponent.exp();

        return firing_rate;
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
                println!("use trapez");
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

fn generate_firing_instances_peterson_2019<T>(cd_t: T, firing_rate: FiringRatePeterson2019)
where
    T: CdT,
{
    let t = 1.0;
    firing_rate.get(3.3);
    cd_t.cd_t(t);
}

#[pyfunction]
fn wrap_generate_firing_instances_peterson_2019<'a>(
    py: Python<'a>,
    kind: &str,
    args: &PyAny,
) -> PyArray1<f64> {
    let a: Option<Box<dyn CdT>> = match kind {
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

    a.unwrap().cd_t(1.0);
}

fn np_vec_test(a: ArrayViewD<'_, f64>, b: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    &a + &b + 100.0
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
        firing_rate[pos] = frp2019.get(*c_t);
    }

    firing_rate.to_pyarray(py)
}

#[pyfunction]
/// This is an example on how to call a generic python function from
/// Rust. This can be used as a template for user to pass python functions
/// that need to be run.
///
/// This approach needs a vaild python gil reference.
///
fn cf<'a>(py: Python<'a>, f: &'a pyo3::types::PyFunction) -> Result<&'a PyAny, PyErr> {
    let args = ("hallo from rust",);
    let kwargs = pyo3::types::PyDict::new(py);
    println!("call python function:");
    let _ = f.call(args, Some(kwargs));
    f.call(args, None)
}

#[pyfunction]
#[pyo3(name = "axpy")]
fn axpy_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
) -> &'py PyArrayDyn<f64> {
    let a = a.as_array();
    let b = b.as_array();

    let z = np_vec_test(a, b);
    z.into_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn lib<'py>(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(axpy_py, m)?)?;
    m.add_function(wrap_pyfunction!(cf, m)?)?;
    m.add_function(wrap_pyfunction!(
        wrap_generate_firing_instances_peterson_2019,
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
