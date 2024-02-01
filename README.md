[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7696161.svg)](https://doi.org/10.5281/zenodo.7696161)

# EMG-concentrated-current-sources

This software package contains a Python module for the simulation of electromyography (EMG) signals.
The present model is described in detail in the method paper by Mechtenberg and Schneider (2023, currently in rewiew).
With this simulation, it is possible to simulate a large number of muscle fibers for large electrode arrays in a short computation time.
This relatively short computation time, compared to other available EMG simulations (like [ime-luebeck/semgsim](https://github.com/ime-luebeck/semgsim)) is possible due to the implementation with TensorFlow and the following physical assumptions:

The EMG signal source is the transmembrane current of myoelectric action potentials.
These action potentials travel along the muscle fibers after neuronal excitation of the muscle fibers.
In this simulation, the transmembrane current is modeled with concentrated current sources (point current sources and sinks).
The medium in which the muscle fibers are located is assumed to be homogeneous, isotropic, and purely
resistive. The EMG electrodes are simulated as point electrodes. The electrode potentials are
calculated under qusistatic conditions.

## Installation

An Anaconda enviroment file, containing all nessecary dependencies, is provided with `./environment.yml`.
To install the environement you need to have anaconda installed. If anaconda is installed and in your
system `PATH` execute:

```bash
conda env create -f ./environment.yml
conda activate semg-model_hiu_v0_2_2
```

If you are using jupyter you can install a jupyter kernel for this environement
with the following instructions.
First make shure that you are in the conda environemnt that you want to add to you jupyter kernels.
Than execute the command:

```bash
python3 -m ipykernel install --user --name <kernel name>
```

## Usage

An introduction to the simulation framework is given by the notebook `introduction.ipynb`.
Example simulation scripts are provided in `./examples`.

# License

This poject is provided under the [Apache License 2.0](LICENSE)

## Authors

- Malte Mechtenberg

# References

Mechtenberg, M and Schneider, A (2023) "A method for the estimation of a motor unitinnervation zone center evaluated with acomputational sEMG model" Frontiers in Neurorobotics. in rewiew.
