# LIPS : Learning Industrial physical simulation benchmark suite: the power grid case
This repository implements the benchmarking platform called LIPS and provides the necessary utilities to reproduce the generated datasets used in research.

The Readme file is organized as follows:

*   [1 Introduction](#introduction)
    * [1.1 What is LIPS?](#what-is-lips)
*   [2 Usage example](#usage-example)
    * [2.1 Train a simulator](#train-a-simulator)
    * [2.2 Reproducibility and evaluation](#reproducibility-and-evaluation)  
*   [3 Installation](#installation)
    *   [3.1 Setup a Virtualenv (optional)](#setup-a-virtualenv-optional)
    *   [3.2 Install from source](#install-from-source)
    *   [3.3 To contribute](#to-contribute)
*   [4 Getting Started](#getting-started)
*   [5 Documentation](#documentation)
*   [6 Contribution](#contribution)
*   [7 License information](#license-information)

## Introduction
Nowdays, the simulators are used in every domain to emulate a real-world situation or event or to reproduce the critical situations for which further investigation may be required. The simulators are based generally on physics equations and are costly in terms of time complexity. 

### What is LIPS
The learning industrial physical simulation benchmark suite allows to evaluate the performance of augmented simulators (aka surrogate models) specialized in a physical domain with respect to various evaluation criteria. The implementation is enough flexible to allow its adaptation to various domains such as power grids, transport, aeronotics etc. To do so, as it is depicted in the scheme provided in the figure below, the platform is designed to be modular and includes following modules: 

- the **Data** module of the platform may be used to import the required datasets or to generate some synthetic data (for power grids for now) 
- A **simulator** may access the provided data to train or evaluate the peformance. The developed platform gives also the flexibility to its users to design and implement their own simulators and evaluate its performance with baselines. Various baseline simulators are already implemented and could be used, e.g., Direct Current (DC) approximation and neural netowrk based simulators which are _Fully Connected_ (FC) model and _Latent Encoding of Atypical Perturbations network_ [(LEAP net)](https://github.com/BDonnot/leap_net).
- The **Evaluation** module allows to select the appropriate criteria among the available implemented metrics. Four category of metrics are provided, which are : 
  - ML-related metrics 
  - Physic compliance
  - Industrial readiness
  - Generalization metrics 
  
![Scheme](./img/Benchmarking_scheme_v2.png)

## Usage example
### Instantiate a benchmark for power grid use case
The paths should correctly point-out to generated data ([DATA_PATH](https://github.com/Mleyliabadi/LIPS/tree/main/reference_data)) and benchmark associated config file ([CONFIG_PATH](https://github.com/Mleyliabadi/LIPS/blob/main/lips/config/conf.ini)). The log path (`LOG_PATH`) could be set by the user.

```python
from lips.benchmark import PowerGridBenchmark

benchmark1 = PowerGridBenchmark(benchmark_name="Benchmark1",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH
                               )
```
### Train a simulator
A simulator (based on tensorflow) could be instantiated and trained if required easily as follows:
```python
from lips.augmented_simulators.tensorflow_models import TfFullyConnected
from lips.dataset.scaler import StandardScaler

tf_fc = TfFullyConnected(name="tf_fc",
                         bench_config_name="Benchmark1",
                         scaler=StandardScaler,
                         log_path=LOG_PATH)
                           
tf_fc.train(train_dataset=benchmark1.train_dataset,
            val_dataset=benchmark1.val_dataset,
            epochs=100
           )

```
For each architecture a config file is attached which are available [here](https://github.com/Mleyliabadi/LIPS/tree/main/lips/augmented_simulators/configurations).
### Reproducibility and evaluation 
The following script show how to use the evaluation capacity of the platform to reproduce the results on all the datasets. A config file (see [here](https://github.com/Mleyliabadi/LIPS/blob/main/lips/config/conf.ini)) is associated with this benchmark and all the required evaluation criteria can be set in this configuration file. 

```Python
tf_fc_metrics = benchmark1.evaluate_simulator(augmented_simulator=tf_fc,
                                              eval_batch_size=128,
                                              dataset="all",
                                              shuffle=False
                                             )                  
```

## Installation
To be able to run the experiments in this repository, the users should install the last lips package from its github repository. The following steps show how to install this package and its dependencies from source.

### Requirements
- Python >= 3.6

### Setup a Virtualenv (optional)
#### Create a virtual environment

```commandline
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv venv_lips
```
#### Enter virtual environment
```commandline
source venv_lips/bin/activate
```

### Install from source
```commandline
git clone https://github.com/Mleyliabadi/LIPS
cd LIPS
pip3 install -U .
cd ..
```

### To contribute
```commandline
pip3 install -e .[recommended]
```

# Getting Started
Some Jupyter notebook are provided as tutorials for LIPS package. They are located in the 
[getting_started](getting_started) directories.   

# Documentation
The documentation is accessible from [here](https://lips.readthedocs.io/en/latest/index.html).

To generate locally the documentation:
```commandline
pip install sphinx
pip install sphinx-rtd-theme
cd docs
make clean
make html
```

# Contribution
* Supplementary features could be requested using github issues. 
* Other contributions are welcomed and can be integrated using pull requests.

# FAQ
To be able to use the torch library with GPU, you should consider multiple factors: 
* if you have a compatible GPU, in this case you can install the last cuda driver (11.6) and install torch using the following command:
```commandline
pip install torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```
To take the advantage of the GPU when training models, you should indicate it via the `device` parameter as follows:
```python
from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected
from lips.augmented_simulators.torch_simulator import TorchSimulator
from lips.dataset.scaler import StandardScaler

torch_sim = TorchSimulator(name="torch_fc",
                           model=TorchFullyConnected,
                           scaler=StandardScaler,
                           device="cuda:0",
                          )
```

* Otherwise, if you want use only CPU for the training of augmented simulators, you could simply use the version installed following the the requirements and set the device parameter to `cpu` when training as follows:
```python
torch_sim = TorchSimulator(name="torch_fc",
                           model=TorchFullyConnected,
                           scaler=StandardScaler,
                           device="cpu",
                          )
```

# License information
Copyright 2022-2023 IRT SystemX & RTE

    IRT SystemX: https://www.irt-systemx.fr/
    RTE: https://www.rte-france.com/

This Source Code is subject to the terms of the Mozilla Public License (MPL) v2 also available 
[here](https://www.mozilla.org/en-US/MPL/2.0/)
