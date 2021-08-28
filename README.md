# LIPS : Learning Industrial physical simulation benchmark suite: the power grid case
This repository implements the benchmarking platform called LIPS and provides the necessary utilities to reproduce the generated datasets used in research.

- [ ] add a readme in each module to explain how it works
- [ ] create a folder to add suplementary results like data volume experiment (and other)
- 
## Introduction
Nowdays, the simulators are used in every domain to emulate a real-world situation or event or to reproduce the critical situations for which further investigation may be required. The simulators are based generally on physics equations and are costly in terms of time complexity. 

## What is LIPS
The learning industrial physical simulation benchmark suite allows to evaluate the performance of augmented simulators (aka surrogate models) specialized in a physical domain with respect to various evaluation criteria. The implementation is enough flexible to allow its adaptation to various domains such as power grids, transport, aeronotics etc. To do so, as it is depicted in the scheme provided in the figure below, the platform is designed to be modular and includes following modules: 

- the **Data** module of the platform may be used to import the required datasets or to generate some synthetic data (for power grids for now) 
- A **simulator** may access the provided data to train or evaluate the peformance. The developed platform gives also the flexibility to its users to design and implement their own simulators and evaluate its performance with baselines. Various baseline simulators are already implemented and could be used, e.g., Direct Current (DC) approximation and neural netowrk based simulators which are _Fully Connected_ (FC) model and _Latent Encoding of Atypical Perturbations network_ [(LEAP net)](https://github.com/BDonnot/leap_net).
- The **Evaluation** module allows to select the appropriate criteria among the available implemented metrics. Four category of metrics are provided, which are : 
  - ML-related metrics 
  - Physic compliance
  - Industrial readiness
  - Generalization metrics 
  
![Scheme](./img/Benchmarking_scheme_v2.png)

## Requirements
To be able to use the LIPS platform, the following packages should be installed via `pip`:
- `leap_net` package using the instructions provided at [this link](https://github.com/BDonnot/leap_net)
- `Grid2Op` package using the instructions provided at [this link](https://github.com/rte-france/Grid2Op)

## Train a simulator

A simulator could be instantiated and trained if required easily as follows :
```python
from lips.augmented_simulators import FullyConnectedAS

my_simulator = FullyConnectedAS(name="test_FullyConnectedAS",
                                attr_x=("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect"),
                                attr_y=("a_or", "a_ex"),
                                sizes_layer=(300, 300, 300, 300),
                                lr=3e-4, 
                                layer=Dense,
                                layer_act="relu",
                                loss,
                                batch_size)
                           
my_simulator.train(nb_iter,
                   train_dataset,
                   val_dataset)

```

## Reproducibility and evaluation 
To reproduce the results of the submitted paper at _NeurIPS2021 Benchmark and Dataset track_, a class called `NeuripsBenchmark1` is provided which is sub class of a more general `Benchmark` class and whose purpose is to facilitate the data generation and the evaluation process. All the experimented datasets are already generated and provided under the _reference_data_ folder. The following script show how to use it quickly to reproduce the results : 

```Python
from lips.neurips_benchmark import NeuripsBenchmark1

# load the data from the provided path
path_benchmark = os.path.join("reference_data")
neurips_benchmark1 = NeuripsBenchmark1(path_benchmark=path_benchmark,
                                       load_data_set=True)

# evaluate an augmented simulator
metrics_per_dataset = neurips_benchmark1.evaluate_augmented_simulator(my_simulator)                   
```
