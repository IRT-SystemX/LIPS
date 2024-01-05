"""
Licence:
    Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""

import os
import shutil
import warnings
import copy
import json
import time
from typing import Union
import pathlib

import numpy as np

from lips.benchmark import Benchmark
from lips.augmented_simulators import AugmentedSimulator
from lips.dataset.airfransDataSet import AirfRANSDataSet,extract_dataset_by_simulations
from lips.evaluation.airfrans_evaluation import AirfRANSEvaluation
from lips.dataset.scaler.standard_scaler_iterative import iterative_fit
from lips.utils import NpEncoder

def reynolds_filter(dataset):
    simulation_names=dataset.extra_data["simulation_names"]
    reynolds=np.array([float(name.split('_')[2])/1.56e-5 for name,numID in simulation_names])
    simulation_indices=np.where((reynolds>3e6) & (reynolds<5e6))[0]
    return simulation_indices

class AirfRANSBenchmark(Benchmark):
    """AirfRANS Benchmark class

    This class allows to benchmark a power grid scenario which are defined in a config file.

    Parameters
    ----------
    benchmark_path : Union[``str``, ``None``], optional
        path to the benchmark, it should be indicated
        if not indicated, the data remains only in the memory
    config_path : Union[``pathlib.Path``, ``str``, ``None``], optional
        path to the configuration file. If config_path is ``None``, the default config file
        present in config module will be used by using the benchmark_name as the section, by default None
    benchmark_name : ``str``, optional
        the benchmark name which is used in turn as the config section, by default "Benchmark1"
    evaluation : Union[``AirfRANSEvaluation``, ``None``], optional
        a ``AirfRANSEvaluation`` instance. If not indicated, the benchmark creates its
        own evaluation instance using appropriate config, by default None
    log_path : Union[``pathlib.Path``, ``str``, ``None``], optional
        path to the logs, by default None

    Warnings
    --------
    An independent class for each benchmark is maybe a better idea.
    This class can be served as the base class for powergrid and a specific class for each benchmark
    can extend this class.
    """
    def __init__(self,
                 benchmark_path: Union[pathlib.Path, str, None],
                 config_path: Union[pathlib.Path, str],
                 benchmark_name: str,
                 log_path: Union[pathlib.Path, str, None]=None,
                ):
        super().__init__(benchmark_name = benchmark_name,
                         benchmark_path = benchmark_path,
                         config_path = config_path,
                         dataset = None,
                         augmented_simulator = None,
                         log_path = log_path
                        )

        self.is_loaded=False
        self.evaluation = AirfRANSEvaluation(config_path = config_path,
                                             scenario = benchmark_name,
                                             data_path = benchmark_path,
                                             log_path = log_path)
        self.training_simulator = None
        self.test_simulator = None
        attr_names = self.config.get_option("attr_x") + \
                     self.config.get_option("attr_y")
        self.logger.info("Loading train dataset")
        self.train_dataset = AirfRANSDataSet(name = "train",
                                             config = self.config,
                                             attr_names = attr_names,
                                             task = 'scarce',
                                             split = "training",
                                             log_path = log_path
                                            )

        self.logger.info("Loading test dataset")
        self._test_dataset = AirfRANSDataSet(name = "test",
                                             config = self.config,
                                             attr_names = attr_names,
                                             task = 'full',
                                             split = "testing",
                                             log_path = log_path
                                            )

        self._test_ood_dataset = AirfRANSDataSet(name = "test_ood",
                                             config = self.config,
                                             attr_names = attr_names,
                                             task = 'reynolds',
                                             split = "testing",
                                             log_path = log_path
                                            )
        self.ml_normalization = dict()


    def load(self,path):
        """
        load the already generated datasets
        """
        if self.is_loaded:
            self.logger.info("Previously saved data will be freed and new data will be reloaded")
        if not os.path.exists(path):
            raise RuntimeError(f"No data are found in {path}. Have you generated or downloaded "
                                   f"some data ?")
        self.train_dataset.load(path = path)
        simulation_indices_train=reynolds_filter(self.train_dataset)
        self.train_dataset=extract_dataset_by_simulations(newdataset_name=self.train_dataset.name,
                                                          dataset=self.train_dataset,
                                                          simulation_indices=simulation_indices_train)

        self._test_dataset.load(path = path)

        self._test_ood_dataset.load(path = path)
        self.is_loaded = True

    def evaluate_simulator(self,
                           dataset: str = "all",
                           augmented_simulator: Union[AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           save_predictions: bool=False,
                           **kwargs) -> dict:

        """evaluate a trained augmented simulator on one or multiple test datasets

        Parameters
        ----------
        dataset : str, optional
            dataset on which the evaluation should be performed, by default "all"
        augmented_simulator : Union[AugmentedSimulator, None], optional
            An instance of the class augmented simulator, by default None
        save_path : Union[str, None], optional
            the path that the evaluation results should be saved, by default None
        save_predictions: bool
            Whether to save the predictions made by an augmented simulator
            The predictions will be saved at the same directory of the generated data
            additional arguments that will be passed to the augmented simulator

        Returns
        -------
        dict
            the results dictionary

        Raises
        ------
        RuntimeError
            Unknown dataset selected

        """
        self.augmented_simulator = augmented_simulator

        field_names = self.train_dataset._attr_y
        chunk_sizes = [int(simulation[1]) for simulation in self.train_dataset.extra_data["simulation_names"]]
        flattened_train = np.concatenate([self.train_dataset.data[field_name][:, None] for field_name in field_names], axis = 1)
        mean_observ,std_observ = iterative_fit(flattened_train,chunk_sizes)
        self.ml_normalization["mean"] = mean_observ
        self.ml_normalization["std"] = std_observ

        if dataset == "all":
            li_dataset = [self._test_dataset, self._test_ood_dataset]
            keys = ["test", "test_ood"]
        elif dataset == "test":
            li_dataset = [self._test_dataset]
            keys = ["test"]
        elif dataset == "test_ood":
            li_dataset = [self._test_ood_dataset]
            keys = ["test_ood"]
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        res = {}
        for dataset_, nm_ in zip(li_dataset, keys):
            # call the evaluate simulator function of Benchmark class
            tmp = self._aux_evaluate_on_single_dataset(dataset=dataset_,
                                                       augmented_simulator=augmented_simulator,
                                                       save_path=save_path,
                                                       save_predictions=save_predictions,
                                                       **kwargs)
            res[nm_] = tmp

        return res

    def _aux_evaluate_on_single_dataset(self,
                                        dataset: AirfRANSDataSet,
                                        augmented_simulator: Union[AugmentedSimulator, None]=None,
                                        save_path: Union[str, None]=None,
                                        save_predictions: bool=False,
                                        **kwargs) -> dict:
        """Evaluate a single dataset
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be overloaded or called to evaluate the performance on multiple datasets

        Parameters
        ------
        dataset : AirfRANSDataSet
            the dataset
        augmented_simulator : Union[AugmentedSimulator, None], optional
            a trained augmented simulator, by default None
        batch_size : int, optional
            batch_size used for inference, by default 32
        active_flow : bool, optional
            whether to compute KCL on active (True) or reactive (False) powers, by default True
        save_path : Union[str, None], optional
            the path where the predictions should be saved, by default None
        save_predictions: bool
            Whether to save the predictions made by an augmented simulator
            The predictions will be saved at the same directory of the generated data
        Returns
        -------
        dict
            the results dictionary
        """
        self.logger.info("Benchmark %s, evaluation using %s on %s dataset", self.benchmark_name,
                                                                            augmented_simulator.name,
                                                                            dataset.name
                                                                            )

        begin_ = time.perf_counter()
        predictions = self.augmented_simulator.predict(dataset, **kwargs)
        end_ = time.perf_counter()
        self.augmented_simulator.predict_time = end_ - begin_
        observation_metadata = dataset.extra_data

        res = self.evaluation.evaluate(observations=dataset.data,
                                       predictions=predictions,
                                       observation_metadata=observation_metadata,
                                       ml_normalization = self.ml_normalization
                                       )
        if save_path:
            if not isinstance(save_path, pathlib.Path):
                save_path = pathlib.Path(save_path)
            save_path = save_path / augmented_simulator.name / dataset.name
            if save_path.exists():
                self.logger.warning("Deleting path %s that might contain previous runs", save_path)
                shutil.rmtree(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            with open((save_path / "eval_res.json"), "w", encoding="utf-8") as f:
                json.dump(obj=res, fp=f, indent=4, sort_keys=True, cls=NpEncoder)
            if save_predictions:
                for attr_nm in predictions.keys():
                    np.savez_compressed(f"{os.path.join(save_path, attr_nm)}.npz", data=predictions[attr_nm])
        elif save_predictions:
            warnings.warn(message="You indicate to save the predictions, without providing a path. No predictions will be saved!")

        return res

if __name__ == '__main__':
    from lips import get_root_path
    from lips.dataset.airfransDataSet import AirfRANSDataSet,download_data
    from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected
    from lips.augmented_simulators.torch_simulator import TorchSimulator

    directory_name='Dataset'
    if not os.path.isdir("Dataset"):
         download_data(root_path=".", directory_name=directory_name)

    config_path_benchmark=get_root_path()+os.path.join("..","configurations","airfrans","benchmarks","confAirfoil.ini")
    benchmark=AirfRANSBenchmark(benchmark_path = directory_name,
                                config_path = config_path_benchmark,
                                benchmark_name = "Case1",
                                log_path = "log_benchmark")
    benchmark.load(path=directory_name)

    sim_config_path=get_root_path()+os.path.join("..","configurations","airfrans","simulators","torch_fc.ini")
    augmented_simulator = TorchSimulator(name="torch_fc",
                                         model=TorchFullyConnected,
                                         log_path="log_benchmark",
                                         device="cuda:0",
                                         seed=42,
                                         bench_config_path=config_path_benchmark,
                                         bench_config_name="Case1",
                                         sim_config_path=sim_config_path,
                                         sim_config_name="DEFAULT",
                                         architecture_type="Classical",
                                        )

    augmented_simulator.train(train_dataset=benchmark.train_dataset, epochs=1, train_batch_size=128000)
    res=benchmark.evaluate_simulator(augmented_simulator=augmented_simulator,eval_batch_size=128000)
    print(res)