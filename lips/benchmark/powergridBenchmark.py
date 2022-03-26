# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking
import os
import shutil
import warnings
import copy
from typing import Union
import importlib
import numpy as np

from lips.physical_simulator.dcApproximationAS import DCApproximationAS

from . import Benchmark

from ..augmented_simulators import AugmentedSimulator
from ..physical_simulator import PhysicalSimulator
from ..physical_simulator import Grid2opSimulator
from ..physical_simulator.grid2opSimulator import get_env
from ..dataset import PowerGridDataSet
from ..evaluation import PowerGridEvaluation


class PowerGridBenchmark(Benchmark):
    """
    This class allows to benchmark a power grid scenario which are defined in a config file.
    """
    def __init__(self,
                 benchmark_path: str,
                 config_path: Union[str, None]=None,
                 benchmark_name="Benchmark1",
                 load_data_set=False,
                 evaluation=None,
                 log_path: Union[str, None]=None,
                 train_env_seed: int = 1,
                 val_env_seed: int = 2,
                 test_env_seed: int = 3,
                 test_ood_topo_env_seed: int = 4,
                 initial_chronics_id: int = 0,
                 train_actor_seed: int = 5,
                 val_actor_seed: int = 6,
                 test_actor_seed: int = 7,
                 test_ood_topo_actor_seed: int = 8,

                 ):
        # init the super class
        super().__init__(benchmark_name=benchmark_name,
                         dataset=None,
                         augmented_simulator=None,
                         evaluation=evaluation,
                         benchmark_path=benchmark_path,
                         log_path=log_path,
                         config_path=config_path
                        )

        self.is_loaded=False
        # TODO : it should be reset if the config file is modified on the fly
        if evaluation is None:
            self.evaluation = PowerGridEvaluation.from_benchmark(self)

        # importing the right module from which the scenarios and actors could be used
        if self.config.get_option("utils_lib") is not None:
            try:
                module_name = self.config.get_option("utils_lib")
                module = ".".join(("lips", "benchmark", "utils", module_name))
                self.utils = importlib.import_module(module)
            except ImportError as error:
                self.logger.error("The module %s could not be accessed! %s", module_name, error)

        self.training_simulator = None
        self.val_simulator = None
        self.test_simulator = None
        self.test_ood_topo_simulator = None

        self.training_actor = None
        self.val_actor = None
        self.test_actor = None
        self.test_ood_topo_actor = None

        self.train_env_seed = train_env_seed
        self.val_env_seed = val_env_seed
        self.test_env_seed = test_env_seed
        self.test_ood_topo_env_seed = test_ood_topo_env_seed

        self.train_actor_seed = train_actor_seed
        self.val_actor_seed = val_actor_seed
        self.test_actor_seed = test_actor_seed
        self.test_ood_topo_actor_seed = test_ood_topo_actor_seed

        self.initial_chronics_id = initial_chronics_id
        # concatenate all the variables for data generation
        attr_names = self.config.get_option("attr_x") + \
                     self.config.get_option("attr_tau") + \
                     self.config.get_option("attr_y")


        self.train_dataset = PowerGridDataSet("train",
                                              attr_names=attr_names,
                                              log_path=log_path
                                              )

        self.val_dataset = PowerGridDataSet("val",
                                            attr_names=attr_names,
                                            log_path=log_path
                                            )

        self._test_dataset = PowerGridDataSet("test",
                                              attr_names=attr_names,
                                              log_path=log_path
                                              )

        self._test_ood_topo_dataset = PowerGridDataSet("test_ood_topo",
                                                       attr_names=attr_names,
                                                       log_path=log_path
                                                       )

        if load_data_set:
            self.load()

    def load(self):
        """
        load the already generated datasets
        """
        if self.is_loaded:
            #print("Previously saved data will be freed and new data will be reloaded")
            self.logger.info("Previously saved data will be freed and new data will be reloaded")
        if not os.path.exists(self.path_datasets):
            raise RuntimeError(f"No data are found in {self.path_datasets}. Have you generated or downloaded "
                               f"some data ?")
        self.train_dataset.load(path=self.path_datasets)
        self.val_dataset.load(path=self.path_datasets)
        self._test_dataset.load(path=self.path_datasets)
        self._test_ood_topo_dataset.load(path=self.path_datasets)
        self.is_loaded = True

    def generate(self, nb_sample_train, nb_sample_val,
                 nb_sample_test, nb_sample_test_ood_topo):
        """
        generate the different datasets required for the benchmark
        """
        if self.is_loaded:
            self.logger.warning("Previously saved data will be erased by this new generation")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._fills_actor_simulator()
        if os.path.exists(self.path_datasets):
            self.logger.warning("Deleting path %s that might contain previous runs", self.path_datasets)
            shutil.rmtree(self.path_datasets)

        self.logger.info("Creating path %s to save the current data", self.path_datasets)
        os.mkdir(self.path_datasets)

        self.train_dataset.generate(simulator=self.training_simulator,
                                    actor=self.training_actor,
                                    path_out=self.path_datasets,
                                    nb_samples=nb_sample_train
                                    )
        self.val_dataset.generate(simulator=self.val_simulator,
                                  actor=self.val_actor,
                                  path_out=self.path_datasets,
                                  nb_samples=nb_sample_val
                                  )
        self._test_dataset.generate(simulator=self.test_simulator,
                                    actor=self.test_actor,
                                    path_out=self.path_datasets,
                                    nb_samples=nb_sample_test
                                    )
        self._test_ood_topo_dataset.generate(simulator=self.test_ood_topo_simulator,
                                             actor=self.test_ood_topo_actor,
                                             path_out=self.path_datasets,
                                             nb_samples=nb_sample_test_ood_topo
                                             )

    def evaluate_simulator(self,
                           dataset: str = "all",  # TODO
                           augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                           batch_size: int=32,
                           save_path: Union[str, None]=None,
                           active_flow: bool=True
                           ):
        """evaluate a trained augmented simulator on one or multiple test datasets

        Parameters
        ----------
        dataset, optional
            _description_, by default "all"
        batch_size, optional
            _description_, by default 32
        save_path, optional
            _description_, by default None
        active_flow, optional
            _description_, by default True

        Returns
        -------
            evaluation_results: dict

        Raises
        ------
        RuntimeError
            _description_
        """
        self._create_training_simulator()
        li_dataset = []
        if dataset == "all":
            li_dataset = [self.val_dataset, self._test_dataset, self._test_ood_topo_dataset]
            keys = ["val", "test", "test_ood_topo"]
        elif dataset == "val" or dataset == "val_dataset":
            li_dataset = [self.val_dataset]
            keys = ["val"]
        elif dataset == "test" or dataset == "test_dataset":
            li_dataset = [self._test_dataset]
            keys = ["test"]
        elif dataset == "test_ood_topo" or dataset == "test_ood_topo_dataset":
            li_dataset = [self._test_ood_topo_dataset]
            keys = ["test_ood_topo"]
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        res = {}
        for dataset_, nm_ in zip(li_dataset, keys):
            # call the evaluate simulator function of Benchmark class
            tmp = self._aux_evaluate_on_single_dataset(dataset=dataset_,
                                                       augmented_simulator=augmented_simulator,
                                                       batch_size=batch_size,
                                                       active_flow=active_flow,
                                                       save_path=save_path
                                                      )
            res[nm_] = copy.deepcopy(tmp)
        return res

    def _aux_evaluate_on_single_dataset(self,
                                        dataset: str,
                                        augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                                        batch_size: int=32,
                                        active_flow: bool=True,
                                        save_path: Union[str, None]=None):
        """
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be overloaded or called to evaluate the performance on multiple datasets

        params
        ------
            active_flow: ``bool``
                whether to compute KCL on active (True) or reactive (False) powers

            save_path: ``str`` or ``None``
                if indicated the evaluation results will be saved to indicated path
        """
        self.logger.info("Benchmark %s, evaluation using %s on %s dataset", self.benchmark_name,
                                                                            augmented_simulator.name,
                                                                            dataset.name
                                                                            )
        self.augmented_simulator = augmented_simulator
        # TODO: however, we can introduce the batch concept in DC, to have equitable comparison for time complexity
        if isinstance(self.augmented_simulator, DCApproximationAS):
            predictions = self.augmented_simulator.evaluate(dataset)
        else:
            predictions = self.augmented_simulator.evaluate(dataset, batch_size)
        observations = dataset.get_data(np.arange(len(dataset)))
        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = observations
        self.dataset = dataset

        res = self.evaluation.evaluate(observations=observations,
                                       predictions=predictions,
                                       save_path=save_path
                                       )

        # res = self.evaluation.do_evaluations(env=get_env(self.utils.get_kwargs_simulator_scenario()),
        #                                      env_name=None,
        #                                      predictions=predictions,
        #                                      observations=observations,
        #                                      choice="predictions",  # we want to evaluate only the predictions here
        #                                      active_flow=active_flow,
        #                                      save_path=save_path  # TODO currently not used
        #                                      )
        self.logger.info("Evaluation on %s dataset was successful!", dataset.name)
        return res

    def _create_training_simulator(self):
        """"""
        if self.training_simulator is None:
            self.training_simulator = Grid2opSimulator(self.utils.get_kwargs_simulator_scenario(),
                                                       initial_chronics_id=self.initial_chronics_id,
                                                       # i use 994 chronics out of the 904 for training
                                                       chronics_selected_regex="^((?!(.*9[0-9][0-9].*)).)*$"
                                                       )

    def _fills_actor_simulator(self):
        """This function is only called when the data are simulated"""
        self._create_training_simulator()
        self.training_simulator.seed(self.train_env_seed)

        self.val_simulator = Grid2opSimulator(self.utils.get_kwargs_simulator_scenario(),
                                              initial_chronics_id=self.initial_chronics_id,
                                              # i use 50 full chronics for testing
                                              chronics_selected_regex=".*9[0-4][0-9].*")
        self.val_simulator.seed(self.val_env_seed)

        self.test_simulator = Grid2opSimulator(self.utils.get_kwargs_simulator_scenario(),
                                               initial_chronics_id=self.initial_chronics_id,
                                               # i use 25 full chronics for testing
                                               chronics_selected_regex=".*9[5-9][0-4].*")
        self.test_simulator.seed(self.test_env_seed)

        self.test_ood_topo_simulator = Grid2opSimulator(self.utils.get_kwargs_simulator_scenario(),
                                                        initial_chronics_id=self.initial_chronics_id,
                                                        # i use 25 full chronics for testing
                                                        chronics_selected_regex=".*9[5-9][5-9].*")
        self.test_ood_topo_simulator.seed(self.test_ood_topo_env_seed)

        self.training_actor = self.utils.get_actor_training_scenario(self.training_simulator)
        self.training_actor.seed(self.train_actor_seed)

        self.val_actor = self.utils.get_actor_test_scenario(self.val_simulator)
        self.val_actor.seed(self.val_actor_seed)

        self.test_actor = self.utils.get_actor_test_scenario(self.test_simulator)
        self.test_actor.seed(self.test_actor_seed)

        self.test_ood_topo_actor = self.utils.get_actor_test_ood_topo_scenario(self.test_ood_topo_simulator)
        self.test_ood_topo_actor.seed(self.test_ood_topo_actor_seed)