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
from typing import Union
#import importlib

import grid2op

from . import Benchmark
from .utils.powergrid_utils import get_kwargs_simulator_scenario, XDepthAgent, get_action_list
from ..augmented_simulators import AugmentedSimulator
from ..physical_simulator import PhysicalSimulator
from ..physical_simulator import Grid2opSimulator
from ..physical_simulator.dcApproximationAS import DCApproximationAS
from ..dataset import PowerGridDataSet
from ..evaluation import PowerGridEvaluation



class PowerGridBenchmark(Benchmark):
    """PowerGrid Benchmark class

    This class allows to benchmark a power grid scenario which are defined in a config file.

    Parameters
    ----------
    benchmark_path : ``str``
        path to the benchmark
    config_path : Union[``str``, ``None``], optional
        path to the configuration file. If config_path is ``None``, the default config file
        present in config module will be used by using the benchmark_name as the section, by default None
    benchmark_name : ``str``, optional
        the benchmark name which is used in turn as the config section, by default "Benchmark1"
    load_data_set : ``bool``, optional
        whether to load the already generated datasets, by default False
    evaluation : Union[``PowerGridEvaluation``, ``None``], optional
        a ``PowerGridEvaluation`` instance. If not indicated, the benchmark creates its
        own evaluation instance using appropriate config, by default None
    log_path : Union[``str``, ``None``], optional
        path to the logs, by default None

    Todo
    ----
    Add all the seeds into the config file

    Warnings
    --------
    An independent class for each benchmark is maybe a better idea.
    This class can be served as the base class for powergrid and a specific class for each benchmark
    can extend this class.
    """
    def __init__(self,
                 benchmark_path: str,
                 config_path: Union[str, None]=None,
                 benchmark_name: str="Benchmark1",
                 load_data_set: bool=False,
                 evaluation: Union[PowerGridEvaluation, None]=None,
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
        """
        if self.config.get_option("utils_lib") is not None:
            try:
                module_name = self.config.get_option("utils_lib")
                module = ".".join(("lips", "benchmark", "utils", module_name))
                self.utils = importlib.import_module(module)
            except ImportError as error:
                self.logger.error("The module %s could not be accessed! %s", module_name, error)
        """
        self.env_name = self.config.get_option("env_name")
        self.env = None
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
                                              config=self.config,
                                              log_path=log_path
                                              )

        self.val_dataset = PowerGridDataSet("val",
                                            attr_names=attr_names,
                                            config=self.config,
                                            log_path=log_path
                                            )

        self._test_dataset = PowerGridDataSet("test",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self._test_ood_topo_dataset = PowerGridDataSet("test_ood_topo",
                                                       attr_names=attr_names,
                                                       config=self.config,
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

    def generate(self, nb_sample_train: int, nb_sample_val: int,
                 nb_sample_test: int, nb_sample_test_ood_topo: int):
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
                           dataset: str = "all",
                           augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           **kwargs) -> dict:
        """evaluate a trained augmented simulator on one or multiple test datasets

        Parameters
        ----------
        dataset : str, optional
            dataset on which the evaluation should be performed, by default "all"
        augmented_simulator : Union[PhysicalSimulator, AugmentedSimulator, None], optional
            An instance of the class augmented simulator, by default None
        save_path : Union[str, None], optional
            the path that the evaluation results should be saved, by default None
        **kwargs: ``dict``
            additional arguments that will be passed to the augmented simulator
        Todo
        ----
        TODO: add active flow in config file

        Returns
        -------
        dict
            the results dictionary

        Raises
        ------
        RuntimeError
            Unknown dataset selected

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
                                                       save_path=save_path,
                                                       **kwargs)
            res[nm_] = copy.deepcopy(tmp)
        return res

    def _aux_evaluate_on_single_dataset(self,
                                        dataset: PowerGridDataSet,
                                        augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                                        save_path: Union[str, None]=None,
                                        **kwargs) -> dict:
        """Evaluate a single dataset
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be overloaded or called to evaluate the performance on multiple datasets

        Parameters
        ------
        dataset : PowerGridDataSet
            the dataset
        augmented_simulator : Union[PhysicalSimulator, AugmentedSimulator, None], optional
            a trained augmented simulator, by default None
        batch_size : int, optional
            batch_size used for inference, by default 32
        active_flow : bool, optional
            whether to compute KCL on active (True) or reactive (False) powers, by default True
        save_path : Union[str, None], optional
            if indicated the evaluation results will be saved to indicated path, by default None

        Returns
        -------
        dict
            the results dictionary
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
            predictions = self.augmented_simulator.evaluate(dataset, **kwargs)

        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = dataset.data
        self.dataset = dataset

        res = self.evaluation.evaluate(observations=dataset.data,
                                       predictions=predictions,
                                       save_path=save_path
                                       )
        return res

    def _create_training_simulator(self):
        """
        Initialize the simulator used for training

        """
        if self.training_simulator is None:
            self.training_simulator = Grid2opSimulator(get_kwargs_simulator_scenario(self.config),
                                                       initial_chronics_id=self.initial_chronics_id,
                                                       chronics_selected_regex=self.config.get_option("chronics").get("train")
                                                       )

    def _fills_actor_simulator(self):
        """This function is only called when the data are simulated"""
        self.env = get_env(get_kwargs_simulator_scenario(self.config))
        self._create_training_simulator()
        self.training_simulator.seed(self.train_env_seed)

        self.val_simulator = Grid2opSimulator(get_kwargs_simulator_scenario(self.config),
                                              initial_chronics_id=self.initial_chronics_id,
                                              chronics_selected_regex=self.config.get_option("chronics").get("val")
                                              )
        self.val_simulator.seed(self.val_env_seed)

        self.test_simulator = Grid2opSimulator(get_kwargs_simulator_scenario(self.config),
                                               initial_chronics_id=self.initial_chronics_id,
                                               chronics_selected_regex=self.config.get_option("chronics").get("test")
                                               )
        self.test_simulator.seed(self.test_env_seed)

        self.test_ood_topo_simulator = Grid2opSimulator(get_kwargs_simulator_scenario(self.config),
                                                        initial_chronics_id=self.initial_chronics_id,
                                                        chronics_selected_regex=self.config.get_option("chronics").get("test_ood")
                                                        )
        self.test_ood_topo_simulator.seed(self.test_ood_topo_env_seed)

        all_topo_actions = get_action_list(self.env.action_space)
        self.training_actor = XDepthAgent(self.env.action_space,
                                          all_topo_actions=all_topo_actions,
                                          reference_params=self.config.get_option("dataset_create_params").get("reference_args", None),
                                          scenario_params=self.config.get_option("dataset_create_params")["train"],
                                          seed=self.train_actor_seed,
                                          log_path=self.log_path)
        self.val_actor = XDepthAgent(self.env.action_space,
                                     all_topo_actions=all_topo_actions,
                                     reference_params=self.config.get_option("dataset_create_params").get("reference_args", None),
                                     scenario_params=self.config.get_option("dataset_create_params")["test"],
                                     seed=self.val_actor_seed,
                                     log_path=self.log_path)
        self.test_actor = XDepthAgent(self.env.action_space,
                                      all_topo_actions=all_topo_actions,
                                      reference_params=self.config.get_option("dataset_create_params").get("reference_args", None),
                                      scenario_params=self.config.get_option("dataset_create_params")["test"],
                                      seed=self.test_actor_seed,
                                      log_path=self.log_path)
        self.test_ood_topo_actor = XDepthAgent(self.env.action_space,
                                               all_topo_actions=all_topo_actions,
                                               reference_params=self.config.get_option("dataset_create_params").get("reference_args", None),
                                               scenario_params=self.config.get_option("dataset_create_params")["test_ood"],
                                               seed=self.test_ood_topo_actor_seed,
                                               log_path=self.log_path)

def get_env(env_kwargs: dict):
    """Getter for the environment

    Parameters
    ----------
    env_kwargs : dict
        environment parameters

    Returns
    -------
    grid2op.Environment
        A grid2op environment with the given parameters
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = grid2op.make(**env_kwargs)
    # env.deactivate_forecast()
    return env