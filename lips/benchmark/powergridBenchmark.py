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
import json
import time
from typing import Union
import pathlib

import numpy as np
import grid2op

from . import Benchmark
from ..augmented_simulators import AugmentedSimulator
from ..physical_simulator import PhysicalSimulator
from ..physical_simulator import Grid2opSimulator
from ..physical_simulator.dcApproximationAS import DCApproximationAS
from ..dataset.powergridDataSet import PowerGridDataSet
from ..dataset.utils.powergrid_utils import get_kwargs_simulator_scenario
from ..dataset.utils.powergrid_utils import XDepthAgent, get_action_list
from ..evaluation.powergrid_evaluation import PowerGridEvaluation
from ..utils import NpEncoder

class PowerGridBenchmark(Benchmark):
    """PowerGrid Benchmark class

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
    load_data_set : ``bool``, optional
        whether to load the already generated datasets, by default False
    evaluation : Union[``PowerGridEvaluation``, ``None``], optional
        a ``PowerGridEvaluation`` instance. If not indicated, the benchmark creates its
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
                 benchmark_name: str="Benchmark1",
                 load_data_set: bool=False,
                 load_ybus_as_sparse: bool=False,
                 evaluation: Union[PowerGridEvaluation, None]=None,
                 log_path: Union[pathlib.Path, str, None]=None,
                 **kwargs
                 ):
        super().__init__(benchmark_name=benchmark_name,
                         benchmark_path=benchmark_path,
                         config_path=config_path,
                         dataset=None,
                         augmented_simulator=None,
                         evaluation=evaluation,
                         log_path=log_path,
                         **kwargs
                        )

        self.is_loaded=False
        # TODO : it should be reset if the config file is modified on the fly
        if evaluation is None:
            self.evaluation = PowerGridEvaluation.from_benchmark(self)

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

        self.train_env_seed = kwargs.get("train_env_seed") if "train_env_seed" in kwargs else self.config.get_option("benchmark_seeds").get("train_env_seed", 0)
        self.val_env_seed = kwargs.get("val_env_seed") if "val_env_seed" in kwargs else self.config.get_option("benchmark_seeds").get("val_env_seed", 1)
        self.test_env_seed = kwargs.get("test_env_seed") if "test_env_seed" in kwargs else self.config.get_option("benchmark_seeds").get("test_env_seed", 2)
        self.test_ood_topo_env_seed = kwargs.get("test_ood_topo_env_seed") if "test_ood_topo_env_seed" in kwargs else self.config.get_option("benchmark_seeds").get("test_ood_topo_env_seed", 3)

        self.train_actor_seed = kwargs.get("train_actor_seed") if "train_actor_seed" in kwargs else self.config.get_option("benchmark_seeds").get("train_actor_seed", 4)
        self.val_actor_seed = kwargs.get("val_actor_seed") if "val_actor_seed" in kwargs else self.config.get_option("benchmark_seeds").get("val_actor_seed", 5)
        self.test_actor_seed = kwargs.get("test_actor_seed") if "test_actor_seed" in kwargs else self.config.get_option("benchmark_seeds").get("test_actor_seed", 6)
        self.test_ood_topo_actor_seed = kwargs.get("test_ood_topo_actor_seed") if "test_ood_topo_actor_seed" in kwargs else self.config.get_option("benchmark_seeds").get("test_ood_topo_actor_seed", 7)

        self.initial_chronics_id = kwargs.get("initial_chronics_id") if "initial_chronics_id" in kwargs else self.config.get_option("benchmark_seeds").get("initial_chronics_id", 0)

        # concatenate all the variables for data generation
        attr_names = ()
        if self.config.get_option("attr_x") is not None:
            attr_names += self.config.get_option("attr_x")
        if self.config.get_option("attr_tau") is not None:
            attr_names += self.config.get_option("attr_tau")
        if self.config.get_option("attr_y") is not None:
            attr_names += self.config.get_option("attr_y")
        
        # attr_names = self.config.get_option("attr_x") + \
        #              self.config.get_option("attr_tau") + \
        #              self.config.get_option("attr_y")

        self.train_dataset = PowerGridDataSet(name="train",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self.val_dataset = PowerGridDataSet(name="val",
                                            attr_names=attr_names,
                                            config=self.config,
                                            log_path=log_path
                                            )

        self._test_dataset = PowerGridDataSet(name="test",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self._test_ood_topo_dataset = PowerGridDataSet(name="test_ood_topo",
                                                       attr_names=attr_names,
                                                       config=self.config,
                                                       log_path=log_path
                                                       )

        if load_data_set:
            self.load(load_ybus_as_sparse)

    def load(self, load_ybus_as_sparse: bool = False):
        """
        load the already generated datasets
        """
        if self.is_loaded:
            #print("Previously saved data will be freed and new data will be reloaded")
            self.logger.info("Previously saved data will be freed and new data will be reloaded")
        if not os.path.exists(self.path_datasets):
            raise RuntimeError(f"No data are found in {self.path_datasets}. Have you generated or downloaded "
                               f"some data ?")
        self.train_dataset.load(path=self.path_datasets, load_ybus_as_sparse=load_ybus_as_sparse)
        self.val_dataset.load(path=self.path_datasets, load_ybus_as_sparse=load_ybus_as_sparse)
        self._test_dataset.load(path=self.path_datasets, load_ybus_as_sparse=load_ybus_as_sparse)
        self._test_ood_topo_dataset.load(path=self.path_datasets, load_ybus_as_sparse=load_ybus_as_sparse)
        self.is_loaded = True

    def generate(self, nb_sample_train: int, nb_sample_val: int,
                 nb_sample_test: int, nb_sample_test_ood_topo: int,
                 do_store_physics: bool=False, is_dc: bool=False,
                 store_as_sparse: bool=False):
        """
        generate the different datasets required for the benchmark
        """
        if self.path_datasets is not None:
            if self.is_loaded:
                self.logger.warning("Previously saved data will be erased by this new generation")
            if os.path.exists(self.path_datasets):
                self.logger.warning("Deleting path %s that might contain previous runs", self.path_datasets)
                shutil.rmtree(self.path_datasets)
            self.logger.info("Creating path %s to save the current data", self.path_datasets)
            os.mkdir(self.path_datasets)

        # it should be done before `_fills_actor_simulator`
        if is_dc:
            params = self.config.get_option("env_params")
            params["ENV_DC"] = True
            self.config.edit_config_option(option="env_params", value=str(params), scenario_name=self.benchmark_name)


        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._fills_actor_simulator()

        self.train_dataset.generate(simulator=self.training_simulator,
                                    actor=self.training_actor,
                                    path_out=self.path_datasets,
                                    nb_samples=nb_sample_train,
                                    nb_samples_per_chronic=self.config.get_option("samples_per_chronic").get("train", 864),
                                    do_store_physics=do_store_physics,
                                    is_dc=is_dc,
                                    store_as_sparse=store_as_sparse
                                    )
        self.val_dataset.generate(simulator=self.val_simulator,
                                  actor=self.val_actor,
                                  path_out=self.path_datasets,
                                  nb_samples=nb_sample_val,
                                  nb_samples_per_chronic=self.config.get_option("samples_per_chronic").get("val", 288),
                                  do_store_physics=do_store_physics,
                                  is_dc=is_dc,
                                  store_as_sparse=store_as_sparse
                                  )
        self._test_dataset.generate(simulator=self.test_simulator,
                                    actor=self.test_actor,
                                    path_out=self.path_datasets,
                                    nb_samples=nb_sample_test,
                                    nb_samples_per_chronic=self.config.get_option("samples_per_chronic").get("test", 288),
                                    do_store_physics=do_store_physics,
                                    is_dc=is_dc,
                                    store_as_sparse=store_as_sparse
                                    )
        self._test_ood_topo_dataset.generate(simulator=self.test_ood_topo_simulator,
                                             actor=self.test_ood_topo_actor,
                                             path_out=self.path_datasets,
                                             nb_samples=nb_sample_test_ood_topo,
                                             nb_samples_per_chronic=self.config.get_option("samples_per_chronic").get("test_ood", 288),
                                             do_store_physics=do_store_physics,
                                             is_dc=is_dc,
                                             store_as_sparse=store_as_sparse
                                             )

    def evaluate_simulator(self,
                           dataset: str = "all",
                           augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           save_predictions: bool=False,
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
        save_predictions: bool
            Whether to save the predictions made by an augmented simulator
            The predictions will be saved at the same directory of the generated data
            # TODO : to save predictions, the directory shoud look like ``benchmark_name\augmented_simulator.name\``
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
        if self.env is None:
            self.env = get_env(get_kwargs_simulator_scenario(self.config))
        kwargs["env"] = self.env
        self.augmented_simulator = augmented_simulator
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
                                                       save_predictions=save_predictions,
                                                       **kwargs)
            res[nm_] = tmp

        return res

    def _aux_evaluate_on_single_dataset(self,
                                        dataset: PowerGridDataSet,
                                        augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None]=None,
                                        save_path: Union[str, None]=None,
                                        save_predictions: bool=False,
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
        # TODO: however, we can introduce the batch concept in DC, to have equitable comparison for time complexity
        if isinstance(self.augmented_simulator, DCApproximationAS):
            predictions = self.augmented_simulator.compute(dataset)
        else:
            sim_kwargs = copy.deepcopy(kwargs)
            sim_kwargs.pop("env")
            begin_ = time.perf_counter()
            predictions = self.augmented_simulator.predict(dataset, **kwargs)
            end_ = time.perf_counter()
            self.augmented_simulator.predict_time = end_ - begin_

        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = dataset.data
        self.dataset = dataset

        kwargs["augmented_simulator"] = self.augmented_simulator
        kwargs["dataset"] = dataset
        res = self.evaluation.evaluate(observations=dataset.data,
                                       predictions=predictions,
                                       **kwargs
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
        self.training_actor = XDepthAgent(self.env,
                                          all_topo_actions=all_topo_actions,
                                          reference_params=self.config.get_option("dataset_create_params").get("reference_args", None),
                                          scenario_params=self.config.get_option("dataset_create_params")["train"],
                                          seed=self.train_actor_seed,
                                          log_path=self.log_path)
        self.val_actor = XDepthAgent(self.env,
                                     all_topo_actions=all_topo_actions,
                                     reference_params=self.config.get_option("dataset_create_params").get("reference_args", None),
                                     scenario_params=self.config.get_option("dataset_create_params")["test"],
                                     seed=self.val_actor_seed,
                                     log_path=self.log_path)
        self.test_actor = XDepthAgent(self.env,
                                      all_topo_actions=all_topo_actions,
                                      reference_params=self.config.get_option("dataset_create_params").get("reference_args", None),
                                      scenario_params=self.config.get_option("dataset_create_params")["test"],
                                      seed=self.test_actor_seed,
                                      log_path=self.log_path)
        self.test_ood_topo_actor = XDepthAgent(self.env,
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
