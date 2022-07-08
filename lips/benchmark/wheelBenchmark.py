"""
Usage:
    Implementation of pneumatic benchmarks
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
import numpy as np

from lips.benchmark import Benchmark
from lips.evaluation.pneumatic_evaluation import PneumaticEvaluation
from lips.physical_simulator.getfemSimulator import PhysicalSimulator,GetfemSimulator
from lips.augmented_simulators.augmented_simulator import AugmentedSimulator
from lips.dataset.pneumaticWheelDataSet import SamplerStaticWheelDataSet,QuasiStaticWheelDataSet

class WheelBenchmark(Benchmark):
    """Pneumatic Benchmark class

    This class allows to benchmark a pneumatic scenario which are defined in a config file.

    Parameters
    ----------
    benchmark_path : Union[str, None], optional
        path to the benchmark, it should be indicated
        if not indicated, the data remains only in the memory
    config_path : Union[str, None], optional
        path to the configuration file. If config_path is None, the default config file
        present in config module will be used by using the benchmark_name as the section, by default None
    benchmark_name : str, optional
        the benchmark name which is used in turn as the config section, by default "Benchmark0"
    load_data_set : bool, optional
        whether to load the already generated datasets, by default False
    evaluation : Union[PneumaticEvaluation, None], optional
        a PneumaticEvaluation instance. If not indicated, the benchmark creates its
        own evaluation instance using appropriate config, by default None
    log_path : Union[str, None], optional
        path to the logs, by default None

    This class is used as the base class for pneumatic usecases and each pneumatic problem
    can extend this class.
    """
    def __init__(self,
                 benchmark_path: str,
                 config_path: Union[str, None]=None,
                 benchmark_name: str="Benchmark0",
                 load_data_set: bool=False,
                 evaluation: Union[PneumaticEvaluation, None]=None,
                 log_path: Union[str, None]=None
                 ):
        super().__init__(benchmark_name=benchmark_name,
                         dataset=None,
                         augmented_simulator=None,
                         evaluation=evaluation,
                         benchmark_path=benchmark_path,
                         log_path=log_path,
                         config_path=config_path
                        )
        self.training_simulator = None


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

    def _create_training_simulator(self):
        """
        Initialize the simulator used for training

        """
        if self.training_simulator is None:
            scenario_params=self.config.get_option("env_params")
            self.training_simulator = GetfemSimulator(**scenario_params)

class WeightSustainingWheelBenchmark(WheelBenchmark):
    """WeightSustainingWheelBenchmark class

    This class allows to benchmark a pneumatic WeightSustaining Wheel problem whose features are 
    defined in a config file.

    Parameters
    ----------
    benchmark_path : Union[str, None], optional
        path to the benchmark, it should be indicated
        if not indicated, the data remains only in the memory
    config_path : Union[str, None], optional
        path to the configuration file. If config_path is None, the default config file
        present in config module will be used by using the benchmark_name as the section, by default None
    benchmark_name : str, optional
        the benchmark name which is used in turn as the config section, by default "Benchmark1"
    load_data_set : bool, optional
        whether to load the already generated datasets, by default False
    evaluation : Union[PneumaticEvaluation, None], optional
        a PneumaticEvaluation instance. If not indicated, the benchmark creates its
        own evaluation instance using appropriate config, by default None
    log_path : Union[str, None], optional
        path to the logs, by default None

    This class derived from WheelBenchmark
    """
    def __init__(self,
                 benchmark_path: str,
                 config_path: Union[str, None]=None,
                 benchmark_name: str="Benchmark1",
                 load_data_set: bool=False,
                 evaluation: Union[PneumaticEvaluation, None]=None,
                 log_path: Union[str, None]=None,
                 ):
        super().__init__(benchmark_name=benchmark_name,
                         evaluation=evaluation,
                         benchmark_path=benchmark_path,
                         log_path=log_path,
                         config_path=config_path
                        )

        self.is_loaded=False
        if evaluation is None:
            myEval=PneumaticEvaluation(config_path=config_path,scenario=benchmark_name,log_path=log_path)
            self.evaluation = myEval.from_benchmark(benchmark=self)

        self.env_name = self.config.get_option("env_name")
        self.val_simulator = None
        self.test_simulator = None
        #self.test_ood_topo_simulator = None

        self.training_actor = None
        self.val_actor = None
        self.test_actor = None
        #self.test_ood_topo_actor = None

        # concatenate all the variables for data generation
        attr_names = self.config.get_option("attr_x")\
                     +self.config.get_option("attr_y")


        self.train_dataset = SamplerStaticWheelDataSet("train",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self.val_dataset = SamplerStaticWheelDataSet("val",
                                            attr_names=attr_names,
                                            config=self.config,
                                            log_path=log_path
                                            )

        self._test_dataset = SamplerStaticWheelDataSet("test",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self._test_ood_topo_dataset = SamplerStaticWheelDataSet("test_ood_topo",
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
            self.logger.info("Previously saved data will be freed and new data will be reloaded")
        if not os.path.exists(self.path_datasets):
            raise RuntimeError(f"No data are found in {self.path_datasets}. Have you generated or downloaded "
                               f"some data ?")
        self.train_dataset.load(path=self.path_datasets)
        self.val_dataset.load(path=self.path_datasets)
        self._test_dataset.load(path=self.path_datasets)
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

    def evaluate_predictor(self,
                           dataset: str = "all",
                           augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           **kwargs) -> dict:
        """Compute prediction on one or multiple test datasets
        This function will predict a solution using simulator (physical or augmented)
        on multiple datasets.

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
        ----

        Returns
        -------
        dict
            the results dictionary

        Raises
        ------
        RuntimeError
            Unknown dataset selected

        """
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
            # call the evaluate predictor function of Benchmark class
            tmp = self._aux_predict_on_single_dataset(dataset=dataset_,
                                                       augmented_simulator=augmented_simulator,
                                                       save_path=save_path,
                                                       **kwargs)
            res[nm_] = copy.deepcopy(tmp)
        return res

    def _aux_predict_on_single_dataset(self,
                                       dataset: SamplerStaticWheelDataSet,
                                       augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                                       save_path: Union[str, None]=None,
                                       **kwargs) -> dict:
        """Compute prediction on one dataset
        This function will predict a solution using simulator (physical or augmented)
        on one datasets.

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
        ----

        Returns
        -------
        predictions: dict
            the prediction dictionary

        """
        self.logger.info("Benchmark %s, evaluation using %s on %s dataset", self.benchmark_name,
                                                                            augmented_simulator.name,
                                                                            dataset.name
                                                                            )
        self.augmented_simulator = augmented_simulator
        predictions = self.augmented_simulator.predict(dataset)

        return predictions


    def evaluate_simulator_from_predictions(self,
                                            predictions: dict,
                                            observations: dict,
                                            dataset: str = "all",
                                            save_path: Union[str, None]=None,
                                            **kwargs) -> dict:
        """Evaluate simulator from predictions
        This function compare a prediction to an observation predict a solution using simulator (physical or augmented)
        on one or multiple datasets.

        Parameters
        ----------
        predictions : dict
            predictions by a simulator
        observations : dict
            actual reference observation
        dataset : str, optional
            dataset on which the evaluation should be performed, by default "all"
        save_path : Union[str, None], optional
            the path that the evaluation results should be saved, by default None
        **kwargs: ``dict``
            additional arguments that will be passed to the augmented simulator
        ----

        Returns
        -------
        res: dict
            the evaluation dictionary

        """
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
            # call the evaluate predictor function of Benchmark class
            tmp = self._aux_evaluate_on_single_dataset_from_prediction(dataset=dataset_,
                                                       predictions=predictions[nm_],
                                                       observations=observations[nm_],
                                                       save_path=save_path,
                                                       **kwargs)
            res[nm_] = copy.deepcopy(tmp)
        return res

    def _aux_evaluate_on_single_dataset_from_prediction(self,
                                                        dataset: SamplerStaticWheelDataSet,
                                                        predictions: dict,
                                                        observations: dict,
                                                        save_path: Union[str, None]=None,
                                                        **kwargs) -> dict:
        """Evaluate single dataset from prediction
        This function evaluate a prediction using various criteria on one or multiple datasets.

        Parameters
        ----------
        dataset : SamplerStaticWheelDataSet
            dataset on which the evaluation should be performed
        predictions : dict
            predictions by a simulator
        observations : dict
            actual reference observation
        save_path : Union[str, None], optional
            the path that the evaluation results should be saved, by default None
        **kwargs: ``dict``
            additional arguments that will be passed to the augmented simulator
        ----

        Returns
        -------
        res: dict
            the evaluation dictionary
        """
        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = observations
        self.dataset = dataset

        res = self.evaluation.evaluate(observations=observations,
                                       predictions=predictions,
                                       save_path=save_path
                                       )
        return res

    def _aux_evaluate_on_single_dataset(self,
                                        dataset: SamplerStaticWheelDataSet,
                                        augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                                        save_path: Union[str, None]=None,
                                        **kwargs) -> dict:
        """Evaluate a single dataset
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be called to evaluate the performance on multiple datasets

        Parameters
        ------
        dataset : SamplerStaticWheelDataSet
            the dataset
        augmented_simulator : Union[PhysicalSimulator, AugmentedSimulator, None], optional
            a trained augmented simulator, by default None
        batch_size : int, optional
            batch_size used for inference, by default 32
        save_path : Union[str, None], optional
            if indicated the evaluation results will be saved to indicated path, by default None

        Returns
        -------
        res: dict
            the results dictionary
        """
        self.logger.info("Benchmark %s, evaluation using %s on %s dataset", self.benchmark_name,
                                                                            augmented_simulator.name,
                                                                            dataset.name
                                                                            )
        self.augmented_simulator = augmented_simulator
        predictions = self.augmented_simulator.predict(dataset)

        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = dataset.data
        self.dataset = dataset

        res = self.evaluation.evaluate(observations=dataset.data,
                                       predictions=predictions,
                                       save_path=save_path
                                       )
        return res

    def _fills_actor_simulator(self):
        """This function is only called when the data are simulated"""
        self._create_training_simulator()

        scenario_params=self.config.get_option("env_params")
        self.val_simulator = GetfemSimulator(**scenario_params)

        self.test_simulator = GetfemSimulator(**scenario_params)

        self.test_ood_topo_simulator = GetfemSimulator(**scenario_params)


class DispRollingWheelBenchmark(WheelBenchmark):
    """WeightSustainingWheelBenchmark class

    This class allows to benchmark a pneumatic WeightSustaining Wheel problem whose features are 
    defined in a config file.

    Parameters
    ----------
    benchmark_path : Union[str, None], optional
        path to the benchmark, it should be indicated
        if not indicated, the data remains only in the memory
    config_path : Union[str, None], optional
        path to the configuration file. If config_path is None, the default config file
        present in config module will be used by using the benchmark_name as the section, by default None
    benchmark_name : str, optional
        the benchmark name which is used in turn as the config section, by default "Benchmark2"
    load_data_set : bool, optional
        whether to load the already generated datasets, by default False
    evaluation : Union[PneumaticEvaluation, None], optional
        a PneumaticEvaluation instance. If not indicated, the benchmark creates its
        own evaluation instance using appropriate config, by default None
    log_path : Union[str, None], optional
        path to the logs, by default None

    This class derived from WheelBenchmark
    """
    def __init__(self,
                 benchmark_path: str,
                 config_path: Union[str, None]=None,
                 benchmark_name: str="Benchmark2",
                 load_data_set: bool=False,
                 evaluation: Union[PneumaticEvaluation, None]=None,
                 log_path: Union[str, None]=None,
                 **kwargs
                 ):
        super().__init__(benchmark_name=benchmark_name,
                         evaluation=evaluation,
                         benchmark_path=benchmark_path,
                         log_path=log_path,
                         config_path=config_path
                        )

        self.input_required_for_post_process= False if "input_required_for_post_process" not in kwargs else kwargs["input_required_for_post_process"]

        self.is_loaded=False
        if evaluation is None:
            myEval=PneumaticEvaluation(config_path=config_path,scenario=benchmark_name,log_path=log_path)
            self.evaluation = myEval.from_benchmark(benchmark=self)

        self.env_name = self.config.get_option("env_name")

        # concatenate all the variables for data generation
        attr_names = self.config.get_option("attr_x")\
                     +self.config.get_option("attr_y")


        self.base_dataset = QuasiStaticWheelDataSet("base",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self.train_dataset = QuasiStaticWheelDataSet("train",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self._test_dataset = QuasiStaticWheelDataSet("test",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self.val_dataset = QuasiStaticWheelDataSet("valid",
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
            self.logger.info("Previously saved data will be freed and new data will be reloaded")
        if not os.path.exists(self.path_datasets):
            raise RuntimeError(f"No data are found in {self.path_datasets}. Have you generated or downloaded "
                               f"some data ?")
        self.base_dataset.load(path=self.path_datasets)
        self.is_loaded = True

    def split_train_test_valid(self):
        """
        split base dataset in train/test/validation datasets
        """
        split_ratio=self.config.get_option("split_ratio")
        train_ratio = split_ratio.get("train_ratio")
        test_ratio= split_ratio.get("test_ratio")
        valid_ratio= split_ratio.get("valid_ratio")

        if sum([train_ratio,test_ratio,valid_ratio])>1.0:
            raise Exception("Sum of splitted ratio can not exceed 1")

        base_dataset_size=len(self.base_dataset)
        train_dataset_size=np.floor(train_ratio*base_dataset_size).astype(int)
        test_dataset_size=np.floor(test_ratio*base_dataset_size).astype(int)
        valid_dataset_size=np.floor(valid_ratio*base_dataset_size).astype(int)

        if sum([train_dataset_size,test_dataset_size,valid_dataset_size]) != base_dataset_size:
            num_data_excluded= base_dataset_size- sum([train_dataset_size,test_dataset_size,valid_dataset_size])
            print("Warning: Train/Test/Valid split does not cover the whole dataset!")
            print("Number of sets transfered to validation: %d" %num_data_excluded)
            valid_dataset_size += num_data_excluded

        indices_by_dataset={
            "train":np.arange(0,train_dataset_size),
            "test":np.arange(train_dataset_size,train_dataset_size+test_dataset_size),
            "valid":np.arange(train_dataset_size+test_dataset_size,train_dataset_size+test_dataset_size+valid_dataset_size)
        }
        indices_by_dataset={key:val for key,val in indices_by_dataset.items() if val.size!=0 }

        nb_data_max=max([data.shape[0] for data in indices_by_dataset.values()])
        indices_by_dataset_extended=dict()
        for key,val in indices_by_dataset.items():
            init_extended_val=np.full((nb_data_max), -1)
            init_extended_val[:val.shape[0]]=val
            indices_by_dataset_extended[key]=init_extended_val

        datasets={name:[] for name in indices_by_dataset_extended.keys()}
        indices_name,indices_val=indices_by_dataset_extended.keys(),indices_by_dataset_extended.values()
        for index in range(len(self.base_dataset)):
            indices_dataset=np.array(list(indices_val))
            indices_dataset_loc=np.where(indices_dataset==index)
            if not indices_dataset_loc: 
                raise Exception("index %d not found" %d)
            num_linked_to_index=indices_dataset_loc[0][0]
            name_linked_to_index=list(indices_name)[num_linked_to_index]
            datasets[name_linked_to_index].append(self.base_dataset.get_data(index=index))

        stacked_dataset=dict()
        for name,dataset in datasets.items():
            stacked_dataset[name]={variable: np.squeeze(np.array([data[variable] for data in dataset])) for variable in dataset[0]}

        internal_datasets=dict(zip(["train","test","valid"],[self.train_dataset,self._test_dataset,self.val_dataset]))
        for internal_name,internal_dataset in internal_datasets.items():
            if internal_name in indices_by_dataset.keys():
                internal_dataset.load_from_data(data=stacked_dataset[internal_name])


    def _aux_evaluate_on_single_dataset(self,
                                        dataset: SamplerStaticWheelDataSet,
                                        augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                                        save_path: Union[str, None]=None,
                                        **kwargs) -> dict:
        """Evaluate a single dataset
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be called to evaluate the performance on multiple datasets

        Parameters
        ------
        dataset : SamplerStaticWheelDataSet
            the dataset
        augmented_simulator : Union[PhysicalSimulator, AugmentedSimulator, None], optional
            a trained augmented simulator, by default None
        batch_size : int, optional
            batch_size used for inference, by default 32
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
        predictions = self.augmented_simulator.predict(dataset=dataset,input_required_for_post_process=self.input_required_for_post_process)

        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = dataset.data
        self.dataset = dataset

        res = self.evaluation.evaluate(observations=dataset.data,
                                       predictions=predictions,
                                       save_path=save_path
                                       )
        return res

