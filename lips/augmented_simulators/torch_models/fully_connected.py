"""
Torch fully connected model
"""
import os
import pathlib
from typing import Union
import json

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from ...dataset import DataSet
from ...dataset.scaler import Scaler
from ...logger import CustomLogger
from ...config import ConfigManager
from ...utils import NpEncoder

class TorchFullyConnected(nn.Module):
    """_summary_

    Parameters
    ----------
    sim_config_path : Union[``pathlib.Path``, ``str``]
        The path to the configuration file for simulator.
        It should contain all the required hyperparameters for this model.
    sim_config_name : Union[str, None], optional
        _description_, by default None
    name : Union[str, None], optional
        _description_, by default None
    scaler : Union[Scaler, None], optional
        _description_, by default None
    bench_config_path : Union[str, pathlib.Path, None], optional
        _description_, by default None
    bench_config_name : Union[str, None], optional
        _description_, by default None
    log_path : Union[None, str], optional
        _description_, by default None

    Raises
    ------
    RuntimeError
        You should provide a path to the configuration file for this augmented simulator
    """
    def __init__(self,
                 sim_config_path: Union[pathlib.Path, str],
                 sim_config_name: Union[str, None]=None,
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
                 bench_config_path: Union[str, pathlib.Path, None]=None,
                 bench_config_name: Union[str, None]=None,
                 log_path: Union[None, pathlib.Path, str]=None,
                 **kwargs):
        super().__init__()
        if not os.path.exists(sim_config_path):
            raise RuntimeError("Configuration path for the simulator not found!")
        if not str(sim_config_path).endswith(".ini"):
            raise RuntimeError("The configuration file should have `.ini` extension!")
        sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        self.sim_config = ConfigManager(section_name=sim_config_name, path=sim_config_path)
        self.bench_config = ConfigManager(section_name=bench_config_name, path=bench_config_path)
        self.name = name if name is not None else self.sim_config.get_option("name")
        # scaler
        self.scaler = scaler
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)

        self.activation = {
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh
        }

        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]

        self.input_layer = None
        self.input_dropout = None
        self.fc_layers = None
        self.dropout_layers = None
        self.output_layer = None

        #self.__build_model()

    def build_model(self):
        """Build the model flow
        """
        linear_sizes = list(self.params["layers"])

        self.input_layer = nn.Linear(self.input_size, linear_sizes[0])
        self.input_dropout = nn.Dropout(p=self.params["input_dropout"])

        self.fc_layers = nn.ModuleList([nn.Linear(in_f, out_f) \
            for in_f, out_f in zip(linear_sizes[:-1], linear_sizes[1:])])

        self.dropout_layers = nn.ModuleList([nn.Dropout(p=self.params["dropout"]) \
            for _ in range(len(self.fc_layers))])

        self.output_layer = nn.Linear(linear_sizes[-1], self.output_size)

    def forward(self, data):
        out = self.input_layer(data)
        out = self.input_dropout(out)
        for _, (fc_, dropout) in enumerate(zip(self.fc_layers, self.dropout_layers)):
            out = fc_(out)
            out = self.activation[self.params["activation"]](out)
            out = dropout(out)
        out = self.output_layer(out)
        return out

    def process_dataset(self, dataset: DataSet, training: bool):
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            _description_
        scaler : Scaler, optional
            _description_, by default True
        training : bool, optional
            _description_, by default False

        Returns
        -------
        DataLoader
            _description_
        """
        if training:
            self._infer_size(dataset)
            batch_size = self.params["train_batch_size"]
            extract_x, extract_y = dataset.extract_data()
            if self.scaler is not None:
                extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
        else:
            batch_size = self.params["eval_batch_size"]
            extract_x, extract_y = dataset.extract_data()
            if self.scaler is not None:
                extract_x, extract_y = self.scaler.transform(extract_x, extract_y)

        torch_dataset = TensorDataset(torch.from_numpy(extract_x).float(), torch.from_numpy(extract_y).float())
        data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"])
        return data_loader

    def _post_process(self, data):
        if self.scaler is not None:
            processed = self.scaler.inverse_transform(data)
        return processed

    def _infer_size(self, dataset: DataSet):
        """Infer the size of the model

        Parameters
        ----------
        dataset : DataSet
            _description_

        Returns
        -------
        None
            _description_
        """
        *dim_inputs, self.output_size = dataset.get_sizes()
        self.input_size = np.sum(dim_inputs)

    def get_metadata(self):
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        return res_json

    def _save_metadata(self, path: str):
        #super()._save_metadata(path)
        #if self.scaler is not None:
        #    self.scaler.save(path)
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        with open((path / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_metadata(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        #super()._load_metadata(path)
        #if self.scaler is not None:
        #    self.scaler.load(path)
        with open((path / "metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.input_size = res_json["input_size"]
        self.output_size = res_json["output_size"]
