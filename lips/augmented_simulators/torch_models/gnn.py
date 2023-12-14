"""
Torch fully connected model
"""
import os
import pathlib
from typing import Union
import json

import numpy as np

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.nn as pyg_nn

from .utils import LOSSES
from .gnn_powergrid_utils import *
from ...dataset import DataSet
from ...dataset.scaler import Scaler
from ...logger import CustomLogger
from ...config import ConfigManager
from ...utils import NpEncoder
from ...benchmark.powergridBenchmark import get_env
from ...dataset.utils.powergrid_utils import get_kwargs_simulator_scenario


class TorchGCN(nn.Module):
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
                 bench_config_path: Union[str, pathlib.Path],
                 sim_config_name: Union[str, None]=None,
                 bench_config_name: Union[str, None]=None,
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
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
        self.default_dtype = torch.float32 if "default_dtype" not in kwargs else kwargs.get("default_dtype")
        self.attr_x = self.bench_config.get_option("attr_x") if "attr_x" not in kwargs else kwargs.get("attr_x")
        self.attr_y = self.bench_config.get_option("attr_y") if "attr_y" not in kwargs else kwargs.get("attr_y")
        # scaler
        self.scaler = scaler
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)

        # TODO: verify that these parameters exist in config file
        self.encoder_sizes = self.params["encoder_sizes"]
        self.hidden_sizes = self.params["hidden_sizes"]
        self.decoder_sizes = self.params["decoder_sizes"]

        self.activation = {
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh
        }

        self.conv_layer = {
            "Linear": nn.Linear,
            "GCN": pyg_nn.GCNConv,
            "GAT": pyg_nn.GATConv,
            "DenseGCN": pyg_nn.DenseGCNConv,
            "PNA": pyg_nn.PNAConv
        }
        self.edge_dim = self.params["conv_layer_params"].get("edge_dim", None)
        self.consider_edge_weights = False if self.params["consider_edge_weights"] is False else True
        self.device = "cpu" if self.params.get("device") is None else self.params["device"]

        self.input_size = None if self.params.get("input_size") is None else self.params["input_size"]
        self.output_size = None if self.params.get("output_size") is None else self.params["output_size"]        

        self.input_layer = None
        self.encoder_layers = None
        self.conv_layers = None
        self.decoder_layers = None
        self.output_layer = None

        # batch information
        self._batch = None
        self._data = None
        self._target = None

        #self.__build_model()

    def build_model(self, **kwargs):
        """Build the model architecture

        Parameters
        ------
        kwargs : ``dict`` 
            including all the required arguments for each specific GNN layer
            it could be read from the config file
            The dictionary could contain the elements like:
                - ``node_dim=1`` to indicate the dimension of batch data where the node dimension could be obtained
                - ``edge_dim=x`` to indicate the dimensionality of edge features
        """
        torch.set_default_dtype(self.default_dtype)
        conv_layer = self.conv_layer.get(self.params["conv_layer"], pyg_nn.GCNConv)
        if self.encoder_sizes:
            self.input_layer = nn.Linear(self.input_size, self.encoder_sizes[0])
            self.encoder_layers = nn.ModuleList([nn.Linear(in_, out_) \
                                                 for in_, out_ in zip(self.encoder_sizes[:-1], self.encoder_sizes[1:])])
            self.encoder_layers.append(conv_layer(self.encoder_sizes[-1], self.hidden_sizes[0]))
        else:
            self.input_layer = conv_layer(self.input_size, self.hidden_sizes[0])
        self.conv_layers = nn.ModuleList([conv_layer(in_, out_, **kwargs) \
                                            for in_, out_ in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])])
        if self.decoder_sizes:
            self.decoder_sizes = list(self.decoder_sizes)
            self.decoder_sizes.insert(0, self.hidden_sizes[-1])
            self.decoder_layers = nn.ModuleList([nn.Linear(in_, out_) \
                                                 for in_, out_ in zip(self.decoder_sizes[:-1], self.decoder_sizes[1:])])
            self.output_layer = nn.Linear(self.decoder_sizes[-1], self.output_size)
        else:
            self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, batch):
        if batch.edge_attr is not None:
            features, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr
        else:
            features, edge_index = batch.x, batch.edge_index
            edge_weight=None
        
        activation = self.activation[self.params["activation"]]

        if self.encoder_sizes:
            out = self.input_layer(features)
            out = activation(out)
            for encoder in self.encoder_layers[:-1]:
                out = encoder(out)
                out = activation(out)
            out = self.encoder_layers[-1](out, edge_index, edge_weight=edge_weight)
            out = activation(out)
        else:
            out = self.input_layer(features, edge_index, edge_weight=edge_weight)
            out = activation(out)

        for conv in self.conv_layers:
            out = conv(out, edge_index, edge_weight=edge_weight)
            out = activation(out)
        if self.decoder_sizes:
            for decoder in self.decoder_layers:
                out = decoder(out)
                out = activation(out)
        out = self.output_layer(out)
        return out

    def process_dataset(self, dataset: DataSet, training: bool, **kwargs):
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
        load_from_file = kwargs.get("load_from_file", None)
        if load_from_file is not None:
            loader_name = dataset.name + "_loader.pth"
            load_from_file = pathlib.Path(load_from_file)
            path_to_loader = load_from_file / loader_name
            if not path_to_loader.exists():
                raise RuntimeError("Data loader path not found for the benchmark!")
            self._infer_size(dataset)
            data_loader = torch.load(path_to_loader)
        else:
            print("processing the dataset: ", dataset.name)
            obs = self._get_obs()
            device = kwargs.get("device", self.device)
            if training:
                self._infer_size(dataset)
                batch_size = self.params["train_batch_size"]
                features, targets, edge_indices, edge_weights = prepare_dataset(obs, dataset.data, return_edge_weights=self.consider_edge_weights)
                #extract_x, extract_y = dataset.extract_data()
                #TODO: adapt the normalization for GNN
                # if self.scaler is not None:
                #     extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
            else:
                batch_size = self.params["eval_batch_size"]
                #extract_x, extract_y = dataset.extract_data()
                features, targets, edge_indices, edge_weights = prepare_dataset(obs, dataset.data, return_edge_weights=self.consider_edge_weights)
                # if self.scaler is not None:
                #     extract_x, extract_y = self.scaler.transform(extract_x, extract_y)

            data_loader = get_batches_pyg(edge_indices=edge_indices,
                                        features=features,
                                        targets=targets,
                                        edge_weights=edge_weights,
                                        batch_size=batch_size,
                                        #ybus=dataset.data["YBus"],
                                        #line_status=dataset.data["line_status"],
                                        #topo_vect=dataset.data["topo_vect"],
                                        device=device)
        #data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"])
        return data_loader
    
    # def _post_process_inplace(self, predictions):
    #     """
    #     Post process if the conversion of theta_bus to active powers should happen here

    #     Parameters
    #     ----------
    #     predictions : _type_
    #         _description_

    #     Returns
    #     -------
    #     _type_
    #         _description_
    #     """
    #     outputs = []
    #     obs = self._get_obs()
    #     p_ors_pred, p_exs_pred = get_all_active_powers(self._batch,
    #                                                    obs,
    #                                                    theta_bus=predictions.view(-1, 14).cpu())
    #     #outputs.append(dataset.data["theta_or"])
    #     #outputs.append(dataset.data["theta_ex"])
    #     outputs.append(p_ors_pred)
    #     outputs.append(p_exs_pred)
    #     return outputs
    
    def _post_process(self, data):
        if self.scaler is not None:
            try:
                processed = self.scaler.inverse_transform(data)
            except TypeError:
                processed = self.scaler.inverse_transform(data.cpu())
        else:
            processed = data
        return processed
    
    def _reconstruct_output(self, dataset: DataSet, data: npt.NDArray[np.float64]):
        """Reconstruct the outputs to obtain the desired shape for evaluation

        In the simplest form, this function is implemented in DataSet class. It supposes that the predictions 
        obtained by the augmented simulator are exactly the same as the one indicated in the configuration file

        However, if some transformations required by each specific model, the extra operations to obtained the
        desired output shape should be done in this function.

        Parameters
        ----------
        dataset : DataSet
            An object of the `DataSet` class 
        data : npt.NDArray[np.float64]
            the data which should be reconstructed to the desired form
        """
        outputs = []
        obs = self._get_obs()
        theta_bus = data.reshape(-1, obs.n_sub)
        p_ors_pred, p_exs_pred = get_all_active_powers(dataset, obs, theta_bus=theta_bus)
        theta_ors_pred, theta_exs_pred = reconstruct_theta_line(dataset, obs, theta_sub=theta_bus)
        outputs.append(theta_ors_pred)
        outputs.append(theta_exs_pred)
        outputs.append(p_ors_pred)
        outputs.append(p_exs_pred)
        outputs = np.hstack(outputs)
        outputs = dataset.reconstruct_output(outputs)
        return outputs
        

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
        #*dim_inputs, self.output_size = dataset.get_sizes(attr_x=self.attr_x, attr_y=self.attr_y)
        #self.input_size = np.sum(dim_inputs)
        self.input_size = self.params["input_size"]
        self.output_size = self.params["output_size"]

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

    def _do_forward(self, batch, **kwargs):
        """Do the forward step through a batch of data

        This step could be very specific to each augmented simulator as each architecture
        takes various inputs during the learning procedure. 

        Parameters
        ----------
        batch : _type_
            A batch of data including various information required by an architecture
        device : _type_
            the device on which the data should be processed

        Returns
        -------
        ``tuple``
            returns the predictions made by the augmented simulator and also the real targets
            on which the loss function should be computed
        """
        self._batch = batch
        self._data = batch.x
        self._target = batch.y
        predictions = self.forward(batch)
        
        return self._data, predictions, self._target

    def get_loss_func(self, loss_name: str, **kwargs) -> Tensor:
        """
        Helper to get loss. It is specific to each architecture
        """
        # if len(args) > 0:
        #     # for Masked RNN loss. args[0] is the list of sequence lengths
        #     loss_func = LOSSES[self.params["loss"]["name"]](args[0], self.params["device"])
        # else:
        loss_func = LOSSES[loss_name](**kwargs)
        
        return loss_func
    
    def _get_obs(self):
        env = get_env(get_kwargs_simulator_scenario(self.bench_config))
        obs = env.reset()
        return obs
    