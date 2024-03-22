"""
Pytorch based augmented simulators
"""
import os
import pathlib
from typing import Union
import shutil
import time
import json
import tempfile
from tqdm import tqdm

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import torch
from torch import optim
from torch import nn
from torch import Tensor
#from torch.nn.modules.loss import _Loss as Loss
from torch.utils.data import DataLoader

from . import AugmentedSimulator
from .torch_models.utils import LOSSES, OPTIMIZERS
from ..dataset import DataSet
from ..dataset.scaler import Scaler
from ..logger import CustomLogger
from ..utils import NpEncoder

class TorchSimulator(AugmentedSimulator):
    """Pytorch based simulators

        .. code-block:: python

            from lips.augmented_simulators.torch_simulator import TorchSimulator
            from lips.augmented_simulators.torch_models import TorchFullyConnected

            params = {"input_size": 784, "output_size": 10}
            torch_sim = TorchSimulator(name="torch_fc",
                                       model=TorchFullyConnected,
                                       **params)
        Parameters
        ----------
        model : nn.Module
            _description_
        name : str, optional
            _description_, by default None
        scaler : Scaler, optional
            scaler used to scale the data, by default None
        **kwargs : dict
            supplementary parameters for the model
            It should contain input_size and output_size
            # TODO: infer from dataset
            It will replace the configs in the config file

        """
    def __init__(self,
                 model: nn.Module,
                 sim_config_path: Union[pathlib.Path, str],
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
                 log_path: Union[str, None]=None,
                 seed: int=42,
                 **kwargs):
        #super().__init__(model, name, scaler, log_path, **kwargs)
        super().__init__(name, log_path, model, **kwargs)
        # logger
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        # scaler
        if "scalerParams" not in kwargs.keys():
            self.scaler = scaler() if scaler else None
        else:
            self.scaler = scaler(**kwargs["scalerParams"])
        self._model = self.model(name=name, sim_config_path=sim_config_path, scaler=self.scaler, **kwargs)
        sim_config_name = self._model.sim_config.section_name
        self.name = self.name + '_' + sim_config_name
        self.params.update(self._model.params)

        # torch devices
        self.device = torch.device(self.params["device"])
        # torch seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


    def build_model(self):
        """build torch model

        Parameters
        ----------
        **kwargs : dict
            if parameters indicated, it will replace config parameters

        Returns
        -------
        nn.Module
            a torch model
        """
        #model_ = self.torch_model(**kwargs)
        #self.params.update(self.torch_model.params)
        self._model.build_model()

    def train(self,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet] = None,
              save_path: Union[None, str] = None,
              **kwargs):
        """Function used to train a neural network

        Parameters
        ----------
        train_dataset : DataSet
            training dataset
        val_dataset : Union[None, DataSet], optional
            validation dataset, by default None
        save_path : Union[None, str], optional
            the path where the trained model should be saved, by default None
        """
        self.params.update(kwargs)
        self._model.params.update(kwargs)
        super().train(train_dataset, val_dataset)

        train_loader = self._model.process_dataset(train_dataset, training=True, **kwargs)
        if val_dataset is not None:
            val_loader = self._model.process_dataset(val_dataset, training=False, **kwargs)
        else:
            val_loader = None

        # build the model
        self.build_model()
        self._model.to(self.params["device"])
        dtype = kwargs.get("dtype", torch.float32)
        self._model.to(dtype)
        
        optimizer = self._get_optimizer(optimizer=OPTIMIZERS[self.params["optimizer"]["name"]],
                                        **self.params["optimizer"]["params"])
        for metric_ in self.params["metrics"]:
            self.train_metrics[metric_] = list()
            if val_loader is not None:
                self.val_metrics[metric_] = list()

        self.logger.info("Training of {%s} started", self.name)
        #losses, elapsed_time = train_model(self.model, data_loaders=data)
        #pbar = tqdm(range(1, self.params["epochs"]+1))
        #for epoch in pbar:
        for epoch in range(self.params["epochs"]):
            #pbar.set_description("Epoch %s" % str(epoch))
            train_loss_epoch, train_metrics_epoch = self._train_one_epoch(epoch, train_loader, optimizer, **kwargs)
            self.train_losses.append(train_loss_epoch)
            for nm_, arr_ in self.train_metrics.items():
                arr_.append(train_metrics_epoch[nm_])

            if val_loader is not None:
                val_loss_epoch, val_metrics_epoch = self._validate(val_loader)
                self.val_losses.append(val_loss_epoch)
                for nm_, arr_ in self.val_metrics.items():
                    arr_.append(val_metrics_epoch[nm_])

            # check point
            if self.params["save_freq"] and (save_path is not None):
                if epoch % self.params["ckpt_freq"] == 0:
                    self.save(save_path, epoch)

        self.trained = True
        # save the final model
        if save_path:
            self.save(save_path)

    def _train_one_epoch(self, epoch:int, train_loader: DataLoader, optimizer: optim.Optimizer, **kwargs) -> set:
        """
        train the model at a epoch
        """
        self._model.train()
        torch.set_grad_enabled(True)

        total_loss = 0
        metric_dict = dict()
        for metric in self.params["metrics"]:
            metric_dict[metric] = 0

        #pbar=tqdm(train_loader)
        for batch_ in train_loader:
        #for batch_ in pbar:
            #pbar.set_description("Batch within epoch (Training)")
            optimizer.zero_grad()
            _, prediction, target = self._model._do_forward(batch_, **kwargs)
            loss_func = self._model.get_loss_func(loss_name=self.params["loss"]["name"], **self.params["loss"]["params"])
            loss = loss_func(prediction, target)
            loss.backward()
            optimizer.step()
            total_loss += (loss*len(target))

            for metric in self.params["metrics"]:
                metric_func = self._model.get_loss_func(loss_name=metric, reduction="mean")
                metric_value = metric_func(prediction, target)
                metric_value = metric_value.item()*len(target)
                metric_dict[metric] += metric_value

        mean_loss = total_loss.item()/len(train_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= len(train_loader.dataset)
        print(f"Train Epoch: {epoch}   Avg_Loss: {mean_loss:.5f}",
              [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])
        return mean_loss, metric_dict

    def _validate(self, val_loader: DataLoader, **kwargs) -> set:
        """function used for validation of the model

        It is separated from evaluate function, because it should be called at each epoch during training

        Parameters
        ----------
        val_loader : DataLoader
            _description_

        Returns
        -------
        set
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        self.params.update(kwargs)
        self._model.eval()
        total_loss = 0
        metric_dict = dict()
        for metric in self.params["metrics"]:
            metric_dict[metric] = 0

        with torch.no_grad():

            #pbar=tqdm(val_loader)
            #for batch_ in pbar:
            for batch_ in val_loader:
                #pbar.set_description("Batch within epoch (Evaluation)")
                _, prediction, target = self._model._do_forward(batch_, **kwargs)
                loss_func = self._model.get_loss_func(loss_name=self.params["loss"]["name"], **self.params["loss"]["params"])
                loss = loss_func(prediction, target)
                total_loss += loss.item()*len(target)

                for metric in self.params["metrics"]:
                    metric_func = self._model.get_loss_func(loss_name=metric, reduction="mean")
                    metric_value = metric_func(prediction, target)
                    metric_value = metric_value.item()*len(target)
                    metric_dict[metric] += metric_value

        mean_loss = total_loss/len(val_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= len(val_loader.dataset)
        print(f"Eval:   Avg_Loss: {mean_loss:.5f}",
              [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])

        return mean_loss, metric_dict

    def predict(self, dataset: DataSet, reconstruct_output: bool=True, **kwargs) -> Union[dict, npt.NDArray[np.float64]]:
        """_summary_

        Parameters
        ----------
        dataset : DataSet
            test datasets to evaluate

        reconstruct_output: ``bool``
            Whether to reconstruct the ouput of the model, by default True
            This should be set to ``True`` if the evaluation is done using the evaluation module 
            of LIPS platform. To be set to ``False`` for custom predictions and custom evaluations

        Returns
        -------
        ``dict``
            a dictionary including the predictions of the model with keys indicating the variables
            name that are set in the configuration file as desired ouputs of the model
        """
        super().predict(dataset)
        if "eval_batch_size" in kwargs:
            self.params["eval_batch_size"] = kwargs["eval_batch_size"]
            self._model.params["eval_batch_size"] = kwargs["eval_batch_size"]

        test_loader = self._model.process_dataset(dataset, training=False, **kwargs)
        
        # activate the evaluation mode
        self._model.eval()
        predictions = []
        observations = []
        total_loss = 0
        loss_func = self._model.get_loss_func(loss_name=self.params["loss"]["name"], **self.params["loss"]["params"])
        metric_dict = dict()
        for metric in self.params["metrics"]:
            metric_dict[metric] = 0

        total_time = 0
        with torch.no_grad():
            #pbar=tqdm(test_loader)
            #for batch_ in pbar:
            for batch_ in test_loader:
                #pbar.set_description("Batch (Prediction)")
                data, prediction, target = self._model._do_forward(batch_, **kwargs)
                
                if "input_required_for_post_process" in kwargs and kwargs["input_required_for_post_process"]:
                    input_model=data
                    prediction = self._model._post_process_with_input(input_model,prediction)
                    target = self._model._post_process_with_input(input_model,target)
                else:
                    prediction = self._model._post_process(prediction)
                    target = self._model._post_process(target)

                try:
                    predictions.append(prediction.numpy())
                    observations.append(target.numpy())
                except TypeError:
                    predictions.append(prediction.cpu().data.numpy())
                    observations.append(target.cpu().data.numpy())

                loss = loss_func(prediction, target)
                total_loss += loss.item()*len(target)

                for metric in self.params["metrics"]:
                    metric_func = self._model.get_loss_func(loss_name=metric, reduction="mean")
                    metric_value = metric_func(prediction, target)
                    metric_value = metric_value.item()*len(target)
                    metric_dict[metric] += metric_value

        mean_loss = total_loss/len(test_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= len(test_loader.dataset)
        #print(f"Eval:   Avg_Loss: {mean_loss:.5f}",
        #      [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])

        predictions = np.concatenate(predictions)
        observations = np.concatenate(observations)
        if reconstruct_output:
            predictions = self._model._reconstruct_output(dataset, predictions)
            observations = self._model._reconstruct_output(dataset, observations)

        self._predictions[dataset.name] = predictions
        self._observations[dataset.name] = observations
        
        return predictions

    def _get_loss_func(self, *args) -> Tensor:
        """
        Helper to get loss
        """
        if len(args) > 0:
            # for Masked RNN loss. args[0] is the list of sequence lengths
            loss_func = LOSSES[self.params["loss"]["name"]](args[0], self.params["device"])
        else:
            loss_func = LOSSES[self.params["loss"]["name"]](**self.params["loss"]["params"])
        return loss_func

    def _get_optimizer(self, optimizer: optim.Optimizer=optim.Adam, **kwargs):
        """get the optimizer

        Parameters
        ----------
        optimizer : optim.Optimizer, optional
            _description_, by default optim.Adam
        **kwargs : dict
            the parameters for optimizer
        Returns
        -------
        _type_
            _description_
        """
        return optimizer(self._model.parameters(), **kwargs)

    ###############################################
    # function used to save and restore the model #
    ###############################################
    def save(self, path: Union[str, pathlib.Path], epoch: Union[int, None]=None, save_metadata: bool=True):
        """_summary_

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            _description_
        epoch : Union[int, None], optional
            _description_, by default None
        save_metadata : bool, optional
            _description_, by default True
        """
        save_path =  pathlib.Path(path) / self.name
        super().save(save_path)

        epoch_ = str(epoch) if epoch is not None else "_last"
        self._save_model(epoch_, save_path)


        if epoch is None and save_metadata:
            self._save_metadata(save_path)

        self.logger.info("Model {%s} is saved at {%s}", self.name, save_path)

    def _save_model(self, epoch: str, path: str):
        file_name = path / ("model" + epoch + ".pt")
        torch.save(self._model.state_dict(), file_name)

    def _save_metadata(self, path: Union[str, pathlib.Path]):
        """save model's metadata
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if self.scaler is not None:
            self.scaler.save(path)
        self._model._save_metadata(path)
        self._save_losses(path)
        self.params["bench_config_path"] = str(self.params["bench_config_path"])
        with open((path / "config.json"), "w", encoding="utf-8") as f:
            json.dump(obj=self.params, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def restore(self, path: Union[str, pathlib.Path], epoch: Union[int, None]=None):
        """
        restore the model
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        full_path = path / self.name
        if not full_path.exists():
            raise FileNotFoundError(f"path {full_path} not found")
        # load the metadata
        self._load_metadata(full_path)
        self._load_model(epoch, full_path)
        self.logger.info("Model {%s} is loaded from {%s}", self.name, full_path)

    def _load_metadata(self, path: str):
        """
        load the model metadata
        """
        # load scaler parameters
        if self.scaler is not None:
            self.scaler.load(path)
        self._load_losses(path)
        with open((path / "config.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.params.update(res_json)
        #self.device = torch.device(self.params["device"])
        return self.params

    def _load_model(self, epoch, path: str):
        epoch = str(epoch) if epoch is not None else "_last"
        nm_file = "model" + epoch + ".pt"
        path_weights = path / nm_file
        if not path_weights.exists():
            raise FileNotFoundError(f"Weights file {path_weights} not found")
        self._model._load_metadata(path)
        self.build_model()
        self._model.to(self.device)
        # load the weights
        with tempfile.TemporaryDirectory() as path_tmp:
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(path_weights, nm_tmp)
            # load this copy (make sure the proper file is not corrupted even if the loading fails)
            self._model.load_state_dict(torch.load(nm_tmp), strict=False)

    def _save_losses(self, path: Union[str, pathlib.Path]):
        """
        save the losses
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        res_losses = {}
        res_losses["train_losses"] = self.train_losses
        res_losses["train_metrics"] = self.train_metrics
        res_losses["val_losses"] = self.val_losses
        res_losses["val_metrics"] = self.val_metrics
        with open((path / "losses.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_losses, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_losses(self, path: Union[str, pathlib.Path]):
        """
        load the losses
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "losses.json"), "r", encoding="utf-8") as f:
            res_losses = json.load(fp=f)
        self.train_losses = res_losses["train_losses"]
        self.train_metrics = res_losses["train_metrics"]
        self.val_losses = res_losses["val_losses"]
        self.val_metrics = res_losses["val_metrics"]
    #########################
    # Some Helper functions #
    #########################
    def summary(self):
        """summary of the model
        """
        print(self._model)

    def count_parameters(self):
        """
        count the number of parameters in the model
        """
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def visualize_convergence(self, figsize=(15,5), save_path: str=None):
        """Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = len(self.params["metrics"])
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)
        ax[0].set_title("MSE")
        ax[0].plot(self.train_losses, label='train_loss')
        if len(self.val_losses) > 0:
            ax[0].plot(self.val_losses, label='val_loss')
        for idx_, metric_name in enumerate(self.params["metrics"]):
            ax[idx_+1].set_title(metric_name)
            ax[idx_+1].plot(self.train_metrics[metric_name], label=f"train_{metric_name}")
            if len(self.val_metrics[metric_name]) > 0:
                ax[idx_+1].plot(self.val_metrics[metric_name], label=f"val_{metric_name}")
        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            fig.savefig(save_path)
