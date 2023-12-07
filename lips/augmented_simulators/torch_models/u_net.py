"""
Usage:
    Torch CNN U-net model
Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""

import pathlib
from typing import Union
import json

import numpy as np
import numpy.typing as npt

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .utils import LOSSES
from ...dataset import DataSet
from ...dataset.scaler import Scaler
from ...logger import CustomLogger
from ...config import ConfigManager
from ...utils import NpEncoder

#Credits goes to https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    """Implement double convolution:
    (convolution => [BatchNorm2d] => ReLU) * 2

    Attributes
    ----------
    in_channels : int
        number of input channels for the convolutional
    out_channels : int
        number of output channels for the convolutional
    mid_channels : Union[int, None], optional
        number of middle channels for the convolutional (after the first convolution)

    """
    def __init__(self, in_channels:int, out_channels:int, mid_channels:Union[int, None] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.tensor):
        """forward run
        
        Attributes
        ----------
        in_channels : torch.tensor
            input data
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv

    Attributes
    ----------
    in_channels : int
        number of input channels for the convolutional
    out_channels : int
        number of output channels for the convolutional
    """
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x:torch.tensor):
        """forward run
        
        Attributes
        ----------
        in_channels : torch.tensor
            input data
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv

    Attributes
    ----------
    in_channels : int
        number of input channels for the convolutional
    out_channels : int
        number of output channels for the convolutional
    bilinear : bool, optional
        if True (default), use the normal convolutions to reduce the number of channels
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1:torch.tensor, x2:torch.tensor):
        """forward run
        
        Attributes
        ----------
        in_channels : torch.tensor
            input data 1
        in_channels : torch.tensor
            input data 2
        """
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Outer convolution

    Attributes
    ----------
    in_channels : int
        number of input channels for the convolutional
    out_channels : int
        number of output channels for the convolutional
    """
    def __init__(self, in_channels:int, out_channels:int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x:torch.tensor):
        """forward run
        
        Attributes
        ----------
        in_channels : torch.tensor
            input data
        """
        return self.conv(x)


class TorchUnet(nn.Module):
    """Implementation of Unet using pytorch

    Attributes
    ----------
    name : Union[str, None], optional
        model name
    scaler : Union[Scaler, None]
        scaler to be used within model
    bench_config_path : Union[str, pathlib.Path, None], optional
        benchmark configuration path
    bench_config_name : Union[str, None], optional
        benchmark configuration name
    sim_config_path : Union[str, None], optional
        augmented simulator configuration path
    sim_config_name : Union[str, None], optional
        augmented simulator configuration name
    log_path : Union[str, None], optional
        log path
    """
    def __init__(self,
                 sim_config_path: str,
                 bench_config_path: Union[str, pathlib.Path],
                 sim_config_name: Union[str, None]=None,
                 bench_config_name: Union[str, None]=None,
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
                 log_path: Union[None, str]=None,
                 **kwargs):
        super().__init__()#name=name, log_path=log_path, **kwargs)

        # Benchmark configurations
        self.bench_config = ConfigManager(section_name=bench_config_name, path=bench_config_path)
        # The config file associoated to this model
        sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        # sim_config_path_default = pathlib.Path(__file__).parent.parent / "configurations" / "torch_unet.ini"
        # sim_config_path = sim_config_path if sim_config_path is not None else sim_config_path_default
        self.sim_config = ConfigManager(section_name=sim_config_name, path=sim_config_path)
        self.name = name if name is not None else self.sim_config.get_option("name")
        # scaler
        self.scaler = scaler
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)

        self.bilinear = True

        self.n_channels = None if kwargs.get("input_channel_size") is None else kwargs["input_channel_size"]
        self.n_classes = None if kwargs.get("output_channel_size") is None else kwargs["output_channel_size"]

        # batch information
        self._data = None
        self._target = None


    def build_model(self):
        """Build the model flow
        """
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x:torch.tensor):
        """forward run
        
        Attributes
        ----------
        in_channels :torch.tensor
            input data
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def process_dataset(self, dataset: DataSet, training: bool, **kwargs):
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            dataset
        training : bool, optional
            training flag for the type of dataset considered, by default False

        Returns
        -------
        DataLoader
            dataloader
        """
        pin_memory = kwargs.get("pin_memory", True)
        num_workers = kwargs.get("num_workers", None)
        dtype = kwargs.get("dtype", torch.float32)

        if training:
            self._infer_size(dataset)
            batch_size = self.params["train_batch_size"]
            extract_x, extract_y = dataset.extract_data(concat=False)

            nb_x_var_axis=len(dataset.data[dataset._attr_x[0]].shape)
            extract_x = np.stack(extract_x, axis=nb_x_var_axis)
            axis_indexes=list(range(nb_x_var_axis+1))
            new_axis_indexes=tuple([axis_indexes[0]]+[axis_indexes[-1]]+axis_indexes[1:len(axis_indexes)-1])
            extract_x=np.transpose(extract_x,new_axis_indexes)
            extract_y = np.concatenate(extract_y, axis=1)

            if self.scaler is not None:
                extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
        else:
            batch_size = self.params["eval_batch_size"]
            extract_x, extract_y = dataset.extract_data(concat=False)

            nb_x_var_axis=len(dataset.data[dataset._attr_x[0]].shape)
            extract_x = np.stack(extract_x, axis=nb_x_var_axis)
            axis_indexes=list(range(nb_x_var_axis+1))
            new_axis_indexes=tuple([axis_indexes[0]]+[axis_indexes[-1]]+axis_indexes[1:len(axis_indexes)-1])
            extract_x=np.transpose(extract_x,new_axis_indexes)
            extract_y = np.concatenate(extract_y, axis=1)


            if self.scaler is not None:
                extract_x, extract_y = self.scaler.transform(extract_x, extract_y)

        torch_dataset = TensorDataset(torch.tensor(extract_x, dtype=dtype), torch.tensor(extract_y, dtype=dtype))
        data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"], pin_memory=pin_memory, num_workers=num_workers)
        return data_loader

    def _post_process(self, data:torch.tensor):
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        data :torch.tensor
            data

        Returns
        ----------
        data :torch.tensor
            data
        """
        if self.scaler is not None:
            data=data.numpy()
            processed = self.scaler.inverse_transform(data)
            data=torch.from_numpy(processed)
        return data

    def _infer_size(self, dataset: DataSet):
        """Infer the size of the model

        Parameters
        ----------
        dataset : DataSet
            dataset
        """
        *dim_inputs, self.n_classes = dataset.get_sizes()
        self.n_channels = np.sum(dim_inputs)

    def _reconstruct_output(self, dataset: DataSet, data: npt.NDArray[np.float64]) -> dict:
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
        data_rec = dataset.reconstruct_output(data)
        return data_rec
    
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
        non_blocking = kwargs.get("non_blocking", True)
        device = self.params.get("device", "cpu")

        self._data, self._target = batch
        self._data = self._data.to(device, non_blocking=non_blocking)
        self._target = self._target.to(device, non_blocking=non_blocking)

        predictions = self.forward(self._data)
        
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

    def get_metadata(self):
        """Retrieve metadata

        Here, it retrieve the sizes related to the data

        Returns
        ----------
        metadata :dict
            specific data related to the model architecture
        """
        res_json = {}
        res_json["input_size"] = self.n_channels
        res_json["output_size"] = self.n_classes
        return res_json

    def _save_metadata(self, path: str):
        """Save metadata

        Here, it saves the sizes related to the data

        Parameters
        ----------
        path : str
            path where the metadata are saved
        """
        res_json = {}
        res_json["input_size"] = self.n_channels
        res_json["output_size"] = self.n_classes
        with open((path / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_metadata(self, path: str):
        """Load metadata

        Here, it loads and initialize the sizes related to the data

        Parameters
        ----------
        path : str
            path to load the metadata from
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.input_size = res_json["input_size"]
        self.output_size = res_json["output_size"]
