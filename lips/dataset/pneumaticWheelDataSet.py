""" This module is used to generate pneumatic datasets

Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""

import os
import warnings
import numpy as np
import shutil

import copy
from typing import Union
from tqdm import tqdm 

from lips.dataset.dataSet import DataSet
from lips.logger import CustomLogger
from lips.config.configmanager import ConfigManager

class WheelDataSet(DataSet):
    """Base class for pneumatic datasets
    This class represent a single dataset dedicated to the pneumatic usecase, that comes from a database.

    Parameters
    ----------
    name: str
        the name of the dataset
    attr_names: tuple
        collection of attributes specific to the dataset
    config: ConfigManager
        instance of configuration
    log_path: str
        path to write the log file
    """
    def __init__(self,
                 name="train",
                 attr_names=("disp",),
                 config: Union[ConfigManager, None]=None,
                 log_path: Union[str, None]=None,
                 **kwargs):
        super(WheelDataSet,self).__init__(name=name)
        self._attr_names = attr_names
        self.size = 0
        self._inputs = []

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        self.config = config

        # number of dimension of x and y (number of columns)
        self._size_x = None
        self._size_y = None
        self._sizes_x = None  # dimension of each variable
        self._sizes_y = None  # dimension of each variable
        self._attr_x = kwargs["attr_x"] if "attr_x" in kwargs.keys() else self.config.get_option("attr_x")
        self._attr_y = kwargs["attr_y"] if "attr_y" in kwargs.keys() else self.config.get_option("attr_y")

    def _infer_sizes(self):
        """Infer the data sizes"""
        data = self.data
        attrs_x=np.array([np.expand_dims(data[el], axis=1) for el in self._attr_x], dtype=int)
        self._sizes_x = np.array([attr_x.shape[1] for attr_x in attrs_x], dtype=int)
        self._size_x = np.sum(self._sizes_x)

        self._sizes_y = np.array([data[el].shape[1] for el in self._attr_y], dtype=int)
        self._size_y = np.sum(self._sizes_y)

    def get_sizes(self):
        """Get the sizes of the dataset

        Returns
        -------
        tuple
            A tuple of size (size_x, size_y)

        """
        return self._size_x, self._size_y


    def load(self, path:str):
        """Load the internal data

        Parameters
        ----------
        path: input path
            A str to indicate where to load the data from.
        """
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        full_path = os.path.join(path, self.name)
        if not os.path.exists(full_path):
            raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")

        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            if not os.path.exists(path_this_array):
                raise RuntimeError(f"Impossible to load data {attr_nm}. Have you called `dataset.generate()` with "
                                   f"a given `path_out` and such that `dataset` is built with the right `attr_names` ?")

        if self.data is not None:
            warnings.warn(f"Deleting previous run in attempting to load the new one located at {path}")
        self.data = {}
        self.size = None

        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.data[attr_nm] = np.load(path_this_array)["data"]
            self.size = self.data[attr_nm].shape[0]

        inputs = {attr_x:self.data[attr_x] for attr_x in self._attr_x}
        self._inputs = [dict(zip(inputs,t)) for t in zip(*inputs.values())]

        self._infer_sizes()

    def load_from_data(self, data:dict,attr_names_to_keep=None):
        """Load the internal data from external data

        Parameters
        ----------
        data: mapping between the data field names and values 
            dict
        """
        self.data = {}
        self.size = None

        if attr_names_to_keep is None:
            attr_names=self._attr_names

        for attr_nm in attr_names_to_keep:
            self.data[attr_nm] = data[attr_nm]
            self.size = self.data[attr_nm].shape[0]

        inputs = {attr_x:self.data[attr_x] for attr_x in self._attr_x}
        self._inputs = [dict(zip(inputs,t)) for t in zip(*inputs.values())]

        self._infer_sizes()

    def get_data(self, index:Union[int, list]):
        """
        This function returns the data in the data that match the index `index`

        Parameters
        ----------
        index:
            A list of integer

        Returns
        -------

        """
        super().get_data(index)  # check that everything is legit

        # make sure the index are numpy array
        if isinstance(index, list):
            index = np.array(index, dtype=int)
        elif isinstance(index, int):
            index = np.array([index], dtype=int)

        # init the results
        res = {}
        nb_sample = index.size
        for el in self._attr_names:
            if len(self.data[el].shape)==1:
                self.data[el]=np.expand_dims(self.data[el], axis=1)
            res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)


        for el in self._attr_names:
            res[el][:] = self.data[el][index, :]

        return res

    def _save_internal_data(self, path_out:str):
        """Save the internal data in a proper format

        Parameters
        ----------
        path_out: output path
            A str to indicate where to save the data.
        """
        full_path_out = os.path.join(os.path.abspath(path_out), self.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))
            # TODO logger
            #print(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")
            self.logger.info(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")

        if os.path.exists(full_path_out):
            # deleting previous saved data
            # TODO logger
            #print(f"Deleting previous run at {full_path_out}")
            self.logger.warning(f"Deleting previous run at {full_path_out}")
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)
        # TODO logger
        #print(f"Creating the path {full_path_out} to store the dataset name {self.name}")
        self.logger.info(f"Creating the path {full_path_out} to store the dataset name {self.name}")

        for attr_nm in self._attr_names:
            np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data=self.data[attr_nm])


    def reconstruct_output(self, data: "np.ndarray") -> dict:
        """It reconstruct the data from the extracted data

        Parameters
        ----------
        data : ``np.ndarray``
            the array that should be reconstruted

        Returns
        -------
        dict
            the reconstructed data with variable names in a dictionary
        """
        predictions = {}
        prev_ = 0
        for var_id, this_var_size in enumerate(self._sizes_y):
            attr_nm = self._attr_y[var_id]
            predictions[attr_nm] = data[:, prev_:(prev_ + this_var_size)]
            prev_ += this_var_size
        return predictions

    def extract_data(self, concat: bool=True) -> tuple:
        """extract the x and y data from the dataset

        Parameters
        ----------
        concat : ``bool``
            If True, the data will be concatenated in a single array.
        Returns
        -------
        tuple
            extracted inputs and outputs
        """
        # init the sizes and everything
        data = self.data
        extract_x = [data[el].astype(np.float32) for el in self._attr_x]
        extract_y = [data[el].astype(np.float32) for el in self._attr_y]

        if concat:
            if len(extract_x[0].shape)==1:
                extract_x = [single_x.reshape((single_x.shape[0],1)) for single_x in extract_x]
            extract_x = np.concatenate(extract_x, axis=1)
            extract_y = np.concatenate(extract_y, axis=1)
        return extract_x, extract_y

    def __getitem__(self, item:int)->tuple:
        currentInput = {inputName:self.data[inputName][item] for inputName in self._attr_x}
        currentOutput = {outputName:self.data[outputName][item] for outputName in self._attr_y}
        return currentInput,currentOutput

    def __str__(self)->str:
        s_info="Instance of "+type(self).__name__+"\n"
        s_info+="Dataset name: "+str(self.name)+"\n"
        for paramName, paramVal in self.data.items():
            s_info+="\t"+str(paramName)+"\n"
            s_info+="\t\t Data size: "+str(paramVal.shape)+"\n"
        return s_info