# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union
from abc import ABC, abstractmethod
import pathlib
import shutil

from ..dataset import DataSet

class AugmentedSimulator(ABC):
    """
    This class is the Base class that is used to create some "augmented simulator". These "augmented simulator" can be
    anything that emulates the behaviour of some "simulator".

    They are meant to use data coming from a `DataSet` to learn from it.
    """
    def __init__(self,
                 name: Union[str, None]=None,
                 log_path: Union[str, None]=None,
                 model: Union["torch.nn.Module", None]=None,
                 **kwargs):
        self.name = name
        self.model = model #unused for tensorflow models
        self.trained = False
        self._model = None
        self.log_path = log_path
        self.params = kwargs

        self._observations = dict()
        self._predictions = dict()

        # history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}

        self.predict_time = 0

    @abstractmethod
    def train(self,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet]=None):
        """
        Train the Augmented simulator using the provided datasets (parameters `train_dataset` and
        `val_dataset`) for a given number of iterations (`nb_iter`)
        """
        if not isinstance(train_dataset, DataSet):
            raise RuntimeError(f"The \"train_dataset\" should be an instance of DataSet. "
                               f"We found {type(train_dataset)}")
        if val_dataset is not None:
            if not isinstance(val_dataset, DataSet):
                raise RuntimeError(f"The \"val_dataset\" should be an instance of DataSet. "
                                   f"We found {type(val_dataset)}")

    @abstractmethod
    def predict(self, dataset: DataSet):
        """
        evaluate the model on the full dataset
        """
        if not isinstance(dataset, DataSet):
            raise RuntimeError(f"The \"test_dataset\" should be an instance of DataSet. "
                               f"We found {type(dataset)}")

    @abstractmethod
    def build_model(self):
        """Build the model

        This is where a neural network is initialized or built.
        """
        pass

    def process_dataset(self, dataset: DataSet, training: bool):
        """
        This function transforms one state of a dataset (one row if you want) into something that can be used by
        the neural network (for example)
        """
        if not isinstance(dataset, DataSet):
            raise RuntimeError(f"The \"dataset\" should be an instance of DataSet. "
                               f"We found {type(dataset)}")

        if not isinstance(training, bool):
            raise RuntimeError(f"The \"training\" should be a boolean. "
                               f"We found {type(training)}")

    def save(self, path: Union['str', pathlib.Path]):
        """save the model at a given path"""
        if not self.trained:
            raise RuntimeError("Model is not trained yet, cannot save it")
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        else:
            shutil.rmtree(path)
            path.mkdir(parents=True)

    def restore(self, path: str):
        """
        restore the model from the given path. It is expected to raise an error if it cannot be initialized
        by the data located at `path`
        """
        pass

    def _save_metadata(self, path: str):
        """
        Saves the "metadata" of the model.

        Metadata should be, if possible, serializable to json.

        For example, this is the perfect function to save the meta parameters of a neural network
        """
        pass

    def _load_metadata(self, path: str):
        """load the metada from the given path."""
        pass