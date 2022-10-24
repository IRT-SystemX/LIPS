# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""
The leap_net model
"""
import os
import pathlib
from typing import Union
import json
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras
    import tensorflow as tf

try:
    from leap_net.proxy import ProxyLeapNet
except ImportError as err:
    raise RuntimeError("You need to install the leap_net package to use this class") from err

from ..tensorflow_simulator import TensorflowSimulator
from ...logger import CustomLogger
from ...config import ConfigManager
from ...dataset import DataSet
from ...dataset.scaler import Scaler

class LeapNet(TensorflowSimulator):
    """LeapNet architecture

    This class wraps the `ProxyLeapNet` of the `leap_net` module (see https://github.com/BDonnot/leap_net) to be
    used as an `AugmentedSimulator`.

    This is just a wrapper, for modification of said "leap_net" or its description, please visit the original
    github repository.


    Parameters
    ----------
    sim_config_path : ``str``
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
        _description_
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

        super().__init__(name=name, log_path=log_path, **kwargs)
        if not os.path.exists(sim_config_path):
            raise RuntimeError("Configuration path for the simulator not found!")
        if not str(sim_config_path).endswith(".ini"):
            raise RuntimeError("The configuration file should have `.ini` extension!")
        sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        self.sim_config = ConfigManager(section_name=sim_config_name, path=sim_config_path)
        self.bench_config = ConfigManager(section_name=bench_config_name, path=bench_config_path)
        self.name = name if name is not None else self.sim_config.get_option("name")
        self.name = name + '_' + sim_config_name
        # scaler
        self.scaler = scaler() if scaler else None
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # Define layer to be used for the model
        self.layers = {"linear": keras.layers.Dense}
        self.layer = self.layers[self.sim_config.get_option("layer")]
        # get tau attributes
        self.attr_tau = kwargs['attr_tau'] if "attr_tau" in kwargs else self.bench_config.get_option("attr_tau")
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)

        # optimizer
        # optimizer
        if "optimizer" in kwargs:
            if not isinstance(kwargs["optimizer"], keras.optimizers.Optimizer):
                raise RuntimeError("If an optimizer is provided, it should be a type tensorflow.keras.optimizers")
            self._optimizer = kwargs["optimizer"](self.params["optimizer"]["params"])
        else:
            self._optimizer = keras.optimizers.Adam(learning_rate=self.params["optimizer"]["params"]["lr"])
        # Init the leap net model
        self._leap_net_model: Union[ProxyLeapNet, None] = None
        self._model: Union[keras.Model, None] = None
        self._create_proxy()

    def build_model(self):
        """Build the model"""
        self._leap_net_model.build_model()
        self._model = self._leap_net_model._model

    def _create_proxy(self):
        """part of the "wrapper" part, this function will initialize the leap net model"""
        self._leap_net_model = ProxyLeapNet(
            name=f"{self.name}_model",
            train_batch_size=self.params["train_batch_size"],
            attr_x=self.bench_config.get_option("attr_x"),
            attr_y=self.bench_config.get_option("attr_y"),
            attr_tau=self.attr_tau,#self.bench_config.get_option("attr_tau"),
            sizes_enc=self.params["sizes_enc"],
            sizes_main=self.params["sizes_main"],
            sizes_out=self.params["sizes_out"],
            lr=self.params["optimizer"]["params"]["lr"],
            layer=self.layer,  # TODO (for save and restore)
            topo_vect_to_tau=self.params["topo_vect_to_tau"],  # see code for now
            # kwargs_tau: optional kwargs depending on the method chosen for building tau from the observation
            kwargs_tau=self.params["kwargs_tau"],
            mult_by_zero_lines_pred=self.params["mult_by_zero_lines_pred"],
            # TODO there for AS
            scale_main_layer=self.params["scale_main_layer"],  # increase the size of the main layer
            scale_input_dec_layer=self.params["scale_input_dec_layer"],  # scale the input of the decoder
            scale_input_enc_layer=self.params["scale_input_enc_layer"],  # scale the input of the encoder
            layer_act=self.params["activation"]

        )

    def process_dataset(self, dataset: DataSet, training: bool=False) -> tuple:
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Warning
        -------
        It works only with PowerGridScaler scaler for the moment

        Parameters
        ----------
        dataset : DataSet
            _description_
        Scaler : bool, optional
            _description_, by default True
        training : bool, optional
            _description_, by default False

        Returns
        -------
        tuple
            the normalized dataset with features and labels
        """
        if training:
            obss = self._make_fake_obs(dataset)
            self._leap_net_model.init(obss)
            if self.scaler is not None:
                (extract_x, extract_tau), extract_y = self.scaler.fit_transform(dataset)
            else:
                (extract_x, extract_tau), extract_y = dataset.extract_data(concat=False)

        else:
            if self.scaler is not None:
                (extract_x, extract_tau), extract_y = self.scaler.transform(dataset)
            else:
                (extract_x, extract_tau), extract_y = dataset.extract_data(concat=False)


        if self._leap_net_model.kwargs_tau is not None:
            is_given_topo_list = (len(self._leap_net_model.kwargs_tau) >= 1)
            if (is_given_topo_list):
                extract_tau = self._transform_tau_given_list(extract_tau)
            else:
                extract_tau = self._transform_tau(dataset, extract_tau)

        #if len(self._leap_net_model.attr_tau) > 1 :
        #   extract_tau = np.concatenate((extract_tau[0], extract_tau[1]), axis=1)


        return (extract_x, extract_tau), extract_y

    def _post_process(self, dataset, predictions):
        """Do some post processing on the predictions

        Parameters
        ----------
        predictions : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        return predictions

    def _transform_tau(self, dataset, tau):
        """Transform only the tau vector with respect to LeapNet encodings
        """
        obss = self._make_fake_obs(dataset)
        tau.pop()
        tau.append(np.array([self._leap_net_model.topo_vect_handler(obs)
                            for obs in obss],
                            dtype=np.float32))
        return tau

    def _transform_tau_given_list(self, tau,with_tf=True):
        """Transform only the tau vector with respect to LeapNet encodings given a list of predefined topological actions
                Parameters
        ----------
        tau : list of raw topology representations (line_status, topo_vect)

        with_tf : transformation using tensorflow or numpy operations

        Returns
        -------
        tau
            list of encoded topology representations (line_status, topo_vect_encoded)
        """
        ##############
        #WARNING: TO DO
        # if we find two topology matches at a same substation, the current code attribute one bit for each
        # But only one should be choosen in the end (we are not in a quantum state, or it does not make sense to combine topologies at a same substation in the encoding here
        #This can happen when there are several lines disconnected at a substation on which we changed the topology, probably in benchmark 3, but probably not in benchmark 1 and 2

        subs_index=self._leap_net_model.subs_index

        list_topos=[]
        sub_length=[]
        for topo_action in self._leap_net_model.kwargs_tau:
            topo_vect = np.zeros(tau[1].shape[1], dtype=np.int32)
            sub_id=topo_action[0]
            sub_topo=np.array(topo_action[1])
            sub_index=subs_index[sub_id][0]
            n_elements=len(sub_topo)
            topo_vect[sub_index:sub_index+n_elements]=sub_topo
            list_topos.append(topo_vect)
            sub_length.append(n_elements)

        list_topos=np.array(list_topos)

        #we are here looking for the number of matches for every element of a substation topology in the predefined list for a new topo_vect observation
        #if the count is equal to the number of element, then the predefined topology is present in topo_vect observation
        #in that case, the binary encoding of that predefined topology is equal to 1, otherwise 0

        import time
        start = time.time()
        if with_tf:
            #count the number of disconnected lines for each substation of topologies in the prefdefined list.
            #These lines could have been connected to either bus_bar1 or bus_bar2, we consider it as a match for that element
            line_disconnected_sub = tf.linalg.matmul((list_topos >0).astype(np.int32),(np.transpose(tau[1]) < 0).astype(np.int32))

            #we look at the number of elements on bus_bar1 that match, same for the number of elements on bus_bar2
            match_tensor_bus_bar1=tf.linalg.matmul((list_topos==1).astype(np.int32),(np.transpose(tau[1])==1).astype(np.int32))
            match_tensor_bus_bar2 =tf.linalg.matmul((list_topos==2).astype(np.int32), (np.transpose(tau[1])==2).astype(np.int32))

            #the number of matches is equal to the sum of those 3 category of matches
            match_tensor_adjusted=match_tensor_bus_bar1+match_tensor_bus_bar2+line_disconnected_sub

            #we see if all elements match by dividing by the number of elements. If this proportion is equal to one, we found a topology match
            normalised_tensor = match_tensor_adjusted / tf.reshape(np.array(sub_length).astype(np.int32), (-1, 1))

        else:#with_numpy

            line_disconnected_sub = np.matmul((list_topos >0),1*(np.transpose(tau[1]) < 0))

            match_tensor_bus_bar1=np.matmul((list_topos==1),1*(np.transpose(tau[1])==1))
            match_tensor_bus_bar2 =np.matmul((list_topos==2), 1*(np.transpose(tau[1])==2))

            match_tensor_adjusted=match_tensor_bus_bar1+match_tensor_bus_bar2+line_disconnected_sub

            normalised_tensor = match_tensor_adjusted / np.array(sub_length).reshape((-1, 1))

        boolean_match_tensor = np.array(normalised_tensor == 1.0).astype(np.int8)

        duration_matches = time.time() -start

        #############"
        ## do correction if multiple topologies of a same substation have a match on a given state
        # as it does not make sense to combine topologies at a same substation
        start = time.time()
        boolean_match_tensor=self._unicity_tensor_encoding(boolean_match_tensor)

        duration_correction = time.time() - start
        if(duration_correction>duration_matches):
            print("warning, correction time if longer that matches time: maybe something to better optimize there")
        tau[1] = np.transpose(boolean_match_tensor)

        return tau

    def _unicity_tensor_encoding(self, tensor):
        """
        do correction if multiple topologies of a same substation have a match on a given state
        as it does not make sense to combine topologies at a same substation
        """
        sub_encoding_pos = np.array([topo_action[0] for topo_action in self._leap_net_model.kwargs_tau])

        # in case of multiple matches of topology for a given substation, encode only one of those topologies as an active bit, not several
        def per_col(a):  # to only have one zero per row
            idx = a.argmax(0)
            out = np.zeros_like(a)
            r = np.arange(a.shape[1])
            out[idx, r] = a[idx, r]
            return out

        for sub in set(sub_encoding_pos):
            indices = np.where(sub_encoding_pos == sub)[0]
            if (len(indices) >= 2):
                tensor[indices,:]=per_col(tensor[indices,:])

        return tensor


    def _make_fake_obs(self, dataset: DataSet):
        """
        the underlying _leap_net_model requires some 'class' structure to work properly. This convert the
        numpy dataset into these structures.

        Definitely not the most efficient way to process a numpy array...
        """
        all_data = dataset.data
        class FakeObs(object):
            pass

        if "topo_vect" in all_data:
            setattr(FakeObs, "dim_topo", all_data["topo_vect"].shape[1])

        setattr(FakeObs, "n_sub", dataset.env_data["n_sub"])
        setattr(FakeObs, "sub_info", np.array(dataset.env_data["sub_info"]))

        nb_row = all_data[next(iter(all_data.keys()))].shape[0]
        obss = [FakeObs() for k in range(nb_row)]
        for attr_nm in all_data.keys():
            arr_ = all_data[attr_nm]
            for ind in range(nb_row):
                setattr(obss[ind], attr_nm, arr_[ind, :])
        return obss

    def write_history(self, history):
        """write the history of the training

        It should have its own history writer as it includes multiple variables

        Parameters
        ----------
        history_callback : keras.callbacks.History
            the history of the training
        """
        hist = history.history

        self.train_losses = hist["loss"]
        self.val_losses = hist["val_loss"]
        metrics=self.params["metrics"]

        for metric in metrics:#["mae"]:
            tmp_train = []
            tmp_val = []
            for key in hist.keys():
                if (metric in key) and ("val" in key):
                    tmp_val.append(hist[key])
                if (metric in key) and ("val" not in key):
                    tmp_train.append(hist[key])
            self.val_metrics[metric] = np.vstack(tmp_val).mean(axis=0)
            self.train_metrics[metric] = np.vstack(tmp_train).mean(axis=0)


    def _save_model(self, path: str):
        if self._leap_net_model is not None:
            # save the weights
            self._leap_net_model.save_data(path, ext=".h5")

    def _load_model(self, path: str):
        self._leap_net_model.build_model()
        self._leap_net_model.load_data(path, ext=".h5")
        self._model = self._leap_net_model._model

    def _save_metadata(self, path: str):
        super()._save_metadata(path)

        res_json = self._leap_net_model.get_metadata()
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True)

        if self.scaler is not None:
            self.scaler.save(path)

    def _load_metadata(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        super()._load_metadata(path)
        with open(os.path.join(path, f"metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self._leap_net_model.load_metadata(res_json)

        if self.scaler is not None:
            self.scaler.load(path)
