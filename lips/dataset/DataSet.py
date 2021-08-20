# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import re
import numpy as np
from tqdm import tqdm
import json
import copy
from collections.abc import Iterable

import grid2op
from grid2op.PlotGrid import PlotMatplot


from lips.dataset.utils import create_env, reproducible_exp
from lips.dataset.utils import get_reference_action, apply_reference_action, get_agent


class DataSet():
    """
    DataSet class allowing to generate data with different distributions and from 
    various topolgy references

    Attributes
    ----------
    env_name : ```str``
        the grid2op environment name to be used for the generation of data
        adjust the power network parameters

    nb_samples : ``ìnt``
        number of observations to be samples with the selected configuration

    tag : ```str``
        indicating the phase `training` `validation` `test`

    expe_type : ``str``
        experiment type focused whether on `powerlines` or `topology`        
    """

    def __init__(self,
                 experiment_name="Scenario"
                 # expe_type="powerlines"
                 ):

        # create the environment
        self.experiment_name = experiment_name
        self.env_name = None
        self.env = None

        self._obs = None
        self.init_obs = None # observation list used for initialization of augmented simulator

        self._nb_samples = None

        self.attr_names = None

        # generator agent
        self.agent_generator = dict()

        # get the reference topology number
        self.reference_number = dict()
        self.reference_action = dict()

        self.variable_size = dict()

        # a dictionary including three datasets (training, validation(optional) and test)
        self.dataset = dict()

        self.dataset_available = dict()

        self.tag_list = list()

        self.dataset_size = dict()

        self.data_path = None

    def init(self, tag="training", nb_samples=None):
        self.dataset[tag] = {}
        
        for attr_nm in self.attr_names:
            tmp = len(getattr(self._obs, attr_nm))
            self.variable_size[attr_nm] = tmp
            self.dataset[tag][attr_nm] = np.full(
                (nb_samples, tmp), fill_value=np.NaN, dtype=np.float)

    def create_environment(self, env_name="l2rpn_case14_sandbox", use_lightsim_if_available=True):
        self.env_name = env_name
        if isinstance(self.env_name, str):
            self.env = create_env(env_name=self.env_name, use_lightsim_if_available=use_lightsim_if_available)
        else:
            raise NotImplementedError


    def generate(self,
                 env_name="l2rpn_case14_sandbox",
                 nb_samples=int(1024) * int(128),
                 tag="training",
                 val_regex=".*99[0-9].*",
                 attr_names=("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                             "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", 
                             "v_or", "v_ex"),
                 use_lightsim_if_available=True,
                 reference_number=0,
                 agent_generator_name="random_nn1",
                 agent_parameters={"p": 0.5},
                 _nb_obs_init=512,
                 skip_gameover=True,
                 env_seed=1234,
                 agent_seed=14,
                 verbose=True):
        """
        main function to generate samples and store the results into numpy arrays

        Attributes
        -----------
            reference_number : ``ìnt``
                the experiment number corresponding to following experiments
                0 : reference topology is a connected network
                1 : reference topology is a power network with a disconnection at line 3
                2 : reference topology is a power network with a change in buses at node 3
                3 : reference topology is a power network with a change in buses at node 5
                4 : combination of 2 and 3

            agent_generator_name : ``string``  
                referring to grid2op.Agent ``BaseAgent`` class which is used for the generation of data

            agent_parameters : ``dict``
                agent parameters to be used for generation of reference topology
                the dict keys are p for the probability of an action or a list indicating the action zone

            _nb_init_obss : ``int``
                a few Grid2op observations to be used for AugmentedSimulator initialization

            skip_gameover : ``bool``
                a boolean variable to keep the environment gameover situations or not

            env_seed : ``int``
                the environment seed for reproducibility of sampling

            agent_seed : ``int``
                the agent seed for reproducibility of agent actions

            verbose : ``bool``
                whether or not to show the data generation progression bar
        """
        self.create_environment(env_name, use_lightsim_if_available)

        self.attr_names = attr_names

        self._obs = self.env.reset()
        self._nb_samples = nb_samples
        self.init(tag, self._nb_samples)
        self.dataset_size[tag] = nb_samples  # length of each dataset
        self.dataset_available[tag] = True
        self.tag_list.append(tag)
        self.reference_number[tag] = reference_number

        # select different chronics for training and evaluation parts
        if (tag == "training"):
            self.env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is None)
            self.env.chronics_handler.real_data.reset()
            self.init_obs = []
        elif tag == "validation":
            self.env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is None)
            self.env.chronics_handler.real_data.reset()
        else:
            self.env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is not None)
            self.env.chronics_handler.real_data.reset()

        self._obs = self.env.reset()

        agent_generator = get_agent(self.env, agent_generator_name, agent_parameters)
        self.agent_generator[tag] = agent_generator_name

        # get the reference action for reference topology
        #reference_action = get_reference_action(self.env, reference_number)
        #self.reference_action[tag] = reference_action

        # set seeds for reprodicibility
        reproducible_exp(env=self.env, agent=agent_generator,
                         env_seed=env_seed, agent_seed=agent_seed)

        # generate observations and store their corresponding values
        obs = self.env.reset()
        done = False
        t = 0
        nb_obs = 0 # counter to keep _nb_obs_init observations for augmented simulator initialization

        if verbose:
            pbar = tqdm(total=self._nb_samples)

        while t < self._nb_samples:
            # generator_agent which guide the data distribution
            act = agent_generator.act(None, None, None)
            action = apply_reference_action(
                self.env, act, reference_number, seed=14)
            #action = act + reference_action
            # sum of reference and generator agent's actions
            obs, _, done, _ = self.env.step(action)

            # Reset if action was not valid and the gameover situation should be skipped
            if (done) & (skip_gameover):
                obs = self.env.reset()
                done = False
                continue

            for attr_nm in self.attr_names:
                self.dataset[tag][attr_nm][t, :] = getattr(obs, attr_nm)

            if tag == "training" and nb_obs <= _nb_obs_init:
                self.init_obs.append(obs)
                nb_obs += 1


            t += 1
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()

    def add_tag(self, tag=None):
        """
        to add more datasets, like super test set
        """
        self.dataset_available[tag] = False
        self.tag_list.append(tag)

    def save(self, path=None):
        """
        save the generated data on disk
        """
        dir_out = os.path.join(path, self.experiment_name)
        if not os.path.exists(dir_out):
            try:
                os.makedirs(dir_out)
            except OSError as err:
                raise err
                #print("OS error {0}".format(err))

        dir_out = os.path.join(dir_out, "Data")
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        self.data_path = dir_out

        self._save_metadata(dir_out)

        if self.init_obs:
            np.save(os.path.join(dir_out,"init_obs.npy"), self.init_obs)

        # save the data for each existing tag
        for key_, val_ in self.dataset.items():
            tag_dir = os.path.join(dir_out, key_)
            if not os.path.exists(tag_dir):
                os.makedirs(tag_dir)
            for key__, val__ in val_.items():
                np.save(os.path.join(tag_dir, f"{key__}_real.npy"), val__)

    def load(self, path=None):
        """
        load the already generated data
        """
        dir_in = os.path.join(path, self.experiment_name, "Data")
        dir_in = os.path.abspath(dir_in)
        if not os.path.exists(dir_in):
            raise RuntimeError("The indicated path does not exists")

        self._load_metadata(dir_in)
        
        self.create_environment(self.env_name)

        self.init_obs = list(np.load(os.path.join(dir_in,"init_obs.npy"), allow_pickle=True))

        # load all the available numpy files to corresponding dictionaries
        for tag in self.tag_list:
            self.dataset[tag] = {}
            self.dataset_available[tag] = True
            tag_dir = os.path.join(dir_in, tag)
            if not os.path.exists(tag_dir):
                raise RuntimeError(f"the directory for dataset {tag} not found")
            for key_ in self.variable_size.keys():
                self.dataset[tag][key_] = np.load(os.path.join(tag_dir, f"{key_}_real.npy"))

    def _save_metadata(self, path):
        res = self._get_metadata()
        json_nm = "metadata_DataSet.json"
        with open(os.path.join(path, json_nm), "w", encoding="utf-8") as f:
            json.dump(obj=res, fp=f)

    def _load_metadata(self, path):
        json_nm = "metadata_DataSet.json"
        with open(os.path.join(path, json_nm), "r", encoding="utf-8") as f:
            res = json.load(f)

        self.env_name = res["env_name"]
        self.attr_names = tuple(res["attr_names"])
        self.variable_size = res["variable_size"]
        self.tag_list = res["tag_list"]
        self.dataset_size = res["dataset_size"]
        self.agent_generator = res["agent_generator"]
        self.reference_number = res["reference_number"]
        self.data_path = res["data_path"]

    def _get_metadata(self):
        res = dict()
        res["env_name"] = self.env_name
        res["attr_names"] = self.attr_names
        res["variable_size"] = self.variable_size
        res["tag_list"] = self.tag_list
        res["dataset_size"] = self.dataset_size
        res["agent_generator"] = self.agent_generator
        res["reference_number"] = self.reference_number
        res["data_path"] = self.data_path

        return res

    def _save_dict(self, li, val):
        """
        save the metadata of generator into a valid representation of `self`. It's a utility function to convert
        into "json serializable object" the numpy data.

        It is a helper to convert data in float format from either a list or a single numpy floating point.

        Parameters
        ----------
        li
        val

        """
        if isinstance(val, Iterable):
            li.append([float(el) for el in val])
        else:
            li.append(float(val))

    def visualize_network(self):
        """
        This functions shows the network state evolution over time for a given dataset
        """
        if self.env is None: 
            print("Generate some data from an environment to visualize the network")
            return
        
        plot_helper = PlotMatplot(self.env.observation_space)

        obs = self.env.reset()
        fig = plot_helper.plot_obs(obs)
        #fig.show()

    def visualize_network_reference_topology(self, tag):
        """
        visualize the power network's reference topology
        """
        if not self.dataset_available["training"]:
            print(
                "the dataset with tag {} should be generated before visualization of reference topology".format(tag))
            return

        tmp_env = self.env.copy()
        plot_helper = PlotMatplot(tmp_env.observation_space)
        action = get_reference_action(
            tmp_env, reference_number=self.reference_number[tag])
        obs = tmp_env.reset()
        obs, _, _, _ = tmp_env.step(action)

        # obs.line_status = np.asarray(
        #    self.dataset_original[tag]["line_status"][0], dtype=np.bool)
        # obs.topo_vect = np.asarray(
        #    self.dataset_original[tag]["topo_vect"][0], dtype=np.int)
        fig = plot_helper.plot_obs(obs)
        #fig.show()
