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
import copy

from grid2op.Runner import Runner
from grid2op.PlotGrid import PlotMatplot


from lips.dataset.utils import create_env, reproducible_exp
from lips.dataset.utils import get_reference_action, apply_reference_action, get_agent

from lips.dataset import DataSet


class PowerGridDataSet(DataSet):
    """
    PowerGridDataSet is a sub-class of the more generic DataSet class specialized for generation of data related to power networks 

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
                 env_name="l2rpn_case14_sandbox",
                 ):

        DataSet.__init__(self)

        # create the environment
        self.env_name = env_name
        self.env = None

        self._obs = None

        # reference number
        self.reference_number = None
        
        # generator agent
        self.agent_generator_name = None
        

    def create_environment(self, env_name="l2rpn_case14_sandbox", use_lightsim_if_available=True):
        self.env_name = env_name
        if isinstance(self.env_name, str):
            self.env = create_env(env_name=self.env_name, use_lightsim_if_available=use_lightsim_if_available)
        else:
            raise NotImplementedError


    def generate(self,
                 nb_samples=int(1024) * int(128),
                 tag = "test",
                 val_regex=".*99[0-9].*",
                 use_lightsim_if_available=True,
                 reference_number=0,
                 agent_generator_name="random_nn1",
                 agent_parameters={"p": 0.5},
                 skip_gameover=True,
                 env_seed=1234,
                 agent_seed=14,
                 verbose=True):
        """
        the generate function allows to generate Grid2Op `observations` where an Observation is an entity of Grid2Op simulation platform 
        including all the data related to power grid at one time stamp 

        Params
        -----------
            nb_samples: ``int``
                number of samples to generate 
            
            tag: ``str``
                a tag is associated to a dataset to identify for which purpose it has been created

            val_regex: ``str``
                a regular expression used to distinguish between training and test chronics

            use_lightsim_if_available: ``bool``
                a boolean to indicate whether more performant lightsim2grid platform for generation of data
                Note that the lightsim2grid should have been installed to be able to use it via `pip install lightsim2grid`

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

            skip_gameover : ``bool``
                a boolean variable to keep the environment gameover situations or not

            env_seed : ``int``
                the environment seed for reproducibility of sampling

            agent_seed : ``int``
                the agent seed for reproducibility of agent actions

            verbose : ``bool``
                whether or not to show the data generation progression bar
        """
        self.create_environment(env_name=self.env_name, use_lightsim_if_available=use_lightsim_if_available)
        self.nb_samples = nb_samples
        self.reference_number = reference_number
        self.agent_generator_name = agent_generator_name
        self.tag = tag

        # re-initialize the data
        self.data = list()
        

        # select different chronics for training and evaluation parts
        if (tag == "training") or (tag == "validation"):
            self.env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is None)
            self.env.chronics_handler.real_data.reset()
        else:
            self.env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is not None)
            self.env.chronics_handler.real_data.reset()

        self._obs = self.env.reset()

        agent_generator = get_agent(self.env, agent_generator_name, agent_parameters)

        # set seeds for reprodicibility
        reproducible_exp(env=self.env, agent=agent_generator, env_seed=env_seed, agent_seed=agent_seed)

        # generate observations and store their corresponding values
        done = False
        t = 0

        if verbose:
            pbar = tqdm(total=self.nb_samples)

        # TODO : to replace the loop with Runner in future
        # runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=my_agent)
        # runner.run(nb_episode=NB_EPISODE, nb_process=NB_CORE, path_save=PATH_SAVE)

        while t < self.nb_samples:
            # generator_agent which guide the data distribution
            act = agent_generator.act(None, None, None)
            action = apply_reference_action(self.env, act, reference_number, seed=14)
            
            # advance one step in calculation of power network state using Grid2Op Step function of Environment module
            obs, _, done, _ = self.env.step(action)

            # Reset if action was not valid and the gameover situation should be skipped
            if (done) & (skip_gameover):
                obs = self.env.reset()
                done = False
                continue

            self.data.append(obs)

            t += 1
            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        return copy.deepcopy(self.data)

    def read_from_file(self, path):
        """
        Read only the data without the meta data
        """
        self.data = list(np.load(path, allow_pickle=True))

    def write_to_file(self, path):
        """
        save only the generated data on disk without any metadata
        """
        np.save(path, self.data)

    def save(self, path=None):
        """
        save the generated data on disk
        """
        tag_dir = super().save(path)

        # save the data (list of observations)
        np.save(os.path.join(tag_dir, self.tag), self.data)

    def load(self, path=None):
        """
        load the already generated data
        """
        dir_in = super().load(path)
        
        self.create_environment(self.env_name)

        # load all the available numpy files to corresponding dictionaries
        
        self.data = list(np.load(os.path.join(dir_in, self.tag + ".npy"), allow_pickle=True))

        return copy.deepcopy(self.data)

    def _load_metadata(self, path):
        res = super()._load_metadata(path)

        self.env_name = res["env_name"]
        self.tag = res["tag"]
        self.agent_generator_name = res["agent_generator_name"]
        self.reference_number = res["reference_number"]

        self.is_available = True

    def _get_metadata(self):
        res = super()._get_metadata()
        res["env_name"] = self.env_name
        res["tag"] = self.tag
        res["agent_generator_name"] = self.agent_generator_name
        res["reference_number"] = self.reference_number

        return res

    def visualize_network(self):
        """
        This functions shows the network state evolution over time for a given dataset
        """
        if self.env is None: 
            print("the environment is not still initialized, generate some data")
            return
        
        plot_helper = PlotMatplot(self.env.observation_space)

        obs = self.env.reset()
        plot_helper.plot_obs(obs)

    def visualize_network_reference_topology(self, tag):
        """
        visualize the power network's reference topology
        """
        if self.env is None:
            print("the environment is not still initialized")

        plot_helper = PlotMatplot(self.env.observation_space)
        action = get_reference_action(self.env, reference_number=self.reference_number)
        obs = self.env.reset()
        obs, _, _, _ = self.env.step(action)

        plot_helper.plot_obs(obs)
