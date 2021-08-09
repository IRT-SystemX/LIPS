# Copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
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


from lips.generate_data.utils import create_env, reproducible_exp
from lips.generate_data.utils import get_reference_action, apply_reference_action, get_agent


class GenerateData():
    """
    GenerateData class allowing to generate data with different distributions and from 
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
                 experiment_name="Scenario",
                 env_name="l2rpn_case14_sandbox",
                 attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                 attr_tau=("line_status", "topo_vect"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or",
                         "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                 use_lightsim_if_available=True
                 # expe_type="powerlines"
                 ):

        # create the environment
        self.experiment_name = experiment_name
        self.env_name = env_name
        if isinstance(self.env_name, str):
            self.env = create_env(
                env_name=self.env_name, use_lightsim_if_available=use_lightsim_if_available)
        else:
            raise NotImplementedError

        self._obs = self.env.reset()

        self._nb_samples = None

        # generator agent
        self.agent_generator = dict()

        # get the reference topology number
        self.reference_number = dict()
        self.reference_action = dict()

        self.attr_x = attr_x
        self.attr_tau = attr_tau
        self.attr_y = attr_y

        self.variable_size = dict()

        # a dictionary including three datasets (training, validation(optional) and test)
        self.dataset = dict()
        self.dataset_original = dict()

        self.dataset_available = dict()
        self.dataset_available["training"] = False
        self.dataset_available["validation"] = False
        self.dataset_available["test"] = False

        self.tag_list = list()

        self.dataset_size = dict()

        self.preprocessed = dict()
        self.preprocessed["training"] = False
        self.preprocessed["validation"] = False
        self.preprocessed["test"] = False

        self.mean_dict = dict()
        self.std_dict = dict()

    def init(self, tag="training", nb_samples=None):
        self.dataset[tag] = {}
        self.dataset_original[tag] = {}
        for attr_nm in (*self.attr_x, *self.attr_tau, *self.attr_y):
            tmp = len(getattr(self._obs, attr_nm))
            self.variable_size[attr_nm] = tmp
            self.dataset[tag][attr_nm] = np.full(
                (nb_samples, tmp), fill_value=np.NaN, dtype=np.float)

    def generate(self,
                 nb_samples=int(1024) * int(128),
                 tag="training",
                 val_regex=".*99[0-9].*",
                 reference_number=0,
                 agent_generator_name="random_nn1",
                 agent_parameters={"p": 0.5},
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

            skip_gameover : ``bool``
                a boolean variable to keep the environment gameover situations or not

            env_seed : ``int``
                the environment seed for reproducibility of sampling

            agent_seed : ``int``
                the agent seed for reproducibility of agent actions

            verbose : ``bool``
                whether or not to show the data generation progression bar
        """
        self._nb_samples = nb_samples
        self.init(tag, self._nb_samples)
        # if the generated datasets are preprocessed
        self.preprocessed[tag] = False
        self.dataset_size[tag] = nb_samples  # length of each dataset
        self.dataset_available[tag] = True
        self.tag_list.append(tag)
        self.reference_number[tag] = reference_number

        # select different chronics for training and evaluation parts
        if (tag == "training") | (tag == "validation"):
            self.env.chronics_handler.set_filter(
                lambda path: re.match(val_regex, path) is None)
            self.env.chronics_handler.real_data.reset()
        else:
            self.env.chronics_handler.set_filter(
                lambda path: re.match(val_regex, path) is not None)
            self.env.chronics_handler.real_data.reset()

        self._obs = self.env.reset()

        agent_generator = get_agent(
            self.env, agent_generator_name, agent_parameters)
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

            for attr_nm in (*self.attr_x, *self.attr_tau, *self.attr_y):
                self.dataset[tag][attr_nm][t, :] = getattr(obs, attr_nm)

            t += 1
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
        if tag == "training":
            for attr_nm in (*self.attr_x, *self.attr_tau, *self.attr_y):
                self.mean_dict[attr_nm], self.std_dict[attr_nm] = self.get_mean_std(
                    attr_nm)

        self.dataset_original[tag] = copy.deepcopy(self.dataset[tag])

    def get_mean_std(self, attr_nm):
        obss = self.dataset["training"]
        mean_tmp = np.mean(obss.get(attr_nm), axis=0).astype(np.float32)
        std_tmp = np.std(obss.get(attr_nm), axis=0).astype(np.float32) + 1e-1

        if attr_nm in ["prod_p"]:
            # mult_tmp = np.array([max((pmax - pmin), 1.) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)],
            #                     dtype=np.float32)
            # default values are good enough
            pass
        elif attr_nm in ["prod_q"]:
            # default values are good enough
            pass
        elif attr_nm in ["load_p", "load_q"]:
            # default values are good enough
            pass
        elif attr_nm in ["load_v", "prod_v"]:
            # default values are good enough
            # stds are almost 0 for loads, this leads to instability
            std_tmp = 1.0
        elif attr_nm in ["v_or", "v_ex"]:
            # default values are good enough
            # because i multiply by the line status, so i don't want any bias
            mean_tmp = np.float32(0.)
            std_tmp = np.mean(obss.get(attr_nm), axis=0).astype(np.float32)
        elif attr_nm in ["p_or", "p_ex", "q_or", "q_ex"]:
            # because i multiply by the line status, so i don't want any bias
            mean_tmp = np.float32(0.)
            std_tmp = np.array([max(np.abs(val), 1.0)
                               for val in obss.get(attr_nm)[0]], dtype=np.float32)
        elif attr_nm in ["a_or", "a_ex"]:
            # because i multiply by the line status, so i don't want any bias
            mean_tmp = np.float32(0.0)
            # which is equal to the thermal limit
            std_tmp = np.abs(obss.get("a_or")[0] / (self._obs.rho + 1e-2))
            std_tmp[std_tmp <= 1.0] = 1.0
        elif attr_nm == "line_status":
            # encode back to 0: connected, 1: disconnected
            mean_tmp = np.float32(1.0)
            std_tmp = np.float32(-1.0)
        # START
        # Added by Milad
        elif attr_nm == "topo_vect":
            # encode back to 0 = on bus 1, 1 = on bus 2
            mean_tmp = np.float32(1.)
            std_tmp = np.float32(1.)
        elif attr_nm == "connectivity_matrix":
            # in connectivity matrix 0: no connection, 1: connected to bus1 2: connected to bus2
            # Not modify this encoding
            mean_tmp = np.float32(0.)
            std_tmp = np.float32(1.)
        # END

        return mean_tmp, std_tmp

    def preprocess_data(self):
        # if not(self.preprocessed):
        #    for attr_nm in ()
        for tag in self.tag_list:
            if not(self.preprocessed[tag]):
                for attr_nm in (*self.attr_x, *self.attr_y):
                    self.dataset[tag][attr_nm] = (
                        self.dataset[tag][attr_nm] - self.mean_dict[attr_nm]) / self.std_dict[attr_nm]
                for attr_nm in self.attr_tau:
                    self.dataset[tag][attr_nm] = abs(
                        (self.dataset[tag][attr_nm] - self.mean_dict[attr_nm]) / self.std_dict[attr_nm])
                self.preprocessed[tag] = True

    def add_tag(self, tag=None):
        """
        to add more datasets, like super test set
        """
        self.preprocessed[tag] = False
        self.dataset_available[tag] = False
        self.tag_list.append(tag)

    def save(self, path=None):
        """
        save the generated data on disk
        """
        dir_out = os.path.join(path, self.experiment_name)
        if not os.path.exists(dir_out):
            try:
                os.mkdir(dir_out)
            except OSError as err:
                raise err
                #print("OS error {0}".format(err))

        dir_out = os.path.join(dir_out, "Data")
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)

        self._save_metadata(dir_out)

        # save the data for each existing tag
        for key_, val_ in self.dataset_original.items():
            tag_dir = os.path.join(dir_out, key_)
            if not os.path.exists(tag_dir):
                os.mkdir(tag_dir)
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

        # load all the available numpy files to corresponding dictionaries
        for tag in self.tag_list:
            self.dataset[tag] = {}
            self.dataset_original[tag] = {}
            self.dataset_available[tag] = True
            tag_dir = os.path.join(dir_in, tag)
            if not os.path.exists(tag_dir):
                raise RuntimeError(
                    f"the directory for dataset {tag} not found")
            for key_ in self.variable_size.keys():
                self.dataset[tag][key_] = np.load(
                    os.path.join(tag_dir, f"{key_}_real.npy"))
                self.dataset_original[tag][key_] = np.load(
                    os.path.join(tag_dir, f"{key_}_real.npy"))

    def _save_metadata(self, path):
        res = self._get_metadata()
        json_nm = "metadata_GenerateData.json"
        with open(os.path.join(path, json_nm), "w", encoding="utf-8") as f:
            json.dump(obj=res, fp=f)

    def _load_metadata(self, path):
        json_nm = "metadata_GenerateData.json"
        with open(os.path.join(path, json_nm), "r", encoding="utf-8") as f:
            res = json.load(f)

        self.env_name = res["env_name"]
        self.attr_x = tuple(res["attr_x"])
        self.attr_y = tuple(res["attr_y"])
        self.attr_tau = tuple(res["attr_tau"])
        self.variable_size = res["variable_size"]
        #self.mean_dict = res["normalization"]["mean_dict"]
        #self.std_dict = res["normalization"]["std_dict"]
        for nm_, val_ in zip(res["variable_size"].keys(), res["normalization"]["mean_dict"]):
            self.mean_dict[nm_] = val_
        for nm_, val_ in zip(res["variable_size"].keys(), res["normalization"]["std_dict"]):
            self.std_dict[nm_] = val_
        self.tag_list = res["tag_list"]
        self.dataset_size = res["dataset_size"]
        self.agent_generator = res["agent_generator"]
        self.reference_number = res["reference_number"]

        for tag in self.tag_list:
            self.preprocessed[tag] = False

    def _get_metadata(self):
        res = dict()
        res["env_name"] = self.env_name
        res["attr_x"] = self.attr_x
        res["attr_y"] = self.attr_y
        res["attr_tau"] = self.attr_tau
        res["variable_size"] = self.variable_size
        res["normalization"] = {}
        res["normalization"]["mean_dict"] = []
        for el in self.mean_dict.values():
            self._save_dict(res["normalization"]["mean_dict"], el)
        res["normalization"]["std_dict"] = []
        for el in self.std_dict.values():
            self._save_dict(res["normalization"]["std_dict"], el)
        res["tag_list"] = self.tag_list
        res["dataset_size"] = self.dataset_size
        res["agent_generator"] = self.agent_generator
        res["reference_number"] = self.reference_number

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
        plot_helper = PlotMatplot(self.env.observation_space)

        obs = self.env.reset()
        fig = plot_helper.plot_obs(obs)
        fig.show()

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
        fig.show()
