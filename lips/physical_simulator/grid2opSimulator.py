"""
Usage:
    Grid2opSimulator implementing powergrid physical simulator
Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""

import re
from typing import Union
import numpy as np

import grid2op
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.PlotGrid import PlotMatplot

from . import PhysicalSimulator

class Grid2opSimulator(PhysicalSimulator):
    """
    This simulator uses the `grid2op` package to implement a physical simulator.
    It accepts both grid2op BaseAction and grid2op BaseAgent to modify the internal state of the simulator

    Parameters
    ----------
    env_kwargs : dict
        Grid2op.Environment parameters
    initial_chronics_id : Union[int, None], optional
        the initial episode identifier, by default None
    chronics_selected_regex : str, optional
        the chronics to keep for this simulator, by default None
    """
    def __init__(self,
                 env_kwargs: dict,
                 initial_chronics_id: Union[int, None] = None,
                 chronics_selected_regex: str = None
                 ):
        PhysicalSimulator.__init__(self, actor_types=(BaseAction, BaseAgent))
        self._simulator = get_env(env_kwargs)
        if chronics_selected_regex is not None:
            # special case of the grid2Op environment: data are read from chronics that should be part of the dataset
            # here i keep only certain chronics for the training, and the other for the test
            chronics_selected_regex = re.compile(chronics_selected_regex)
            self._simulator.chronics_handler.set_filter(lambda path:
                                                        re.match(chronics_selected_regex, path) is not None)
            self._simulator.chronics_handler.real_data.reset()

        if initial_chronics_id is not None:
            self._simulator.set_id(initial_chronics_id)

        self._chronic_id = 0
        self.chronics_id = list()
        self.chronics_name = list()
        self.chronics_name_unique = list()
        self.time_stamps = list()
        self._chronic_counter = -1
        self._max_episode_duration = None
        self._visited_ts = list()

        self._obs = None
        self._reward = None
        self._info = None
        self._max_episode_duration = self._simulator.chronics_handler.max_episode_duration()
        self._reset_simulator()
        self._plot_helper = None


        self._nb_divergence = 0  # number of failures of modify_state
        self._nb_output = 0  # number of time get_state is called

        self._time_powerflow = 0
        self.comp_time = 0
        self._timer_solver = 0
        self._timer_preproc = 0
        self._timer_postproc = 0

        try:
            from lightsim2grid import LightSimBackend
            self.lightsim_backend = LightSimBackend
        except ImportError as exc_:
            print(exc_)
            self.lightsim_backend = None

    def seed(self, seed: int):
        """
        It seeds the environment, for reproducible experiments.

        Parameters
        ----------
        seed: int
            An integer representing the seed.
        """
        seeds = self._simulator.seed(seed)
        self._reset_simulator()
        return seeds

    def sample_chronics(self):
        """Sample a chronic among all the chronics available
        """
        self._chronic_id = self._simulator.chronics_handler.real_data.sample_next_chronics()[0]
        self._simulator.set_id(self._chronic_id)
        self._obs = self._simulator.reset()
        self.time_stamps.append([])
        self.chronics_name_unique.append(self._simulator.chronics_handler.get_name())
        self._max_episode_duration = self._simulator.chronics_handler.max_episode_duration()
        self._visited_ts = []
        self._chronic_counter += 1
        self._reset_simulator()

    def get_state(self) -> tuple:
        """
        The state of the powergrid is, for this class, represented by a tuple:
        - grid2op observation.
        - extra information (can be empty)

        Returns
        -------
        tuple
            observation and extra information
        """
        self._nb_output += 1
        return self._obs, self._info

    def _get_time_powerflow(self) -> tuple:
        """getter for powerflow execution time

        Returns
        -------
        tuple
            powerflow execution time & computation time
        """
        return self._time_powerflow, self.comp_time

    def modify_state(self, actor: BaseAgent):
        """
        It calls `env.step` until a convergence is obtained.

        Parameters
        ----------
        actor : BaseAgent
            a grid2op agent which performs a specific action on the grid

        Raises
        ------
        RuntimeError
            _description_
        """
        super().modify_state(actor)  # perform the check that the actor is legit
        done = True
        while done:
            # simulate data (resimulate in case of divergence of the simulator)
            act = actor.act(self._obs, self._reward, done)
            _beg_time_pf = self._simulator._time_powerflow
            _beg_time_cp = self._simulator.backend.comp_time
            # LightSimBackend specific timers
            if isinstance(self._simulator.backend, self.lightsim_backend):
                beg_preproc_time = self._simulator.backend._timer_preproc
                beg_solver_time = self._simulator.backend._timer_solver
                beg_postproc_time = self._simulator.backend._timer_postproc

            self._obs, self._reward, done, self._info = self._simulator.step(act)

            if isinstance(self._simulator.backend, self.lightsim_backend):
                diff_timer_preproc = self._simulator.backend._timer_preproc - beg_preproc_time
                diff_timer_solver = self._simulator.backend._timer_solver - beg_solver_time
                diff_timer_postproc = self._simulator.backend._timer_postproc - beg_postproc_time

            _diff_time_pf = self._simulator._time_powerflow - _beg_time_pf
            _diff_time_cp = self._simulator.backend.comp_time - _beg_time_cp

            # verify for isolated injections
            check = self.__any_isolated_injections(self._obs)
            if check:
                done=True

            check = self.__any_nan_values(self._obs)
            if check:
                done=True

            check = self.__any_isolated_nodes(self._simulator)
            if check:
                done = True

            self._time_powerflow += _diff_time_pf
            self.comp_time += _diff_time_cp
            if isinstance(self._simulator.backend, self.lightsim_backend):
                self._timer_preproc += diff_timer_preproc
                self._timer_solver += diff_timer_solver
                self._timer_postproc += diff_timer_postproc

            if self._info["is_illegal"]:
                self._time_powerflow -= _diff_time_pf
                self.comp_time -= _diff_time_cp
                if isinstance(self._simulator.backend, self.lightsim_backend):
                    self._timer_preproc -= diff_timer_preproc
                    self._timer_solver -= diff_timer_solver
                    self._timer_postproc -= diff_timer_postproc
                raise RuntimeError("Your `actor` should not take illegal action. Please modify the environment "
                                   "or your actor.")

            if done:
                self._nb_divergence += 1
                self._reset_simulator()
                self._time_powerflow -= _diff_time_pf
                self.comp_time -= _diff_time_cp
                if isinstance(self._simulator.backend, self.lightsim_backend):
                    self._timer_preproc -= diff_timer_preproc
                    self._timer_solver -= diff_timer_solver
                    self._timer_postproc -= diff_timer_postproc
            else:
                self.time_stamps[self._chronic_counter].append(self._simulator.chronics_handler.real_data.data.current_datetime)
                self.chronics_id.append(self._chronic_id)
                self.chronics_name.append(self._simulator.chronics_handler.get_name())

    def visualize_network(self) -> PlotMatplot:
        """
        This functions shows the network state evolution over time for a given dataset
        """
        if self._plot_helper is None:
            self._plot_helper = PlotMatplot(self._simulator.observation_space)
        return self._plot_helper.plot_layout()

    def visualize_network_reference_topology(self,
                                             action: Union[BaseAction, None] = None,
                                             **plot_kwargs):
        """
        visualize the power network's reference topology
        """
        if self._plot_helper is None:
            self._plot_helper = PlotMatplot(self._simulator.observation_space)
        env = self._simulator.copy()

        obs = env.reset()
        if action is not None:
            obs, _, _, _ = env.step(action)

        fig = self._plot_helper.plot_obs(obs, **plot_kwargs)
        return fig

    def _reset_simulator(self, fast_forward: bool = True):
        self._obs = self._simulator.reset()
        if fast_forward:
            # exclude the already selected time stamps from which the scenario should started
            remaining_timesteps = set(np.arange(self._max_episode_duration)) - set(self._visited_ts)
            # randomly skip a given number of steps in the first day (to improve the randomness)
            nb_ff = self._simulator.space_prng.choice(list(remaining_timesteps))
            if nb_ff > 0:
                self._simulator.fast_forward_chronics(nb_ff)
                self._obs = self._simulator.get_obs()
                self._visited_ts.append(nb_ff)
            else:
                self._visited_ts.append(0)
        self._reward = self._simulator.reward_range[0]
        self._info = {}

    def __any_isolated_injections(self, obs):
        """Verifies if there are any isolated injections in current obs

        Parameters
        ----------
        obs : _type_
            the current observation for which this verification should be performed

        Returns
        -------
        ``bool``
            If `True` there are some isolated injections
        """
        check_list = []
        for sub_id in range(obs.n_sub):
            sub_obj = obs.get_obj_connect_to(substation_id=sub_id)
            gens = sub_obj["generators_id"]
            loads = sub_obj["loads_id"]
            if len(gens) < 1 and len(loads) < 1:
                continue
            lines_or = sub_obj["lines_or_id"]
            lines_ex = sub_obj["lines_ex_id"]
            lines_topo = []
            injections_topo = []
            if len(lines_or) > 0:
                for line_or in lines_or:
                    line_pos = obs.line_or_pos_topo_vect[line_or]
                    line_topo_vect = obs.topo_vect[line_pos]
                    lines_topo.append(line_topo_vect)
            if len(lines_ex) > 0:
                for line_ex in lines_ex:
                    line_pos = obs.line_ex_pos_topo_vect[line_ex]
                    line_topo_vect = obs.topo_vect[line_pos]
                    lines_topo.append(line_topo_vect)
            if len(loads) > 0:
                for load in loads:
                    load_pos = obs.load_pos_topo_vect[load]
                    load_topo_vect = obs.topo_vect[load_pos]
                    injections_topo.append(load_topo_vect)
            if len(gens) > 0:
                for gen in gens:
                    gen_pos = obs.gen_pos_topo_vect[gen]
                    gen_topo_vect = obs.topo_vect[gen_pos]
                    injections_topo.append(gen_topo_vect)

            check = any(item in lines_topo for item in injections_topo)
        
            #if check is not True:
            #    print(f"At least one isolated injection found at substation {sub_id}")
            check_list.append(check)
        
        check_all = not(all(check_list))
        #if check_all:
        #    print("There are some isolated injections")
        return check_all

    def __any_nan_values(self, obs):
        """Verify if there are any nan values during data generation

        Parameters
        ----------
        obs : _type_
            _description_
        """
        check = False
        if any(np.isnan(obs.p_or)):
            check = True
        return check
    
    def __any_isolated_nodes(self, env):
        """verify if there are isolated nodes in the graph

        Parameters
        ----------
        env : _type_
            _description_
        """
        check = False
        grid = env.backend._grid
        Sbus = grid.get_Sbus()
        if len(Sbus) < env.n_sub:
            check = True
        return check
    
def get_env(env_kwargs: dict):
    """Getter for the environment

    Parameters
    ----------
    env_kwargs : dict
        environment parameters

    Returns
    -------
    grid2op.Environment
        A grid2op environment with the given parameters
    """
    env = grid2op.make(**env_kwargs)
    # env.deactivate_forecast()
    return env
