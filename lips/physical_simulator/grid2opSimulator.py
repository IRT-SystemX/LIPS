# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union

import grid2op
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent

from lips.physical_simulator.physicalSimulator import PhysicalSimulator


class Grid2opSimulator(PhysicalSimulator):
    """
    This simulator uses the `grid2op` package to implement a physical simulator.

    It accepts both grid2op BaseAction and grid2op BaseAgent to modify the internal state of the simulator
    """
    def __init__(self,
                 env_kwargs: dict,
                 initial_chronics_id: Union[int, None] = None):
        PhysicalSimulator.__init__(self, actor_types=(BaseAction, BaseAgent))
        self._simulator = grid2op.make(**env_kwargs)
        self._simulator.deactivate_forecast()
        if initial_chronics_id is not None:
            self._simulator.set_id(initial_chronics_id)

        self._obs = None
        self._reward = None
        self._info = None
        self._reset_simulator()

        self._nb_divergence = 0  # number of failures of modify_state
        self._nb_output = 0  # number of time get_state is called

    def seed(self, seed: int):
        """
        It seeds the environment, for reproducible experiments.
        Parameters
        ----------
        seed:
            An integer representing the seed.

        """
        seeds = self._simulator.seed(seed)
        self._reset_simulator()
        return seeds

    def get_state(self):
        """
        The state of the powergrid is, for this class, represented by a tuple:
        - grid2op observation.
        - extra information (can be empty)

        """
        self._nb_output += 1
        return self._obs, self._info

    def modify_state(self, actor):
        """
        It calls `env.step` until a convergence is obtained.
        """
        super().modify_state(actor)  # perform the check that the actor is legit
        done = True
        while done:
            # simulate data (resimulate in case of divergence of the simulator)
            act = actor.act(self._obs, self._reward, done)
            obs, reward, done, info = self._simulator.step(act)
            if info["is_illegal"]:
                raise RuntimeError("Your `actor` should not take illegal action. Please modify the environment "
                                   "or your actor.")
            if done:
                self._nb_divergence += 1
                self._reset_simulator()

    def _reset_simulator(self):
        self._obs = self._simulator.reset()
        self._reward = self._simulator.reward_range[0]
        self._info = {}
