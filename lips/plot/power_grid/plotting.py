# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union
import numpy as np
from matplotlib import pyplot as plt

import grid2op
from grid2op.PlotGrid import PlotMatplot
from grid2op.PlotGrid.PlotPlotly import PlotPlotly

class PlotData():
    def __init__(self, dataset: dict, env_name="l2rpn_case14_sandbox"):
        self.dataset = dataset
        self.chronics_name = self.dataset.chronics_info["chronics_name"]
        self.time_stamps = np.hstack(self.dataset.chronics_info["time_stamps"])
        self.data = dataset.data
        self.size = dataset.size
        self.env_name = env_name
        self.env = grid2op.make(self.env_name)
        self.id_obs = 0

    def visualize_obs(self,
                      id_obs: Union[int, None]=None,
                      line_info=None,
                      load_info=None,
                      gen_info=None):
        if id_obs is not None:
            self.id_obs = id_obs
        obs = self.set_obs(self.id_obs)
        plot_helper = PlotMatplot(self.env.observation_space)

        fig_custom = plot_helper.plot_obs(obs,
                                          line_info=line_info,
                                          load_info=load_info,  # i don't plot anything concerning the load
                                          gen_info=gen_info  # i draw the voltage setpoint of the generators
                                          )
        plt.title(f"Chronic: {self.chronics_name[self.id_obs]}, "
                  f"Time stamp: {str(self.time_stamps[self.id_obs])}, "
                  f"index {self.id_obs}")

    def visualize_next(self, line_info=None, load_info=None, gen_info=None):
        self.id_obs += 1
        if self.id_obs > self.size-1:
            self.id_obs = self.size-1
            # TODO: replace the print by warning once in implementation
            print("You reached the end of data. The index is set to the last observation.")

        self.visualize_obs(self.id_obs, line_info=line_info, load_info=load_info, gen_info=gen_info)

    def visualize_prev(self, line_info=None, load_info=None, gen_info=None):
        self.id_obs -= 1
        if self.id_obs < 0:
            self.id_obs = 0
            # TODO: replace the print by warning once in implementation
            print("You reached the first observation. The index is set to the first observation!")
        self.visualize_obs(self.id_obs, line_info=line_info, load_info=load_info, gen_info=gen_info)


    def set_obs(self, id_obs):
        fake_obs = self.env.reset()
        #change obs values to the ones we want
        for key in self.data.keys():
            values_key = self.data[key][id_obs]
            try:
                setattr(fake_obs, key, values_key)
            finally:
                continue
        fake_obs.rho=fake_obs.a_or/fake_obs.thermal_limit
        return fake_obs
    
class PlotDataPlotly():
    def __init__(self, dataset: dict, env_name="l2rpn_case14_sandbox"):
        self.dataset = dataset
        self.chronics_name = self.dataset.chronics_info["chronics_name"]
        self.time_stamps = np.hstack(self.dataset.chronics_info["time_stamps"])
        self.data = dataset.data
        self.size = dataset.size
        self.env_name = env_name
        self.env = grid2op.make(self.env_name)
        self.id_obs = 0

    def visualize_obs(self,
                      id_obs: Union[int, None]=None,
                      line_info=None,
                      load_info=None,
                      gen_info=None):
        if id_obs is not None:
            self.id_obs = id_obs
        obs = self.set_obs(self.id_obs)
        plot_helper = PlotPlotly(self.env.observation_space)

        fig_custom = plot_helper.plot_obs(obs,
                                          line_info=line_info,
                                          load_info=load_info,  # i don't plot anything concerning the load
                                          gen_info=gen_info  # i draw the voltage setpoint of the generators
                                          )
        fig_custom.update_layout(title=dict(text=f"Chronic: {self.chronics_name[self.id_obs]}, "
                                                 f"Time stamp: {str(self.time_stamps[self.id_obs])}, "
                                                 f"index {self.id_obs}",
                                            automargin=True))
        fig_custom.show()

    def visualize_next(self, line_info=None, load_info=None, gen_info=None):
        self.id_obs += 1
        if self.id_obs > self.size-1:
            self.id_obs = self.size-1
            # TODO: replace the print by warning once in implementation
            print("You reached the end of data. The index is set to the last observation.")

        self.visualize_obs(self.id_obs, line_info=line_info, load_info=load_info, gen_info=gen_info)

    def visualize_prev(self, line_info=None, load_info=None, gen_info=None):
        self.id_obs -= 1
        if self.id_obs < 0:
            self.id_obs = 0
            # TODO: replace the print by warning once in implementation
            print("You reached the first observation. The index is set to the first observation!")
        self.visualize_obs(self.id_obs, line_info=line_info, load_info=load_info, gen_info=gen_info)


    def set_obs(self, id_obs):
        fake_obs = self.env.reset()
        #change obs values to the ones we want
        for key in self.data.keys():
            values_key = self.data[key][id_obs]
            try:
                setattr(fake_obs, key, values_key)
            finally:
                continue
        fake_obs.rho=fake_obs.a_or/fake_obs.thermal_limit
        return fake_obs
