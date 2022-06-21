# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import pathlib
from configparser import ConfigParser
import ast
from typing import Union

class ConfigManager(object):
    """
    This class ease the use of config parser for the framework
    """
    def __init__(self,
                 section_name: str="DEFAULT",
                 path: Union[str, None] = None
                ):

        self.section_name = section_name
        self.path_config = None
        if path is None:
            self.path_config = pathlib.Path(__file__).parent.absolute().joinpath("conf.ini")
        else:
            self.path_config = path

        self.config = ConfigParser()
        # if a config file exists already try to load it
        if os.path.exists(self.path_config):
            self.config = self._read_config(self.path_config)

    def create_config(self,
                      scenario_name: Union[str, None] = None,
                      path: Union[str, None] = None,
                      **kwargs):
        """
        function to create a config file if it does not exist already
        """
        if scenario_name is None:
            scenario_name = self.section_name

        if path is None:
            path = self.path_config

        # do not allow to re-create a config for existing scenario_name
        try:
            self.config[scenario_name] = {}
        except KeyError as err:
            raise RuntimeError(f"Invalid scenario_name {scenario_name}. \
                     A configuration with this name is already exists in config file.") from err

        for key, value in kwargs.items():
            self.config[scenario_name][key] = str(value)
        # save the config file
        self._write_config(path)
        return self.config

    def get_option(self, option:str):
        """
        return the value for an option under a list format
        """
        option = self.config[self.section_name].get(option, fallback=None)
        if option is not None:
            option = self._str_to_list(option)
        return option

    def get_options_dict(self):
        """
        retrun a dictionary of all the config options
        """
        my_dict = {}
        for key_, item_ in self.config[self.section_name].items():
            my_dict[key_] = self._str_to_list(item_)
        return my_dict

    def edit_config_option(self,
                           option: str,
                           value: Union[str, None]=None,
                           scenario_name: Union[str, None]=None):
        """
        to add or edit an option for a scenario
        """
        if scenario_name is None:
            scenario_name = self.section_name

        if value is None:
            value = ""
        self.config.set(scenario_name, option, value)
        return self.config

    def remove_section(self, section_name: str):
        """
        remove a section from the config file

        Parameters
        ----------
        section_name : str
            the section to remove
        """
        self.config.remove_section(section_name)
        return self.config

    def remove_config_option(self,
                             option: Union[str, None],
                             scenario_name: Union[str, None] = None):
        """
        to remove an option from a config
        """
        if scenario_name is None:
            scenario_name = self.section_name
        if option is None:
            self.config.remove_section(scenario_name)
        else:
            self.config.remove_option(scenario_name, option)
        return self.config

    def _write_config(self, path: Union[str, None] = None):
        """
        function to write the config to the file
        """
        if path is None:
            path = self.path_config

        with open(path, "w", encoding="utf-8") as configfile:
            self.config.write(configfile)

    def _read_config(self, path: Union[str, None] = None):
        """
        read a config from file
        """
        if path is None:
            path = self.path_config

        # TODO : verify if the indicated path exists

        self.config.read(path)
        return self.config

    @staticmethod
    def _str_to_list(string: str):
        return ast.literal_eval(string)
