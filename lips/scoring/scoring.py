# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking
from abc import ABC, abstractmethod
from typing import Union

from ..config import ConfigManager
from ..logger import CustomLogger


class Scoring(ABC):

    def __init__(self,
                 config: Union[ConfigManager, None] = None,
                 config_path: Union[str, None] = None,
                 config_section: Union[str, None] = None,
                 log_path: Union[str, None] = None
                 ):
        if config is None:
            self.config = ConfigManager(section_name=config_section, path=config_path)
        else:
            self.config = config

        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

    @abstractmethod
    def scoring(self):
        pass
