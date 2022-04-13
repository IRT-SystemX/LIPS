# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking


import logging
from typing import Union

class CustomLogger:
    def __init__(self,
                 name: Union[str, None]=None,
                 log_path: Union[str, None]=None):
        if name is None:
            name = ""
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter(fmt='%(asctime)s - %(name)-25s - %(levelname)-7s : %(message)s',
                                          datefmt="%Y-%m-%d %H:%M:%S")
            if log_path is None:
                log_path = "logs.log"
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.logger.propagate = False

    def info(self, message):
        self.logger.info("{}".format(message))
    
    def warning(self, message):
        self.logger.warning("{}".format(message))

    def error(self, message):
        self.logger.error("{}".format(message))