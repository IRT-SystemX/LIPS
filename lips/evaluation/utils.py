# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Callable

class MetricFactory:
    """Factory method to register the metrics
    """
    def __init__(self):
        self._creators = {}

    def register_metric(self, metric_name: str, creator: Callable):
        """Register new metrics

        Parameters
        ----------
        metric_name : str
            _description_
        creator : Callable
            _description_
        """
        self._creators[metric_name] = creator

    def register_metric_dict(self, metric_dict: dict):
        """register a dictionary including metrics

        Parameters
        ----------
        metric_dict : dict
            _description_
        """
        for key_, fun_ in metric_dict.items():
            self.register_metric(key_, fun_)

    def get_metric(self, metric_name:str):
        """Get the required metric

        Parameters
        ----------
        metric_name : str
            _description_

        Returns
        -------
        Callable
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        creator = self._creators.get(metric_name)
        if not creator:
            raise ValueError(metric_name)
        return creator

metric_factory = MetricFactory()
