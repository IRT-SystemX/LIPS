# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import json
import numpy
from .benchmark import Benchmark


class NpEncoder(json.JSONEncoder):
    """
    taken from : https://java2blog.com/object-of-type-int64-is-not-json-serializable/
    """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        # if the object is a function, save it as a string
        if callable(obj):
            return str(obj)
        return super(NpEncoder, self).default(obj)


def get_path(root_path: str, benchmark: Benchmark):
    """get the path where the model or evaluation results should be saved

    Parameters
    ----------
    root_path : str
        _description_
    benchmark : Benchmark
        _description_
    """
    path = root_path / benchmark.env_name / benchmark.benchmark_name
    return path
