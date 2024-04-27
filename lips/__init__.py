# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking
import pathlib

__version__ = "0.2.4"

here = pathlib.Path(__file__).parent.resolve()

def get_version(rel_path="__init__.py"):
    init_content = (here / rel_path).read_text(encoding='utf-8')
    for line in init_content.split('\n'):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

def get_root_path(pathlib_format=False):
    """Returns the root path of LIPS."""
    import os
    path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    if pathlib_format:
        return pathlib.Path(path)
    return path
