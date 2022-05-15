# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

__all__ = ["DataSet"]

from lips.dataset.dataSet import DataSet
# try:
#     from lips.dataset.powergridDataSet import PowerGridDataSet
#     __all__.append("PowerGridDataSet")
# except ImportError as exc_:
#     # grid2op package is not installed i cannot used this augmented simulator
#     pass
