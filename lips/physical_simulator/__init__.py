# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

__all__ = ["PhysicalSimulator", "PhysicsSolver"]

from lips.physical_simulator.physicalSimulator import PhysicalSimulator
from lips.physical_simulator.physicsSolver import PhysicsSolver

try:
    from lips.physical_simulator.dcApproximationAS import DCApproximationAS
    __all__.append("DCApproximationAS")
except ImportError:
    # grid2op or lightsim2grid package is not installed i cannot used this augmented simulator
    pass

try:
    from lips.physical_simulator.grid2opSimulator import Grid2opSimulator
    __all__.append("Grid2opSimulator")
except ImportError as exc_:
    # grid2op or lightsim2grid are not installed
    pass

