# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

__all__ = ["AugmentedSimulator"]

from lips.augmented_simulators.augmentedSimulator import AugmentedSimulator


try:
    from lips.augmented_simulators.dcApproximationAS import DCApproximationAS
    __all__.append("DCApproximationAS")
except ImportError:
    # grid2op or lightsim2grid package is not installed i cannot used this augmented simulator
    pass


try:
    from lips.augmented_simulators.hyperParameterTuner import HyperParameterTuner
    __all__.append("HyperParameterTuner")
except ImportError:
    # tensorflow or sklearn package is not installed i cannot used this class
    pass


try:
    from lips.augmented_simulators.fullyConnectedAS import FullyConnectedAS
    __all__.append("FullyConnectedAS")
except ImportError:
    # tensorflow package is not installed i cannot used this augmented simulator
    pass


try:
    from lips.augmented_simulators.leapNetAS import LeapNetAS
    __all__.append("LeapNetAS")
except ImportError:
    # leap_net package is not installed i cannot used this augmented simulator
    pass
