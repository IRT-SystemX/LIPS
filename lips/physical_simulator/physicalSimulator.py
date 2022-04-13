# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import copy


class PhysicalSimulator(object):
    """
    This is the base class representing a physical simulator.

    A physical simulator is an abstraction that is able to perform some computation for
    a given domain, for example powergrid.

    It should have two main functions:

    - `get_state()` that returns the results of the simulation
    - `modify_state(actor)` that modifies its internal state. This one is optional.

    In order to use `modify_state` the `actor` should be compatible with
    the simulator `actor_types`.
    """
    def __init__(self, actor_types):
        self.actor_types = copy.deepcopy(actor_types)

    def get_state(self):
        """
        This function return the state of the simulator.

        It should be implemented for all `PhysicalSimulator` subclasses.
        """
        pass

    def modify_state(self, actor):
        """
        This function allows to modify the internal state of the simulator. It is optional.

        Parameters
        ----------
        actor:
            Something that is able to modify the internal state of the simulator. Once modified, another
            call to `physical_simulator.get_results()` might give another results.
        """
        if not isinstance(actor, self.actor_types):
            raise RuntimeError(f"The actor used is of class {type(actor)} which is not supported "
                               f"by this simulator. Accepted actor types are {self.actor_types}")
