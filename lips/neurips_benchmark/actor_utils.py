# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import copy
from leap_net.agents import RandomAct1


class ChangeTopoRefN1(RandomAct1):
    """
    This agent change the reference topology (from the fully connected topology plus some
    topology given by `ref_topo`), and is able to perform (uniformly at random) some action (from list_act)
    on top of that.

    """

    def __init__(self, action_space, ref_topo, list_act=()):
        RandomAct1.__init__(self, action_space, list_act)
        # now i clean the provided `ref_topo`
        tmp_random_act = RandomAct1(action_space, list_act=ref_topo)

        self.ref_topo = [copy.deepcopy(self.all_to_one)]
        self.ref_topo += [copy.deepcopy(self.all_to_one) + el for el in tmp_random_act._all_actions]

    def act(self, obs, reward, done=False):
        # sample a reference topology
        self.all_to_one = self.space_prng.choice(self.ref_topo)
        # play some action on top of that
        return super().act(obs, reward, done=done)


class ChangeTopoRefN1Ref(RandomAct1):
    """
    This agent change the reference topology (chose one uniformly at random among the complete one and the one
    specified in `ref_topo`)

    And with probability `p` will also apply, on top of that, an action from `list_act`
    """

    def __init__(self, action_space, p, ref_topo, list_act=()):
        RandomAct1.__init__(self, action_space, list_act)
        # now i clean the provided `ref_topo`
        tmp_random_act = RandomAct1(action_space, list_act=ref_topo)

        self.ref_topo = [copy.deepcopy(self.all_to_one)]
        self.ref_topo += [copy.deepcopy(self.all_to_one) + el for el in tmp_random_act._all_actions]
        self.p = float(p)
        self._1_p = 1. - self.p

    def act(self, obs, reward, done=False):
        # sample a reference topology
        self.all_to_one = self.space_prng.choice(self.ref_topo)
        ur = self.space_prng.uniform()
        if ur < self._1_p:
            # return it with proba p
            res = self.all_to_one
        else:
            # otherwise disconnect a powerline on top of that
            res = RandomAct1.act(self, obs, reward, done)
        return res


class ChangeTopoRefN2(ChangeTopoRefN1):
    """
    This agent change the reference topology (chose one uniformly at random among the complete one and the one
    specified in `ref_topo`)

    And, on top of that, will apply 2 actions among `list_act`
    """
    def __init__(self, action_space, ref_topo, list_act=()):
        ChangeTopoRefN1.__init__(self, action_space, ref_topo, list_act)

    def act(self, obs, reward, done=False):
        # choose reference topology
        self.all_to_one = self.space_prng.choice(self.ref_topo)
        # choose first action
        act_id1, sub_id_act1, this_random1 = self.sample_act()
        if sub_id_act1 is not None:
            sub_id_act1 = sub_id_act1[0]
        # choose second action
        act_id2, sub_id_act2, this_random2 = self.sample_act(act_id1, sub_id_act1)
        # combine everything
        res = self._combine_actions(self.all_to_one, this_random1)
        res = self._combine_actions(res, this_random2)
        return res
