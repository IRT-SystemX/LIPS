"""
PowerGrid general scenario utilities
"""
from typing import Union
import itertools
import warnings
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Parameters import Parameters

from ...config import ConfigManager

def get_kwargs_simulator_scenario(config: ConfigManager) -> dict:
    """Return environment parameters for Benchmark1

    Parameters
    ----------
    config : ConfigManager
        ``ConfigManager`` instance comprising the options for a scenario

    Returns
    -------
    dict
        the dictionary of parameters
    """
    try:
        from lightsim2grid import LightSimBackend
        BkCls = LightSimBackend
    except ImportError:
        from grid2op.Backend import PandaPowerBackend
        BkCls = PandaPowerBackend
    env_name = config.get_option("env_name")
    param = Parameters()
    param.init_from_dict(config.get_option("env_params"))
    # env_name = "l2rpn_case14_sandbox"
    # create a temporary environment to retrieve the default parameters of this specific environment
    #with warnings.catch_warnings():
    #    warnings.filterwarnings("ignore")
    #    env_tmp = grid2op.make(config.get_option("env_name"))
    #param = env_tmp.parameters
    #param.NO_OVERFLOW_DISCONNECTION = True
    # i can act on all powerline / substation at once
    #param.MAX_LINE_STATUS_CHANGED = 999999
    #param.MAX_SUB_CHANGED = 999999
    # i can act every step on every line / substation (no cooldown)
    #param.NB_TIMESTEP_COOLDOWN_LINE = 0
    #param.NB_TIMESTEP_COOLDOWN_SUB = 0
    return {"dataset": env_name,
            "param": param,
            "backend": BkCls()}

class XDepthAgent(BaseAgent):
    """This agent allows to generate the combinatory action of any depth based on probabilities.

    This new agent works as follows:

    1. Apply the reference topology if reference_args is not None by doing following steps
    2. Uniformly based on probability (`prob_do_nothing`) select a DoNothing or a combinatory action.
    3. If uniform prob is less than $1 - $`prob_do_nothing`:
        - Select a depth on the basis of probability vector `prob_depth`
            1. if the maximum number of authorized disconnections is reached
                - sample an action from provided topology actions
            2. Otherwise
                - sample among the disconnection actions and topology action based on `prob_type` probability vector
    4. Apply a scenario based combinatory action based on the given ``subs_to_change`` and ``lines_to_change`` (from step 2)

    Parameters
    ----------
    action_space : ``Environment.action_space``
        Grid2Op environment action space
    params : Union[dict, None], optional
        a dictionary containing all the required parameters for this agent, by default None
        reference_args : Union[``dict``, None], optional
            if provided the reference topology will be applied to the generated action, by default None
        prob_depth: `list`
            a list of probabilities with length `k`, corresponding to the sampling probability of a combinatory action
            with a specific depth. [P_depth_1, P_depth_2, ..., P_depth_k]
        prob_type: ``list``
            Porbability to select an action from types ``DoNothing``, ``topo_change`` and ``line_disc``
            It should contain two or three probabilities, one for each of these types in order
        prob_do_nothing: ``float``
            Probability to select a do nothing action
        max_disc: ``int``
            Maximum disconnection allowed per combinatory action
    **kwargs: ``dict``
        the parameters provided by the user will replace the default parameters of this agent

    Raises
    ------
    RuntimeError
        _description_

    Todo
    -----
    # TODO: verify that number of actions listed in subs_to_change is more than ``prob_depth``
    # TODO: verify that when params is None, kwargs include all the required parameters
    """
    def __init__(self,
                 action_space,
                 params: Union[dict, None] = None,
                #  subs_to_change: Union[list, None]=None,
                #  lines_to_disc: Union[list, None]=None,
                #  prob_depth: list=(0.4, 0.3, 0.3), # max_depth : len(prob_depth)
                #  prob_type: list=(0.7, 0.3), # (TopoChange, LineDisc)
                #  prob_do_nothing=0.2,
                #  max_disc: int=1, # max authorized disconnection per action
                #  reference_args: Union[None, dict]=None,
                 **kwargs
                ):

        super().__init__(action_space)
        self.params = params if params is not None else {}
        self.params.update(kwargs)
        subs_to_change = self.params.get("subs_to_change", None)
        lines_to_disc = self.params.get("lines_to_disc", None)
        prob_depth = self.params.get("prob_depth", (0.4, 0.3, 0.3))
        prob_type = self.params.get("prob_type", (0.7, 0.3))
        prob_do_nothing = self.params.get("prob_do_nothing", 0.2)
        max_disc = self.params.get("max_disc", 1)
        reference_args = self.params.get("reference_args", None)

        if (sum(prob_depth) != 1) or (sum(prob_type) != 1):
            raise RuntimeError("The probabilities should sum to one")

        self.subs_to_change = subs_to_change
        self.all_topo_actions = self.get_action_list()
        if self.subs_to_change is None:
            self.topo_actions = self.all_topo_actions
        else:
            self.topo_actions = self._filter_topo_actions()
        self.prob_depth = prob_depth
        self.max_depth = len(self.prob_depth)
        self.prob_type = prob_type
        self.prob_do_nothing = prob_do_nothing
        self.max_disc = max_disc

        # find the substations for which there is no actions
        self._sub_empty_action_list = np.where([(len(action_list) == 0) for action_list in self.topo_actions])[0]

        # it aims to verify if the maximum number of authorized line disconnections is reached
        # and to avoid the duplicate actions
        self.disconnected_lines_id = []
        self.impacted_subs_id = []

        # get a list of all the substations on which we can do topology actions
        self.sub_ids = np.arange(action_space.n_sub)
        # get a list of all the line actions (line identifiers)
        self.line_ids = np.arange(action_space.n_line) if lines_to_disc is None else lines_to_disc
        self._remaining_lines = self.line_ids # if it remains some lines in the list to disconnect
        # self.line_ids = np.arange(action_space.n_line)
        self._disc_actions = [{"set_line_status": [(l_id, -1)]} for l_id in self.line_ids]#range(self.action_space.n_line)]
        self._disc_actions = [self.action_space(el) for el in self._disc_actions]
        # get a DoNothing action
        self._do_nothing = self.action_space({})
        # Connect all the elements to busbar one (reference topology)
        self.ref_topo = self.action_space({"set_bus":
                                           {"substations_id":
                                            [(sub_id, np.ones(self.action_space.sub_info[sub_id], dtype=int))
                                             for sub_id in range(self.action_space.n_sub)]}})
        self.reference_args = reference_args
        self.ref_lines_to_disc = None
        self.ref_subs_to_change = None
        self.ref_prob_depth = None
        self.ref_prob_type = None
        self.ref_prob_do_nothing = None
        self.ref_max_disc = None


    def act(self, obs=None, reward=None, done=None):
        if self.reference_args is not None:
            self.ref_topo = self._apply_reference_topo()

        uniform_prob = self.space_prng.uniform()
        if uniform_prob < (1. - self.prob_do_nothing):
            # reset the counters for each action
            self.disconnected_lines_id = []
            self.impacted_subs_id = []

            current_depth = self.space_prng.choice(range(1,self.max_depth+1), 1, p=self.prob_depth)[0]
            #print("current_depth : ", current_depth)

            previous_action = self.ref_topo
            for i in range(current_depth):
                action = self.sample_act()
                action = self._combine_actions(previous_action, action)
                previous_action = action
        else:
            # DoNothing
            action = self.ref_topo

        return action

    def sample_act(self):
        """
        Sample an action among the X required actions
        """
        if (len(self.disconnected_lines_id) < self.max_disc) and np.any(self._remaining_lines):
            # the maximum authorized number of disconnections is not yet reached
            current_type = self.space_prng.choice(range(2), 1, p=self.prob_type)[0]
            if current_type == 0:
                # select a substation among those not yet selected
                sub_id = self.space_prng.choice(list(set(np.arange(self.action_space.n_sub)) -
                                                     set(self.impacted_subs_id) -
                                                     set(self._sub_empty_action_list)))
                self.impacted_subs_id.append(sub_id)
                action = self._select_topo_action(sub_id)
                #print(f"Sub {sub_id} changed")
            elif current_type == 1:
                line_id, action = self._select_line_action()
                self.disconnected_lines_id.append(line_id)
                #print(f"line {line_id} disconnected")
        else:
            # the maximum authorized disconnection is reached
            # select a substation among those not yet selected
            sub_id = self.space_prng.choice(list(set(np.arange(self.action_space.n_sub)) -
                                                 set(self.impacted_subs_id) -
                                                 set(self._sub_empty_action_list)))
            self.impacted_subs_id.append(sub_id)
            action = self._select_topo_action(sub_id)
            #print(f"Sub {sub_id} changed")

        return action

    def _combine_actions(self, act1, act2):
        """some kind of "overload" of the `+` operator of grid2op to take into account the disconnected powerline"""
        res = act1 + act2
        for act in (act1, act2):
            set_status = act.line_set_status
            li = [(l_id, -1) for l_id, el in enumerate(set_status) if el == -1]
            if li:
                # force disconnection of the disconnected powerline in this action
                res.line_or_set_bus = li
                res.line_ex_set_bus = li
                res.line_set_status = li
        return res

    def _select_line_action(self):
        """
        select randomly one line to disconnect
        """
        self._remaining_lines = list(set(np.arange(len(self.line_ids))) - set(self.disconnected_lines_id))
        id_ = self.space_prng.choice(self._remaining_lines)
        return self.line_ids[id_], self._disc_actions[id_]

    def _select_topo_action(self, sub_id):
        """
        select randomly one possible action for one substation with id ``sub_id``
        """
        id_ = self.space_prng.choice(len(self.topo_actions[sub_id]))
        return self.topo_actions[sub_id][id_]

    def _filter_topo_actions(self):
        scen_action_subs = [[] for i in range(self.action_space.n_sub)]
        # append the sub topo changes from topo_change list of actions
        for i,j in self.subs_to_change:
            if not self.all_topo_actions[i]:
                warnings.warn('We did not find any action for substation {i}.' \
                             ' It has been skipped.'
                            )
            else:
                if j > len(self.all_topo_actions[i]):
                    raise RuntimeError(f"The action id {j} does not exist.")
                scen_action_subs[i].append(self.all_topo_actions[i][j])

        return scen_action_subs

    def _apply_reference_topo(self):
        self.ref_lines_to_disc = self.reference_args["lines_to_disc"]
        self.ref_subs_to_change = self.reference_args["subs_to_change"]
        self.ref_prob_depth = self.reference_args["prob_depth"]
        self.ref_prob_type = self.reference_args["prob_type"]
        self.ref_prob_do_nothing = self.reference_args["prob_do_nothing"]
        self.ref_max_disc = self.reference_args["max_disc"]

        # REMOVED FOR BETTER PERFORMANCE
        # append line disconnection action
        #for i in self.ref_lines_to_disc:
        #    if i > self.action_space.n_line:
        #        raise RuntimeError(f"Line with id {i} does not exist.")
        #    self.ref_action_lines.append(self._disc_actions[i])

        ref_agent = self.__class__(action_space=self.action_space,
                                   subs_to_change=self.ref_subs_to_change,
                                   lines_to_disc=self.ref_lines_to_disc,
                                   prob_depth=self.ref_prob_depth,
                                   prob_type=self.ref_prob_type,
                                   prob_do_nothing=self.ref_prob_do_nothing,
                                   max_disc=self.ref_max_disc
                                  )
        action = ref_agent.act()
        return action

    ######################################################
    #   Function to compute all the action combinations  #
    ######################################################
    def compute_all_combinations(self, sub_id):
        """
        It computes all the possible combination of topological changes at a substation
        - It prunes the symetrical topologies
        - It ignore the cases where a node is only connected to loads or productions

        Parameters
        ----------
        obs: ``Grid2Op.Observation``
            an observation of Grid2Op environment
        sub_id: ``int``
            substation identifier for which we want get all the topological combinations
        Returns
        -------
        ``np.ndarray``
            a 2d array including all the possible (legal) combinations at a node
            Each vector is binary, where 0 means no change from the reference (busbar 1)
            and 1 represented a topology change (busbar 2)
        """
        sub_topo = self.get_sub_topo(sub_id=sub_id)
        n_elements = len(sub_topo)
        if n_elements == 0 or n_elements == 1:
            raise ValueError("Cannot generate combinations out of a configuration with len = 1 or 2")
        elif n_elements == 2:
            return np.array([[2, 2], [1, 1]])
        else:
            l = [0, 1]
            allcomb = [list(i) for i in itertools.product(l, repeat=n_elements)]

            #we also want to filter combs that only have prods and loads connected to a node
            n_load = sum(self.action_space.load_to_subid==sub_id)
            n_prod = sum(self.action_space.gen_to_subid==sub_id)
            nProd_loads= n_load + n_prod

            # we get rid of symetrical topologies by fixing the first element to busbar 0.
            uniqueComb = [np.array(allcomb[i])+1 for i in range(len(allcomb)) if self.legal_comb(allcomb[i], n_elements, nProd_loads)]

        return np.array(uniqueComb)

    @staticmethod
    def legal_comb(comb, n_elements, nProd_loads):
        """
        verify if all the combination is legal
        """
        sum_comb=np.sum(comb)
        busBar_prods_loads=set(comb[0:nProd_loads])
        busBar_lines = set(comb[nProd_loads:])

        areProdsLoadsIsolated=False
        if(nProd_loads>=2) and (sum_comb != 1) and (sum_comb != n_elements - 1):
            busbar_diff=set(busBar_prods_loads)-set(busBar_lines)
            if(len(busbar_diff)!=0):
                areProdsLoadsIsolated=True

        legal_condition=((comb[0] == 0) & # Keep only actions starting by 0 for symmetrical reasons
                         (sum_comb != 1) &
                         (sum_comb != n_elements - 1) &
                         (areProdsLoadsIsolated==False) &
                         (sum_comb != 0)) # remove the first doNothing (all elements to busbar one) action

        return legal_condition

    def get_sub_topo(self, sub_id):
        """
        This function give the topology vector per substation
        The same functionality is implemented in Grid2Op as ``obs.sub_topology``

        Parameters
        ----------
        obs: ``grid2op.Observation``
            An observation of the environment
        sub_id: ``int``
            The identifier of a substation for which required its topology vector

        Return
        ------
        ``np.ndarray``
            the topology vector for substation ``sub_id``
        """
        sub_info = self.action_space.sub_info
        return np.ones(sub_info[sub_id], dtype=int)

    def get_action(self, sub_id, comb):
        """
        Gets the corresponding action for a topological combination
        """
        reconfig_sub = self.action_space({"set_bus": {"substations_id": [(sub_id, comb)] } })
        return reconfig_sub

    def get_action_list(self):
        action_list = []
        action_list_sub = []
        for s_id in range(self.action_space.n_sub):
            action_list_sub.append([])
            all_comb = self.compute_all_combinations(sub_id=s_id)
            for comb in all_comb:
                tmp_action = self.get_action(sub_id=s_id, comb=comb)
                action_list.append(tmp_action)
                action_list_sub[s_id].append(tmp_action)
        return action_list_sub
