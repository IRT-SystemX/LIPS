"""
PowerGrid general scenario utilities
"""
from typing import Union
import itertools
import warnings
import numpy as np

# from grid2op.Chronics import GridStateFromFile
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Action import DontAct
from grid2op.Action import PlayableAction
from grid2op.Agent import BaseAgent
from grid2op.Parameters import Parameters

from ...logger import CustomLogger
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

    return {"dataset": env_name,
            "param": param,
            # disable maintenances
            "data_feeding_kwargs": {"gridvalueClass": GridStateFromFileWithForecasts},
            # inhibit the opponent
            "action_class": PlayableAction,
            "opponent_init_budget": 0,
            "opponent_action_class": DontAct,
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
    reference_params: Union[dict, None], optional
        a dictionary including the reference topology parameters
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
    # TODO: verify that we do not disconnect the same lines and change the same subs as the reference
    # TODO: Do not take the same actions as the opponent (if opponent is avoided, this is not necessary)
    """
    def __init__(self,
                 env,
                 all_topo_actions: Union[list, None]=None,
                 reference_params: Union[dict, None]=None,
                 scenario_params: Union[dict, None]=None,
                 log_path: Union[str, None]=None,
                 seed: Union[int, None]=None,
                 **kwargs
                ):

        super().__init__(env.action_space)
        self.env = env
        action_space = self.env.action_space
        self.params = scenario_params if scenario_params is not None else {}
        self.params.update(kwargs)
        self.topo_actions = self.params.get("topo_actions", None)
        self.lines_to_disc = self.params.get("lines_to_disc", None)
        self.prob_depth = self.params.get("prob_depth", (1., ))
        self.max_depth = len(self.prob_depth)
        self.prob_type = self.params.get("prob_type", (1., 0.))
        self.prob_do_nothing = self.params.get("prob_do_nothing", 1)
        self.max_disc = self.params.get("max_disc", 0)
        self.action_by_area = self.params.get("action_by_area", False)
        if self.action_by_area:
            self.lines_id_by_area = list(env._game_rules.legal_action.lines_id_by_area.values())

        if reference_params == {}:
            self.reference_args = None
        else:
            self.reference_args = reference_params

        self.seed(seed)


        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger

        if (1. - sum(self.prob_depth) > 1e-3) or (1. - sum(self.prob_type) > 1e-3):
            raise RuntimeError("The probabilities should sum to one")

        self.all_topo_actions =  self.get_action_list() if all_topo_actions is None else all_topo_actions
        self.topo_actions = self.all_topo_actions if self.topo_actions is None else self._filter_topo_actions()

        # find the substations for which there is no actions
        self._sub_empty_action_list = np.where([(len(action_list) == 0) for action_list in self.topo_actions])[0]

        # it aims to verify if the maximum number of authorized line disconnections is reached
        # and to avoid the duplicate actions
        self.disconnected_lines_id = []
        self.impacted_subs_id = []
        self.opponent_attack_line = []
        self._info = {}

        # get a list of all the line actions (line identifiers)
        self.line_ids = np.arange(action_space.n_line) if self.lines_to_disc is None else self.lines_to_disc
        self._remaining_lines = self.line_ids # if it remains some lines in the list to disconnect
        self._disc_actions = [{"set_line_status": [(l_id, -1)]} for l_id in self.line_ids]#range(self.action_space.n_line)]
        self._disc_actions = [self.action_space(el) for el in self._disc_actions]
        # get a DoNothing action
        self._do_nothing = self.action_space({})
        # use do nothing as reference topology, because connecting the elements to busbar one can produce some illegal
        # action reports when the environment present some opponent attacks on the grid
        # self.ref_topo = self._do_nothing # IT IS NOT GOOD
        # Connect all the elements to busbar one (reference topology)
        self.ref_topo = self.action_space({"set_bus":
                                           {"substations_id":
                                            [(sub_id, np.ones(self.action_space.sub_info[sub_id], dtype=int))
                                             for sub_id in range(self.action_space.n_sub)]}})

        # it allows to avoid a depth if struggling to find a combination
        self._depth_tries = 30 # try max 30 times to find a combination
        self._depth_fails = 0 # count the number of fails to find a combination

        if self.reference_args is not None:
            self.ref_lines_to_disc = self.reference_args.get("lines_to_disc", None)
            self.ref_topo_actions = self.reference_args.get("topo_actions", None)
            self.ref_prob_depth = self.reference_args.get("prob_depth", (1., ))
            self.ref_prob_type = self.reference_args.get("prob_type", (1., 0.))
            self.ref_prob_do_nothing = self.reference_args.get("prob_do_nothing", 1.)
            self.ref_max_disc = self.reference_args.get("max_disc", 0)

            self.ref_agent = self.__class__(self.env,
                                            all_topo_actions=self.all_topo_actions,
                                            log_path=self.log_path,
                                            seed=seed,
                                            topo_actions=self.ref_topo_actions,
                                            lines_to_disc=self.ref_lines_to_disc,
                                            prob_depth=self.ref_prob_depth,
                                            prob_type=self.ref_prob_type,
                                            prob_do_nothing=self.ref_prob_do_nothing,
                                            max_disc=self.ref_max_disc
                                            )

    def act(self, obs=None, reward=None, done=None):
        if self.reference_args is not None:
            # self.logger.info("BEGIN REF")
            self.ref_topo = self._apply_reference_topo(obs)
            # self.logger.info("END REF")

        uniform_prob = self.space_prng.uniform()
        if uniform_prob < (1. - self.prob_do_nothing):
            selected_depth = self.space_prng.choice(range(1,self.max_depth+1), 1, p=self.prob_depth)[0]
            #self.logger.info("current_depth : %s", selected_depth)
            # ensure that an action is provided from this depth
            # and don't stuck here if the combination is impossible
            done = True
            nb_try = 0
            while done and (nb_try < self._depth_tries):
                action = self._combine_at_depth(selected_depth)
                done = self._verify_convergence(obs, action)
                #action = self._apply_opponent_topo(action) # this aims to avoid the illegal actions
                nb_try += 1
                if (nb_try >= self._depth_tries) and (done is True):
                    self.logger.error("Impossible to find an action at depth %s", selected_depth)
                    self._depth_fails += 1
                # self.logger.info("nb_try : %s", str(nb_try))
                # self.logger.info("done : %s", done)
        else:
            # DoNothing
            action = self.ref_topo
            #done = self._verify_convergence(obs, action)
            #action = self._apply_opponent_topo(action)

        return action

    def _combine_at_depth(self, selected_depth):
        # self.logger.info("combine at depth : %s", selected_depth)
        # reset the counters for each action
        self.disconnected_lines_id = []
        self.impacted_subs_id = []
        #self.opponent_attack_line = [int(el) for el in np.where(~obs.line_status)[0] if el.size > 0] # include the opponent attacks in the list

        previous_action = self.ref_topo
        current_depth = 0
        while current_depth < selected_depth:
            action = self.sample_act()
            action = self._combine_actions(previous_action, action)
            previous_action = action
            current_depth += 1

        return action

    def _verify_convergence(self, obs, action):
        _, _, done, self._info = obs.simulate(action, time_step=0)
        ambiguous, _ = action.is_ambiguous()
        if ambiguous:
            done = True
        # if info["is_illegal"]:
        #     done = True
        return done

    def _apply_opponent_topo(self, action):
        """re-apply the opponent action to avoid illegal action report

        Parameters
        ----------
        action : ``grid2op.Action``
            the final action to be proposed after verification of its convergence
        info : dict
            supplementary information after applying an action in an environment

        Returns
        -------
        ``grid2op.Action``
            the modified action if the ``is_illegal`` flag is true
            the original action if no illegal action
        """
        #if self._info["is_illegal"]:
        lines_attacked = (np.where(self._info["opponent_attack_line"])[0])
        if lines_attacked.size > 0:
            for line_id in lines_attacked:
                action_or = self.action_space({"set_bus": {"lines_or_id": [(line_id,-1)]}})
                action_ex = self.action_space({"set_bus": {"lines_ex_id": [(line_id,-1)]}})
                line_action = self._combine_actions(action_or, action_ex)
                action = self._combine_actions(action, line_action)

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
                # self.logger.info("Sub %s changed", sub_id)
            elif current_type == 1:
                line_id, action = self._select_line_action()
                self.disconnected_lines_id.append(line_id)
                # self.logger.info("line %s disconnected", line_id)
        else:
            # the maximum authorized disconnection is reached
            # select a substation among those not yet selected
            # check if sub change is authorized
            if self.prob_type[0] > 0.:
                sub_id = self.space_prng.choice(list(set(np.arange(self.action_space.n_sub)) -
                                                    set(self.impacted_subs_id) -
                                                    set(self._sub_empty_action_list)))
                self.impacted_subs_id.append(sub_id)
                action = self._select_topo_action(sub_id)
                # self.logger.info("Sub %s changed", sub_id)
            else:
                action = self._do_nothing

        return action

    @staticmethod
    def _combine_actions(act1, act2):
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
        self._remaining_lines = list(set(np.arange(len(self.line_ids))) -
                                     set(self.disconnected_lines_id)) #-
                                     #set(self.opponent_attack_line))
        #id_ = self.space_prng.choice(self._remaining_lines)
        
        # select a line if it is in the same region of the first disconnection
        if (len(self.disconnected_lines_id) > 0) & (self.action_by_area): # ensure that there is already a disconnected line
            # get the area in which we have already disconnected a line
            area_id_ = [i for i, area in enumerate(self.lines_id_by_area) if self.disconnected_lines_id[-1] in area][0]
            remaining_lines = list(set(self.lines_id_by_area[area_id_]) - set(self.disconnected_lines_id))
            id_ = self.space_prng.choice(remaining_lines)
        else:
            id_ = self.space_prng.choice(self._remaining_lines)
        return self.line_ids[id_], self._disc_actions[id_]

    def _select_topo_action(self, sub_id):
        """
        select randomly one possible action for one substation with id ``sub_id``
        """
        id_ = self.space_prng.choice(len(self.topo_actions[sub_id]))
        return self.topo_actions[sub_id][id_]

    def _filter_topo_actions(self):
        action_list = [[] for i in range(self.action_space.n_sub)]
        for action in self.topo_actions:
            action = self.action_space(action)
            impacted_sub = int(np.where(action.get_topological_impact()[1])[0])
            action_list[impacted_sub].append(action)
        return action_list

    def _apply_reference_topo(self, obs):
        action = self.ref_agent.act(obs=obs)
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
            #return np.array([[2, 2], [1, 1]])
            #return np.array([[2, 2]])
            return np.array([]) # issue #80
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

def compute_all_combinations(action_space, sub_id):
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
    sub_topo = get_sub_topo(action_space, sub_id=sub_id)
    n_elements = len(sub_topo)
    #if n_elements == 0 or n_elements == 1:
    #    raise ValueError("Cannot generate combinations out of a configuration with len = 1 or 2")
    if n_elements <= 2:
        #return np.array([[2, 2], [1, 1]])
        #return np.array([[2, 2]])
        return np.array([])
    else:
        l = [0, 1]
        allcomb = [list(i) for i in itertools.product(l, repeat=n_elements)]

        #we also want to filter combs that only have prods and loads connected to a node
        n_load = sum(action_space.load_to_subid==sub_id)
        n_prod = sum(action_space.gen_to_subid==sub_id)
        nProd_loads= n_load + n_prod

        # we get rid of symetrical topologies by fixing the first element to busbar 0.
        uniqueComb = [np.array(allcomb[i])+1 for i in range(len(allcomb)) if legal_comb(allcomb[i], n_elements, nProd_loads)]

    return np.array(uniqueComb)

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

def get_sub_topo(action_space, sub_id):
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
    sub_info = action_space.sub_info
    return np.ones(sub_info[sub_id], dtype=int)

def get_action(action_space, sub_id, comb):
    """
    Gets the corresponding action for a topological combination
    """
    reconfig_sub = action_space({"set_bus": {"substations_id": [(sub_id, comb)] } })
    return reconfig_sub

def get_action_list(action_space):
    action_list = []
    action_list_sub = []
    for s_id in range(action_space.n_sub):
        action_list_sub.append([])
        all_comb = compute_all_combinations(action_space, sub_id=s_id)
        for comb in all_comb:
            tmp_action = get_action(action_space, sub_id=s_id, comb=comb)
            action_list.append(tmp_action)
            action_list_sub[s_id].append(tmp_action)
    return action_list_sub
