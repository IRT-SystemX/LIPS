import pathlib
from pathlib import Path
from typing import Union

import grid2op
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Action.CompleteAction import CompleteAction
from grid2op.Backend import PandaPowerBackend

from lips.config import ConfigManager
from lips.physical_simulator import Grid2opSimulator
from lips.physical_simulator.dcApproximationAS import PhysicsSolver, DCApproximationAS


# Modifications: the class BenchmarkedSimulator shall be a subclass of AugmentedSimulator or of PhysicsSolver


class BenchmarkedSimulator(DCApproximationAS):
    """
    This class implements a specific augmented simulator to be evaluated through the LIPS benchmarks.
    Its implementation shall follow a specific set of rules:
    - be a subclass of AugmentedSimulator
    - implements at least the functions train, predict, restore
    - the specific parameters of the simulator shall be specified in a dedicated file simulator_config.json
    """

    def __init__(self,
                 name: str = "dc_approximation",
                 benchmark_name: str = "Benchmark1",
                 config_path: Union[str, None] = None,
                 simulator: Union[Grid2opSimulator, None] = None,
                 ):
        """
        DCApproximationAS.__init__(
            self,
            name=name,
            **kwargs,
        )
        """
        PhysicsSolver.__init__(self, name=name)
        self.bench_config = ConfigManager(path=config_path, section_name=benchmark_name)
        # input that will be given to the augmented simulator
        self._attr_x = ("prod_p", "prod_v", "load_p", "load_q", "topo_vect")
        # output that we want the proxy to predict
        #self._attr_y = ("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex")
        attr_y = {"a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"}
        self._attr_y = tuple(set(self.bench_config.get_option("attr_y")).intersection(attr_y))
        # TODO : this attribute is not already used
        self._attr_fix_gen_p = "__prod_p_dc"

        #FIXME: change w.r.t. DCApproximationAS: initialization of grid_path from benchmark config

        env = grid2op.make(ConfigManager(path=config_path, section_name="DEFAULT").get_option("env_name"), test=True)
        grid_path = pathlib.Path(env.get_path_env()) / "grid.json"

        if grid_path is not None:
            self._raw_grid_simulator = PandaPowerBackend()
            self._raw_grid_simulator.load_grid(grid_path)
            self._raw_grid_simulator.assert_grid_correct()
        elif simulator is not None:
            assert isinstance(simulator, Grid2opSimulator), "To make the DC approximation augmented simulator, you " \
                                                            "should provide the reference grid as a Grid2opSimulator"

            self._raw_grid_simulator = simulator._simulator.backend.copy()
        else:
            raise RuntimeError("Impossible to initialize the DC approximation with a grid2op simulator or"
                               "a powergrid")

        self._bk_act_class = _BackendAction.init_grid(self._raw_grid_simulator)
        self._act_class = CompleteAction.init_grid(self._raw_grid_simulator)

        self.comp_time = 0
