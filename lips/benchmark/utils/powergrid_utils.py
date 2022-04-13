"""
PowerGrid general scenario utilities
"""
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