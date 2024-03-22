import numpy as np
from lips import get_root_path
from lips.dataset.utils.powergrid_utils import XDepthAgent, get_kwargs_simulator_scenario, get_action_list
from lips.benchmark.powergridBenchmark import get_env
from lips.config import ConfigManager

LIPS_PATH = get_root_path(pathlib_format=True).parent
CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_idf_2023.ini"
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_idf_2023"
LOG_PATH = LIPS_PATH / "lips_logs.log"

def test_action_by_area():
    config = ConfigManager(path=CONFIG_PATH, section_name="Benchmark_competition")
    env_kwargs = get_kwargs_simulator_scenario(config=config)
    env = get_env(env_kwargs)

    all_topo_actions = get_action_list(env.action_space)

    test_ood_actor = XDepthAgent(env,
                                all_topo_actions=all_topo_actions,
                                reference_params=config.get_option("dataset_create_params").get("reference_args", None),
                                scenario_params=config.get_option("dataset_create_params")["test_ood"],
                                seed=1234,
                                log_path=None
                                )

    obs = env.reset()
    action = test_ood_actor.act(obs)
    obs, _, _, _ = env.step(action)

    lines_disc_list = list(np.where(obs.line_status==False)[0])

    lines_id_by_area = list(env._game_rules.legal_action.lines_id_by_area.values())

    contained = []

    for area in lines_id_by_area:
        flag = True if set(lines_disc_list) <= set(area) else False
        contained.append(flag)
    print(contained)
    assert(any(contained))

# benchmark_competition = PowerGridBenchmark(benchmark_path=DATA_PATH,
#                                            benchmark_name="Benchmark_competition",
#                                            load_data_set=False,
#                                            config_path=CONFIG_PATH,
#                                            log_path=LOG_PATH)

if __name__ =="__main__":
    test_action_by_area()