import numpy as np
import re
from itertools import combinations
from tqdm.auto import tqdm


import grid2op
from grid2op.Parameters import Parameters
import warnings
from lightsim2grid import LightSimBackend, ContingencyAnalysis

from lips.config import ConfigManager

def reset_simulator(env, max_episode_duration, visited_ts):
    # select chronics and shuffle them
    _ = env.reset()
    env.chronics_handler.shuffle()
    _ = env.reset()

    # Fast forward the environment to reach a different time stamps
    remaining_timesteps = set(np.arange(max_episode_duration)) - set(visited_ts)
    nb_ff = env.space_prng.choice(list(remaining_timesteps))

    if nb_ff > 0:
        env.fast_forward_chronics(nb_ff)
        obs = env.get_obs()
        visited_ts.append(nb_ff)
    else:
        visited_ts.append(0)

    return env

def compute_solver_time(nb_samples: int, config: ConfigManager):
    #env_name = "l2rpn_neurips_2020_track2_small"
    env_name = config.get_option("env_name")
    test = False

    # Create the grid2op environment
    param = Parameters()
    param.init_from_dict(config.get_option("env_params"))

    initial_chronics_id = config.get_option("benchmark_seeds").get("initial_chronics_id", 0)
    chronics_selected_regex = config.get_option("chronics").get("train")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = grid2op.make(env_name,
                           backend=LightSimBackend(),
                           # ignore the protection, that are NOT simulated
                           # by the TimeSerie module !
                           param=param,
                           test=test
                           )
        chronics_selected_regex = re.compile(chronics_selected_regex)
        env.chronics_handler.set_filter(lambda path:
                                        re.match(chronics_selected_regex, path) is not None)
        env.chronics_handler.real_data.reset()
        env.set_id(initial_chronics_id)

    actions_list = config.get_option("dataset_create_params").get("reference_args").get("topo_actions")
    all_combinations = list(combinations(actions_list, 2))
    all_combinations.extend(actions_list)
    solver_times = []
    max_episode_duration = env.chronics_handler.max_episode_duration()
    nb_divergence = 0

    for actions in tqdm(all_combinations):
        visited_ts = []
        for _ in range(3):
            env = reset_simulator(env, max_episode_duration, visited_ts)

            if len(actions) > 1:
                action = env.action_space()
                for action_ in actions:
                    action += env.action_space(action_)
            else:
                action = env.action_space(actions)
            
            # ensure that the action does not cause gameover
            done = True
            while done:
                _, _, done, _ = env.step(action)
                if done:
                    nb_divergence += 1
                    env = reset_simulator(env, max_episode_duration, visited_ts)
            
            # Prints to verify that the chronics and timestamps changes over actions
            # print(env.chronics_handler.get_name())
            # print(env.chronics_handler.real_data.data.current_datetime)

            # Run the environment on a scenario using the TimeSerie module
            security_analysis = ContingencyAnalysis(env)
            security_analysis.add_all_n1_contingencies()
            p_or, a_or, voltages = security_analysis.get_flows()

            computer = security_analysis.computer
            time_solver_one_pf = computer.solver_time() / computer.nb_solved()

            solver_times.append(time_solver_one_pf * nb_samples)
    # print(solver_times)
    # print(f"Number of divergent cases : {nb_divergence}")
    return np.mean(solver_times)

if __name__ == "__main__":
    from lips import get_root_path
    LIPS_PATH = get_root_path(pathlib_format=True).parent
    config_path = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_idf_2023.ini"
    config = ConfigManager(path=config_path, section_name="Benchmark_competition")
    solver_time = compute_solver_time(nb_samples=int(1e5), config=config)
    print(solver_time)
