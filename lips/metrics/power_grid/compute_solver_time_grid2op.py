from lips import get_root_path
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.dataset.powergridDataSet import PowerGridDataSet
from lips.config import ConfigManager
from lips.physical_simulator import Grid2opSimulator
from lips.dataset.utils.powergrid_utils import get_kwargs_simulator_scenario, get_action_list, XDepthAgent
from lips.benchmark.powergridBenchmark import get_env

def compute_solver_time_grid2op(config_path, benchmark_name, nb_samples=1e5):

    # benchmark_competition = PowerGridBenchmark(benchmark_path=None,
    #                                            benchmark_name=benchmark_name,
    #                                            load_data_set=False,
    #                                            config_path=config_path)
    
    # benchmark_competition.generate(nb_sample_train=int(1),
    #                                nb_sample_val=int(1),
    #                                nb_sample_test=int(1e3),
    #                                nb_sample_test_ood_topo=int(1)
    #                               )
    
    # print(benchmark_competition.test_simulator._timer_solver)
    # print((benchmark_competition.test_simulator._timer_solver / int(1e3)) * nb_samples)


    config = ConfigManager(section_name=benchmark_name, path=config_path)
    env = get_env(get_kwargs_simulator_scenario(config))
    attr_names = config.get_option("attr_x") + config.get_option("attr_tau") + config.get_option("attr_y")
    test_dataset = PowerGridDataSet(name="test",
                                    attr_names=attr_names,
                                    config=config,
                                    )
    test_env_seed = config.get_option("benchmark_seeds").get("test_env_seed", 2)
    test_actor_seed = config.get_option("benchmark_seeds").get("test_actor_seed", 6)
    initial_chronics_id = config.get_option("benchmark_seeds").get("initial_chronics_id", 0)

    test_simulator = Grid2opSimulator(get_kwargs_simulator_scenario(config),
                                      initial_chronics_id=initial_chronics_id,
                                      chronics_selected_regex=config.get_option("chronics").get("test")
                                      )
    test_simulator.seed(test_env_seed)
    all_topo_actions = get_action_list(env.action_space)
    test_actor = XDepthAgent(env,
                             all_topo_actions=all_topo_actions,
                             reference_params=config.get_option("dataset_create_params").get("reference_args", None),
                             scenario_params=config.get_option("dataset_create_params")["test"],
                             seed=test_actor_seed)
                                 
    test_dataset.generate(simulator=test_simulator,
                         actor=test_actor,
                         path_out=None,
                         nb_samples=int(1e3),
                         nb_samples_per_chronic=config.get_option("samples_per_chronic").get("test", 288),
                         do_store_physics=False,
                         is_dc=False,
                         store_as_sparse=False
                         )
    print("Time required to solve one power flow: ", test_simulator._timer_solver / int(1e3))
    print(f"Time required to solve {nb_samples} power flows:  {(test_simulator._timer_solver / int(1e3)) * nb_samples}")

    return (test_simulator._timer_solver / int(1e3)) * nb_samples
    

if __name__ == "__main__":
    LIPS_PATH = get_root_path(pathlib_format=True).parent
    CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_idf_2023.ini"
    compute_solver_time_grid2op(CONFIG_PATH, "Benchmark_competition", nb_samples=int(1e5))
