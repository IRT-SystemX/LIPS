# """
# Test case for lips
# """
# import unittest
# import pathlib as pl

# class TestData(unittest.TestCase):
#     """
#     Testing the benchmark classes
#     """
#     def test(self):
#         """
#         testing if data for first benchmark has been provided
#         """
#         path = pl.Path("reference_data/NeuripsBenchmark1/train")
#         self.assertEqual((str(path), path.is_dir()), (str(path), True))

# #if __name__ == "__main__":
# #    unittest.main(verbosity=2)

# suite = unittest.TestLoader().loadTestsFromTestCase(TestData)
# unittest.TextTestRunner(verbosity=2).run(suite)

import pathlib
import numpy as np
from sklearn.metrics import mean_absolute_error

from lips.config import ConfigManager
from lips.benchmark.powergridBenchmark import PowerGridBenchmark

LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
DATA_PATH = LIPS_PATH / "reference_data" / "test"
LOG_PATH = LIPS_PATH / "lips_logs.log"

def test_generation_reproducibiltiy():
    """This test aims at executing the same experiment several times and verifying that the results are the same
    """
    benchmark1_ex1 = PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark1",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    benchmark1_ex1.generate(nb_sample_train=int(1e4),
                            nb_sample_val=int(1e3),
                            nb_sample_test=int(1e3),
                            nb_sample_test_ood_topo=int(1e3)
                           )

    benchmark1_ex2 = PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark1",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    benchmark1_ex2.generate(nb_sample_train=int(1e4),
                            nb_sample_val=int(1e3),
                            nb_sample_test=int(1e3),
                            nb_sample_test_ood_topo=int(1e3)
                           )

    data_ex1 = benchmark1_ex1.train_dataset.data
    data_ex2 = benchmark1_ex2.train_dataset.data
    keys = data_ex1.keys()
    errors = list()
    for key_ in keys:
        if key_ == "line_status":
            data_ex1["line_status"] = np.asarray(data_ex1.get("line_status"), dtype=int)
            data_ex2["line_status"] = np.asarray(data_ex2.get("line_status"), dtype=int)
        error = mean_absolute_error(data_ex1.get(key_), data_ex2.get(key_))
        errors.append(error)

    assert(np.sum(errors) < 1e-3)

def test_generation_seeds():
    """This test aims at changing the seeds and verifying the heterogeneity of data
    """
    benchmark_seeds = {
        "train_env_seed": 1,
        "val_env_seed": 2,
        "test_env_seed": 3,
        "test_ood_topo_env_seed": 4,
        "train_actor_seed": 5,
        "val_actor_seed": 6,
        "test_actor_seed": 7,
        "test_ood_topo_actor_seed": 8,
        }

    benchmark1_ex1 = PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark1",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH,
                                        **benchmark_seeds
                                        )

    benchmark1_ex1.generate(nb_sample_train=int(1e4),
                            nb_sample_val=int(1e3),
                            nb_sample_test=int(1e3),
                            nb_sample_test_ood_topo=int(1e3)
                           )

    benchmark_seeds = {
        "train_env_seed": 9,
        "val_env_seed": 10,
        "test_env_seed": 11,
        "test_ood_topo_env_seed": 12,
        "train_actor_seed": 13,
        "val_actor_seed": 14,
        "test_actor_seed": 15,
        "test_ood_topo_actor_seed": 16,
        }
    benchmark1_ex2 = PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark1",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH,
                                        **benchmark_seeds
                                        )

    benchmark1_ex2.generate(nb_sample_train=int(1e4),
                            nb_sample_val=int(1e3),
                            nb_sample_test=int(1e3),
                            nb_sample_test_ood_topo=int(1e3)
                           )

    data_ex1 = benchmark1_ex1.train_dataset.data
    data_ex2 = benchmark1_ex2.train_dataset.data
    keys = data_ex1.keys()
    errors = list()
    for key_ in keys:
        if key_ == "line_status":
            data_ex1["line_status"] = np.asarray(data_ex1.get("line_status"), dtype=int)
            data_ex2["line_status"] = np.asarray(data_ex2.get("line_status"), dtype=int)
        error = mean_absolute_error(data_ex1.get(key_), data_ex2.get(key_))
        errors.append(error)

    assert(np.sum(errors) > 0)


# if __name__ == "__main__":
#     benchmark_seeds = {
#     "train_env_seed": 7,
#     "val_env_seed": 2,
#     "test_env_seed": 3,
#     "test_ood_topo_env_seed": 4,
#     "train_actor_seed": 5,
#     "val_actor_seed": 6,
#     "test_actor_seed": 7,
#     "test_ood_topo_actor_seed": 8,
#     }

#     benchmark1_ex1 = PowerGridBenchmark(benchmark_path=None,
#                                         benchmark_name="Benchmark1",
#                                         load_data_set=False,
#                                         config_path=CONFIG_PATH,
#                                         log_path=LOG_PATH,
#                                         #**benchmark_seeds
#                                         )
#     print(benchmark1_ex1.train_env_seed)
#     #test_generation_reproducibiltiy()
#     #test_generation_seeds()
#     print(CONFIG_PATH)
#     config = ConfigManager(path=CONFIG_PATH, section_name="Benchmark1")
#     seed = config.get_option("benchmark_seeds").get("train_env_seed", 0)
#     print(seed)
#     benchmark1_ex1 = PowerGridBenchmark(benchmark_path=None,
#                                         benchmark_name="Benchmark1",
#                                         load_data_set=False,
#                                         config_path=CONFIG_PATH,
#                                         log_path=LOG_PATH)
