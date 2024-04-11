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
from cmath import exp
from math import pi
import numpy as np
from lips import get_root_path

from lips.benchmark.powergridBenchmark import PowerGridBenchmark

LIPS_PATH = get_root_path(pathlib_format=True).parent
CONFIG_PATH = LIPS_PATH / "lips" / "tests" / "configs" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
#DATA_PATH = LIPS_PATH / "reference_data" / "test"
#DATA_PATH = LIPS_PATH / "reference_data" / "powergrid" / "l2rpn_case14_sandbox"
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_case14_sandbox"
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
        if key_ == "line_status" or key_ == "PV_nodes":
            data_ex1[key_] = np.asarray(data_ex1.get(key_), dtype=int)
            data_ex2[key_] = np.asarray(data_ex2.get(key_), dtype=int)
        error = np.mean(np.abs(data_ex1.get(key_) - data_ex2.get(key_)))
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
        if key_ == "line_status" or key_ == "PV_nodes":
            data_ex1[key_] = np.asarray(data_ex1.get(key_), dtype=int)
            data_ex2[key_] = np.asarray(data_ex2.get(key_), dtype=int)
        error = np.mean(np.abs(data_ex1.get(key_) - data_ex2.get(key_)))
        errors.append(error)

    assert(np.sum(errors) > 0)

def test_bench1_data_reproducibility():
    """This test aims at verifying if the same exact data could be reproduced after each FrameWork update
    BENCHMARK 1
    """
    benchmark1_ex1 = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                        benchmark_name="Benchmark1",
                                        load_data_set=True,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    benchmark1_ex2 = PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark1",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    data_size = int(2e3)
    benchmark1_ex2.generate(nb_sample_train=data_size,
                            nb_sample_val=data_size,
                            nb_sample_test=data_size,
                            nb_sample_test_ood_topo=data_size
                           )
    dataset_labels = ("train_dataset", "val_dataset", "_test_dataset", "_test_ood_topo_dataset")
    for label_ in dataset_labels:
        data_ex1 = getattr(benchmark1_ex1, label_).data
        data_ex2 = getattr(benchmark1_ex2, label_).data
        keys = data_ex1.keys()
        errors = list()
        for key_ in keys:
            print(key_)
            if key_ == "line_status" or key_ == "PV_nodes":
                data_ex1[key_] = np.asarray(data_ex1.get(key_), dtype=int)
                data_ex2[key_] = np.asarray(data_ex2.get(key_), dtype=int)
            error = np.mean(np.abs(data_ex1.get(key_)[:data_size, :] - data_ex2.get(key_)))
            errors.append(error)
        assert(np.sum(errors) < 1e-3)

def test_bench2_data_reproducibility():
    """This test aims at verifying if the same exact data could be reproduced after each FrameWork update
    BENCHMARK 2
    """
    benchmark2_ex1 = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                        benchmark_name="Benchmark2",
                                        load_data_set=True,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    benchmark2_ex2 = PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark2",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    data_size = int(2e3)
    benchmark2_ex2.generate(nb_sample_train=data_size,
                            nb_sample_val=data_size,
                            nb_sample_test=data_size,
                            nb_sample_test_ood_topo=data_size
                           )
    dataset_labels = ("train_dataset", "val_dataset", "_test_dataset", "_test_ood_topo_dataset")
    #data_ex1 = benchmark1_ex1.train_dataset.data
    #data_ex2 = benchmark1_ex2.train_dataset.data
    for label_ in dataset_labels:
        data_ex1 = getattr(benchmark2_ex1, label_).data
        data_ex2 = getattr(benchmark2_ex2, label_).data
        keys = data_ex1.keys()
        errors = list()
        for key_ in keys:
            print(key_)
            if key_ == "line_status" or key_ == "PV_nodes":
                data_ex1[key_] = np.asarray(data_ex1.get(key_), dtype=int)
                data_ex2[key_] = np.asarray(data_ex2.get(key_), dtype=int)
            error = np.mean(np.abs(data_ex1.get(key_)[:data_size, :] - data_ex2.get(key_)))
            errors.append(error)
        assert(np.sum(errors) < 1e-3)

def test_bench3_data_reproducibility():
    """This test aims at verifying if the same exact data could be reproduced after each FrameWork update
    BENCHMARK 3
    """
    benchmark3_ex1 = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                        benchmark_name="Benchmark3",
                                        load_data_set=True,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    benchmark3_ex2 = PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark3",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    data_size = int(1e2)
    benchmark3_ex2.generate(nb_sample_train=data_size,
                            nb_sample_val=data_size,
                            nb_sample_test=data_size,
                            nb_sample_test_ood_topo=data_size,
                            do_store_physics=True
                           )
    dataset_labels = ("train_dataset", "val_dataset", "_test_dataset", "_test_ood_topo_dataset")
    #data_ex1 = benchmark1_ex1.train_dataset.data
    #data_ex2 = benchmark1_ex2.train_dataset.data
    for label_ in dataset_labels:
        data_ex1 = getattr(benchmark3_ex1, label_).data
        data_ex2 = getattr(benchmark3_ex2, label_).data
        keys = data_ex1.keys()
        errors = list()
        for key_ in keys:
            print(key_)
            if key_ == "line_status" or key_ == "PV_nodes":
                data_ex1[key_] = np.asarray(data_ex1.get(key_), dtype=int)
                data_ex2[key_] = np.asarray(data_ex2.get(key_), dtype=int)
            error = np.mean(np.abs(data_ex1.get(key_)[:data_size, :] - data_ex2.get(key_)))
            errors.append(error)
        assert(np.sum(errors) < 1e-3)

def test_bench1_consistency():
    """Consistency check wrt. config

    This test verifies if generated variables and those indicated in config file are consistent.
    """
    benchmark1 = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                    benchmark_name="Benchmark1",
                                    load_data_set=True,
                                    config_path=CONFIG_PATH,
                                    log_path=LOG_PATH)
    config_keys = benchmark1.config.get_option("attr_x") + \
                  benchmark1.config.get_option("attr_tau") + \
                  benchmark1.config.get_option("attr_y")
    if benchmark1.config.get_option("attr_physics"):
        config_keys = config_keys + benchmark1.config.get_option("attr_physics")
    dataset_labels = ("train_dataset", "val_dataset", "_test_dataset", "_test_ood_topo_dataset")
    for label_ in dataset_labels:
        data = getattr(benchmark1, label_).data
        dataset_keys = tuple(data.keys())
        assert dataset_keys == config_keys

def test_bench2_consistency():
    """Consistency check wrt. config

    This test verifies if generated variables and those indicated in config file are consistent.
    """
    benchmark2 = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                    benchmark_name="Benchmark2",
                                    load_data_set=True,
                                    config_path=CONFIG_PATH,
                                    log_path=LOG_PATH)
    config_keys = benchmark2.config.get_option("attr_x") + \
                  benchmark2.config.get_option("attr_tau") + \
                  benchmark2.config.get_option("attr_y")
    if benchmark2.config.get_option("attr_physics"):
        config_keys = config_keys + benchmark2.config.get_option("attr_physics")
    dataset_labels = ("train_dataset", "val_dataset", "_test_dataset", "_test_ood_topo_dataset")
    dataset_labels = ("train_dataset", "val_dataset", "_test_dataset", "_test_ood_topo_dataset")
    for label_ in dataset_labels:
        data = getattr(benchmark2, label_).data
        dataset_keys = tuple(data.keys())
        assert dataset_keys == config_keys

def test_bench3_consistency():
    """Consistency check wrt. config

    This test verifies if generated variables and those indicated in config file are consistent.
    """
    benchmark3 = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                    benchmark_name="Benchmark3",
                                    load_data_set=True,
                                    config_path=CONFIG_PATH,
                                    log_path=LOG_PATH)
    config_keys = benchmark3.config.get_option("attr_x") + \
                  benchmark3.config.get_option("attr_tau") + \
                  benchmark3.config.get_option("attr_y")
    if benchmark3.config.get_option("attr_physics"):
        config_keys = config_keys + benchmark3.config.get_option("attr_physics")
    dataset_labels = ("train_dataset", "val_dataset", "_test_dataset", "_test_ood_topo_dataset")
    for label_ in dataset_labels:
        data = getattr(benchmark3, label_).data
        dataset_keys = tuple(data.keys())
        assert dataset_keys == config_keys

def test_power_grid_physics_informed_data():
    """This test aims at verifying that the generated data structure representing the physics (admittance matrix YBus, SBus vector)
    are consistent with the physical variables obtained
    """
    benchmark3= PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark3",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=None)

    data_size = int(10)
    benchmark3.generate(nb_sample_train=data_size,
                            nb_sample_val=1,
                            nb_sample_test=1,
                            nb_sample_test_ood_topo=1,
                            do_store_physics=True
                           )
    data = benchmark3.train_dataset.data

    #############"
    # we are going to check that V.(Ybus.V)* = Sbus_after_pf
    YBuses=data["YBus"]
    SBuses = data["SBus"]
    #pv_nodes = data["PV_nodes"]
    theta_ors=data["theta_or"]
    theta_exs = data["theta_ex"]
    v_ors=data["v_or"]
    v_exs = data["v_ex"]
    prod_qs=data["prod_q"]

    for ix in range(data_size):
        print(ix)
        obs = benchmark3.training_simulator._simulator.reset()
        obs.topo_vect = data["topo_vect"][ix]

        Ybus=YBuses[ix]
        SBus_init=SBuses[ix]
        theta_or=theta_ors[ix]
        theta_ex=theta_exs[ix]

        lines_or_pu_to_kv=benchmark3.training_simulator._simulator.backend.lines_or_pu_to_kv
        lines_ex_pu_to_kv=benchmark3.training_simulator._simulator.backend.lines_ex_pu_to_kv
        v_or=v_ors[ix]/lines_or_pu_to_kv
        v_ex=v_exs[ix]/lines_ex_pu_to_kv

        #get SBus after powerflow: reactive power of generators have been computed and should be added
        SBus_after_pf=SBus_init
        prod_bus, prod_conn = obs._get_bus_id(obs.gen_pos_topo_vect, obs.gen_to_subid)
        prod_q=prod_qs[ix]
        for i in range(len(prod_bus)):
            SBus_after_pf[prod_bus[i]] += 1j * prod_q[i] / obs._obs_env.backend._grid.get_sn_mva()

        ##############
        #we want to get V in complex form per bus_bar
        bus_theta = np.zeros(Ybus.shape[0])
        bus_v=np.zeros(Ybus.shape[0])

        lor_bus, lor_conn = obs._get_bus_id(
            obs.line_or_pos_topo_vect, obs.line_or_to_subid
        )
        lex_bus, lex_conn = obs._get_bus_id(
            obs.line_ex_pos_topo_vect, obs.line_ex_to_subid
        )

        bus_theta[lor_bus] = theta_or
        bus_theta[lex_bus] = theta_ex

        bus_v[lor_bus]=v_or
        bus_v[lex_bus]=v_ex
        bus_v_complex=np.array([bus_v[k]*exp(1j*(bus_theta[k]*2*pi)/360) for k in range(len(bus_v))])

        ####
        #we now compute V.(Ybus.V)*
        SBus_computed = np.conj(np.matmul(Ybus, bus_v_complex))*bus_v_complex
        #we check for all nodes execpt for the slack. For the slack, the active power has been adjusted to compensate for the losses, hence we should have adjusted for it in SBus_after_pf.
        #but we cannot recover the adjustment from the data we saved, so we ignore that index
        slack_id=prod_bus[-1]
        index_to_keep=np.ones(len(SBus_after_pf),dtype=bool)
        index_to_keep[slack_id]=False
        assert np.all(np.abs(np.round(SBus_after_pf - SBus_computed,2))[index_to_keep]<=0.01) #[pv_nodes]
        
if __name__ == "__main__":
    benchmark2_ex1 = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                        benchmark_name="Benchmark2",
                                        load_data_set=True,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)
    print(len(benchmark2_ex1.train_dataset))
    print(benchmark2_ex1.train_dataset[0])

    benchmark2_ex2 = PowerGridBenchmark(benchmark_path=None,
                                        benchmark_name="Benchmark2",
                                        load_data_set=False,
                                        config_path=CONFIG_PATH,
                                        log_path=LOG_PATH)

    data_size = int(2e3)
    benchmark2_ex2.generate(nb_sample_train=data_size,
                            nb_sample_val=data_size,
                            nb_sample_test=data_size,
                            nb_sample_test_ood_topo=data_size)
    
    print(benchmark2_ex2.train_dataset.size)
    print(benchmark2_ex2.train_dataset[0])
