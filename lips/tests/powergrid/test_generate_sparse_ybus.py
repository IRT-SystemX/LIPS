import numpy as np
from scipy.sparse import csr_matrix
from lips import get_root_path
from lips.benchmark.powergridBenchmark import PowerGridBenchmark


LIPS_PATH = get_root_path(pathlib_format=True).parent
CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_idf_2023.ini"
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_idf_2023"
LOG_PATH = LIPS_PATH / "lips_logs.log"

def test_generate_data_sparse():
    """
    verify that the generation with ybus as sparse works correctly
    """
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(mode=511, parents=False)

    benchmark_competition = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                               benchmark_name="Benchmark_competition",
                                               load_data_set=False,
                                               config_path=CONFIG_PATH,
                                               log_path=LOG_PATH)
    
    benchmark_competition.generate(nb_sample_train=int(1e2),
                                   nb_sample_val=int(1e2),
                                   nb_sample_test=int(1e2),
                                   nb_sample_test_ood_topo=int(1e2),
                                   do_store_physics=True,
                                   store_as_sparse=True)
    assert(benchmark_competition.train_dataset.data["YBus"].shape[0] == 1e2)
    assert(isinstance(benchmark_competition.train_dataset.data["YBus"], csr_matrix))


def test_load_generated_sparse_data():
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(mode=511, parents=False)

    benchmark_competition = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                               benchmark_name="Benchmark_competition",
                                               load_data_set=False,
                                               config_path=CONFIG_PATH,
                                               log_path=LOG_PATH)
    
    benchmark_competition.generate(nb_sample_train=int(1e2),
                                   nb_sample_val=int(1e2),
                                   nb_sample_test=int(1e2),
                                   nb_sample_test_ood_topo=int(1e2),
                                   do_store_physics=True,
                                   store_as_sparse=True)
    
    benchmark_competition = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                               benchmark_name="Benchmark_competition",
                                               load_data_set=True,
                                               load_ybus_as_sparse=True,
                                               config_path=CONFIG_PATH,
                                               log_path=LOG_PATH)

    assert(benchmark_competition.train_dataset.data["YBus"].shape[0] == 1e2)
    assert(isinstance(benchmark_competition.train_dataset.data["YBus"], csr_matrix))

def test_equality_sparse_and_np_array():
    benchmark_competition_sparse = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                                      benchmark_name="Benchmark_competition",
                                                      load_data_set=False,
                                                      config_path=CONFIG_PATH,
                                                      log_path=LOG_PATH)
    
    benchmark_competition_sparse.generate(nb_sample_train=int(1e2),
                                          nb_sample_val=int(1e1),
                                          nb_sample_test=int(1e1),
                                          nb_sample_test_ood_topo=int(1e1),
                                          do_store_physics=True,
                                          store_as_sparse=True)
    
    benchmark_competition = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                               benchmark_name="Benchmark_competition",
                                               load_data_set=False,
                                               config_path=CONFIG_PATH,
                                               log_path=LOG_PATH)
    
    benchmark_competition.generate(nb_sample_train=int(1e2),
                                   nb_sample_val=int(1e1),
                                   nb_sample_test=int(1e1),
                                   nb_sample_test_ood_topo=int(1e1),
                                   do_store_physics=True,
                                   store_as_sparse=False)
    n_bus = benchmark_competition.training_simulator._simulator.n_sub * 2

    for i in range(benchmark_competition.train_dataset.size):
        assert(np.array_equal(benchmark_competition.train_dataset.data["YBus"][i],
                              benchmark_competition_sparse.train_dataset.data["YBus"][i].toarray().reshape(n_bus, n_bus)))
    

if __name__ == "__main__":
    test_equality_sparse_and_np_array()
                                  