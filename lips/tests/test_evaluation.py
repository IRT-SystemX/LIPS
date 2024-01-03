import pathlib

import numpy as np

from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.augmented_simulators.tensorflow_models import TfFullyConnected
from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected
from lips.augmented_simulators.torch_simulator import TorchSimulator
from lips.dataset.scaler import StandardScaler


# indicate required paths
LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_case14_sandbox"
BENCH_CONFIG_PATH = LIPS_PATH / "lips" / "tests" / "configs" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
SIM_CONFIG_PATH = LIPS_PATH / "lips" / "tests" / "configs" / "powergrid" / "simulators"

benchmark1 = PowerGridBenchmark(benchmark_name="Benchmark1",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=None,
                                config_path=BENCH_CONFIG_PATH
                               )


def test_evaluation_criteria():
    """Test if the computed evaluation criteria are not overwritten by new execution of evaluation
    """
    torch_sim = TorchSimulator(name="torch_fc",
                               model=TorchFullyConnected,
                               scaler=StandardScaler,
                               log_path=None,
                               device="cpu",
                               seed=42,
                               bench_config_path=BENCH_CONFIG_PATH,
                               bench_config_name="Benchmark1",
                               sim_config_path=SIM_CONFIG_PATH / "torch_fc.ini",
                               sim_config_name="DEFAULT" # use the default set of hyper parameters
                               )
    torch_sim.train(benchmark1.train_dataset, benchmark1.val_dataset, save_path=None, epochs=2, train_batch_size=128)
    test_metrics = benchmark1.evaluate_simulator(augmented_simulator=torch_sim,
                                                 eval_batch_size=128,
                                                 dataset="test",
                                                 shuffle=False,
                                                 save_path=None,
                                                 save_predictions=False
                                                 )
    test_ood_metrics = benchmark1.evaluate_simulator(augmented_simulator=torch_sim,
                                                     eval_batch_size=128,
                                                     dataset="test_ood_topo",
                                                     shuffle=False,
                                                     save_path=None,
                                                     save_predictions=False
                                                     )
    assert np.not_equal(test_metrics["test"]["ML"]["MAPE_avg"]["a_or"], test_ood_metrics["test_ood_topo"]["ML"]["MAPE_avg"]["a_or"])
