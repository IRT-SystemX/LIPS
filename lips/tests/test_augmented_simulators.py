import pathlib

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


def test_train_tf_fc():
    """Test the fully connected architecture implemented using Tensorflow simulator
    """
    tf_fc = TfFullyConnected(name="tf_fc",
                            sim_config_path=SIM_CONFIG_PATH / "tf_fc.ini",
                            sim_config_name="DEFAULT",
                            bench_config_path=BENCH_CONFIG_PATH,
                            bench_config_name="Benchmark1",
                            scaler=StandardScaler,
                            log_path=None)
    
    tf_fc.train(train_dataset=benchmark1.train_dataset,
            val_dataset=benchmark1.val_dataset,
            epochs=2
           )

    assert tf_fc.trained is True

def test_train_torch_fc():
    """Test the fully connected architecture implemented using Torch simulator
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
                               sim_config_name="DEFAULT", # use the default set of hyper parameters
                               architecture_type="Classical"
                               )
    torch_sim.train(benchmark1.train_dataset, benchmark1.val_dataset, save_path=None, epochs=2, train_batch_size=128)
    assert torch_sim.trained is True
    