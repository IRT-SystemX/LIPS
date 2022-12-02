from lips.augmented_simulators.tensorflow_models import TfFullyConnected

import tensorflow as tf
from loguru import logger
from pathlib import Path

physical_devices = tf.config.list_physical_devices("GPU")
logger.info(f"Detected GPU devices: {physical_devices}")
for el in physical_devices:
    tf.config.experimental.set_memory_growth(el, True)


class BenchmarkedSimulator(TfFullyConnected):
    """
    This class implements a specific augmented simulator to be evaluated through the LIPS benchmarks.
    Its implementation shall follow a specific set of rules:
    - be a subclass of AugmentedSimulator
    - implements at least the functions train, predict, restore
    - the specific parameters of the simulator shall be specified in a dedicated file simulator_config.json
    """

    def __init__(
        self,
        name: str = "my_augmented_simulator",
        **kwargs,
    ):
        TfFullyConnected.__init__(
            self,
            name=name,
            sim_config_path=Path(__file__).parent / "simulator_archi.ini",
            sim_config_name="DEFAULT",
            **kwargs,
        )
