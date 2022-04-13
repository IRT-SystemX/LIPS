__all__ = ["TfFullyConnected"]

from lips.augmented_simulators.tensorflow_models.fully_connected import TfFullyConnected

try:
    from lips.augmented_simulators.tensorflow_models.leap_net import LeapNet
    __all__.append("LeapNet")
except ImportError as err:
    pass