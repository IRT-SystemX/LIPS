"""
Usage :
    Unitary testing of Physics Compliances
"""
import os
import pickle
import pathlib
import warnings
from pprint import pprint
from collections.abc import Iterable
import grid2op

from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.config import ConfigManager
from lips.metrics.power_grid import physics_compliances
from lips.metrics.power_grid.verify_voltage_equality import verify_voltage_at_bus
from lips.metrics.power_grid.global_conservation import global_conservation
from lips.metrics.power_grid.local_conservation import local_conservation
from lips.metrics.power_grid.ohm_law import verify_ohm_law
from lips.dataset.utils.powergrid_utils import get_kwargs_simulator_scenario

LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_case14_sandbox"
LOG_PATH = LIPS_PATH / "lips_logs.log"

def load_data():
    """
    load some data for test
    """
    # load some data
    # test data observations
    obs_path = pathlib.Path(__file__).parent.parent.absolute().joinpath("data", "ref_obs.pkl")
    # os.path.join(os.path.pardir, "data", "ref_obs.pkl")
    a_file = open(obs_path, "rb")
    observations = pickle.load(a_file)
    a_file.close()

    # predictions
    pred_path = pathlib.Path(__file__).parent.parent.absolute().joinpath("data", "predictions_FC_test.pkl")
    # os.path.join(os.path.pardir, "data", "predictions_FC_test.pkl")
    a_file = open(pred_path, "rb")
    predictions = pickle.load(a_file)
    a_file.close()

    return observations, predictions

def get_env(env_kwargs: dict):
    """Getter for the environment

    Parameters
    ----------
    env_kwargs : dict
        environment parameters

    Returns
    -------
    grid2op.Environment
        A grid2op environment with the given parameters
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = grid2op.make(**env_kwargs)
    # env.deactivate_forecast()
    return env


def test_loss_shapes():
    """
    test the loss equation verification function
    """
    observations, predictions = load_data()
    tolerance = 0.04
    verifications = physics_compliances.verify_loss(predictions, observations=observations, tolerance=tolerance)
    assert isinstance(verifications, dict)
    for key_, item_ in verifications.items():
        if isinstance(item_, Iterable):
            if key_ == "Law_values":
                assert item_.shape[0] == observations["prod_p"].shape[0]

def test_loss_function():
    """
    verify if the loss function is respected using real observations
    """
    observations, _ = load_data()
    tolerance = 1e-1
    verifications = physics_compliances.verify_loss(observations,
                                                    observations=observations,
                                                    tolerance=tolerance)
    assert verifications["violation_percentage"] < 0.1

def test_global_conservation_matrix_shapes():
    """
    test the law of conservation of energy function
    """
    observations, predictions = load_data()
    tolerance = 1e-3
    verifications = global_conservation(predictions,
                                        observations=observations,
                                        tolerance=tolerance)
    assert isinstance(verifications, dict)
    for _, item_ in verifications.items():
        if isinstance(item_, Iterable):
            assert item_.shape[0] == observations["prod_p"].shape[0]

def test_global_conservation():
    """
    verify if the loss function is respected using real observations
    """
    observations, _ = load_data()
    tolerance = 1e-3
    verifications = global_conservation(observations,
                                        observations=observations,
                                        tolerance=tolerance)
    assert verifications["violation_percentage"] == float(0)

def test_local_conservation():
    """verify if the local conservation is respected using real observations
    """
    benchmark2 = PowerGridBenchmark(benchmark_name="Benchmark2",
                                    benchmark_path=DATA_PATH,
                                    load_data_set=True,
                                    log_path=LOG_PATH,
                                    config_path=CONFIG_PATH
                                   )
    config = ConfigManager(section_name="Benchmark3", path=CONFIG_PATH)
    observations = benchmark2._test_dataset.data
    env = get_env(get_kwargs_simulator_scenario(config))
    verifications = local_conservation(predictions=observations,
                                       observations=observations,
                                       env=env,
                                       tolerance=1e-3
                                      )
    print(verifications["mape"])
    assert verifications["violation_percentage"] == float(0)

def test_verify_voltage_equality():
    """
    Verify if the voltage equality at bus is respected using real data
    """
    observations, _ = load_data()
    LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
    CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
    config = ConfigManager(section_name="Benchmark3", path=CONFIG_PATH)
    verifications = verify_voltage_at_bus(predictions=observations,
                                          observations=observations,
                                          config=config)
    # Assert that violation of voltage equality at bus is zero on simulation data
    assert verifications["prop_voltages_violation"] == 0
    assert verifications["prop_theta_violation"] == 0

def test_verify_ohm_law():
    """
    Verify if the Ohm law is respected with respect to the indicated threshold
    """
    benchmark3= PowerGridBenchmark(benchmark_path=DATA_PATH,
                                   benchmark_name="Benchmark3",
                                   load_data_set=True,
                                   config_path=CONFIG_PATH,
                                   log_path=LOG_PATH)

    data = benchmark3._test_dataset.data
    # env = benchmark3.training_simulator._simulator
    env = get_env(get_kwargs_simulator_scenario(benchmark3.config))

    verifications = verify_ohm_law(predictions=data, observations=data, env=env, result_level=0, tolerance=1e-4)#, tolerance=0.01)
    assert verifications["violation_prop_p_or"] < 1e-3
    assert verifications["violation_prop_p_ex"] < 1e-3
'''
if __name__ == "__main__":
    #test_evaluation()
    #test_evaluation_from_benchmark()
    """
    print("**********test_loss***************")
    test_loss_shapes()
    print("**********test_lce****************")
    test_lce()
    """
    #test_loss_function()
    #test_loss_function()
    #test_energy_conservation_function()
    # test verify voltage equality
    test_verify_voltage_equality()
'''
