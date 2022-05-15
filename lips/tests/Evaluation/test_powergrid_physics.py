"""
Usage :
    Unitary testing of Physics Compliances
"""
import os
import pickle
import pathlib
from pprint import pprint
from collections.abc import Iterable

from lips.config import ConfigManager
from lips.metrics.power_grid import physics_compliances
from lips.metrics.power_grid.verify_voltage_equality import verify_voltage_at_bus


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

def test_lce_shapes():
    """
    test the law of conservation of energy function
    """
    observations, predictions = load_data()
    tolerance = 1e-3
    verifications = physics_compliances.verify_energy_conservation(predictions,
                                                                   observations=observations,
                                                                   tolerance=tolerance)
    assert isinstance(verifications, dict)
    for _, item_ in verifications.items():
        if isinstance(item_, Iterable):
            assert item_.shape[0] == observations["prod_p"].shape[0]

def test_energy_conservation_function():
    """
    verify if the loss function is respected using real observations
    """
    observations, _ = load_data()
    tolerance = 1e-3
    verifications = physics_compliances.verify_energy_conservation(observations,
                                                                   observations=observations,
                                                                   tolerance=tolerance)
    assert verifications["violation_percentage"] == float(0)

def test_verify_voltage_equality():
    """
    Verify if the voltage equality at bus is respected using real data
    """
    observations, _ = load_data()
    LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
    CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
    config = ConfigManager(section_name="Benchmark3", path=CONFIG_PATH)
    voltages, thetas = verify_voltage_at_bus(predictions=observations,
                                             observations=observations,
                                             config=config)
    # Assert that violation of voltage equality at bus is zero on simulation data
    assert voltages[-1] == 0
    assert thetas[-1] == 0

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
