import re
import os
import grid2op
from grid2op.Parameters import Parameters
from grid2op.Chronics import GridStateFromFile
from grid2op.Chronics import GridStateFromFileWithForecasts
from lightsim2grid import LightSimBackend
from grid2op.Action import DontAct
from grid2op.Action import PlayableAction


def test_env_idf():

	env_params = {
		"NO_OVERFLOW_DISCONNECTION": True,
		"MAX_LINE_STATUS_CHANGED": 999999,
		"MAX_SUB_CHANGED": 999999,
		"NB_TIMESTEP_COOLDOWN_LINE": 0,
		"NB_TIMESTEP_COOLDOWN_SUB": 0}

	parameters = Parameters()
	parameters.init_from_dict(env_params)
	BkCls = LightSimBackend

	env = grid2op.make("l2rpn_idf_2023",
					param=parameters,
					backend=BkCls(),
					action_class=PlayableAction,
					opponent_init_budget=0,
					opponent_action_class=DontAct,
					data_feeding_kwargs={"gridvalueClass": GridStateFromFileWithForecasts})

	env.deactivate_forecast()

	chronics = {
		"train": ".*_[0-6]$",#6 chronics per month for 12 months
		"val": ".*_[7-9]",#3 chronics per month for 12 months
		"test": ".*_1[0-2]",#3 chronics per month for 12 months
		"test_ood": ".*_1[3-5]"#3 chronics per month  for 12 months
		}

	# chronics = {
	# 	"train": "^((?!(.*9[0-9][0-9].*)).)*$", # # i use 994 chronics out of the 904 for training
	# 	"val": ".*9[0-4][0-9].*", # i use 50 full chronics for validation
	# 	"test": ".*9[5-9][0-4].*", # i use 25 full chronics for testing
	# 	"test_ood": ".*9[5-9][5-9].*" # i use 25 full chronics for testing
	# }

	# special case of the grid2Op environment: data are read from chronics that should be part of the dataset
	# here i keep only certain chronics for the training, and the other for the test
	chronics_selected_regex = re.compile(chronics.get("test_ood"))
	env.chronics_handler.set_filter(lambda path:
									re.match(chronics_selected_regex, path) is not None)
	env.chronics_handler.real_data.reset()
	env.set_id(0)

	assert(env.env_name == "l2rpn_idf_2023")
	#print(len(os.listdir(env.chronics_handler.path)))



	# from lips.physical_simulator import Grid2opSimulator
	# from lips.dataset.utils.powergrid_utils import get_kwargs_simulator_scenario
	# from lips import get_root_path
	# from lips.config import ConfigManager

	# LIPS_PATH = get_root_path(pathlib_format=True).parent#pathlib.Path().resolve().parent
	# # CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_idf_2023.ini"
	# CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"

	# config = ConfigManager(path=CONFIG_PATH, section_name="DEFAULT")
	# print(config)
	# training_simulator = Grid2opSimulator(get_kwargs_simulator_scenario(config),
	#                                       initial_chronics_id=0,
	#                                       chronics_selected_regex=chronics["train"])


if __name__ == "__main__":
	test_env_idf()
