import os
import sys
sys.path.insert(0, "../")

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)

from IPython.display import display
import pandas as pd
import warnings

import numpy as np
import grid2op
import copy

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', None)
import pathlib
from lips.augmented_simulators.tensorflow_models import TfFullyConnected#, TfFullyConnectedTopoEncoding
from lips.dataset.scaler import StandardScaler

from pprint import pprint
from matplotlib import pyplot as plt
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.utils import get_path
from lips.augmented_simulators.tensorflow_models import LeapNet
from lips.dataset.scaler import PowerGridScaler, StandardScaler
from sklearn.preprocessing import MinMaxScaler

from lips.config import ConfigManager

#from execution_jobs.utils import init_df_bench1, append_metrics_to_df_bench1, init_df_bench2, append_metrics_to_df_bench2, init_df_bench3, append_metrics_to_df_bench3, filter_bench1, filter_bench2_3

# indicate required paths
# indicate required paths
LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_case14_sandbox"
BENCH_CONFIG_PATH = LIPS_PATH / "lips" / "tests" / "configs" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
SIM_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "simulators"
BASELINES_PATH = LIPS_PATH / "trained_baselines" / "powergrid"
TRAINED_MODEL_PATH = LIPS_PATH / "trained_models" / "powergrid"
EVALUATION_PATH = LIPS_PATH / "evaluation_results" / "PowerGrid"
LOG_PATH = LIPS_PATH / "lips_logs.log"

benchmark1 = PowerGridBenchmark(benchmark_name="Benchmark1",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=BENCH_CONFIG_PATH
                               )

from lips.augmented_simulators.tensorflow_models import LeapNet

from lips.augmented_simulators.tensorflow_models import LeapNet

def test_fast_transform_tau():

    bench_config = ConfigManager(section_name="Benchmark1", path=BENCH_CONFIG_PATH)
    topo_actions = bench_config.get_option("dataset_create_params")["reference_args"]["topo_actions"]

    kwargs_tau = []
    for el in topo_actions:
         kwargs_tau.append(el["set_bus"]["substations_id"][0])

    leap_net1 = LeapNet(name="tf_leapnet",

                        bench_config_path=BENCH_CONFIG_PATH,
                        bench_config_name="Benchmark1",
                        sim_config_path=SIM_CONFIG_PATH / "tf_leapnet.ini",
                        sim_config_name="DEFAULT",
                        log_path=LOG_PATH,

                        loss={"name": "mse"},
                        lr=1e-4,
                        activation=tf.keras.layers.LeakyReLU(alpha=0.01),

                        sizes_enc=(),
                        sizes_main=(150, 150),
                        sizes_out=(),
                        topo_vect_to_tau="given_list",
                        is_topo_vect_input=False,
                        kwargs_tau=kwargs_tau,
                        layer="resnet",
                        attr_tau=("line_status","topo_vect"),
                        scale_main_layer=150,
                        # scale_input_enc_layer = 40,
                        #scale_input_dec_layer=200,
                        # topo_vect_as_input = True,
                        mult_by_zero_lines_pred=False,
                        topo_vect_as_input=True,
                        scaler=PowerGridScaler,

                        )

    ## add topo_vect (temporary ) in attr_x in benchmark config file


    ## add topo_vect (temporary ) in attr_x in benchmark config file
    ##############
    nb_timesteps_to_test=1000
    indices=[i for i in range(nb_timesteps_to_test)]
    for key in benchmark1.train_dataset.data.keys():
        benchmark1.train_dataset.data[key]=benchmark1.train_dataset.data[key][indices]

    benchmark1.train_dataset.size=len(indices)

    leap_net1._leap_net_model.max_row_training_set=len(indices)
    dataset=benchmark1.train_dataset
    obss = leap_net1._make_fake_obs(dataset)
    leap_net1._leap_net_model.init(obss)


    (extract_x, extract_tau), extract_y = leap_net1.scaler.fit_transform(dataset)

    #####
    #Launch two different mathods for transformation and check that they match
    tau = copy.deepcopy(extract_tau)
    import time
    start = time.time()
    extract_tau_1 = leap_net1._transform_tau(dataset, extract_tau)
    end = time.time()
    print(end - start) #3.9s pour 10 000 / Pour 100 000 35.63s

    start = time.time()
    extract_tau_bis = leap_net1._transform_tau_given_list(tau)
    end = time.time()
    print(end - start) #0.026s pour 10 000 => 150 fois plus rapide! Pour 100 000, 1,7s
    #0.021s with int32, could we only work with boolean ? 0.32s for 100 000 (or 0.82s with numpy matmult, a bit faster with tensorflow then)

    assert np.all(extract_tau_1[1].astype((np.bool_))==extract_tau_bis[1])
