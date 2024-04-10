import importlib
import json
import os
import pickle
import resource
import sys
import threading
import time
from _thread import interrupt_main
from types import ModuleType

import yaml

from argparse import ArgumentParser, Namespace
from filelock import FileLock
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.config import ConfigManager
from loguru import logger
from pathlib import Path
from sys import path
from typing import Dict, Tuple, Union

from lips.augmented_simulators import AugmentedSimulator
from lips.benchmark import Benchmark
from lips.physical_simulator import PhysicsSolver

BENCHMARK_CONFIG = "conf.ini"
SUBMISSION_CONFIG = "simulator_config.json"
METRICS_FILE = "metrics.pkl"

PNEUMATICS_UC = "pneumatic"
POWERGRID_UC = "powergrid"

TRAINING_TIMEOUT = 600  # unit = seconds


def quit_function(fn_name):
    logger.error(f"{fn_name} function took too long. exiting...")
    interrupt_main()  # raises KeyboardInterrupt


def exit_after(s: int):
    """Use as decorator to exit process if function takes longer than s seconds"""

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


class ModelApiError(Exception):
    """Model api error"""

    def __init__(self, msg=""):
        self.msg = msg
        logger.error(msg)


class TimeoutException(Exception):
    """timeoutexception"""


def _here(*args):
    """Helper function for getting the current directory of this script."""
    return Path(__file__).resolve().parent


def _parse_args() -> Tuple[Namespace, Dict]:
    """Parse arguments and basic checks on them"""
    root_dir = _here(os.pardir).parent
    default_dataset_dir = root_dir / "dataset"
    default_output_dir = root_dir / "benchmark_output"
    default_config = root_dir / "conf.ini"
    default_submission_dir = root_dir / "basic_code_submission"
    parser = ArgumentParser()
    parser.add_argument(
        "--use_case",
        type=str,
        default=POWERGRID_UC,
        help=(f"Name of the use-case (allowed values are: {POWERGRID_UC}, {PNEUMATICS_UC})"),
    )
    parser.add_argument(
        "--benchmark_name", type=str, default="DefaultBenchmark", help="Name of the benchmark"
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=default_dataset_dir,
        help="Directory storing the dataset ",
    )
    parser.add_argument(
        "--benchmark_output_dir",
        type=Path,
        default=default_output_dir,
        help="Directory storing the predictions. It will "
        "contain e.g. [start.txt, predictions, end.yaml]"
        "when ingestion terminates.",
    )
    parser.add_argument(
        "--benchmark_config",
        type=Path,
        default=default_config,
        help="Path to file containing the benchmark configuration",
    )
    parser.add_argument(
        "--submission_dir",
        type=Path,
        default=default_submission_dir,
        help="Directory storing the submission code `my_augmented_simulator.py` and other necessary packages.",
    )

    args = parser.parse_args()
    if not args.benchmark_config.exists():
        raise ModelApiError(f"Configuration file {args.benchmark_config} is missing.")
    if not args.benchmark_config.exists():
        raise ModelApiError(f"Dataset dir {args.dataset_dir} not found.")

    logger.debug(f"Parsed args: {args}")
    logger.debug("-" * 50)

    return args


def _init_python_path(args: Namespace):
    """Adds to Python path the paths of the ingestion program and provided submission"""
    path.append(str(args.submission_dir))
    os.makedirs(args.benchmark_output_dir, exist_ok=True)


def _finalize(args: Namespace, ts: int):
    # Finishing ingestion program
    end_time = time.time()
    overall_time_spent = end_time - ts

    # Write overall_time_spent to a end.yaml file
    end_filename = "end.yaml"
    content = {
        "ingestion_duration": overall_time_spent,
        "end_time": end_time,
    }

    with open(args.benchmark_output_dir / end_filename, "w") as ftmp:
        yaml.dump(content, ftmp)
        logger.info(f"Wrote the file {end_filename} marking the end of ingestion.")

        logger.info("[+] Done. Ingestion program successfully terminated.")
        logger.info(f"[+] Overall time spent: {overall_time_spent} sec")

    logger.info("[Ingestion terminated]")


def _init_timer():
    ts = time.time()
    return ts


def write_start_file(output_dir: Path):
    """write start file"""
    start_filepath = output_dir / "start.txt"
    lockfile = output_dir / "start.txt.lock"
    ingestion_pid = os.getpid()

    with FileLock(lockfile):
        with open(start_filepath, "w") as ftmp:
            ftmp.write(str(ingestion_pid))

    logger.info(f'===== Finished writing "start.txt" file, PID is {ingestion_pid}.')


def initialize_benchmark(args: Namespace):
    if args.use_case is POWERGRID_UC:
        """Initializes a power grid benchmark"""
        env_name = ConfigManager(
            section_name=args.benchmark_name, path=args.benchmark_config
        ).get_option("env_name")

        benchmark = PowerGridBenchmark(
            benchmark_name=args.benchmark_name,
            benchmark_path=args.dataset_dir / args.use_case / env_name,
            config_path=args.benchmark_config,
            load_data_set=True,
        )
        logger.info(benchmark.config.get_options_dict())
        return benchmark
    else:
        logger.error(f"Use case {args.use_case} not implemented yet")
        exit(1)


def _define_simulator_bench_args(model_cls: ModuleType, args: Namespace) -> Dict:
    if issubclass(model_cls, AugmentedSimulator):
        return {"bench_config_name": args.benchmark_name, "bench_config_path": args.benchmark_config}
    else: # PhysicsSolver
        #TODO: in LIPS harmonize args names as much as possible
        return {"benchmark_name": args.benchmark_name, "config_path": args.benchmark_config}



def _check_simulator_method(model_cls: ModuleType):
    """Check if the user-provided simulator is compliant with the interface"""
    if not issubclass(model_cls, AugmentedSimulator) and not issubclass(model_cls, PhysicsSolver):
        raise ModelApiError("Your simulator class does not inherit from AugmentedSimulator or PhysicsSolver.")

    if issubclass(model_cls, AugmentedSimulator):
        for attr in ["restore", "train", "predict"]:
            # FIXME: to be improved with exact list of minimum methods to be implemented, not sure that train shall be mandatory
            if not hasattr(model_cls, attr):
                raise ModelApiError(f"Your simulator object doesn't have the method {attr}")
    elif issubclass(model_cls, PhysicsSolver):
        for attr in ["compute"]:
            if not hasattr(model_cls, attr):
                raise ModelApiError(f"Your solver object doesn't have the method {attr}")
    logger.info("Provided implementation of AugmentedSimulator is compliant with the interface.")


def init_simulator(args: Namespace) -> Tuple[Union[AugmentedSimulator,PhysicsSolver], Dict]:
    """Initialize custom user-defined augmented simulator"""
    # check that the file simulator_config exists and loads it
    path_simulator_config = args.submission_dir / SUBMISSION_CONFIG
    logger.debug(f"Path to simulator config: {path_simulator_config}")
    if not path_simulator_config.exists():
        raise ModelApiError(f"Unable to find {SUBMISSION_CONFIG} in the provided submission")
    simulator_config = json.loads(path_simulator_config.read_bytes())

    # Import user-provided simulator code
    simulator_class = importlib.import_module("my_augmented_simulator").BenchmarkedSimulator
    # Check validity of simulator class
    _check_simulator_method(simulator_class)

    # loads constructor parameters for config file (to adapt to custom parameters) and instantiate simulator
    # with those parameters
    kwargs = simulator_config["simulator_config"]
    logger.debug(f"Simulator config: {kwargs}")
    # and pass to the simulator constructor the variables used as input/output for the benchmark
    # this is done using the name of the benchmark and associated configuration  file

    vars = _define_simulator_bench_args(simulator_class, args)

    # FIXME: tweak to handle specific case of the scaler parameter: in the config, we use a "scaler_config" parameter (string) which is turned into a "scaler" class parameter
    if kwargs and kwargs.get("scaler_class"):
        module = importlib.import_module("lips.dataset.scaler")
        class_ = getattr(module, kwargs["scaler_class"])
        vars["scaler"] = class_
    logger.debug(f"Simulator vars: {vars}")

    # Instantiate simulator
    simulator = simulator_class(**kwargs, **vars)

    logger.info(f"[+] Simulator instantiated: {simulator.name}.")

    return simulator, simulator_config


@exit_after(TRAINING_TIMEOUT)
def train_simulator(simulator: AugmentedSimulator, benchmark: Benchmark, config: Dict):
    """Train the simulator (if required) based on the parameters provided in the submission configuration file"""
    logger.info("Training simulator...")
    # retrieves specific parameters for training
    training_config = config["simulator_training_config"]
    logger.debug(f"Training config: {training_config}")
    if not training_config:
        raise RuntimeError("Impossible to train augmented simulator, no config.")

    # FIXME: handle the case of non basic type in the training_config (such as layer=Dense)
    simulator.train(
        train_dataset=benchmark.train_dataset, val_dataset=benchmark.val_dataset, **training_config
    )

    logger.info("[+] Training done.")
    return


def create_simulator(args: Namespace, benchmark: Benchmark) -> Union[AugmentedSimulator, PhysicsSolver]:
    """Create the simulator using the benchmark and provided configuration"""
    simulator, simulator_config = init_simulator(args)

    if isinstance(simulator, AugmentedSimulator):
        pre_trained_path = args.submission_dir / simulator_config.get("trained_model_path")
        if simulator_config and simulator_config.get("requires_training"):
            requires_training = simulator_config["requires_training"]
        else:
            requires_training = False

        if (pre_trained_path / simulator.name).exists():
            # load simulator from pre-trained data
            logger.info(f"Loading Augmented Simulator from {pre_trained_path}")
            # simulator.load_metadata(pre_trained_path)
            # simulator.init()
            simulator.restore(pre_trained_path)
            logger.info(f"[+] Successfully loaded Augmented Simulator from {pre_trained_path}")

        elif requires_training:
            # TODO: maybe add a timer / budget for training
            train_simulator(simulator, benchmark, simulator_config)

            logger.info(f"Saving augmented simulator to {pre_trained_path}")
            # simulator.save_metadata(pre_trained_path)
            simulator.save(pre_trained_path)
            logger.info(f"[+] Augmented Simulator saved to {pre_trained_path}")

    return simulator


def compute_metrics(benchmark: PowerGridBenchmark, simulator: AugmentedSimulator) -> Dict:
    """use the benchmark class to compute the metrics"""
    metrics_per_dataset = benchmark.evaluate_simulator(
        augmented_simulator=simulator, eval_batch_size=128, dataset="all", shuffle=False
    )

    # some logs
    logger.debug(f"Output metric: {metrics_per_dataset['test']['ML']}")
    return metrics_per_dataset


def write_metrics(metrics: Dict, path_dir: Path):
    """Write metrics file (as a pickle because some types such as float32 does not support direct json conversion."""
    metrics_file = path_dir / METRICS_FILE
    with open(metrics_file, "wb") as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Metrics written to: {metrics_file}")


def is_compatible(
    benchmark: PowerGridBenchmark, simulator: AugmentedSimulator, args: Namespace
) -> bool:
    benchmark_y = list(benchmark.config.get_option("attr_y"))
    benchmark_y.sort()
    simulator_y = list(simulator.bench_config.get_option("attr_y"))
    simulator_y.sort()

    if benchmark_y == simulator_y:
        logger.info(f"[+] Benchmark and simulator have the same attr_y: {simulator_y}")
        return True
    elif set(benchmark_y).issubset(simulator_y):
        # FIXME: this should be ok in theory, but generate an error in practice.
        logger.warning(
            f"[!] Simulator has more attr_y attributes ({simulator_y}) than the benchmark data ({benchmark_y})"
        )
        return False
    else:
        logger.error(
            f"[-] Incompatible attr_y attributes between the simulator ({simulator_y}) and the benchmark data ({benchmark_y})"
        )
        return False


def main():
    """Main entry"""
    logger.info("===== Start ingestion program.")
    # Parse directories from input arguments
    logger.info("===== Initialize args.")
    args = _parse_args()
    _init_python_path(args)

    write_start_file(args.benchmark_output_dir)

    logger.info("===== Load data.")

    logger.info("===== Initialize benchmark.")
    benchmark = initialize_benchmark(args)

    logger.info("===== Create simulator.")
    ts = _init_timer()
    simulator: Union[AugmentedSimulator,PhysicsSolver] = create_simulator(args, benchmark)

    if not is_compatible(benchmark, simulator, args):
        raise ValueError(
            "Problem of compatibility between simulator and benchmark data: not the same y!"
        )

    logger.info("===== Compute output metrics.")
    metrics = compute_metrics(benchmark, simulator)
    # write metrics
    write_metrics(metrics, args.benchmark_output_dir)

    logger.info("===== Finalization.")
    _finalize(args, ts)


def memory_limit(max_memory_in_gb: int):
    """Allocates a maximum amount of memory for the process, in GB"""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    logger.debug(f"Available memory: {get_memory()/1024/1024} GB")
    max_allocated_memory = max_memory_in_gb * 1024 * 1024 * 1024
    logger.debug(f"Max allocated memory: {max_memory_in_gb} GB")
    resource.setrlimit(resource.RLIMIT_AS, (max_allocated_memory, hard))


def get_memory():
    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory


if __name__ == "__main__":
    # memory_limit(2)
    try:
        main()
    except MemoryError:
        logger.error("Memory Exception")
        sys.exit(1)
