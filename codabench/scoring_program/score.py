import hashlib
import json
import numpy
import os
import pickle
import time

import yaml

from collections.abc import MutableMapping
from argparse import ArgumentParser, Namespace
from json2table import convert
from lips.config import ConfigManager
from loguru import logger
from pathlib import Path
from typing import Dict, List

PNEUMATICS_UC = "pneumatic"
POWERGRID_UC = "powergrid"

METRICS_FILE = "metrics.pkl"
SCORES_FILE = "scores.json"  # name is enforced by Codabench
SCORES_FILE_ORIG_KEYS = "scores_original_keys.json"  # name is enforced by Codabench
KEYS_FILE = "keys.json"  # mapping of keys (readable / encoded)
SCORES_FILE_DETAILED = "detailed_scores.json"
DETAILED_RESULTS_HTML = "detailed_results.html"  # name is enforced by Codabench
INGESTION_END_FILE = "end.yaml"
TIMEOUT = 60000

VAL = "val"
TEST = "test"
OOD = "test_ood_topo"

ML = "ML"
PHYSICS = "Physics"
INDRED = "IndRed"

EXCLUDED_VARIABLES_PATTERN = "_ex"  # to avoid p_or and p_ex... (too many variables)


class IngestionError(Exception):
    """Ingestion error"""

    def __init__(self, msg=""):
        self.msg = msg
        logger.error(msg)


class ScoringError(Exception):
    """scoring error"""

    def __init__(self, msg=""):
        self.msg = msg
        logger.error(msg)


def _here(*args):
    """Helper function for getting the current directory of this script."""
    return Path(__file__).resolve().parent


def _parse_args() -> Namespace:
    root_dir = _here().parent
    default_score_dir = root_dir / "scoring_output"
    default_output_dir = root_dir / "benchmark_output"
    default_benchmark_config = root_dir / "GridBenchmark_conf.ini"
    parser = ArgumentParser()
    parser.add_argument(
        "--use_case",
        type=Path,
        default=POWERGRID_UC,
        help=(f"Name of the use-case (allowed values are: {POWERGRID_UC}, {PNEUMATICS_UC})"),
    )
    parser.add_argument(
        "--benchmark_output_dir",
        type=Path,
        default=default_output_dir,
        help=("Directory storing the results."),
    )
    parser.add_argument(
        "--score_dir",
        type=Path,
        default=default_score_dir,
        help=(
            "Directory storing the scoring output e.g. `scores.json` and `detailed_results.html`."
        ),
    )
    parser.add_argument(
        "--benchmark_name", type=str, default="DefaultBenchmark", help="Name of the benchmark"
    )
    parser.add_argument(
        "--benchmark_config",
        type=Path,
        default=default_benchmark_config,
        help="Config of the benchmark (metrics and output variables).",
    )

    args = parser.parse_args()
    logger.debug(f"Parsed args are: {args}")
    logger.debug("-" * 50)
    logger.debug(f"Using output_dir: {args.benchmark_output_dir}")
    logger.debug(f"Using score_dir: {args.score_dir}")
    return args


def _init_scores_html(args: Namespace):
    detailed_results_filepath = args.score_dir / DETAILED_RESULTS_HTML

    html_head = '<html><head> <meta http-equiv="refresh" content="5"> </head><body><pre>'
    html_end = "</pre></body></html>"
    with open(detailed_results_filepath, "a") as html_file:
        html_file.write(html_head)
        html_file.write("Starting training process... <br> Please be patient. ")
        html_file.write(html_end)


def _init(args: Namespace):
    if not args.benchmark_config.exists():
        raise ScoringError(f"Configuration file {args.benchmark_config} is missing!")
    if not args.score_dir.exists():
        args.score_dir.mkdir(parents=True)
    # Initialize detailed_results.html
    # _init_scores_html(args)


def check_ingestion_end(output_dir: Path):
    """Check that ingestion is finished; we assume that ingestion and scoring may be run at the same time """
    endfile = output_dir / INGESTION_END_FILE

    waiting_time = 0
    while not endfile.is_file():
        time.sleep(1)
        waiting_time += 1
        if waiting_time > TIMEOUT:
            raise IngestionError(
                f"[-] Failed: scoring didn't detected the end of ingestion after {TIMEOUT} seconds."
            )


def is_process_alive(ingestion_pid: int) -> bool:
    """Detect ingestion alive"""
    try:
        os.kill(ingestion_pid, 0)
    except OSError:
        return False
    else:
        return True


def get_ingestion_info(output_dir: Path) -> Dict:
    """Get ingestion information"""
    endfile_path = output_dir / "end.yaml"
    if not endfile_path.is_file():
        raise IngestionError("[-] No end.yaml exist, ingestion failed. Scoring impossible.")
    logger.info("[+] Detected end.yaml file, get ingestion information")
    with open(endfile_path, "r") as ftmp:
        ingestion_info = yaml.safe_load(ftmp)
    return ingestion_info


def get_results_info(args: Namespace) -> Dict:
    """Get results information"""
    metrics_file = args.benchmark_output_dir / METRICS_FILE
    if not metrics_file.is_file():
        raise IngestionError(
            f"[-] No {metrics_file} exist, something went wrong in ingestion. Scoring impossible."
        )

    metrics = load_metrics(metrics_file)
    logger.info(f"[+] Metrics results {metrics_file} correctly parsed.")
    return metrics


def _finalize(scoring_start: float):
    """Finalize the scoring"""
    duration = time.time() - scoring_start
    logger.info("[+] Successfully finished scoring! " f"Scoring duration: {duration:.2f} sec. ")
    logger.info("[Scoring terminated]")


def _flatten(d: Dict, parent_key: str = "", sep: str = "__") -> Dict:
    """Flatten a dict, compress keys"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _keep_simple_types(d: Dict) -> Dict:

    items = []
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            items.append((k, _keep_simple_types(v)))
        else:
            if not isinstance(v, list):
                if isinstance(v, numpy.float32):
                    # convert to native type for json serialization
                    items.append(((k, v.item())))
                else:
                    items.append((k, v))
    return dict(items)


def load_metrics(metrics_file: Path) -> Dict:
    """Load metrics file, previously created by the """
    with open(metrics_file, "rb") as handle:
        d = pickle.load(handle)
    return d


def logging_scores_html(json_file: Path, detailed_results_filepath: Path):
    """Log the scores as an HTML table"""
    json_object = json.loads(json_file.read_bytes())
    html_head = (
        "<html><head>"
        '<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">'
        "</head><body><pre>"
    )
    html_end = "</pre></body></html>"
    with open(detailed_results_filepath, "w") as html_file:
        html_file.write(html_head)
        build_direction = "LEFT_TO_RIGHT"
        table_attributes = {
            "style": "width:20%",
            "class": "w3-table-all w3-hoverable w3-small",
            "border": 0,
        }
        html_file.write(
            convert(json_object, build_direction=build_direction, table_attributes=table_attributes)
        )
        html_file.write(html_end)


def logging_score(detailed_score: Dict, duration: float, args: Namespace):
    """Log the scores"""
    # TODO: include timestamp in filmename?
    detailed_scoring_file = args.score_dir / SCORES_FILE_DETAILED
    json_str = json.dumps(detailed_score, indent=4) + "\n"
    detailed_scoring_file.write_text(json_str, encoding="utf-8")
    logger.info(f"Detailed scoring data written to: {detailed_scoring_file}")

    # NB. score.json keys shall match what is described in the competition.yaml datafile
    scoring_file = args.score_dir / SCORES_FILE
    scoring_file_orig_keys = args.score_dir / SCORES_FILE_ORIG_KEYS
    keys_file = args.score_dir / KEYS_FILE
    flatten_dict_orig = _flatten(detailed_score)
    flatten_dict_orig["duration"] = duration

    # FIXME: in codabench, the lengths ok keys is limited to 36 characters, when we flatten the keys may be longer
    # encode keys...
    encoded_keys = {k: hashlib.md5(k.encode()).hexdigest() for k in flatten_dict_orig.keys()}
    keys_str = json.dumps(encoded_keys, indent=4) + "\n"
    keys_file.write_text(keys_str, encoding="utf-8")
    json_str_orig = json.dumps(flatten_dict_orig, indent=4) + "\n"
    scoring_file_orig_keys.write_text(json_str_orig, encoding="utf-8")

    flatten_dict = {encoded_keys[k]: v for k, v in flatten_dict_orig.items()}

    json_str = json.dumps(flatten_dict, indent=4) + "\n"
    scoring_file.write_text(json_str, encoding="utf-8")
    logger.info(f"Scoring data written to: {scoring_file}")
    logger.debug(f"Metrics (flatten): {flatten_dict}")

    detailed_results_filepath = args.score_dir / DETAILED_RESULTS_HTML
    logging_scores_html(detailed_scoring_file, detailed_results_filepath)
    logger.info(f"Detailed scoring data written to: {detailed_results_filepath}")


def _filter_relevant_quantities(d: Dict, relevant_physical_quantities: List[str]):
    return {k: v for k, v in d.items() if k in relevant_physical_quantities}


def _sort_by_types(cfg, metrics):
    complex_types = [m_type for m_type in cfg if isinstance(metrics[m_type], dict)]
    simple_types = [m_type for m_type in cfg if not isinstance(metrics[m_type], dict)]
    return complex_types, simple_types


def _update_dict(type, metrics, cfg: ConfigManager):
    metrics_cfg = cfg.get_option("eval_dict")
    relevant_physical_quantities = [
        v for v in cfg.get_option("attr_y") if EXCLUDED_VARIABLES_PATTERN not in v
    ]

    d = {}

    # ML_metrics
    # filter ML metrics keeping only relevant physical quantities if there are in the results
    dict_ml, simple_ml = _sort_by_types(metrics_cfg[ML], metrics[type][ML])
    d[ML] = {
        m_type: dict(
            filter(
                lambda k: k[0] in relevant_physical_quantities, metrics[type][ML][m_type].items(),
            )
        )
        for m_type in dict_ml
    }
    for m_type in simple_ml:
        d[ML][m_type] = metrics[type][ML][m_type]

    # Physics - keep only simple types (aka synthetic results)
    dict_phy, simple_phy = _sort_by_types(metrics_cfg[PHYSICS], metrics[type][PHYSICS])
    physics_res = _keep_simple_types(metrics[type][PHYSICS])
    d[PHYSICS] = {
        p: _filter_relevant_quantities(physics_res.get(p), relevant_physical_quantities)
        for p in dict_phy
        if physics_res.get(p)
    }
    for m_type in simple_phy:
        d[PHYSICS][m_type] = metrics[type][PHYSICS][m_type]

    # IndRed - keep only simple types (aka synthetic results)
    indred_res = _keep_simple_types(metrics[type][INDRED])
    dict_ind, simple_ind = _sort_by_types(metrics_cfg[INDRED], metrics[type][INDRED])
    d[INDRED] = {
        p: _filter_relevant_quantities(indred_res.get(p), relevant_physical_quantities)
        for p in dict_ind
        if indred_res.get(p)
    }
    for m_type in simple_ind:
        d[INDRED][m_type] = metrics[type][INDRED][m_type]

    return d


def process_metrics(metrics: Dict, args: Namespace) -> Dict:
    """
    Process the metrics file by keeping only the most relevant data w.r.t. the benchmark and
    flatten the results for easier display on Codabench (the official dict results can only contain key: value, value
    being a simple type)
    :param metrics: Raw metrics file
    :param args: configuration parameters
    :return: a Dict (flattened)
    """
    config = ConfigManager(section_name=args.benchmark_name, path=args.benchmark_config)
    logger.debug(
        f"Evaluation criteria for {args.benchmark_name}: {config.get_options_dict()['eval_dict']}"
    )

    score_test = _update_dict(TEST, metrics, config)
    score_ood = _update_dict(OOD, metrics, config)

    return {"test": score_test, "ood": score_ood}


def main():
    """main entry"""

    # Important note about mounted volumes on Cadabench:
    # the ingestion program scores output data in /app/output which is then available as /app/input/res for the scoring program

    scoring_start = time.time()
    logger.info("===== Init scoring program")
    args = _parse_args()
    _init(args)

    logger.info("===== Wait for the exit of ingestion.")
    check_ingestion_end(args.benchmark_output_dir)

    # Compute/write score
    logger.info("===== Processing ingestion results.")
    ingestion_info = get_ingestion_info(args.benchmark_output_dir)
    duration = ingestion_info["ingestion_duration"]
    metrics = get_results_info(args)
    detailed_score = process_metrics(metrics, args)

    # Add results to file
    logger.info("===== Logging score results.")
    logging_score(detailed_score, duration, args)
    _finalize(scoring_start)


if __name__ == "__main__":
    main()
