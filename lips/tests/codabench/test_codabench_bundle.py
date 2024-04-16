import pathlib
import subprocess
from loguru import logger
from tempfile import TemporaryDirectory
from pathlib import Path
from lips import get_root_path

def test_ingestion_scoring_programs():
    LIPS_PATH = get_root_path(pathlib_format=True).parent
    dirname = Path(__file__).parent
    #dataset_dir = dirname / ".." / ".." / ".." / "reference_data"
    dataset_dir = dirname.parent / "data"
    benchmark_config = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
    #benchmark_config = dirname.parent / "configs" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
    benchmark_name = "Benchmark1"

    # creates temporary folder for output
    with TemporaryDirectory() as tempdir:
        logger.info(f"Temporary output dir: {tempdir}")

        # check ingestion behaviour
        logger.info(f"Checking ingestion program for {benchmark_name}...")
        ingestion_script = LIPS_PATH / "codabench" / "ingestion_program" / "ingestion.py"
        ans_ingestion = subprocess.call(
            ["python", ingestion_script,
             "--dataset_dir", dataset_dir,
             "--benchmark_config", benchmark_config,
             "--benchmark_name", benchmark_name,
             "--benchmark_output_dir", tempdir
             ]
        )

        # check that execution finished correctly
        assert ans_ingestion==0
        # check that output has been created
        expected_output = Path(tempdir + "/metrics.pkl")
        assert Path.exists(expected_output)

        # check scoring behaviour
        logger.info(f"Checking scoring program for {benchmark_name}...")
        scoring_script = LIPS_PATH / "codabench" / "scoring_program" / "score.py"
        ans_scoring = subprocess.call(
            ["python", scoring_script,
             "--benchmark_config", benchmark_config,
             "--benchmark_name", benchmark_name,
             "--benchmark_output_dir", tempdir,
             "--score_dir", tempdir
             ]
        )
        # check that execution finished correctly
        assert ans_scoring==0
        # check outputs
        results = Path(tempdir + "/scores_original_keys.json")
        assert Path.exists(results)
        assert Path.exists(Path(tempdir + "/keys.json"))
        assert Path.exists(Path(tempdir + "/scores.json"))
        assert Path.exists(Path(tempdir + "/detailed_scores.json"))
        assert Path.exists(Path(tempdir + "/detailed_results.html"))

if __name__ == "__main__":
    test_ingestion_scoring_programs()
