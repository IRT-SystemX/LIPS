from typing import List, Dict

from lips.scoring import PowerGridScoring
from lips.scoring import utils



metrics = utils.read_json(json_path="/mnt/seif/HSA/LIPS/score_scripts/PowerGrid_score/res/json_metrics.json")

# thresholds={"a_or":(0.02,0.05,"min"),
#             "a_ex":(0.02,0.05,"min"),
#             "p_or":(0.02,0.05,"min"),
#             "p_ex":(0.02,0.05,"min"),
#             "v_or":(0.2,0.5,"min"),
#             "v_ex":(0.2,0.5,"min"),
#             "CURRENT_POS":(1., 5.,"min"),
#             "VOLTAGE_POS":(1.,5.,"min"),
#             "LOSS_POS":(1.,5.,"min"),
#             "DISC_LINES":(1.,5.,"min"),
#             "CHECK_LOSS":(1.,5.,"min"),
#             "CHECK_GC":(0.05,0.10,"min"),
#             "CHECK_LC":(0.05,0.10,"min"),
#             "CHECK_JOULE_LAW":(1.,5.,"min")
#            }.keys()


thresholds={"x-velocity":(0.01,0.02,"min"),
            "y-velocity":(0.01,0.02,"min"),
            "pressure":(0.002,0.01,"min"),
            "pressure_surfacic":(0.008,0.02,"min"),
            "turbulent_viscosity":(0.05,0.1,"min"),
            "mean_relative_drag":(0.4,5.0,"min"),
            "mean_relative_lift":(0.1,0.3,"min"),
            "spearman_correlation_drag":(0.8,0.9,"max"),
            "spearman_correlation_lift":(0.96,0.99,"max")
    }.keys()
import json

def transform_json_generic(
    input_json,
    used_metric_list,
    ml_metric,
    in_distrubution_key_path,
    ood_key_path,
    inference_time
):
    """
    Transforms a JSON structure into a desired format based on selected metrics and paths.

    Parameters:
    - input_json (dict): The original JSON data.
    - used_metric_list (list): List of metrics to include in the output (e.g., ["a_or", "a_ex"]).
    - ml_metric (str): Metric key to use in the ML section.
    - in_distribution_key_path (list): Path to the in-distribution (test) section.
    - ood_key_path (list): Path to the OOD test section.
    - inference_time (float): The inference time value.

    Returns:
    - dict: Transformed JSON in the desired format.
    """
    def get_nested_value(data, keys):
        """Retrieve a nested value from a dictionary using a list of keys."""
        for key in keys:
            if key not in data:
                return None
            data = data[key]
        return data

    def filter_metrics(data, metric_list):
        """Filter the data dictionary to include only the specified metrics."""
        return {key: value for key, value in data.items() if key in metric_list}

    def transform_section(section, metric_list, ml_metric):
        """Transform a section (e.g., in-distribution or OOD) based on the desired metrics."""
        transformed = {}
        for subsection, data in section.items():
            if subsection == "ML":
                # Process ML section with the chosen ml_metric
                transformed["ML"] = filter_metrics(data.get(ml_metric, {}), metric_list)
            elif subsection == "Physics":
                # Process Physics section with the chosen metrics
                transformed["Physics"] = filter_metrics(data, metric_list)
            else:
                # Process other subsections if needed
                transformed[subsection] = filter_metrics(data, metric_list)
        return transformed

    # Extract in-distribution section
    in_distribution_section = get_nested_value(input_json, in_distrubution_key_path)
    if not in_distribution_section:
        raise ValueError(f"In-distribution section not found at path: {in_distrubution_key_path}")

    in_distribution_transformed = transform_section(in_distribution_section, used_metric_list, ml_metric)

    # Extract OOD section
    ood_section = get_nested_value(input_json, ood_key_path)
    if not ood_section:
        raise ValueError(f"OOD section not found at path: {ood_key_path}")

    ood_transformed = transform_section(ood_section, used_metric_list, ml_metric)

    # Combine into the desired format
    transformed_json = {
        **in_distribution_transformed,
        "Speed": {"inference_time": inference_time},
        "OOD": ood_transformed,
    }

    return transformed_json

# Example usage:

# used_metric_list = thresholds
# ml_metric = "MSE_avg"
# in_distrubution_key_path = ["test"]
# ood_key_path = ["test_ood_topo"]
# inference_time = 4.163906573085114
#
# transformed_json = transform_json_generic(
#     input_json=metrics,
#     used_metric_list=used_metric_list,
#     ml_metric=ml_metric,
#     in_distrubution_key_path=in_distrubution_key_path,
#     ood_key_path=ood_key_path,
#     inference_time=inference_time
# )

used_metric_list = thresholds

from lips.scoring.utils import filter_metrics, get_nested_value


def construct_ml_metrics(input_json, ml_key_path, used_metric_list):
    """
    Construct ML metrics by retrieving and filtering data from the given JSON.

    Parameters:
    - input_json (dict): The input JSON containing the ML metrics.
    - ml_key_path (list): Path to the ML section in the JSON as a list of keys.
    - used_metric_list (list): List of metrics to include in the output.

    Returns:
    - dict: Filtered ML metrics containing only the specified metrics.
    """
    all_ml_metrics = get_nested_value(input_json, ml_key_path)
    if all_ml_metrics is None:
        raise ValueError(f"Invalid path {ml_key_path}. Could not retrieve ML metrics.")

    if not isinstance(all_ml_metrics, dict):
        raise TypeError(f"Expected a dictionary at {ml_key_path}, but got {type(all_ml_metrics).__name__}.")

    return {"ML" : filter_metrics(all_ml_metrics, used_metric_list)}


def construct_speed_metric(input_json, speed_key_path):
    return {"Speed": {"inference_time": get_nested_value(input_json, speed_key_path)}}

print(construct_speed_metric(metrics, ['test', 'ML', 'TIME_INF']))