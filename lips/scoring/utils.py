# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import json
import logging
from typing import Union, Dict, Any, List


def read_json(json_path: str = "", json_object: Union[Dict, str, None] = None) -> Any:
    """Reads a JSON file from the specified path or a JSON object.

    Args:
        json_path: Path to the JSON file. If empty, `json_object` is used.
        json_object: A JSON object (as a dict or a JSON string).
                     Used if `json_path` is empty.

    Returns:
        Parsed JSON data as a Python object.

    Raises:
        ValueError: If both `json_path` and `json_object` are empty or
                    if `json_object` is not a valid type.
        FileNotFoundError: If the file specified by `json_path`
                           does not exist.
        json.JSONDecodeError: If the JSON data is invalid.
    """
    if json_path:
        try:
            with open(json_path, "r") as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"JSON file not found at path: {json_path}") from e
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in file: {json_path}", e.doc, e.pos) from e
    elif json_object:
        if isinstance(json_object, str):
            try:
                return json.loads(json_object)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError("Invalid JSON string provided.", e.doc, e.pos) from e
        elif isinstance(json_object, dict):
            return json_object
        else:
            raise ValueError("`json_object` must be a valid JSON string or dictionary.")
    else:
        raise ValueError("Both `json_path` and `json_object` are empty. Provide at least one.")


def get_nested_value(data: Dict, keys: List[str]) -> Any:
    """Retrieves a nested value from a dictionary using a list of keys.

    Args:
        data: The dictionary to retrieve the value from.
        keys: A list of keys representing the path to the nested value.

    Returns:
        The nested value if found, otherwise None.
    """
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            logging.warning(f"Path '{keys}' not found in data. Returning None.")
            return None
    return data


def filter_metrics(data: Dict, metrics: List[str]) -> Dict:
    """Filters a dictionary to include only specified keys.

    Args:
        data: The dictionary to filter.
        metrics: A list of keys to keep in the filtered dictionary.

    Returns:
        A new dictionary containing only the specified keys and their values.
    """
    return {key: value for key, value in data.items() if key in metrics}