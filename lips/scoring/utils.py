import json
from typing import Union, Dict


def read_json(json_path: str = "", json_object: Union[Dict, str, None] = None):
    """
        Reads a JSON file from the specified path or a JSON object if the path is empty.

        Args:
            json_path (str): Path to the JSON file. If empty, the json_object is used.
            json_object (Union[Dict, str, None]): A JSON object (as a dict or a JSON string). Used if json_path is empty.

        Returns:
            Any: Parsed JSON data as a Python object (dict, list, etc.).

        Raises:
            ValueError: If both json_path and json_object are empty.
            FileNotFoundError: If the json_path does not exist.
            json.JSONDecodeError: If the JSON data is invalid.
        """
    if json_path:
        # Read from JSON file
        try:
            with open(json_path, "r") as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"JSON file not found at path: {json_path}") from e
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in file: {json_path}", e.doc, e.pos)
    elif json_object:
        # Read from provided JSON object
        if isinstance(json_object, str):
            try:
                return json.loads(json_object)  # Parse JSON string
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError("Invalid JSON string provided.", e.doc, e.pos)
        elif isinstance(json_object, dict):
            return json_object  # Already a parsed dictionary
        else:
            raise ValueError("json_metrics must be a valid JSON string or dictionary.")
    else:
        raise ValueError("Both json_path and json_object are empty. Provide at least one.")
