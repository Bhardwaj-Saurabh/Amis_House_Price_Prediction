import os
import sys
import yaml
from src.exception import CustomException


def read_yaml_file(file_path: str):
    """
    Read a YAML file and return its contents.
    Args:
        file_path (str): Path to the YAML file.
    Returns:
        dict: Data containing the contents of the YAML file.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)