import os
import yaml
from src.datascience import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path 
from typing import Any
from box.exceptions import BoxValueError

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """ Read yaml file and returns
    Args:
        Path_to_yaml (str): path like input
    
    Raise: 
        ValueError: if yaml file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox tyep
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            logger.info((f"yaml file : {path_to_yaml} loaded successfully"))
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    create list of directories

    Args:
        path_to_directories (list) : list of path of directories
    """

    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Args:
        path (path) : Path to json file
        data (dict) : data to be save in json file
    """

    with open(path,"w") as f:
        json.dump(data,f,indent=4)
    
    logger.info(f"json file saved at {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """

    Load json file
    Args:
        path (path) : Path to json file
    Returns:
        ConfigBox : data as class attributes instead of dict
    """

    with open(path) as f:
        content=json.load(f)
    
    logger.info(f"json file loaded successfully from {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any,path: Path):
    """
    Save binary file
    Args:
        path (path) : Path to binary file
        data (any) : data to be save as binary
    """

    joblib.dump(value=data,filename=path)
    
    logger.info(f"binary file saved at {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    load binary file
    Args:
        path (path) : Path to binary file
    Returns:
        Any: object  stored in the file
    """
    data=joblib.load(path)
    
    logger.info(f"binary file saved at {path}")
    return data
