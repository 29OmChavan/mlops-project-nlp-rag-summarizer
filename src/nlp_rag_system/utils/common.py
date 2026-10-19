"""
Common Utilities for YAML, JSON, and Directory Management in Project 03.
"""
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List
from src.nlp_rag_system.logger import logger
from src.nlp_rag_system.exception import CustomException


def read_yaml(path_to_yaml: Path) -> Dict[str, Any]:
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return content or {}
    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_to_directories: List[Any], verbose: bool = True) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e, sys)
