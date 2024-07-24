import json
import os
from pathlib import Path
from easydict import EasyDict as edict


def get_config(model_name, dataset_name, config_file=""):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): Path to config file, if given an empty string, will use default config file.
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / dataset_name /"config.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_args = config_all['modelParams'].get(model_name, {})
    dataset_args = config_all['datasetParams']

    # update config
    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_args)
    
    config = edict(config)

    return config


