"""
Author: Lara Fadel
Date: 2024-12-19
MCGill Composite Center
Department of Chemical Engineering and Material Science, University of Southern California
Email: larafade@usc.edu

Provides functions to parse YAML configuration files.
"""

import yaml

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class ConfigObject:
    """
    A utility class to convert dictionaries to objects with attributes
    and back to dictionaries.
    """
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                value = ConfigObject(value)
            setattr(self, key, value)

    def to_dict(self):
        """
        Recursively converts ConfigObject to a dictionary.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObject):
                value = value.to_dict()  # Recursively convert nested ConfigObjects
            result[key] = value
        return result