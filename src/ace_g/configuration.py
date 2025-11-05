# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Configuration module for ACE-G.

Configuration for ACE-G is done through a combination of yoco and dacite.
YOCO is used to merge yaml files and command-like arguments into a dictionary.
This is done as the first step in each executable script.

Dacite is used to convert the dictionary into a dataclass which is then passed to the actual function / class to be
configured. This ensures the configuration is complete, merged with the defaults, and some type conversions can be
performed automatically (e.g., str to pathlib.Path).
Furthermore, docstring, type and defaults are defined in one place, i.e., the dataclass definition.

Each script produces yaml files with the configuration used to produce the results. This script should be consumable
by the same script to reproduce the result and by other scripts to use the resulting model / map for further processing.
This is facilitated by the GlobalConfig dataclass which is used to store all parameters involved in the process so far.
"""
import dataclasses
import pathlib
from typing import Type, TypeVar

import dacite
import yoco


@dataclasses.dataclass(kw_only=True)
class GlobalConfig:
    """Global configuration keeping track of all parameters involved.

    All scripts should produce a yaml file with GlobalConfig allowing reproducibility and downstream consumption.
    The global configuration can be used to track all parameters involved to produce a set of results, i.e.,
     - single-scene training
     - registration
     - evaluation
     - results
    """

    sst: dict | None = None  # single-scene training configuration
    reg: dict | None = None  # registration configuration
    eva: dict | None = None  # evaluation configuration
    res: dict | None = None  # results


ConfigType = TypeVar("ConfigType", bound=GlobalConfig)


def fromdict(config_type: Type[ConfigType], config_dict: dict, defaults_key: str | None = None) -> ConfigType:
    """Convert dictionary to GlobalConfig.

    Args:
        config_type: Type of the config.
        config_dict: Dictionary to convert.
        defaults_key: Key of the defaults to load.
    """
    if defaults_key is not None and defaults_key in config_dict:
        config_dict = yoco.load_config(config_dict, config_dict[defaults_key])
        config_dict[defaults_key] = None

    return dacite.from_dict(config_type, config_dict, config=dacite.Config(cast=[pathlib.Path]))  # type: ignore


def asdict(config: GlobalConfig, remove_global_config: bool) -> dict:
    """Convert GlobalConfig to dictionary.

    Args:
        config: GlobalConfig instance.
        remove_global_config: If True mst, sst, reg, eva keys are removed from the dictionary.
    """
    dictionary = dataclasses.asdict(config)
    if remove_global_config:
        for key in GlobalConfig.__annotations__.keys():
            dictionary.pop(key, None)
    return dictionary
