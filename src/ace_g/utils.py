# Copyright © Niantic Spatial, Inc. 2025. All rights reserved.
"""Utility functions."""

import argparse
import collections
import copy
import datetime
import inspect
import os
import pathlib
import pickle
import pydoc
import random
import time
from threading import Condition, Lock
from typing import Any, Optional

import filelock
import numpy as np
import torch
import torch.nn.functional as F
from ruamel import yaml as _yaml

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\x1b[31;20m"
RESET = "\x1b[0m"


def set_seed(seed) -> None:
    """Seed all sources of randomness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def augment_features(features: torch.Tensor, drop: float, std: float) -> torch.Tensor:
    """Augment features with diffferent types of noise.

    Args:
        features: Features to augment. Any shape.
        drop: Probability of dropping a feature.
        std: Standard deviation of Gaussian noise to add to each feature.

    Returns:
        Augmented features. Same shape as features.
    """
    if drop > 0:
        features = F.dropout(features, p=drop, training=True)

    if std > 0:
        features += torch.randn_like(features) * std

    return features


def average_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics from multiple dictionaries.

    Returns:
        A dictionary with the same keys as all input dictionaries combined, with values being the mean of all values.
    """
    total_dict = collections.defaultdict(float)
    counter_dict = collections.defaultdict(int)
    for metric_dict in metric_dicts:
        for k, v in metric_dict.items():
            total_dict[k] += v
            counter_dict[k] += 1
    return {k: v / counter_dict[k] for k, v in total_dict.items()}


def primitive(obj: Any) -> Any:
    """Recurse object and convert all non-primitive types to strings."""
    obj = copy.deepcopy(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = primitive(value)
    elif isinstance(obj, list):
        obj = [primitive(value) for value in obj]
    elif not isinstance(obj, (int, float, bool, str, dict, list, tuple, type(None))):
        obj = str(obj)
    return obj
    # for key, value in dictionary.items():
    #     if isinstance(value, pathlib.Path):
    #         dictionary[key] = str(value)
    #     if not isinstance(value, (int, float, bool, str, dict, list, tuple, type(None))):
    #         dictionary[key] = str(value)
    #     if isinstance(value, list):
    #         dictionary[key] = [
    #             str(x) if not isinstance(x, (int, float, bool, str, dict, list, tuple, type(None))) else x
    #             for x in value
    #         ]
    # return dictionary


def load_yaml(file_path: str | pathlib.Path, typ=None, pkl_cache=False) -> Any:
    """Load a yaml file.

    Args:
        file_path: Path to the yaml file.
        typ: Type of yaml loader.
        pkl_cache:
            If True, cache the loaded object using pickle and use the cache if the file has not changed.
            Useful to speed up subsequent loading of large yaml files that are not frequently modified.
    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    pkl_file_path = None
    if pkl_cache:
        abs_path = file_path.absolute()
        pkl_file_path = (pathlib.Path(".yaml_cache") / str(abs_path).replace(os.sep, "-")).with_suffix( ".pkl")
        with filelock.FileLock(pkl_file_path.with_suffix(".lock")):
            if pkl_file_path.exists() and pkl_file_path.stat().st_mtime >= file_path.stat().st_mtime:
                with open(pkl_file_path, "rb") as file:
                    return pickle.load(file)

    yaml = _yaml.YAML(typ=typ)
    with open(file_path, "r") as file:
        data = yaml.load(file)

    if pkl_file_path is not None:
        with filelock.FileLock(pkl_file_path.with_suffix(".lock")):
            with open(pkl_file_path, "wb") as file:
                pickle.dump(data, file)

            os.utime(pkl_file_path, (file_path.stat().st_atime, file_path.stat().st_mtime))  # copy mtime of yaml file

    return data


def save_yaml(object_: Any, file_path: pathlib.Path, typ=None) -> Any:
    """Save object as yaml file.

    Note that roundtrips might fail if type for loading and saving is not the same.
    """
    yaml = _yaml.YAML(typ=typ)
    yaml.representer.ignore_aliases = lambda *data: True
    with open(file_path, "w", encoding="utf-8") as file:
        return yaml.dump(object_, file)


def get_pixel_grid(subsampling_factor: int) -> torch.Tensor:
    """Generate target pixel positions according to a subsampling factor, assuming prediction at center pixel."""
    pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
    yy, xx = torch.meshgrid(pix_range, pix_range, indexing="ij")
    return subsampling_factor * (torch.stack([xx, yy]) + 0.5)


def to_homogeneous(input_tensor, dim=1):
    """Converts tensor to homogeneous coordinates by adding ones to the specified dimension."""
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output


def generate_session_id(
    scene_name: Optional[str] = None,
    split_name: Optional[str] = None,
    encoder_type: Optional[str] = None,
    head_type: Optional[str] = None,
) -> str:
    """Generate a unique session ID for the current run.

    Two runs with same milliseconds and arguments would have the same session ID (guessing this won't be a problem).
    """
    session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]

    if scene_name is not None:
        session_id = f"{session_id}_{scene_name}"
    if split_name is not None:
        session_id = f"{session_id}_{split_name}"
    if encoder_type is not None:
        session_id = f"{session_id}_{encoder_type}"
    if head_type is not None:
        session_id = f"{session_id}_{head_type}"

    return session_id


class Nop:
    """Class that does nothing. Useful for disabling objects without extra wrapping.

    Modified from https://stackoverflow.com/a/24946360.
    """

    def nop(self, *args, **kwargs):
        pass

    def __getattr__(self, _: str) -> Any:
        return self.nop

    def __getitem__(self, _: Any) -> Any:
        pass

    def __setitem__(self, _: Any, __: Any) -> None:
        pass


class MRSWLock:
    """Multi-threading safe write-preferring multi-reader single-writer lock.

    Reference: https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock
    """

    def __init__(self):
        self._cv = Condition(Lock())
        self._num_readers = 0
        self._num_pending_writers = 0
        self._writer_active = False

    def acquire_read(self, blocking=True):
        with self._cv:
            while self._writer_active or self._num_pending_writers > 0:
                if not blocking:
                    return False
                self._cv.wait()
            self._num_readers += 1
        return True

    def release_read(self):
        with self._cv:
            self._num_readers -= 1
            if self._num_readers < 0:
                raise RuntimeError("no active readers to release")
            if self._num_readers == 0:
                self._cv.notify_all()

    def acquire_write(self, blocking=True):
        with self._cv:
            self._num_pending_writers += 1
            while self._writer_active or self._num_readers > 0:
                if not blocking:
                    self._num_pending_writers -= 1
                    return False
                self._cv.wait()
            self._num_pending_writers -= 1
            self._writer_active = True
        return True

    def release_write(self):
        with self._cv:
            if not self._writer_active:
                raise RuntimeError("no active writer to release")
            self._writer_active = False
            self._cv.notify_all()


class MRSWDict:
    """Multi-threading safe dictionary that allows multiple readers and a single writer independently for each key.

    Note that set, get, del, etc. are not thread-safe, and must be manually protected by the user via the
    acquire/release methods if needed.

    Each key has its own write-preferring MRSW lock (see MRSWLock).
    """

    def __init__(self):
        self._locks = {}
        self._dict = {}

    def set_key(self, key, value):
        """Set the value for a given key."""
        if key not in self._locks:
            self._locks[key] = MRSWLock()
        self._dict[key] = value

    def has_key(self, key):
        """Check if the dictionary has a given key."""
        return key in self._dict

    def get_key(self, key):
        """Return the value for a given key."""
        return self._dict[key]

    def del_key(self, key):
        """Delete a key from the dictionary."""
        del self._dict[key]
        del self._locks[key]

    def len(self) -> int:
        """Return the number of keys in the dictionary.

        Not using __len__ because it would need to be manually exposed when registering the object.
        See multiprocessing.managers.BaseManager.register.
        """
        return len(self._dict)

    def keys(self) -> list:
        """Return the keys in the dictionary."""
        return list(self._dict.keys())

    def acquire_read(self, key, blocking=True) -> bool:
        try:
            return self._locks[key].acquire_read(blocking=blocking)
        except KeyError:
            return False

    def release_read(self, key):
        self._locks[key].release_read()

    def acquire_write(self, key, blocking=True) -> bool:
        if key not in self._locks:
            self._locks[key] = MRSWLock()
        return self._locks[key].acquire_write(blocking=blocking)

    def release_write(self, key):
        self._locks[key].release_write()


class InfiniteSampler:
    """Infinite sampler that keeps sampling from a data source without replacement."""

    def __init__(
        self,
        data_source,
        batch_size: int | None = None,
        drop_last: bool = False,
        circular: bool = False,
    ):
        """Initialize the sampler.

        If batch_size is None, elements are returned. Otherwise, elements are returned as lists of size batch_size.
        If data_source is a tensor, it is indexed along the first dimension, and a tensor instead of a list is returned.

        Args:
            data_source: List or tensor of elements to sample from.
            batch_size: Number of elements to return in each batch. If None, elements are returned individually.
            drop_last: If True, the sampler will drop incomplete last batches. Not used if circular is True.
            circular:
                If True, the sampler will always return the same number of elements per batch by cycling through the
                shuffled data source. Note that this can lead to repeated elements in the same batch.
                Only used if batch_size is not None. N
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.circular = circular

        assert len(self.data_source) > 0, "data_source must have at least one element"

        if self.batch_size is not None and self.drop_last and not self.circular:
            assert (
                len(self.data_source) >= self.batch_size
            ), f"data_source must have at least batch_size elements; got {data_source=} and {batch_size=}"

        if self.circular:
            assert not isinstance(self.data_source, torch.Tensor), "circular is not supported for tensor data sources"

    def __iter__(self):
        while True:
            if isinstance(self.data_source, torch.Tensor):
                permutation = torch.randperm(len(self.data_source))
                if self.batch_size is None:
                    for i in permutation:
                        yield self.data_source[i]
                else:
                    full_batches = len(self.data_source) // self.batch_size
                    for i in range(full_batches):
                        indices = permutation[i * self.batch_size : (i + 1) * self.batch_size]
                        yield self.data_source[indices]
                    if not self.drop_last:
                        indices = permutation[full_batches * self.batch_size :]
                        yield self.data_source[indices]
            else:
                random.shuffle(self.data_source)
                if self.batch_size is None:
                    for data in self.data_source:
                        yield data
                elif self.circular:
                    next_batch = []
                    index = 0
                    while True:
                        remaining = self.batch_size - len(next_batch)
                        end_index = index + remaining
                        next_batch.extend(self.data_source[index:end_index])
                        index = end_index
                        if len(next_batch) == self.batch_size:
                            yield next_batch
                            next_batch = []
                        if index >= len(self.data_source):
                            index = 0
                            random.shuffle(self.data_source)
                else:
                    full_batches = len(self.data_source) // self.batch_size
                    for i in range(full_batches):
                        yield self.data_source[i * self.batch_size : (i + 1) * self.batch_size]
                    if not self.drop_last and len(self.data_source) % self.batch_size != 0:
                        yield list(self.data_source[full_batches * self.batch_size :])


class ConfigArgumentParser(argparse.ArgumentParser):
    """Argument parser that supports additional config argument expecting an optional yaml file.

    The configuration from the yaml file will be merged with the command line arguments and practically act similar to
    overwriting default values in a normal argparse.ArgumentParser.

    That is, each argument is resolved in the following order:
        1. Default value from the argument parser.
        2. Value from the configuration file.
        3. Command line argument value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--config", type=none_or_path, help="Path to a yaml file with configuration.")

    def parse_args(self, args=None, namespace=None):
        args_with_defaults = super().parse_args(args)

        config_namespace = argparse.Namespace()
        if args_with_defaults.config is not None:
            config = load_yaml(args_with_defaults.config)

            # Apply type conversion to the config arguments
            for key, value in config.items():
                if isinstance(value, list) or isinstance(value, dict):
                    continue
                argument = f"--{key}"
                if argument in self._option_string_actions:
                    config[key] = self._option_string_actions[argument].type(str(value))
                else:
                    msg = f"unrecognized config argument: {key}"
                    self.error(msg)

            config_namespace = argparse.Namespace(**config)

        return super().parse_args(args, config_namespace)


def stack_and_pad(
    tensors: list[torch.Tensor], pad_dim: int, pad_to_len: int | None = None, stack_dim: int = 0
) -> torch.Tensor:
    """Stack and pad a list of tensors along a new dimension.

    All dimensions except the padding dimension must match.

    Args:
        tensors: List of tensors to stack and pad.
        pad_dim: Dimension to pad along. Dimension is prior to stacking.
        pad_to_shape: Shape to pad the tensors to.
        stack_dim: New dimension to stack the tensors along.

    Returns:
        Stacked and padded tensor.
    """
    if pad_to_len is None:
        pad_to_len = max(tensor.shape[pad_dim] for tensor in tensors)

    padded_tensors = [
        torch.cat(
            (
                tensor,
                tensor.new_zeros(
                    *tensor.shape[:pad_dim], pad_to_len - tensor.shape[pad_dim], *tensor.shape[pad_dim + 1 :]
                ),
            ),
            dim=pad_dim,
        )
        for tensor in tensors
    ]

    return torch.stack(padded_tensors, dim=stack_dim)


class FileSwitch:
    """Context manager that only enters if a file exists.

    Allows to throttle the file checking frequency and caches the file existence status for that duration.
    """

    _last_check_time_dict = {}
    _exists_dict = {}

    def __init__(self, file_path: pathlib.Path | str, check_interval: float = 1.0):
        self.file_path = pathlib.Path(file_path)
        self.check_interval = check_interval

    def __bool__(self) -> bool:
        if (
            self.file_path not in self._last_check_time_dict
            or self.file_path not in self._exists_dict
            or time.time() - self._last_check_time_dict[self.file_path] > self.check_interval
        ):
            self._exists_dict[self.file_path] = self.file_path.exists()
            self._last_check_time_dict[self.file_path] = time.time()

        return self._exists_dict[self.file_path]


def str_to_object(name: str) -> object | None:
    """Try to find object with a given name.

    First scope of calling function is checked for the name, then current environment
    (in which case name has to be a fully qualified name). In the second case, the
    object is imported if found.

    Args:
        name: Name of the object to resolve.

    Returns:
        The object which the provided name refers to. None if no object was found.
    """
    # check callers local variables
    caller_locals = inspect.currentframe().f_back.f_locals  # type: ignore
    if name in caller_locals:
        return caller_locals[name]

    # check callers global variables (i.e., imported modules etc.)
    caller_globals = inspect.currentframe().f_back.f_globals  # type: ignore
    if name in caller_globals:
        return caller_globals[name]

    # check environment
    return pydoc.locate(name)