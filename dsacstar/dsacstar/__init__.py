import importlib
import torch as _torch  # ensure libtorch libraries are loaded before the extension

_C = importlib.import_module("dsacstar._C")

# Re-export all public attributes from the C++ extension
__all__ = [name for name in dir(_C) if not name.startswith("_")]
for name in __all__:
    globals()[name] = getattr(_C, name)
