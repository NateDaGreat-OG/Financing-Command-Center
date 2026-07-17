"""Compatibility aliases for importing the project package from the repo root."""
from importlib import import_module
import sys


for _module_name in ("backtest", "config", "core", "rl", "services", "strategies"):
    sys.modules.setdefault(_module_name, import_module(f".{_module_name}", __name__))
