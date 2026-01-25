"""
Script to print all the available environments in LocoLab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `locolab` extension. They start
with `OmniBot` in their name.
"""

import importlib
import pathlib
import pkgutil
import sys


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror=None,
):
    """Yields ModuleInfo for all modules recursively on path."""

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True

    for info in pkgutil.iter_modules(path, prefix):
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path = getattr(sys.modules[info.name], "__path__", None) or []
                path = [p for p in path if not seen(p)]
                yield from _walk_packages(path, info.name + ".", onerror)


def import_packages():
    # Import the locolab tasks module to register all environments
    try:
        import locolab.source.locomotion.tasks  # noqa: F401
    except ImportError:
        # Fallback: try to import from local path
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        try:
            import source.locomotion.tasks  # noqa: F401
        except ImportError:
            pass
        finally:
            sys.path.pop(0)


import_packages()

"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable


def main():
    """Print all environments registered in `locolab` extension."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in LocoLab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "OmniBot" in task_spec.id and "Isaac" not in task_spec.id:
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
