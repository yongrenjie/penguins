"""
exportdeco.py
-------------

Provides a decorator to add functions, etc. to __all__.
The intention is that any function that is registered with @export will be
found in the top-level penguins namespace.

Taken directly from Aaron Hall's answer at
    https://stackoverflow.com/a/35710527/7115316
"""

import sys

def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn
