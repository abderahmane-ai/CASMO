"""Compatibility shim for tooling that still invokes setup.py directly.

All packaging metadata lives in ``pyproject.toml`` — keep it there, not here, so
there is exactly one place to update.
"""

from setuptools import setup

setup()
