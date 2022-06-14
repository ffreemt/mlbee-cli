"""Refer to https://github.com/pyinstaller/pyinstaller/issues/2560.

Intended for nuitka freezing.
"""
from mlbee import __main__

if __name__ == "__main__":
    __main__.app()
