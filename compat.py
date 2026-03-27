"""Runtime compatibility checks for supported Python versions."""

import sys


MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 11)


def ensure_supported_python(min_version=MIN_PYTHON, max_version=MAX_PYTHON) -> None:
    """Raise RuntimeError if interpreter is outside supported range."""
    current = sys.version_info[:3]
    if current < (*min_version, 0) or current > (*max_version, 999):
        min_str = f"{min_version[0]}.{min_version[1]}"
        max_str = f"{max_version[0]}.{max_version[1]}"
        cur_str = f"{current[0]}.{current[1]}.{current[2]}"
        raise RuntimeError(
            "Unsupported Python version: "
            f"{cur_str}. Supported versions are Python {min_str} to {max_str}."
        )
