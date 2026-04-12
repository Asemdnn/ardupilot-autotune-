from pathlib import Path
import sys


APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import parameters  # noqa: E402


def test_defaults_are_within_declared_ranges():
    for category in parameters.ARDUPILOT_PARAMETERS.values():
        for name, info in category["parameters"].items():
            low, high = info["range"]
            default = info["default"]
            assert low <= default <= high, f"{name} default {default} outside [{low}, {high}]"
