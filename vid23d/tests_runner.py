from __future__ import annotations

import importlib.util
import sys
import unittest


def main(argv: list[str] | None = None) -> int:
    # Gate test functionality behind the [test] extra by checking for pytest
    if importlib.util.find_spec("pytest") is None:
        print("Tests are disabled. Install with extras: 'pip install vid23d[test]' or 'pip install .[test]'.")
        return 2

    loader = unittest.defaultTestLoader
    suite = loader.discover("vid23d.tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
