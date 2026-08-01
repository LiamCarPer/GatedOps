"""Allow ``python -m gatedops`` to behave like the ``gatedops`` command."""

import sys

from gatedops.cli import main

if __name__ == "__main__":
    sys.exit(main())
