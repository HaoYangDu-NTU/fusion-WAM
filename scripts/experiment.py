#!/usr/bin/env python
"""Local experiment entry point; never import a sibling source checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwam.experiments import main

if __name__ == "__main__":
    raise SystemExit(main())
