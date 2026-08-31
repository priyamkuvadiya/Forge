"""Put the repo root on sys.path so `import task_suite` works from anywhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
