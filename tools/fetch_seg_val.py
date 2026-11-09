"""Launcher for ecs_hcm_model.eval.fetch; see that module for usage."""
import sys
from pathlib import Path

# Run from anywhere: the repo root holds both ecs_hcm_model/ and jetson/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecs_hcm_model.eval.fetch import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
