import sys
from pathlib import Path

from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import run_standard_eval


class StandardEvalArgs(Tap):
    config_file: str
    checkpoint_path: str | None = None
    output_dir: str | None = None


def main():
    args = StandardEvalArgs(
        description="Test the trained model on validation data."
    ).parse_args()
    run_standard_eval(args)


if __name__ == "__main__":
    main()
