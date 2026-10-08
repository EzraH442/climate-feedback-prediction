import sys
from pathlib import Path

from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import run_ensemble_eval


class EnsembleEvalArgs(Tap):
    config_file: str
    checkpoint_path: list[str] | None = None
    seeds: list[str] | None = None
    output_dir: str | None = None

    def process_args(self):
        if self.seeds and self.checkpoint_path:
            self.error("--seeds and --checkpoint_path are mutually exclusive")


def main():
    args = EnsembleEvalArgs(
        description="Test the trained model on validation data."
    ).parse_args()
    run_ensemble_eval(args)


if __name__ == "__main__":
    main()
