import sys
from pathlib import Path

from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import run_shakirova_eval

from constants import SHAKIROVA_NORTH_BOUNDARY


class ShakirovaEvalArgs(Tap):
    config_file: str = "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clear.yaml"
    checkpoint_path: str | None = None
    output_dir: str | None = None
    base_date: str = "1992-09"
    perturbed_date: str = "2012-09"
    data_path: str = "data/era5"
    north_boundary: float = SHAKIROVA_NORTH_BOUNDARY
    skip_input_anomaly_plots: bool = False
    overwrite_responses: bool = False


def main():
    args = ShakirovaEvalArgs(
        description="Run one two-date radiative closure experiment."
    ).parse_args()
    run_shakirova_eval(args)


if __name__ == "__main__":
    main()
