from pathlib import Path
from loguru import logger
import sys

# Ensure project root (src) on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.generated_runner import run_generated_directory


if __name__ == "__main__":
    data_path = ROOT / ".." / "data" / "synthetic_business_data.csv"
    code_dir = ROOT / "generated_code"
    out_dir = ROOT / "generated_results"
    logger.info("Running generated analyses with default paths (no argparse)")
    run_generated_directory(data_path=data_path, code_dir=code_dir, out_dir=out_dir, timeout=120)
