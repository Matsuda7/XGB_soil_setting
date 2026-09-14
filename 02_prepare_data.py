"""Run the prepare stage of the unified pipeline."""
import sys
from module.pipeline import main
if __name__ == "__main__":
    raise SystemExit(main(["--stage", "prepare", *sys.argv[1:]]))
