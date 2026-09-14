"""Run the distribute stage of the unified pipeline."""
import sys
from module.pipeline import main
if __name__ == "__main__":
    raise SystemExit(main(["--stage", "distribute", *sys.argv[1:]]))
