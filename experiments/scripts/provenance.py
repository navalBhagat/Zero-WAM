import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.runtime.provenance import fingerprint
parser = argparse.ArgumentParser()
parser.add_argument("--verify", required=True)
args = parser.parse_args()
if fingerprint()[0] != args.verify:
    raise SystemExit("Code changed after planning; create a new campaign")
