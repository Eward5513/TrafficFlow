"""Batch entry for conservation mechanisms on the fixed backbone (no implicit launch)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from MSTT.run_experiments import main as run_tasks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('soft','projected'),required=True)
    parser.add_argument('--output-root',required=True,help='Empty result root for this mechanism; historical results are protected.')
    args,extra=parser.parse_known_args(argv)
    return run_tasks(['--tasks','All2All','--output-root',args.output_root,'--conservation-mode',args.mode,*extra])


if __name__=='__main__':raise SystemExit(main())
