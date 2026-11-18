import argparse
from src.experiment import run_full

def main():
    p=argparse.ArgumentParser(description='Reproducible DEAP EEG experiment')
    p.add_argument('--quick',action='store_true'); p.add_argument('--resume',action='store_true')
    p.add_argument('--subjects',type=int); p.add_argument('--skip-exploratory',action='store_true')
    p.add_argument('--out'); p.add_argument('--gru-subjects',type=int)
    args=p.parse_args()
    try: run_full(args)
    except Exception as e:
        raise SystemExit(f'Fatal input or environment error: {e}')
if __name__=='__main__':main()
