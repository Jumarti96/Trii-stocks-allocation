"""
Stock Allocation Pipeline - Orchestrator

Runs the full pipeline or a subset of steps in sequence.
Each step is executed as a subprocess so it can also be run standalone.

By default every step runs regardless of whether its outputs already exist.
Use --resume to skip steps whose output files are already present.

Usage:
  python orchestrator.py                       # Run all steps
  python orchestrator.py --steps 4            # Run only step 4
  python orchestrator.py --from 2             # Run from step 2 to the end
  python orchestrator.py --resume             # Skip steps that are already cached
  python orchestrator.py --steps 4 --resume   # Run step 4 only if not cached
  python orchestrator.py --list               # Show step status and exit
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline')
sys.path.insert(0, _PIPELINE_DIR)
from config import load_config, PATHS, BASE_DIR  # noqa: E402 - path insert must come first
load_config()  # populates PATHS['04_report'] from params.yaml before STEPS is built

STEPS = {
    1: {
        'script':  '01_download.py',
        'name':    'Download and Preprocess',
        'outputs': [PATHS['01_prices'], PATHS['01_returns']],
    },
    2: {
        'script':  '02_predict.py',
        'name':    'Transformer Prediction and Covariance',
        'outputs': [PATHS['02_expected_returns'], PATHS['02_covmat'], PATHS['02_metadata']],
    },
    3: {
        'script':  '03_allocate.py',
        'name':    'Portfolio Allocation',
        'outputs': [PATHS['03_weights']],
    },
    4: {
        'script':  '04_report.py',
        'name':    'Report Generation',
        'outputs': [PATHS['04_report']],
    },
}


def step_is_cached(step_num):
    outputs = STEPS[step_num]['outputs']
    return bool(outputs) and all(Path(p).exists() for p in outputs)


def run_step(step_num):
    step   = STEPS[step_num]
    script = os.path.join(_PIPELINE_DIR, step['script'])

    print(f"\n{'=' * 60}")
    print(f"  Step {step_num}: {step['name']}")
    print(f"{'=' * 60}")

    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def list_steps():
    print("\nPipeline steps:")
    for num, step in STEPS.items():
        status = "cached" if step_is_cached(num) else "not cached"
        print(f"  {num}. {step['name']:<45} [{status}]")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stock Allocation Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--steps', nargs='+', type=int, metavar='N',
                        help='Run specific steps, e.g. --steps 3 4')
    parser.add_argument('--from', dest='from_step', type=int, metavar='N',
                        help='Run from this step to the end')
    parser.add_argument('--resume', action='store_true',
                        help='Skip steps whose output files already exist')
    parser.add_argument('--list', action='store_true',
                        help='Print step status and exit')
    args = parser.parse_args()

    if args.list:
        list_steps()
        return

    if args.steps:
        steps_to_run = sorted(set(args.steps))
    elif args.from_step:
        steps_to_run = list(range(args.from_step, max(STEPS) + 1))
    else:
        steps_to_run = list(STEPS.keys())

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"[WARN] Unknown step {step_num} - skipping.")
            continue

        if args.resume and step_is_cached(step_num):
            step_name = STEPS[step_num]['name']
            print(f"\n[SKIP] Step {step_num} ({step_name}): outputs cached.")
            continue

        run_step(step_num)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
