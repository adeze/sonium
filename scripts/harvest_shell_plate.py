#!/usr/bin/env python3
"""Archive a completed engine run of the checked-in circular shell plate."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('engine_exe', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parents[1] / 'crates/sonium-qa/decks/shell_clamped_plate'
    names = ['shell_clamped_plate_0000.rad', 'shell_clamped_plate_0001.rad']
    for name in names:
        if sha256(args.run_directory / name) != sha256(fixture / name):
            raise ValueError(f'run deck differs from checked fixture: {name}')
    log = (args.run_directory / 'engine.log').read_text()
    if 'NORMAL TERMINATION' not in log:
        raise ValueError('engine did not terminate normally')
    commit = re.search(r'CommitID:\s*([a-f0-9]{40})', log).group(1)
    history = args.run_directory / 'shell_clamped_plateT01.csv'
    with history.open() as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 100 or float(rows[-1][0]) < .99:
        raise ValueError('incomplete pressure-hold history')
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (fixture / name).resolve() != (out / name).resolve():
            shutil.copyfile(fixture / name, out / name)
    shutil.copyfile(history, out / 'history.csv')
    shutil.copyfile(args.run_directory / 'engine.log', out / 'engine-output.txt')
    golden = {
        'schema': 'sonium-shell-plate-golden-v1', 'engine_normal_termination': True,
        'engine_commit': commit, 'engine_sha256': sha256(args.engine_exe),
        'model': {'radius_m': 1., 'thickness_m': .02, 'youngs_modulus_pa': 200e9,
                  'poisson_ratio': .3, 'density_kg_per_m3': 7800., 'pressure_pa': 1000.,
                  'rings': 8, 'sectors': 32},
        'deck_sha256': {name: sha256(out / name) for name in names},
        'history_sha256': sha256(out / 'history.csv'),
        'engine_log_sha256': sha256(out / 'engine-output.txt'),
        'scope': 'elastic quasistatic pressure-hold; no plastic-impact or transient-ramp claim',
    }
    (out / 'golden.json').write_text(json.dumps(golden, indent=2) + '\n')


if __name__ == '__main__':
    main()
