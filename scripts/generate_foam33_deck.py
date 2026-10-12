#!/usr/bin/env python3
"""Generate an original LAW33 FOAM_PLAS compression/hold/unload fixture."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

NAME = 'foam33_cube'
PATH = [(0., 0.), (.001, -.0005), (.0015, -.0005), (.0025, 0.), (.003, 0.)]


def integer(v):
    return f'{v:10d}'


def real(v):
    return f'{v:20.12E}'


def starter():
    units = ''.join(f'{unit:20}' for unit in ['kg', 'm', 's']) + '\n'
    text = '#RADIOSS STARTER\n/BEGIN\nSonium original LAW33 cube\n' + integer(2022) + '\n' + units * 2
    text += '/NODE\n'
    nodes = [(0,0,0),(.01,0,0),(.01,.01,0),(0,.01,0),(0,0,.01),(.01,0,.01),(.01,.01,.01),(0,.01,.01)]
    for i, p in enumerate(nodes, 1):
        text += integer(i) + ''.join(real(x) for x in p) + '\n'
    text += '/PART/1\nfoam cube\n' + integer(1)*2 + '\n/BRICK/1\n' + ''.join(integer(i) for i in [1,1,2,3,4,5,6,7,8]) + '\n'
    text += '/PROP/SOLID/1\nfull integration brick\n' + integer(18)+integer(0)+real(0)*2+integer(0)*2+real(0)+'\n'+real(0)*3+'\n'+real(0)+integer(0)+'\n'
    text += '/MAT/FOAM_PLAS/1\nplain LAW33 card, same SI parameters as FOAM_V10\n'
    text += real(433.)+real(0.)+'\n'+real(350e6)+integer(0)*2+real(1.)+'\n'
    text += real(1e5)+real(.1)+real(0.)+'\n'+real(-177000.)+real(677000.)+real(-3.34)+'\n'
    for group, ids in [(1,[1,4,5,8]),(2,[2,3,6,7])]:
        text += f'/GRNOD/NODE/{group}\nface {group}\n'+''.join(integer(i) for i in ids)+'\n'
    text += '/BCS/1\nfixed left face\n   111 000'+integer(0)+integer(1)+'\n'
    text += '/BCS/2\ntransverse fixed right face\n   011 000'+integer(0)+integer(2)+'\n'
    text += '/FUNCT/1\ncompression hold unload hold\n'
    for t,u in PATH:
        text += real(t)+real(u)+'\n'
    text += '/IMPDISP/1\nright face axial displacement\n'+integer(1)+f'{"X":>10}'+integer(0)*2+integer(2)+integer(0)+'\n'
    text += real(1.)*2+real(0.)+real(1e30)+'\n'
    text += '/TH/BRIC/1\ncube\nDEF       \n'+integer(1)+'          cube\n'
    text += '/TH/NODE/2\nface\nDEF       \n'+integer(2)+integer(0)+'moving_node\n/END\n'
    return text


def engine():
    return '#RADIOSS ENGINE\n/RUN/'+NAME+'/1\n'+real(.003)+'\n/TFILE\n'+real(.000025)+'\n/DT/BRICK\n'+real(.05)+real(0.)+'\n/PRINT/-1000\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--harvest-from', type=Path)
    parser.add_argument('--engine-exe', type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory/(NAME+'_0000.rad')).write_text(starter())
    (args.output_directory/(NAME+'_0001.rad')).write_text(engine())
    if args.harvest_from:
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        names = [NAME+'_0000.rad', NAME+'_0001.rad']
        for name in names:
            if sha(args.harvest_from/name) != sha(args.output_directory/name):
                raise ValueError(f'harvest deck differs from generator: {name}')
        log = (args.harvest_from/'engine.log').read_text()
        if 'NORMAL TERMINATION' not in log or args.engine_exe is None:
            raise ValueError('completed engine run and executable required')
        if '0 ERROR(S)' not in (args.harvest_from/'starter.log').read_text():
            raise ValueError('starter errors present')
        history = args.harvest_from/(NAME+'T01.csv')
        with history.open() as stream:
            rows = list(csv.reader(stream))
        if len(rows) < 110 or float(rows[-1][0]) < .00295:
            raise ValueError('incomplete compression/hold/unload history')
        shutil.copyfile(history, args.output_directory/'history.csv')
        shutil.copyfile(args.harvest_from/'engine.log', args.output_directory/'engine-output.txt')
        golden = {
            'schema': 'sonium-foam33-golden-v1', 'engine_normal_termination': True,
            'engine_commit': re.search(r'CommitID:\s*([a-f0-9]{40})', log).group(1),
            'engine_sha256': sha(args.engine_exe),
            'model': {'young_pa': 350e6, 'a_pa': -177000., 'b_pa': 677000.,
                      'c': -3.34, 'pressure0_pa': 1e5, 'porosity': .1, 'gamma0': 0.,
                      'density_kg_m3': 433., 'side_m': .01, 'ka': 0, 'isolid': 18},
            'path_s_m': PATH,
            'deck_sha256': {name: sha(args.output_directory/name) for name in names},
            'history_sha256': sha(args.output_directory/'history.csv'),
            'engine_log_sha256': sha(args.output_directory/'engine-output.txt'),
            'scope': 'plain FOAM_PLAS Ka=0 card material replay; all nodal kinematics prescribed',
        }
        (args.output_directory/'golden.json').write_text(json.dumps(golden, indent=2)+'\n')



if __name__ == '__main__':
    main()
