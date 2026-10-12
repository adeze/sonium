#!/usr/bin/env python3
"""Generate original actual LAW76 quadratic, isochoric single-brick fixtures."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from generate_foam33_deck import starter as foam33_starter, integer, real


def starter(amplitude):
    text = foam33_starter().replace('Sonium original LAW33 cube', 'Sonium original LAW76 cube')
    start = text.index('/MAT/FOAM_PLAS/1')
    end = text.index('/GRNOD/NODE/1')
    material = '/MAT/LAW76/1\nquadratic nonassociated isochoric SAMP, convex curves\n'
    material += real(433.)+'\n'+real(350e6)+real(0.)+'\n'
    material += integer(2)+integer(3)+integer(4)+'\n'
    material += real(1.)*3+' '*20+real(1.)+'\n'
    material += real(.5)+integer(0)+real(0.)+integer(0)+real(0.)+'\n'
    material += real(0.)*2+'\n'+integer(0)+' '*30+real(0.)+'\n'
    material += integer(0)+integer(1)+integer(1)+'\n'
    # Engine PLAS = sqrt(3)/2 * equivalent plastic strain at Nu_p=0.5.
    for ident, initial, last_x in [(2,500000.,.5),(3,900000.,.5),(4,420000.,3**.5/4)]:
        material += f'/FUNCT/{ident}\nyield curve {ident}\n'+real(0.)+real(initial)+'\n'+real(last_x)+real(11*initial)+'\n'
    text = text[:start]+material+text[end:]
    start = text.index('/FUNCT/1\n')
    end = text.index('/IMPDISP/1')
    path = [(0.,0.),(.001,amplitude),(.0015,amplitude),(.0025,0.),(.003,0.)]
    text = text[:start]+'/FUNCT/1\nload hold unload hold\n'+''.join(real(t)+real(u)+'\n' for t,u in path)+text[end:]
    return text, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory',type=Path)
    parser.add_argument('--mode',choices=['compression','tension'],required=True)
    parser.add_argument('--harvest-from',type=Path)
    parser.add_argument('--engine-exe',type=Path)
    args=parser.parse_args()
    root=args.output_directory
    root.mkdir(parents=True,exist_ok=True)
    amplitude=-.00004 if args.mode=='compression' else .00002
    text,path=starter(amplitude)
    name='foam76_cube'
    (root/(name+'_0000.rad')).write_text(text)
    (root/(name+'_0001.rad')).write_text('#RADIOSS ENGINE\n/RUN/'+name+'/1\n'+real(.003)+'\n/TFILE\n'+real(.000025)+'\n/DT/BRICK\n'+real(.05)+real(0.)+'\n/PRINT/-1000\n')
    if args.harvest_from:
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        names=[name+'_0000.rad',name+'_0001.rad']
        if any(sha(root/n)!=sha(args.harvest_from/n) for n in names):
            raise ValueError('harvest deck differs from generated LAW76 card')
        log=(args.harvest_from/'engine.log').read_text()
        starter_log=(args.harvest_from/'starter.log').read_text()
        if 'NORMAL TERMINATION' not in log or args.engine_exe is None or '0 ERROR(S)' not in starter_log or '0 WARNING(S)' not in starter_log:
            raise ValueError('clean completed engine/starter run required')
        source=args.harvest_from/(name+'T01.csv')
        with source.open() as stream:
            rows=list(csv.reader(stream))
        if len(rows)<110 or float(rows[-1][0])<.00295:
            raise ValueError('incomplete LAW76 history')
        shutil.copyfile(source,root/'history.csv')
        shutil.copyfile(args.harvest_from/'engine.log',root/'engine-output.txt')
        golden={'schema':'sonium-foam76-golden-v1','mode':args.mode,'engine_normal_termination':True,
                'engine_commit':re.search(r'CommitID:\s*([a-f0-9]{40})',log).group(1),
                'engine_sha256':sha(args.engine_exe),'path_s_m':path,
                'model':{'young_pa':350e6,'poisson':0.,'density_kg_m3':433.,'side_m':.01,
                         'tension_pa':500000.,'compression_pa':900000.,'shear_pa':420000.,
                         'hardening_slope':20.,'nu_p':.5,'iform':0,'iquad':1,'iconv':1,'isolid':18},
                'deck_sha256':{n:sha(root/n) for n in names},'history_sha256':sha(root/'history.csv'),
                'engine_log_sha256':sha(root/'engine-output.txt'),
                'scope':'actual quadratic LAW76 material replay; all nodal kinematics prescribed'}
        (root/'golden.json').write_text(json.dumps(golden,indent=2)+'\n')


if __name__=='__main__':
    main()
