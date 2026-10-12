#!/usr/bin/env python3
"""Extract the original CRASH box cards for a native response comparison.

Coordinates/time use a fixed similarity transform L'=0.001L, t'=0.001t,
stress'=1e9 stress, density'=1e9 density. Mass, velocity, momentum and
energy are unchanged. This is not a reinterpretation of the deck's kg/m/s.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cards(path):
    result = {}
    active = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('/'):
            active = line
            if active in result:
                raise ValueError(f'duplicate card {active}')
            result[active] = []
        elif active:
            result[active].append(line)
    return result


def numbers(line):
    return [float(x) for x in line.split()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('starter', type=Path)
    parser.add_argument('engine', type=Path)
    parser.add_argument('golden', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    golden = json.loads(args.golden.read_text())
    meta = golden['meta']
    if not meta['normal_termination'] or sha(args.starter) != meta['deck_sha256']['starter'] or sha(args.engine) != meta['deck_sha256']['engine']:
        raise ValueError('original deck/golden identity mismatch')
    data = cards(args.starter)
    nodes = []
    for line in data['/NODE']:
        fields = numbers(line)
        if len(fields) != 4:
            raise ValueError('expected explicit node identifier and XYZ')
        nodes.append([int(fields[0]), *(x * .001 for x in fields[1:])])
    quads = []
    for line in data['/SHELL/1']:
        fields = numbers(line)
        if len(fields) != 5:
            raise ValueError('expected original four-node shell connectivity')
        quads.append([int(v) for v in fields])
    groups = {}
    for ident in [2, 3, 4, 6]:
        lines = data[f'/GRNOD/NODE/{ident}']
        groups[str(ident)] = [int(x) for line in lines[1:] for x in line.split()]
    material = data['/MAT/PLAS_JOHNS/1']
    density = numbers(material[1])[0] * 1e9
    e, nu, iflag = numbers(material[2])
    a, b, n, epmax, sigmax = numbers(material[3])
    c, ref_rate, icc, fsmooth, fcut, chard = numbers(material[4])
    thermal = numbers(material[5])
    prop = [numbers(line) for line in data['/PROP/TYPE1/1'][1:]]
    rigid = [numbers(line) for line in data['/RBODY/155'][1:]]
    added = numbers(data['/ADMAS/0/1'][1])
    initial = numbers(data['/INIVEL/TRA/1'][1])
    defaults = [numbers(line) for line in data['/DEF_SHELL']]
    if len(nodes) != 155 or len(quads) != 140 or sorted(n[0] for n in nodes) != list(range(1, 156)):
        raise ValueError('original box geometry changed')
    if any(node not in range(1, 155) for quad in quads for node in quad[1:]):
        raise ValueError('shell node missing')
    if iflag != 0 or fsmooth != 0 or any(thermal) or chard != 0 or epmax != 0:
        raise ValueError('material card no longer matches planned nonthermal isotropic LAW2')
    coordinates = {row[0]: row[1:] for row in nodes}
    nodal_mass = {row[0]: 0. for row in nodes}
    thickness = prop[2][2] * .001
    for quad in quads:
        pts = [coordinates[node] for node in quad[1:]]
        d02 = [pts[2][i]-pts[0][i] for i in range(3)]
        d13 = [pts[3][i]-pts[1][i] for i in range(3)]
        cross = [d02[1]*d13[2]-d02[2]*d13[1], d02[2]*d13[0]-d02[0]*d13[2], d02[0]*d13[1]-d02[1]*d13[0]]
        area = .5 * math.sqrt(sum(x*x for x in cross))
        for node in quad[1:]:
            nodal_mass[node] += density * thickness * area / 4.
    shell_mass = sum(nodal_mass.values())
    wall_mass = sum(nodal_mass[node] for node in groups['3'])
    total_mass = shell_mass + rigid[0][4] + added[0]
    initial_ke = .5 * total_mass * sum(v*v for v in initial[:3]) - .5 * wall_mass * initial[0]**2
    initial_impulse = wall_mass * initial[0]
    channels = golden['channels']
    if abs(total_mass-channels['MASS']['values'][0]) > .0005:
        raise ValueError('extracted mass disagrees with original TH rounding band')
    if abs(initial_ke-channels['KINETIC ENERGY']['values'][0]) > .0005:
        raise ValueError('wall-projected initial kinetic energy mismatch')
    if abs(initial_impulse-channels['Rigid walls 1 rigid_wall_1 ~1']['values'][0]) > 1e-8:
        raise ValueError('initial wall impulse mismatch')
    output = {
        'schema': 'sonium-original-crash-box-v1',
        'source': 'OpenCourant qa-tests/miniqa/CRASH/crashsection/box1_42/data/BOX1BL_0000.rad',
        'model_copyright': 'Copyright (C) 2026 SISW Siemens Industry Software Inc.',
        'model_license': 'CC BY-NC 4.0 https://creativecommons.org/licenses/by-nc/4.0/legalcode',
        'starter_sha256': sha(args.starter), 'engine_deck_sha256': sha(args.engine),
        'golden_sha256': sha(args.golden), 'engine_commit': meta['engine_commit'],
        'original_declared_units': meta['deck_units'],
        'similarity': {'length': .001, 'time': .001, 'stress': 1e9, 'density': 1e9,
                       'mass': 1., 'velocity': 1., 'energy': 1., 'momentum': 1., 'force': 1000.},
        'nodes': nodes, 'quads': quads, 'node_groups': groups,
        'law2': {'density_kg_m3': density, 'young_pa': e * 1e9, 'poisson': nu,
                 'a_pa': a * 1e9, 'b_pa': b * 1e9, 'n': n, 'c': c,
                 'reference_rate_s': ref_rate * 1000., 'max_stress_pa': sigmax * 1e9,
                 'icc': int(icc), 'fcut_hz': fcut * 1000., 'chard': chard,
                 'thermal_card_original': thermal},
        'assembly_contract': {'shell_mass_kg': shell_mass, 'wall_nodal_mass_kg': wall_mass,
                              'total_mass_kg': total_mass, 'initial_wall_projected_ke_j': initial_ke,
                              'initial_wall_impulse_ns': initial_impulse},
        'thickness_m': prop[2][2] * .001, 'thickness_points': int(prop[2][0]),
        'thickness_rule': 'uniform-surface-samples-piecewise-linear-stress',
        'property_flags_original': prop, 'def_shell_original': defaults,
        'initial_velocity_m_s': initial[:3],
        'rigid_body_card_original': rigid,
        'added_mass_kg': added[0], 'added_mass_group': int(added[1]),
        'wall_card_original': data['/RWALL/PLANE/1'],
        'scope': 'original 154 deformable shell nodes plus explicit rigid-body master 155 and 140 Q4 shells; native response comparison still pending',
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory/'model.json').write_text(json.dumps(output, indent=2)+'\n')
    print(f'checked original deck/golden; extracted {len(nodes)} nodes and {len(quads)} Q4 shells')


if __name__ == '__main__':
    main()
