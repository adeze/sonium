#!/usr/bin/env python3
"""Generate an original OpenCourant axisymmetric pressure-cylinder fixture.

The Reference Guide /ANALY convention is Y=radius, Z=axial, N2D3D=1.
CLOAD values integrate pressure over axial length per unit circumference.
The engine applies the radial weight itself (force.F90, n2d=1 branch). All axial translations
are constrained for plane strain; radial translations remain free. A smooth
pressure ramp ends in a hold, allowing a static-response comparison.
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

NAME = "axisym_cylinder"
INNER = 0.05
OUTER = 0.10
LENGTH = 0.05
PRESSURE = 1e6
RAMP_TIME = 0.002
STOP_TIME = 0.0025
RADIAL_ELEMENTS = 32
AXIAL_ELEMENTS = 2


def integer(value):
    return f"{value:10d}"


def real(value):
    return f"{value:20.12E}"


def node(radial, axial):
    return 1 + axial * (RADIAL_ELEMENTS + 1) + radial


def starter():
    nr, nz = RADIAL_ELEMENTS, AXIAL_ELEMENTS
    units = "".join(f"{unit:20}" for unit in ["kg", "m", "s"]) + "\n"
    text = "#RADIOSS STARTER\n/BEGIN\nSonium axisymmetric pressure cylinder\n"
    text += integer(2022) + "\n" + units * 2
    text += "/ANALY\n" + integer(1) + f"{0:20d}" + integer(0) + "\n/NODE\n"
    for j in range(nz + 1):
        for i in range(nr + 1):
            text += integer(node(i, j)) + real(0) + real(INNER + (OUTER - INNER) * i / nr) + real(LENGTH * j / nz) + "\n"
    text += "/PART/1\ncylinder\n" + integer(1) * 2 + integer(0) + "\n/QUAD/1\n"
    for j in range(nz):
        for i in range(nr):
            text += "".join(integer(v) for v in [1 + j * nr + i, node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]) + "\n"
    text += "/PROP/TYPE14/1\naxisymmetric quad\n"
    text += integer(0) + integer(0) + real(0) * 2 + integer(0) * 2 + real(0) + "\n"
    text += real(0) * 5 + "\n" + real(0) + integer(0) * 2 + "\n"
    text += "/MAT/LAW1/1\nelastic steel\n" + real(7800) + "\n" + real(200e9) + real(0.3) + "\n"
    text += "/GRNOD/NODE/1\nall axial fixed\n"
    for first in range(1, node(nr, nz) + 1, 10):
        text += "".join(integer(v) for v in range(first, min(first + 10, node(nr, nz) + 1))) + "\n"
    text += "/BCS/1\nplane strain\n   101 000" + integer(0) + integer(1) + "\n"
    text += "/GRNOD/NODE/2\nbore end nodes\n" + integer(node(0, 0)) + integer(node(0, nz)) + "\n"
    text += "/GRNOD/NODE/3\nbore midpoint\n" + integer(node(0, 1)) + "\n"
    text += "/FUNCT/1\nsmooth pressure ramp and hold\n"
    for k in range(101):
        x = k / 100
        text += real(RAMP_TIME * x) + real(10 * x**3 - 15 * x**4 + 6 * x**5) + "\n"
    text += real(STOP_TIME) + real(1) + "\n"
    for group, scale in [(2, PRESSURE * LENGTH / nz / 2), (3, PRESSURE * LENGTH / nz)]:
        text += f"/CLOAD/{group}\npressure nodal traction per unit circumference\n"
        text += integer(1) + f"{'Y':>10}" + integer(0) * 2 + integer(group) + " " * 10 + real(1) + real(scale) + "\n"
    text += "/TH/NODE/1\nradial response\nDY        VY        \n"
    text += integer(node(0, 1)) + integer(0) + "\n" + integer(node(nr, 1)) + integer(0) + "\n/END\n"
    return text


def engine():
    return (f"#RADIOSS ENGINE\n/RUN/{NAME}/1\n" + real(STOP_TIME)
            + "\n/TFILE\n" + real(5e-6) + "\n/TH/TITLE\n/PRINT/-1000\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for generated starter/engine decks")
    parser.add_argument("--harvest-from", type=Path, help="scratch directory containing a completed engine run and converted CSV")
    parser.add_argument("--engine-exe", type=Path, help="engine binary used for the harvest (checksum provenance)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{NAME}_0000.rad").write_text(starter())
    (args.output / f"{NAME}_0001.rad").write_text(engine())
    if args.harvest_from:
        if not args.engine_exe:
            parser.error("--engine-exe is required with --harvest-from")
        harvest(args.harvest_from, args.output, args.engine_exe)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harvest(scratch, output, engine_exe):
    decks = {}
    for suffix in ["0000", "0001"]:
        name = f"{NAME}_{suffix}.rad"
        if (scratch / name).read_bytes() != (output / name).read_bytes():
            raise ValueError(f"harvest deck differs from current generator: {name}")
        decks[name] = digest(scratch / name)
    log = scratch / "engine.log"
    if "NORMAL TERMINATION" not in log.read_text():
        raise ValueError("engine did not terminate normally")
    csv_path = scratch / f"{NAME}T01.csv"
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    hold = [row for row in rows if float(row["time"]) >= RAMP_TIME + 0.0001]
    if len(hold) < 50:
        raise ValueError("insufficient pressure-hold samples")
    channels = {}
    for label, probe in [("inner_displacement_m", node(0, 1)), ("outer_displacement_m", node(RADIAL_ELEMENTS, 1))]:
        keys = [key for key in rows[0] if str(probe) in key.split() and key.strip().endswith("DY")]
        if len(keys) != 1:
            raise ValueError(f"ambiguous displacement channel for node {probe}")
        values = [float(row[keys[0]]) for row in hold]
        channels[label] = {"mean": sum(values) / len(values), "minimum": min(values), "maximum": max(values)}
    energy = [float(row["INTERNAL ENERGY"]) for row in hold]
    kinetic = [float(row["KINETIC ENERGY"]) for row in hold]
    golden = {
        "schema": "sonium-axisym-golden-v1",
        "model": {"inner_radius_m": INNER, "outer_radius_m": OUTER, "length_m": LENGTH,
                  "internal_pressure_pa": PRESSURE, "external_pressure_pa": 0.0,
                  "youngs_modulus_pa": 200e9, "poisson_ratio": 0.3,
                  "radial_elements": RADIAL_ELEMENTS, "axial_elements": AXIAL_ELEMENTS,
                  "n2d3d": 1, "element": "QUAD", "plane_strain": True},
        "deck_sha256": decks,
        "engine_sha256": digest(engine_exe),
        "engine_log_sha256": digest(log),
        "history_sha256": digest(csv_path),
        "engine_normal_termination": True,
        "hold_samples": len(hold),
        "hold_start_s": RAMP_TIME + 0.0001,
        "channels": channels,
        "strain_energy_per_radian_j": sum(energy) / len(energy),
        "maximum_kinetic_energy_ratio": max(kinetic) / min(energy),
        "scope": "quasistatic pressure-hold response; excludes transient-ramp agreement",
    }
    (output / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
    for source, name in [(csv_path, "history.csv"), (log, "engine-output.txt")]:
        target = output / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)


if __name__ == "__main__":
    main()
