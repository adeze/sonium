#!/usr/bin/env python3
"""Generate `waveguide_optimization.json` (the realistic benchmark).

Rebuilds the typed-driver benchmark from the lumped electro-acoustic
equations documented in `README.md`, so the JSON is reproducible and the
curves stay internally consistent (velocity, electrical impedance, intrinsic
mechanical impedance, and the crossover network all derive from the same
Thiele-Small parameters).

Usage: python3 generate_waveguide_benchmark.py
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "waveguide_optimization.json")


# Z_coil(f) = Re + jw(Le+L_series) - j/(wC_series)
# Z_mech(f) = Rms + jwMms + 1/(jwCms)   (intrinsic, excludes radiation load)
# v(f)      = Bl * V / (Z_coil*Z_mech + Bl^2)          [RMS m/s]
# Z_el(f)   = Z_coil + Bl^2 / Z_mech                   [Ohm]
def driver_curves(ts, v_rms, freqs, l_series=0.0, c_series=None):
    Re, Le, Bl, Mms, Cms, Rms = ts
    vel, zel, zm_curve, le_curve = [], [], [], []
    for f in freqs:
        w = 2 * math.pi * f
        Zc = complex(Re, w * (Le + l_series))
        if c_series is not None:
            Zc -= 1j / (w * c_series)
        Zm = complex(Rms, w * Mms - 1.0 / (w * Cms))
        v = Bl * v_rms / (Zc * Zm + Bl * Bl)
        Ze = Zc + Bl * Bl / Zm
        vel.append({"frequency_hz": f, "velocity_real_m_per_s": v.real,
                    "velocity_imag_m_per_s": v.imag})
        zel.append({"frequency_hz": f, "real": Ze.real, "imaginary": Ze.imag})
        zm_curve.append({"frequency_hz": f, "real": Zm.real, "imaginary": Zm.imag})
        le_curve.append({"frequency_hz": f, "inductance_h": Le + l_series,
                         "eddy_resistance_ohm": 0.0})
    return vel, zel, zm_curve, le_curve


def thiele_small(Re, Le, Bl, Mms, Cms, Rms, Sd):
    ws = 1.0 / math.sqrt(Mms * Cms)
    fs = ws / (2 * math.pi)
    qms = ws * Mms / Rms
    qes = ws * Mms * Re / (Bl * Bl)
    qts = qms * qes / (qms + qes)
    rho, c = 1.21, 343.0
    vas = rho * c * c * Sd * Sd * Cms
    return {
        "voice_coil_resistance_ohm": Re,
        "nominal_inductance_h": Le,
        "force_factor_tm": Bl,
        "moving_mass_kg": Mms,
        "compliance_m_per_n": Cms,
        "mechanical_resistance_kg_per_s": Rms,
        "effective_piston_area_m2": Sd,
        "resonance_hz": fs,
        "mechanical_q": qms,
        "electrical_q": qes,
        "total_q": qts,
        "equivalent_compliance_volume_m3": vas,
    }


# 6.5" mid-woofer (plausible SEAS/Scan-Speak class unit) + 0.8 mH series LP
# inductor (1st-order electrical corner ~780 Hz, adding to the mass roll-off).
WF = dict(Re=5.6, Le=0.35e-3, Bl=6.5, Mms=14.0e-3, Cms=1.1e-3,
          Rms=0.89, Sd=math.pi * 0.065**2)
WF_L = 0.8e-3
# 25 mm dome tweeter on waveguide + 120 uF series HP capacitor (electrical
# resonance with Le ~2.05 kHz; acoustic HP corner lands near 1.8-2 kHz).
TW = dict(Re=4.6, Le=0.05e-3, Bl=2.3, Mms=0.4e-3, Cms=7.83e-5,
          Rms=0.75, Sd=math.pi * 0.013**2)
TW_C = 120.0e-6

V_REF = 2.83  # 1 W into 8 Ohm, RMS
FREQS = [100.0] + [round(200.0 * (8000.0 / 200.0) ** (i / 23), 1) for i in range(24)]


def driver_model(d, l_series=0.0, c_series=None):
    ts = (d["Re"], d["Le"], d["Bl"], d["Mms"], d["Cms"], d["Rms"])
    vel, zel, zm, le_curve = driver_curves(ts, V_REF, FREQS, l_series, c_series)
    model = {
        "thiele_small": thiele_small(*ts, d["Sd"]),
        "bl_curve": [{"displacement_m": 0.0, "force_factor_tm": d["Bl"]}],
        "inductance_curve": le_curve,
        "reference_voltage_rms_v": V_REF,
        "velocity_transfer": vel,
        "electrical_impedance": zel,
        "intrinsic_mechanical_impedance": zm,
    }
    if c_series is not None:
        model["series_capacitance_f"] = c_series
    return model


model = {
    "cabinet": {"height": 0.40, "width": 0.22, "depth": 0.28},
    "woofer": {
        "radius": 0.065,
        "y_offset": -0.08,
        "piston_velocity": 0.001,
        "driver_model": driver_model(WF, l_series=WF_L),
        "cone": {"cone_depth": 0.008, "dustcap_radius": 0.015},
    },
    "tweeter": {
        "radius": 0.013,
        "y_offset": 0.10,
        "piston_velocity": 0.001,
        "driver_model": driver_model(TW, c_series=TW_C),
        "cone": {"cone_depth": 0.003, "dustcap_radius": 0.005},
    },
    "separator": {"y_position": 0.02, "thickness": 0.018},
    "waveguide": {
        "horizontal_spline": [[0.0, 0.050], [0.5, 0.030], [1.0, 0.013]],
        "vertical_spline": [[0.0, 0.040], [0.5, 0.025], [1.0, 0.013]],
        "depth": 0.025,
    },
    "simulation": {
        "freq_start": 200.0,
        "freq_end": 5000.0,
        "freq_points": 50,
        "speed_of_sound": 343.0,
        "air_density": 1.21,
        "max_mesh_frequency": 5000.0,
    },
}

with open(OUT, "w") as fh:
    json.dump(model, fh, indent=2)
    fh.write("\n")

for name, d, kw in (("woofer", WF, dict(l_series=WF_L)),
                    ("tweeter", TW, dict(c_series=TW_C))):
    ts = (d["Re"], d["Le"], d["Bl"], d["Mms"], d["Cms"], d["Rms"])
    vel, _, _, _ = driver_curves(ts, V_REF, FREQS, **kw)

    def at(f):
        return min(vel, key=lambda p: abs(p["frequency_hz"] - f))

    for f in (200.0, 500.0, 1000.0, 2200.0, 5000.0):
        p = at(f)
        mag = math.hypot(p["velocity_real_m_per_s"], p["velocity_imag_m_per_s"])
        ph = math.degrees(math.atan2(p["velocity_imag_m_per_s"], p["velocity_real_m_per_s"]))
        print(f"{name:8s} {f:7.0f} Hz  |v|={mag:.4e} m/s RMS  phase={ph:7.1f} deg")
print("written", OUT)
