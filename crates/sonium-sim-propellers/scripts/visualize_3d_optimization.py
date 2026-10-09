#!/usr/bin/env python3
"""Visualize 3D propeller blade optimization over time with PyVista.

Run from the repo root:

    python crates/sonium-sim-propellers/scripts/visualize_3d_optimization.py

Optional GIF export:

    python crates/sonium-sim-propellers/scripts/visualize_3d_optimization.py --gif /tmp/propeller_opt.gif

The script mirrors the Rust `propeller-3d-opt` surrogate closely enough for
design-review visualization: coordinate-search steps, shorter-only radius,
20% thrust-loss cap, constant-weight thickness scaling, toroidal loop closure,
and LE/TE serrations.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


TAU = 2.0 * math.pi


@dataclass(frozen=True)
class Source:
    n_blades: int = 2
    rpm: float = 18_000.0
    tip_radius: float = 0.0635
    hub_radius: float = 0.008
    thrust: float = 3.0
    torque: float = 0.05
    density: float = 1.204
    speed_of_sound: float = 343.0

    @property
    def omega(self) -> float:
        return TAU * self.rpm / 60.0

    @property
    def bpf(self) -> float:
        return self.n_blades * self.rpm / 60.0

    @property
    def tip_mach(self) -> float:
        return self.omega * self.tip_radius / self.speed_of_sound


@dataclass(frozen=True)
class MeshConfig:
    radial_segments: int = 96
    chord_segments: int = 18
    loop_segments: int = 40
    pitch_deg_root: float = 28.0
    pitch_deg_tip: float = 12.0
    chord_root: float = 0.028
    chord_tip: float = 0.010
    loop_closure_fraction: float = 0.78
    loop_axial_lift: float = 0.012


@dataclass(frozen=True)
class ShapeVars:
    radius_scale: float = 0.96
    pitch_root_delta_deg: float = 0.0
    pitch_tip_delta_deg: float = 1.0
    chord_root_scale: float = 1.08
    chord_tip_scale: float = 1.18
    loop_closure_fraction: float = 0.78
    loop_axial_lift_scale: float = 1.0
    serration_amplitude: float = 0.018
    serration_wavelength: float = 0.055
    serration_span: float = 0.72

    def clamped(self) -> "ShapeVars":
        return ShapeVars(
            radius_scale=float(np.clip(self.radius_scale, 0.82, 1.0)),
            pitch_root_delta_deg=float(np.clip(self.pitch_root_delta_deg, -7.0, 5.0)),
            pitch_tip_delta_deg=float(np.clip(self.pitch_tip_delta_deg, -5.0, 7.0)),
            chord_root_scale=float(np.clip(self.chord_root_scale, 0.78, 1.38)),
            chord_tip_scale=float(np.clip(self.chord_tip_scale, 0.75, 1.55)),
            loop_closure_fraction=float(np.clip(self.loop_closure_fraction, 0.62, 0.90)),
            loop_axial_lift_scale=float(np.clip(self.loop_axial_lift_scale, 0.55, 1.85)),
            serration_amplitude=float(np.clip(self.serration_amplitude, 0.006, 0.035)),
            serration_wavelength=float(np.clip(self.serration_wavelength, 0.035, 0.120)),
            serration_span=float(np.clip(self.serration_span, 0.40, 0.95)),
        )

    def perturb(self, index: int, delta: float) -> "ShapeVars":
        values = list(self.__dict__.values())
        values[index] += delta
        return ShapeVars(*values).clamped()


@dataclass
class State:
    step: int
    variables: ShapeVars
    mesh_config: MeshConfig
    source: Source
    mesh: pv.PolyData
    area: float
    thrust_loss: float
    thickness_scale: float
    band_reduction_db: float
    peak_spread_delta_db: float
    vortex_index: float
    objective: float


def smooth_window(x: float, lo: float, hi: float) -> float:
    if hi <= lo or x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    t = (x - lo) / (hi - lo)
    return t * t * (3.0 - 2.0 * t)


def band_window(freq: float, lo_hz: float, hi_hz: float) -> float:
    if freq <= 0.0:
        return 0.0
    f = math.log10(freq)
    lo = math.log10(lo_hz)
    hi = math.log10(hi_hz)
    rise = smooth_window(f, math.log10(lo_hz * 0.75), lo)
    fall = 1.0 - smooth_window(f, hi, math.log10(hi_hz * 1.4))
    return rise * fall


def harmonic_amplitudes(source: Source, n_harmonics: int) -> np.ndarray:
    loading_force = math.hypot(source.thrust, source.torque / source.tip_radius)
    base = loading_force / (4.0 * math.pi * source.speed_of_sound)
    b = float(source.n_blades)
    values = []
    for harmonic in range(1, n_harmonics + 1):
        nb = harmonic * b
        loading = base * nb * abs(source.omega)
        values.append(loading * abs(bessel_jn(int(nb), nb * source.tip_mach)))
    return np.asarray(values)


def bessel_jn(n: int, x: float) -> float:
    """Small-argument J_n series, matching the Rust source model."""
    half_x = 0.5 * x
    term = 1.0
    for k in range(1, n + 1):
        term *= half_x / k
    total = term
    half_x_sq = half_x * half_x
    for k in range(1, 50):
        term *= -half_x_sq / (k * (n + k))
        total += term
        if abs(term) < 1.0e-18 * max(abs(total), 1.0e-300):
            break
    return total


def reduction_for_shape(source: Source, variables: ShapeVars, n_harmonics: int) -> np.ndarray:
    reductions = []
    for harmonic in range(1, n_harmonics + 1):
        freq = harmonic * source.bpf
        tip_band = band_window(freq, 1_000.0, 5_000.0)
        high = (freq / 700.0) / (1.0 + freq / 700.0)
        serration_strength = 0.15 + 0.85 * smooth_window(variables.serration_amplitude, 0.006, 0.035)
        serration_strength *= 1.0 - smooth_window(variables.serration_wavelength, 0.09, 0.22)
        serration_strength *= variables.serration_span
        loop_relief = 0.08 + 0.27 * tip_band + 0.08 * high
        le = 0.18 * serration_strength * band_window(freq, 800.0, 4_500.0)
        te = 0.24 * serration_strength * band_window(freq, 1_200.0, 6_000.0)
        decoherence = 0.05 * serration_strength * min(math.log1p(harmonic), 2.0)
        reductions.append(float(np.clip(1.0 - loop_relief - le - te - decoherence, 0.18, 1.0)))
    return np.asarray(reductions)


def spectrum_power(source: Source, variables: ShapeVars, n_harmonics: int) -> np.ndarray:
    amps = harmonic_amplitudes(source, n_harmonics)
    return (amps * reduction_for_shape(source, variables, n_harmonics)) ** 2


def spectral_peakiness_db(power: np.ndarray, source: Source) -> float:
    freqs = np.arange(1, len(power) + 1) * source.bpf
    mask = (freqs >= 1_000.0) & (freqs <= 5_000.0)
    if not mask.any():
        return 0.0
    band = power[mask]
    return 10.0 * math.log10(max(band.max(), 1e-30) / max(band.mean(), 1e-30))


def band_reduction_db(base_power: np.ndarray, cand_power: np.ndarray, source: Source) -> float:
    freqs = np.arange(1, len(base_power) + 1) * source.bpf
    mask = (freqs >= 1_000.0) & (freqs <= 5_000.0)
    return 10.0 * math.log10(max(base_power[mask].sum(), 1e-30) / max(cand_power[mask].sum(), 1e-30))


def serration_offset(source: Source, vars_: ShapeVars, eta: float, chord_fraction: float, chord: float) -> float:
    if eta < 1.0 - vars_.serration_span:
        return 0.0
    leading = chord_fraction < 0.18
    trailing = chord_fraction > 0.82
    if not (leading or trailing):
        return 0.0
    treated_span = max(eta - (1.0 - vars_.serration_span), 0.0) * (source.tip_radius - source.hub_radius)
    wavelength = max(vars_.serration_wavelength, 0.005) * max(chord, 1.0e-9)
    tooth = math.sin(TAU * treated_span / wavelength)
    sign = -1.0 if leading else 1.0
    return sign * vars_.serration_amplitude * chord * tooth


def blade_point(source: Source, cfg: MeshConfig, vars_: ShapeVars, eta: float, chord_fraction: float, loop_z: float) -> tuple[float, float, float]:
    radius = source.hub_radius + eta * (source.tip_radius - source.hub_radius)
    chord = cfg.chord_root + eta * (cfg.chord_tip - cfg.chord_root)
    pitch = math.radians(cfg.pitch_deg_root + eta * (cfg.pitch_deg_tip - cfg.pitch_deg_root))
    x_chord = (chord_fraction - 0.5) * chord + serration_offset(source, vars_, eta, chord_fraction, chord)
    theta = 0.35 * eta**1.25
    swept_radius = radius + x_chord * math.sin(pitch)
    axial = x_chord * math.cos(pitch) + loop_z
    return (swept_radius * math.cos(theta), swept_radius * math.sin(theta), axial)


def build_mesh(source: Source, cfg: MeshConfig, vars_: ShapeVars, include_loop: bool = True) -> pv.PolyData:
    points: list[tuple[float, float, float]] = []
    faces: list[int] = []

    def add_quad(vertices: list[tuple[float, float, float]]) -> None:
        start = len(points)
        points.extend(vertices)
        faces.extend([4, start, start + 1, start + 2, start + 3])

    radial_end = vars_.loop_closure_fraction if include_loop else 1.0
    for ir in range(cfg.radial_segments):
        e0 = radial_end * ir / cfg.radial_segments
        e1 = radial_end * (ir + 1) / cfg.radial_segments
        for ic in range(cfg.chord_segments):
            c0 = ic / cfg.chord_segments
            c1 = (ic + 1) / cfg.chord_segments
            add_quad([
                blade_point(source, cfg, vars_, e0, c0, 0.0),
                blade_point(source, cfg, vars_, e1, c0, 0.0),
                blade_point(source, cfg, vars_, e1, c1, 0.0),
                blade_point(source, cfg, vars_, e0, c1, 0.0),
            ])

    if include_loop:
        eta0 = vars_.loop_closure_fraction
        for il in range(cfg.loop_segments):
            u0 = il / cfg.loop_segments
            u1 = (il + 1) / cfg.loop_segments
            for ic in range(cfg.chord_segments):
                c0 = ic / cfg.chord_segments
                c1 = (ic + 1) / cfg.chord_segments
                vertices = []
                for u, c in [(u0, c0), (u1, c0), (u1, c1), (u0, c1)]:
                    phase = TAU * u
                    eta = eta0 + (1.0 - eta0) * (0.5 - 0.5 * math.cos(phase))
                    loop_z = cfg.loop_axial_lift * math.sin(phase)
                    vertices.append(blade_point(source, cfg, vars_, eta, c, loop_z))
                add_quad(vertices)

    pts = np.asarray(points, dtype=float)
    all_points = []
    all_faces = []
    for blade in range(source.n_blades):
        angle = TAU * blade / source.n_blades
        rot = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        offset = len(all_points)
        all_points.extend((pts @ rot.T).tolist())
        raw = np.asarray(faces, dtype=np.int64).reshape((-1, 5)).copy()
        raw[:, 1:] += offset
        all_faces.extend(raw.ravel().tolist())

    mesh = pv.PolyData(np.asarray(all_points), np.asarray(all_faces))
    mesh["radius"] = np.linalg.norm(mesh.points[:, :2], axis=1)
    return mesh


def thrust_proxy(source: Source, cfg: MeshConfig) -> float:
    span = source.tip_radius - source.hub_radius
    thrust = 0.0
    for eta in (np.arange(512) + 0.5) / 512.0:
        radius = source.hub_radius + eta * span
        chord = cfg.chord_root + eta * (cfg.chord_tip - cfg.chord_root)
        pitch = math.radians(cfg.pitch_deg_root + eta * (cfg.pitch_deg_tip - cfg.pitch_deg_root))
        thrust += chord * max(math.sin(pitch), 0.04) * radius * radius * span / 512.0
    return thrust * source.omega**2 * source.density * source.n_blades


def mesh_area(mesh: pv.PolyData) -> float:
    return float(mesh.compute_cell_sizes(length=False, area=True, volume=False)["Area"].sum())


def evaluate(vars_: ShapeVars, baseline: State | None, step: int, base_power: np.ndarray, base_peak: float, base_thrust: float, base_area: float) -> State:
    base_source = Source()
    source = Source(tip_radius=base_source.tip_radius * vars_.radius_scale)
    cfg = MeshConfig(
        pitch_deg_root=28.0 + vars_.pitch_root_delta_deg,
        pitch_deg_tip=12.0 + vars_.pitch_tip_delta_deg,
        chord_root=0.028 * vars_.chord_root_scale,
        chord_tip=0.010 * vars_.chord_tip_scale,
        loop_closure_fraction=vars_.loop_closure_fraction,
        loop_axial_lift=0.012 * vars_.loop_axial_lift_scale,
    )
    candidate_thrust = thrust_proxy(source, cfg)
    loading_scale = candidate_thrust / max(base_thrust, 1.0e-30)
    source = Source(
        tip_radius=source.tip_radius,
        thrust=base_source.thrust * loading_scale,
        torque=base_source.torque * loading_scale,
    )
    mesh = build_mesh(source, cfg, vars_, include_loop=True)
    area = mesh_area(mesh)
    power = spectrum_power(source, vars_, 20)
    thrust_loss = 1.0 - loading_scale
    thickness_scale = base_area / max(area, 1e-12)
    band_db = band_reduction_db(base_power, power, source)
    peak_delta = spectral_peakiness_db(power, source) - base_peak
    sized = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    cell_radii = np.linalg.norm(sized.cell_centers().points[:, :2], axis=1)
    cell_areas = np.asarray(sized["Area"])
    tip_area_fraction = float(cell_areas[cell_radii > source.tip_radius * 0.88].sum() / max(cell_areas.sum(), 1.0e-30))
    vortex_index = 0.18 * tip_area_fraction / 0.08
    thrust_excess = max(thrust_loss - 0.20, 0.0)
    thickness_excess = max(thickness_scale - 1.85, 0.0) + max(0.55 - thickness_scale, 0.0)
    objective = -band_db + 0.45 * peak_delta + 250.0 * thrust_excess**2 + 20.0 * thickness_excess**2
    return State(step, vars_, cfg, source, mesh, area, thrust_loss, thickness_scale, band_db, peak_delta, vortex_index, objective)


def optimization_history() -> list[State]:
    base_source = Source()
    base_vars = ShapeVars(radius_scale=1.0, chord_root_scale=1.0, chord_tip_scale=1.0, serration_span=0.72)
    base_cfg = MeshConfig()
    base_mesh = build_mesh(base_source, base_cfg, base_vars, include_loop=False)
    base_area = mesh_area(base_mesh)
    base_power = harmonic_amplitudes(base_source, 20) ** 2
    base_peak = spectral_peakiness_db(base_power, base_source)
    base_thrust = thrust_proxy(base_source, base_cfg)

    current = evaluate(ShapeVars(), None, 0, base_power, base_peak, base_thrust, base_area)
    history = [current]
    steps = [0.035, 1.6, 1.6, 0.08, 0.10, 0.035, 0.16, 0.0035, 0.010, 0.055]

    step_id = 1
    for _ in range(8):
        improved = False
        for i, step in enumerate(steps):
            for sign in (-1.0, 1.0):
                candidate = evaluate(current.variables.perturb(i, sign * step), current, step_id, base_power, base_peak, base_thrust, base_area)
                if candidate.objective < current.objective and candidate.thrust_loss <= 0.2000001:
                    current = candidate
                    history.append(current)
                    step_id += 1
                    improved = True
        steps = [s * (0.72 if improved else 0.48) for s in steps]
    return history


def state_text(state: State) -> str:
    return (
        f"Accepted step {state.step}\n"
        f"Objective: {state.objective:+.2f}\n"
        f"1-5 kHz: {state.band_reduction_db:+.2f} dB\n"
        f"Peak spread: {state.peak_spread_delta_db:+.2f} dB\n"
        f"Thrust loss: {100.0 * state.thrust_loss:.1f}% / 20%\n"
        f"Radius scale: {state.variables.radius_scale:.3f}\n"
        f"Thickness scale: {state.thickness_scale:.3f}\n"
        f"Serration span: {state.variables.serration_span:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", type=Path, help="Write an animated GIF instead of only showing the window.")
    parser.add_argument("--off-screen", action="store_true", help="Render without opening an interactive window.")
    parser.add_argument("--frame-ms", type=int, default=260, help="Animation frame delay in milliseconds.")
    args = parser.parse_args()

    history = optimization_history()
    pv.set_plot_theme("document")
    off_screen = args.off_screen or args.gif is not None
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1500, 900))
    plotter.set_background("white")
    plotter.add_text("3D Propeller Blade Optimization", position="upper_left", font_size=14, color="black")
    text_actor = plotter.add_text(state_text(history[0]), position="lower_left", font_size=11, color="black")
    mesh_actor = plotter.add_mesh(
        history[0].mesh,
        scalars="radius",
        cmap="viridis",
        smooth_shading=True,
        show_edges=False,
        specular=0.35,
    )
    plotter.add_axes()
    plotter.camera_position = [(0.11, -0.20, 0.10), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    if args.gif:
        plotter.open_gif(str(args.gif), fps=max(1, round(1000 / args.frame_ms)))
    elif not off_screen:
        plotter.show(auto_close=False, interactive_update=True)

    for state in history:
        plotter.remove_actor(mesh_actor)
        plotter.remove_actor(text_actor)
        mesh_actor = plotter.add_mesh(
            state.mesh,
            scalars="radius",
            cmap="viridis",
            smooth_shading=True,
            show_edges=False,
            specular=0.35,
        )
        text_actor = plotter.add_text(state_text(state), position="lower_left", font_size=11, color="black")
        if args.gif:
            plotter.write_frame()
        elif not off_screen:
            plotter.update()
            time.sleep(args.frame_ms / 1000.0)

    if args.gif:
        plotter.close()
        print(f"Wrote {args.gif}")
    elif not off_screen:
        plotter.show(auto_close=True)


if __name__ == "__main__":
    main()
