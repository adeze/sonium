#!/usr/bin/env python3
"""Generate the independent rigid-scattering coefficient fixture.

The Rust implementation evaluates its series in binary64 using Sonium's own
special functions (sphere) or `spec_math` (cylinder). This generator shares no
code with Sonium: all recurrences run in Python's `decimal` module at 110
significant digits, and only the elementary seed values (sin/cos and the
cylindrical J0/J1/Y0/Y1) come from the independent arbitrary-precision
library `mpmath` (120-digit working precision). The seeds are transcendentals,
not the scattering physics under test.

Run from the repository root with::

    python3 crates/sonium-wave/fixtures/generate_scattering_golden.py > \\
        crates/sonium-wave/fixtures/scattering-golden-v1.json
"""

from decimal import Decimal, getcontext
import json

import mpmath

mpmath.mp.dps = 120
getcontext().prec = 110
D = Decimal


def q(value: Decimal) -> str:
    """Emit a stable decimal string with guard digits for binary64."""
    return format(+value, ".45f")


def seeds(ka: D) -> dict:
    x = mpmath.mpf(str(ka))
    return {
        "sin": D(str(mpmath.sin(x))),
        "cos": D(str(mpmath.cos(x))),
        "J0": D(str(mpmath.besselj(0, x))),
        "J1": D(str(mpmath.besselj(1, x))),
        "Y0": D(str(mpmath.bessely(0, x))),
        "Y1": D(str(mpmath.bessely(1, x))),
    }


def miller_downward(x: D, j0_exact: D, nmax: int, top: int):
    """Downward (Miller) recurrence for first-kind Bessel values 0..=nmax."""
    f = [D(0)] * (top + 1)
    f[top] = D(0)
    f[top - 1] = D(1)
    for n in range(top - 1, 0, -1):
        f[n - 1] = D(2 * n + 1) / x * f[n] - f[n + 1]
    scale = f[0] / j0_exact
    return [f[n] / scale for n in range(nmax + 1)]


def upward_second_kind(x: D, y0: D, y1: D, nmax: int, spherical: bool):
    """Upward recurrence (stable direction) for second-kind values 0..=nmax.

    Spherical functions step with (2n+1)/x; cylindrical with 2n/x.
    """
    y = [D(0)] * (nmax + 1)
    y[0] = y0
    if nmax >= 1:
        y[1] = y1
    for n in range(1, nmax):
        factor = D(2 * n + 1) / x if spherical else D(2 * n) / x
        y[n + 1] = factor * y[n] - y[n - 1]
    return y


def complex_pair(re: D, im: D) -> dict:
    return {"re": q(re), "im": q(im)}


def sphere_case(ka_str: str, nmax: int) -> dict:
    ka = D(ka_str)
    s = seeds(ka)
    top = nmax + 40
    j = miller_downward(ka, s["sin"] / ka, nmax, top)
    # Spherical Neumann seeds: y0 = -cos(x)/x, y1 = -cos(x)/x^2 - sin(x)/x.
    y = upward_second_kind(
        ka,
        -s["cos"] / ka,
        -s["cos"] / (ka * ka) - s["sin"] / ka,
        nmax,
        spherical=True,
    )
    j_neg1 = s["cos"] / ka  # j_{-1}(x) = cos(x)/x
    y_neg1 = s["sin"] / ka  # y_{-1}(x) = sin(x)/x
    coefficients = []
    for n in range(nmax + 1):
        nf = D(n)
        j_prev = j[n - 1] if n > 0 else j_neg1
        y_prev = y[n - 1] if n > 0 else y_neg1
        jp = j_prev - (nf + 1) / ka * j[n]
        yp = y_prev - (nf + 1) / ka * y[n]
        denom = jp * jp + yp * yp
        # a_n = j' / (j' + i y')
        coefficients.append(complex_pair(jp * jp / denom, -jp * yp / denom))
    return {
        "ka": ka_str,
        "num_terms": nmax + 1,
        "coefficients": coefficients,
    }


def cylinder_case(ka_str: str, nmax: int) -> dict:
    ka = D(ka_str)
    s = seeds(ka)
    top = nmax + 40
    # Cylindrical J via Miller downward, normalized on the mpmath J0 seed.
    f = [D(0)] * (top + 1)
    f[top] = D(0)
    f[top - 1] = D(1)
    for n in range(top - 1, 0, -1):
        f[n - 1] = D(2 * n) / ka * f[n] - f[n + 1]
    scale = f[0] / s["J0"]
    j = [f[n] / scale for n in range(nmax + 1)]
    y = upward_second_kind(ka, s["Y0"], s["Y1"], nmax, spherical=False)
    coefficients = []
    for n in range(nmax + 1):
        nf = D(n)
        # J_{-1} = -J_1, matching the Rust implementation convention.
        j_prev = j[n - 1] if n > 0 else -s["J1"]
        y_prev = y[n - 1] if n > 0 else -s["Y1"]
        jp = j_prev - nf / ka * j[n]
        yp = y_prev - nf / ka * y[n]
        denom = jp * jp + yp * yp
        # b_n = -J'/H' * i^n with i^n = cos(n pi/2) + i sin(n pi/2).
        base_re = -jp * jp / denom
        base_im = jp * yp / denom
        quadrant = n % 4
        if quadrant == 0:
            re, im = base_re, base_im
        elif quadrant == 1:
            re, im = -base_im, base_re
        elif quadrant == 2:
            re, im = -base_re, -base_im
        else:
            re, im = base_im, -base_re
        coefficients.append(complex_pair(re, im))
    return {
        "ka": ka_str,
        "num_terms": nmax + 1,
        "coefficients": coefficients,
    }


fixture = {
    "schema": "sonium-wave.scattering-golden.v1",
    "case_id": "scattering-coefficients-rigid-v1",
    "precision_digits": 100,
    "generator": "Python decimal recurrences (Miller downward first-kind, upward second-kind); elementary seeds only from mpmath",
    "mpmath_version": mpmath.__version__,
    "radius": "1.0",
    "boundary_condition": "rigid (Neumann)",
    "conventions": {
        "sphere": "total = sum_n (2n+1) i^n [j_n(kr) - a_n h_n^(1)(kr)] P_n(cos theta); a_n = j_n'(ka)/h_n^(1)'(ka)",
        "cylinder": "p = exp(ikr cos theta) + sum_n eps_n b_n H_n^(1)(kr) cos(n theta); b_n = -J_n'(ka)/H_n^(1)'(ka) i^n; eps_0 = 1, eps_n = 2",
    },
    "sphere": [
        sphere_case("1.0", 19),
        sphere_case("5.0", 35),
    ],
    "cylinder": [
        cylinder_case("1.0", 11),
        cylinder_case("5.0", 19),
    ],
}

print(json.dumps(fixture, indent=2))
