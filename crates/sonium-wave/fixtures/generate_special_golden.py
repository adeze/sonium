#!/usr/bin/env python3
"""Generate the independent high-precision special-function fixture.

The Rust implementation is evaluated in binary64, while this generator uses
Python's decimal module with 100 significant digits.  Trigonometric values
are supplied as exact special-angle identities (rather than calling a library
routine), so the fixture does not share the implementation under test.

Run from the repository root with::

    python3 crates/sonium-wave/fixtures/generate_special_golden.py > \\
        crates/sonium-wave/fixtures/special-golden-v1.json
"""

from decimal import Decimal, getcontext
import json


getcontext().prec = 110
D = Decimal
PI = D(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
)
SQRT2 = D(2).sqrt()


def q(value: Decimal) -> str:
    """Emit a stable decimal string with enough guard digits for binary64."""

    return format(+value, ".45f")


def spherical_case(name: str, x: Decimal, sin_x: Decimal, cos_x: Decimal, order: int):
    j = [D(0)] * order
    y = [D(0)] * order
    j[0] = sin_x / x
    y[0] = -cos_x / x
    if order > 1:
        j[1] = sin_x / (x * x) - cos_x / x
        y[1] = -cos_x / (x * x) - sin_x / x
    for n in range(1, order - 1):
        j[n + 1] = D(2 * n + 1) / x * j[n] - j[n - 1]
        y[n + 1] = D(2 * n + 1) / x * y[n] - y[n - 1]
    jp = [D(0)] * order
    yp = [D(0)] * order
    jp[0] = -j[1]
    yp[0] = -y[1]
    for n in range(1, order):
        jp[n] = j[n - 1] - D(n + 1) / x * j[n]
        yp[n] = y[n - 1] - D(n + 1) / x * y[n]
    return {
        "name": name,
        "x": q(x),
        "j": [q(v) for v in j],
        "y": [q(v) for v in y],
        "j_derivative": [q(v) for v in jp],
        "y_derivative": [q(v) for v in yp],
    }


def legendre_case(name: str, x: Decimal, order: int, m: int):
    p = [D(0)] * order
    p[0] = D(1)
    if order > 1:
        p[1] = x
    for n in range(2, order):
        nf = D(n)
        p[n] = ((D(2 * n - 1) * x * p[n - 1]) - D(n - 1) * p[n - 2]) / nf

    dp = [D(0)] * order
    if order > 1:
        denominator = x * x - D(1)
        for n in range(1, order):
            dp[n] = D(n) * (x * p[n] - p[n - 1]) / denominator

    # Condon–Shortley associated Legendre recurrence.
    pm = [D(0)] * (order - m)
    root = (D(1) - x * x).sqrt()
    pmm = D(1)
    for i in range(1, m + 1):
        pmm *= -D(2 * i - 1) * root
    pm[0] = pmm
    if len(pm) > 1:
        pm[1] = x * D(2 * m + 1) * pmm
    for n in range(m + 2, order):
        nf = D(n)
        pm[n - m] = (
            D(2 * n - 1) * x * pm[n - m - 1]
            - D(n + m - 1) * pm[n - m - 2]
        ) / (nf - D(m))

    return {
        "name": name,
        "x": q(x),
        "p": [q(v) for v in p],
        "derivative": [q(v) for v in dp],
        "associated_m": m,
        "associated": [q(v) for v in pm],
    }


fixture = {
    "schema": "sonium-wave.special-golden.v1",
    "precision_digits": 100,
    "generator": "Python decimal recurrence with exact special-angle identities",
    "conventions": {
        "spherical": "j_n and y_n use the real positive-argument convention; derivatives are d/dx",
        "legendre": "P_n and P_n^m use the Condon-Shortley (-1)^m phase",
    },
    "spherical": [
        spherical_case("pi", PI, D(0), D(-1), 14),
        spherical_case("pi_over_2", PI / D(2), D(1), D(0), 14),
        spherical_case("pi_over_4_high_order", PI / D(4), SQRT2 / D(2), SQRT2 / D(2), 14),
        spherical_case("two_pi_high_order", PI * D(2), D(0), D(1), 14),
    ],
    "legendre": [
        legendre_case("x_0.37", D("0.37"), 8, 2),
        legendre_case("x_minus_0.81", D("-0.81"), 8, 2),
    ],
}

print(json.dumps(fixture, indent=2))
