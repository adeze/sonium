# sonium-wave (lib: `sonium_wave`, version: 0.4.0)

Analytical solutions for wave and Helmholtz equations.

## Purpose

Provides reference analytical solutions used to validate BEM and FEM numerical solvers.

## Key Components

- Bessel function evaluation
- Helmholtz equation analytical solutions
- Wave propagation formulas

## Dependencies

- `num-complex` - Complex number support
- `spec_math` - Special mathematical functions

## Testing

```bash
cargo test -p sonium-wave --lib
cargo check -p sonium-wave && cargo clippy -p sonium-wave
```
