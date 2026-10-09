# MeshLang

A modern domain-specific language for computational mesh generation, targeting Gmsh.

## Why

Gmsh's `.geo` scripting language is powerful but dated: no type system, implicit global state, integer entity tags instead of names, cryptic error messages. MeshLang fixes this with:

- **Declarative geometry** with composable CSG and pipe-based transforms
- **Named topology access** (`cabinet.shell.faces.top` instead of `Surface{14}`)
- **Physical units** with compile-time dimensional analysis
- **Separate concerns**: geometry, mesh control, physics groups, and export are distinct blocks
- **Frequency-aware mesh sizing** (acoustics-first, but general)

MeshLang compiles to Gmsh `.geo` files or Python API scripts. No Gmsh linking required — clean GPL firewall via subprocess boundary.

## Quick Example

```meshlang
param f_max = 2000.0
param c = 343.0
param elem_size = c / f_max / 6.0

geometry cabinet {
    let shell = Box(0.20, 0.35, 0.25)
    let port = Cylinder(r: 0.025, h: 0.15)
        |> translate(0.10, 0.05, 0.125)
    let solid = shell - port
}

mesh for cabinet {
    default_size: elem_size
    order: 2
    refine near(cabinet.port, distance: 0.03) {
        size: 0.005
    }
}

physics {
    group wall: neumann = cabinet.solid.faces
    group port_exit: radiation = cabinet.port.faces.top
}

export {
    format: msh4
    filename: "cabinet.msh"
}
```

## Generated Output

**`.geo` (Gmsh OpenCASCADE):**
```
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 0.20, 0.35, 0.25};
Cylinder(2) = {0, 0, 0, 0, 0, 0.15, 0.025};
Translate {0.10, 0.05, 0.125} { Volume{2}; }
BooleanDifference (3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
```

**Python (Gmsh API):**
```python
shell = occ.addBox(0, 0, 0, 0.20, 0.35, 0.25)
port = occ.addCylinder(0, 0, 0, 0, 0, 0.15, 0.025)
occ.translate([(3, port)], 0.10, 0.05, 0.125)
csg_result = occ.cut([(3, shell)], [(3, port)])
solid = csg_result[0][0][1]
```

## Usage

```bash
# Compile to Gmsh .geo (also emits .geo.map sidecar)
meshlang compile input.mlg -o output.geo

# Compile to Python script
meshlang compile input.mlg -o output.py --target python

# Type-check without compiling
meshlang check input.mlg

# Type-check with JSON output (for editors/LSP)
meshlang check input.mlg --format json

# Debug: lex / parse
meshlang lex input.mlg
meshlang parse input.mlg
```

## Project Structure

```
meshlang/
├── docs/SPEC.md                         # Language specification (791 lines)
├── tree-sitter-meshlang/
│   ├── grammar.js                       # Tree-sitter grammar
│   ├── queries/highlights.scm           # Syntax highlighting
│   └── test/corpus/                     # Grammar test cases
├── crates/
│   ├── meshlang-ast/                    # AST, types, units, visitor
│   ├── meshlang-parser/                 # Lexer + recursive-descent parser
│   └── meshlang-compiler/               # Full compiler pipeline
│       ├── codegen_geo.rs               # .geo backend
│       ├── codegen_python.rs            # Python API backend
│       ├── topology.rs                  # Named face/edge resolution
│       ├── type_check.rs                # Dimensional analysis + validation
│       ├── diagnostics.rs               # LSP-compatible + terminal output
│       ├── imports.rs                   # Module resolution + stdlib
│       ├── tag_allocator.rs             # Name ↔ Gmsh tag mapping
│       └── name_resolution.rs           # Scope/symbol resolution
├── stdlib/
│   ├── acoustics.mlg                    # C_AIR, size_for_freq(), wavelength()
│   └── shapes.mlg                       # Tube, RoundedBox, Waveguide
├── examples/
│   ├── speaker_cabinet.mlg              # BEM cabinet with port + driver
│   ├── sphere_scattering.mlg            # Analytical benchmark case
│   └── room_acoustics.mlg               # FEM room with floor/ceiling BCs
└── Cargo.toml
```

## Architecture

```
                          ┌─────────────┐
                          │  .mlg source │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │    Lexer     │  token stream
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │   Parser    │  AST
                          └──────┬──────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐ ┌───▼───┐ ┌──────▼──────┐
              │Name Resolve│ │ Type  │ │  Topology   │
              │   + Scope  │ │ Check │ │   Tracker   │
              └─────┬─────┘ └───┬───┘ └──────┬──────┘
                    │            │            │
                    └────────────┼────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │                         │
              ┌─────▼─────┐           ┌──────▼──────┐
              │ .geo codegen│           │  Python     │
              │ + tag alloc │           │  codegen    │
              └─────┬─────┘           └──────┬──────┘
                    │                         │
              ┌─────▼─────┐           ┌──────▼──────┐
              │ output.geo │           │  output.py  │
              │ output.map │           │  (helpers)  │
              └───────────┘           └─────────────┘
```

### Key Design Decisions

**GPL Firewall**: MeshLang compiles to `.geo` files that Gmsh reads as a subprocess. No linking against Gmsh's GPL code — your solver stays proprietary.

**Topology by Name**: After `let solid = shell - port`, you write `solid.faces` instead of guessing Gmsh surface tags. The compiler tracks topology through CSG operations and emits bounding-box-based face finders for the Python backend.

**Tag Map Sidecar**: The `.geo.map` JSON file maps MeshLang names ↔ Gmsh integer tags, letting your solver reference entities by name.

## Test Suite

41 tests across 4 test suites:
- **Unit tests** (3): dimensional analysis, unit parsing
- **Lexer tests** (5): tokenization, operators, type identifiers
- **Compiler unit tests** (11): topology resolution, type checking, imports
- **Integration tests** (22): parser roundtrips, codegen validation, end-to-end compilation

## License

MIT OR Apache-2.0
