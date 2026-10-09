# MeshLang Specification v0.1

## Overview

MeshLang is a declarative domain-specific language for describing computational meshes.
It compiles to Gmsh `.geo` files (and optionally direct Gmsh API calls) while providing
a modern, type-safe, composable authoring experience.

**Design goals:**
- Separate geometry, mesh control, physics, and export concerns
- Catch errors at parse/compile time, not during meshing
- First-class units, CSG operations, and topology references
- Pipe-based transform chains for readability
- Frequency/wavelength-aware mesh sizing (acoustics-first, but general)

**File extension:** `.mlg`

---

## 1. Lexical Structure

### 1.1 Comments

```
// Single-line comment
/* Multi-line
   comment */
/// Doc comment (attached to next declaration)
```

### 1.2 Identifiers

```
IDENT = [a-zA-Z_][a-zA-Z0-9_]*
```

### 1.3 Literals

```
INTEGER  = [0-9]+
FLOAT    = [0-9]+ '.' [0-9]* ([eE] [+-]? [0-9]+)?
STRING   = '"' [^"]* '"'
BOOL     = 'true' | 'false'
```

### 1.4 Units

Units are optional suffixes on numeric literals. They participate in
dimensional analysis at compile time.

| Category    | Units                          |
|-------------|--------------------------------|
| Length      | `m`, `mm`, `cm`, `um`, `in`    |
| Angle       | `deg`, `rad`                   |
| Frequency   | `Hz`, `kHz`                    |
| Velocity    | `m/s`                          |
| Time        | `s`, `ms`                      |
| Pressure    | `Pa`, `kPa`                    |

Compound units via `/`: `343.0 m/s`

### 1.5 Keywords

```
param    const    import   geometry   mesh     physics  export
group    refine   near     on         default  true     false
boundary_layer   let      fn         for      in       if
else     match    struct   enum       use      as       pub
```

### 1.6 Operators

```
// Arithmetic
+  -  *  /  %  **

// CSG (on geometry solids)
|   // union
-   // difference
&   // intersection

// Pipe
|>

// Comparison
==  !=  <  >  <=  >=

// Logical
&&  ||  !

// Access
.   ::

// Assignment
=

// Range
..  ..=
```

---

## 2. Type System

### 2.1 Primitive Types

```
int      // 64-bit signed integer
float    // 64-bit floating point
bool     // true / false
string   // UTF-8 string
```

### 2.2 Dimensional Types (with units)

```
length       // e.g., 0.3 m
angle        // e.g., 45 deg
frequency    // e.g., 1000 Hz
velocity     // e.g., 343 m/s
time         // e.g., 0.001 s
pressure     // e.g., 101325 Pa
```

Dimensional types carry their unit and are checked at compile time:
```
param a: length = 0.3 m
param b: frequency = 1000 Hz
param c = a * b  // ERROR: length * frequency has no defined type
param d = a / (1.0 s)  // OK: length / time = velocity
```

### 2.3 Geometry Types

```
point       // 0D: position in 3D space
curve       // 1D: line, arc, spline, etc.
surface     // 2D: plane, curved surface, etc.
solid       // 3D: volume

face_set    // collection of surfaces (for BCs)
edge_set    // collection of curves
```

### 2.4 Composite Types

```
// Arrays
[float; 3]          // fixed-size
[point]             // dynamic

// Structs
struct Material {
    density: float,
    speed_of_sound: velocity,
}
```

### 2.5 Type Inference

Types are inferred where unambiguous:
```
param freq = 1000.0 Hz     // inferred: frequency
param c = 343.0 m/s        // inferred: velocity
param wavelength = c / freq // inferred: length (velocity / frequency)
param n = 6                 // inferred: int
```

---

## 3. Declarations

### 3.1 Parameters

```
param <name> (: <type>)? = <expr> (<unit>)?

// Examples
param freq: frequency = 1000.0 Hz
param c = 343.0 m/s
param elem_size = c / freq / 6.0
```

Parameters are immutable. Use `const` for compile-time constants:
```
const PI = 3.14159265358979
const ELEM_PER_WAVELENGTH = 6
```

### 3.2 Imports

```
import "path/to/file.mlg"
import "path/to/file.mlg" as mylib
import "path/to/file.step"  // CAD import (STEP, IGES, STL, OBJ)
```

### 3.3 Functions

```
fn <name>(<params>) -> <return_type> {
    <body>
}

// Example: parameterized port geometry
fn port(radius: length, depth: length) -> solid {
    Cylinder(r: radius, h: depth)
}
```

---

## 4. Geometry Blocks

### 4.1 Syntax

```
geometry <name> {
    <statement>*
}
```

### 4.2 Shape Primitives

All primitives are centered at origin unless otherwise noted.

```
// 3D primitives
Box(x: length, y: length, z: length)
Sphere(r: length)
Cylinder(r: length, h: length)
Cone(r1: length, r2: length, h: length)
Torus(r_major: length, r_minor: length)
Wedge(x: length, y: length, z: length, ltx: length)

// 2D primitives (for extrusion)
Rectangle(w: length, h: length)
Circle(r: length)
Polygon(points: [point])
```

### 4.3 Constructive Solid Geometry (CSG)

```
geometry example {
    let a = Box(1m, 1m, 1m)
    let b = Sphere(r: 0.6m)

    let result = a & b     // intersection
    let result = a | b     // union
    let result = a - b     // difference
}
```

CSG operations are type-checked: both operands must be `solid`.

### 4.4 Transforms (Pipe Syntax)

```
let moved = Cylinder(r: 0.1m, h: 0.5m)
    |> translate(0.2m, 0.0m, 0.1m)
    |> rotate(axis: Z, angle: 45 deg)

// Transforms:
translate(dx: length, dy: length, dz: length)
rotate(axis: X | Y | Z | vector, angle: angle)
scale(factor: float)
scale(sx: float, sy: float, sz: float)
mirror(plane: XY | XZ | YZ | surface)
```

### 4.5 Extrusions & Sweeps

```
let profile = Circle(r: 0.05m)
let tube = profile |> extrude(direction: Z, distance: 1.0m)
let ring = profile |> revolve(axis: X, angle: 360 deg)
let horn = profile |> sweep(path: my_spline)
```

### 4.6 Topology Access (Dot Notation)

This is a key differentiator from Gmsh. Instead of tracking integer entity tags,
you access topology by name and relationship:

```
geometry cab {
    let outer = Box(0.3m, 0.4m, 0.25m)

    // Named face access
    outer.faces           // all 6 faces -> face_set
    outer.faces.top       // +Z face -> surface
    outer.faces.bottom    // -Z face -> surface
    outer.faces.front     // +Y face -> surface
    outer.faces.back      // -Y face -> surface
    outer.faces.left      // -X face -> surface
    outer.faces.right     // +X face -> surface

    outer.edges           // all 12 edges -> edge_set
    outer.vertices        // all 8 vertices -> point set

    // For non-box shapes, index or filter:
    let cyl = Cylinder(r: 0.1m, h: 0.5m)
    cyl.faces.lateral     // curved surface
    cyl.faces.top         // top disk
    cyl.faces.bottom      // bottom disk

    // After CSG, faces from operands are tracked:
    let diff = outer - cyl
    // diff retains references to outer.faces and gains cyl's cut surface
}
```

### 4.7 CAD Import

```
geometry bracket {
    import "bracket.step" as raw

    // Access imported topology
    raw.solids[0]
    raw.faces.where(area > 0.01 m^2)   // filter by property
    raw.edges.where(length < 0.005 m)   // short edges (for cleanup)
}
```

### 4.8 Geometry Array and Repetition

```
geometry array_example {
    let base = Cylinder(r: 0.01m, h: 0.02m)

    let grid = for i in 0..4, j in 0..4 {
        base |> translate(i * 0.05m, j * 0.05m, 0m)
    }  // produces a solid (implicit union)
}
```

---

## 5. Mesh Control Blocks

### 5.1 Syntax

```
mesh {
    <statement>*
}

// or named, referencing a specific geometry:
mesh for <geometry_name> {
    <statement>*
}
```

### 5.2 Default Sizing

```
mesh {
    default_size: 0.01 m
    min_size: 0.001 m
    max_size: 0.05 m
}
```

### 5.3 Frequency-Aware Sizing

```
mesh {
    // Automatically sets element size = wavelength / elements_per_wavelength
    target_frequency: 2000 Hz
    medium_speed: 343 m/s
    elements_per_wavelength: 6
}
```

### 5.4 Refinement Zones

```
mesh {
    refine near(cab.port, distance: 0.02m) {
        size: 0.002 m
        growth_rate: 1.15
    }

    refine on(cab.outer.faces.front) {
        size: 0.005 m
    }

    refine in(Box(0.1m, 0.1m, 0.1m) |> translate(0.5m, 0m, 0m)) {
        size: 0.003 m
    }

    // Field-based (distance from entity)
    refine field(distance_from: cab.port) {
        size_at_entity: 0.001 m
        size_far: 0.02 m
        far_distance: 0.1 m
    }
}
```

### 5.5 Boundary Layers

```
mesh {
    boundary_layer on(cab.outer.faces) {
        first_height: 0.001 m
        layers: 5
        growth_rate: 1.2
    }
}
```

### 5.6 Algorithm Selection

```
mesh {
    algorithm_2d: delaunay | frontal | meshadapt | pack_parallelograms
    algorithm_3d: delaunay | frontal | hxt
    recombine: false         // quad-dominant if true
    optimize: true
    optimize_threshold: 0.3
    order: 2                 // element polynomial order
    smoothing_steps: 5
}
```

---

## 6. Physics Blocks (Boundary Conditions / Physical Groups)

### 6.1 Syntax

```
physics {
    <group_declaration>*
}
```

### 6.2 Group Declarations

```
physics {
    // Named groups with BC type annotations
    group wall: neumann = cab.outer.faces - cab.port.faces
    group inlet: dirichlet = cab.port.faces.top
    group radiation: sommerfeld = cab.outer.faces.outer
    group source: velocity = cab.driver.faces.front

    // Volume groups
    group air: volume = cab.interior
    group absorber: volume = cab.foam
}
```

BC type annotations (`neumann`, `dirichlet`, `sommerfeld`, `impedance`, `velocity`,
`volume`) are metadata — they don't affect meshing but enable downstream solver
validation: if your solver expects a `dirichlet` group and it's missing, the
compiler warns.

### 6.3 Group Operations

```
physics {
    let all_walls = cab.outer.faces | cab.inner.faces   // union
    let inlet_only = all_walls - cab.outlet              // difference

    group walls: neumann = all_walls
    group inlet: dirichlet = inlet_only
}
```

---

## 7. Export Block

```
export {
    format: msh4 | msh2 | vtk | stl | obj | nastran | abaqus | med
    filename: "output.msh"
    order: 2
    binary: true

    // Optional: include only specific groups
    include: [wall, inlet, radiation]
}
```

Multiple exports are allowed:
```
export as "surface_mesh" {
    format: stl
    filename: "surface.stl"
    entities: cab.outer.faces
}

export as "volume_mesh" {
    format: msh4
    filename: "volume.msh"
}
```

---

## 8. Expressions

### 8.1 Arithmetic

```
1 + 2
3.0 * freq
c / freq / 6.0
2.0 ** 0.5          // power
```

### 8.2 Pipe Expressions

```
<expr> |> <function>(<args>)

// Equivalent to function(<expr>, <args>)
Cylinder(r: 0.1m, h: 0.5m) |> translate(1m, 0m, 0m) |> rotate(axis: Z, angle: 90 deg)
```

### 8.3 Conditional

```
if freq > 1000 Hz {
    elem_size / 2.0
} else {
    elem_size
}
```

### 8.4 Dot Access

```
cab.outer.faces.top        // topology traversal
material.density           // struct field
```

---

## 9. Module System

### 9.1 File Imports

```
import "acoustics/port.mlg"                // import all public symbols
import "acoustics/port.mlg" as port        // namespaced
import { BassReflex, Sealed } from "acoustics/enclosures.mlg"
```

### 9.2 Visibility

```
// pub makes declarations available to importers
pub param default_speed = 343.0 m/s

pub geometry StandardPort(radius: length, depth: length) {
    Cylinder(r: radius, h: depth)
}

// without pub, only visible in current file
param internal_detail = 0.001 m
```

### 9.3 Standard Library

The standard library provides common acoustic and mechanical primitives:

```
import std.acoustics       // wavelength(), elem_size_for_freq(), ...
import std.shapes          // RoundedBox, Fillet, Chamfer, ...
import std.materials       // Air, Water, Steel, ...
import std.transforms      // array_linear, array_polar, ...
```

---

## 10. Error Model

### 10.1 Compile-Time Errors

| Error | Example |
|-------|---------|
| **Unit mismatch** | `param x = 1.0 m + 2.0 Hz` |
| **CSG type error** | `Box(...) - Point(...)` (solid - point) |
| **Undefined reference** | `refine near(nonexistent, ...)` |
| **Missing physics group** | Solver expects `inlet` but physics block has no `inlet` |
| **Duplicate names** | Two `geometry` blocks with same name |

### 10.2 Warnings

| Warning | Example |
|---------|---------|
| **Coarse mesh** | `elem_size > wavelength / 3` for target frequency |
| **Unused geometry** | Geometry defined but not exported |
| **Degenerate entity** | Face with near-zero area |

### 10.3 Error Messages

Errors include source location, context, and suggestions:

```
error[E0012]: unit mismatch in addition
  --> speaker.mlg:14:22
   |
14 |     param x = width + freq
   |                ^^^^^ length (m)
   |                        ^^^^ frequency (Hz)
   |
   = note: cannot add length and frequency
   = hint: did you mean `width + c / freq`?
```

---

## 11. Compilation Targets

### 11.1 Gmsh .geo (primary)

The compiler outputs a `.geo` file using Gmsh's OpenCASCADE kernel. Entity tags
are managed automatically. A mapping file (`.mlg.map`) is emitted alongside,
recording the correspondence between MeshLang names and Gmsh integer tags.

### 11.2 Gmsh Python API

Alternative backend generating a Python script using `import gmsh`. Useful for
integration into existing Python workflows.

### 11.3 Direct Mesh (future)

For simple geometries (structured grids, BEM surface meshes), the compiler may
directly emit mesh files without Gmsh, using built-in algorithms.

---

## 12. Full Example

```
/// Bass reflex speaker cabinet mesh for BEM analysis
/// Target: 20 Hz – 2000 Hz frequency range

import std.acoustics

const C_AIR = 343.0 m/s

param f_max = 2000.0 Hz
param elem_size = acoustics.size_for_freq(f_max, C_AIR, epw: 6)

param cab_w = 0.20 m
param cab_h = 0.35 m
param cab_d = 0.25 m
param port_r = 0.025 m
param port_d = 0.15 m
param driver_r = 0.065 m

geometry cabinet {
    let shell = Box(cab_w, cab_h, cab_d)

    let port = Cylinder(r: port_r, h: port_d)
        |> translate(cab_w / 2.0, 0.05m, cab_d / 2.0)

    let driver_hole = Cylinder(r: driver_r, h: 0.01m)
        |> translate(cab_w / 2.0, cab_h - 0.08m, cab_d / 2.0)
        |> rotate(axis: X, angle: 90 deg)

    let solid = shell - port - driver_hole
}

mesh for cabinet {
    default_size: elem_size
    min_size: elem_size / 10.0
    order: 2

    refine near(cabinet.port, distance: 0.03m) {
        size: elem_size / 4.0
        growth_rate: 1.2
    }

    refine near(cabinet.driver_hole, distance: 0.02m) {
        size: elem_size / 3.0
        growth_rate: 1.15
    }
}

physics {
    group cabinet_wall: neumann = cabinet.solid.faces
        - cabinet.port.faces.top
        - cabinet.driver_hole.faces.front

    group port_exit: radiation = cabinet.port.faces.top
    group driver: velocity = cabinet.driver_hole.faces.front
}

export {
    format: msh4
    order: 2
    filename: "cabinet_bem.msh"
}
```

---

## Appendix A: Grammar (EBNF summary)

```ebnf
program        = declaration* ;

declaration    = param_decl
               | const_decl
               | import_decl
               | fn_decl
               | geometry_block
               | mesh_block
               | physics_block
               | export_block
               | struct_decl
               ;

param_decl     = "param" IDENT (":" type)? "=" expr unit? ;
const_decl     = "const" IDENT (":" type)? "=" expr unit? ;
import_decl    = "import" STRING ("as" IDENT)? ;
fn_decl        = "pub"? "fn" IDENT "(" param_list? ")" "->" type block ;

geometry_block = "pub"? "geometry" IDENT ("(" param_list ")")? block ;
mesh_block     = "mesh" (("for" IDENT) | IDENT)? block ;
physics_block  = "physics" block ;
export_block   = "export" ("as" STRING)? block ;

block          = "{" statement* "}" ;
statement      = let_stmt | assign_stmt | expr_stmt | group_decl
               | refine_stmt | bl_stmt | kv_pair | for_stmt | if_stmt ;

let_stmt       = "let" IDENT (":" type)? "=" expr ;
group_decl     = "group" IDENT ":" IDENT "=" expr ;
refine_stmt    = "refine" refine_kind block ;
refine_kind    = "near" "(" expr "," kv_args ")"
               | "on" "(" expr ")"
               | "in" "(" expr ")"
               | "field" "(" kv_args ")" ;

expr           = pipe_expr ;
pipe_expr      = or_expr ("|>" IDENT "(" args? ")")* ;
or_expr        = and_expr ("||" and_expr)* ;
and_expr       = cmp_expr ("&&" cmp_expr)* ;
cmp_expr       = add_expr (cmp_op add_expr)? ;
add_expr       = mul_expr (("+"|"-") mul_expr)* ;
mul_expr       = unary_expr (("*"|"/"|"%") unary_expr)* ;
unary_expr     = ("-"|"!") unary_expr | power_expr ;
power_expr     = postfix_expr ("**" unary_expr)? ;
postfix_expr   = primary ("." IDENT | "[" expr "]")* ;
primary        = IDENT | literal | "(" expr ")" | constructor | if_expr ;

constructor    = IDENT "(" kv_args ")" ;
kv_args        = kv_arg ("," kv_arg)* ;
kv_arg         = (IDENT ":")? expr ;

literal        = INTEGER | FLOAT | STRING | BOOL ;
unit           = IDENT ("/" IDENT)* ;
```

---

## Appendix B: Compilation Model

```
  .mlg source
       │
       ▼
  ┌─────────┐
  │  Parse   │  → Concrete Syntax Tree (tree-sitter)
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │  Lower   │  → Abstract Syntax Tree (typed, resolved)
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │  Check   │  → Unit checking, type checking, name resolution
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │  Emit    │  → .geo file / Python script / direct mesh
  └──────────┘
```
