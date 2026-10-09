# MeshLang grammar contract

This file is the normative precedence and operator contract shared by the
recursive-descent parser and the tree-sitter grammar. The Rust parser remains
the diagnostic authority; `tree-sitter-meshlang/grammar.js` is the editor
incremental parser and must preserve the token spellings below.

## Lexical operator spellings

| spelling | token | meaning |
| --- | --- | --- |
| `|>` | `PipeArrow` | left-to-right function pipeline |
| `|` | `Pipe` | geometry CSG union |
| `||` | `PipePipe` | boolean OR |
| `&` | `Ampersand` | geometry CSG intersection |
| `&&` | `AmpAmp` | boolean AND |
| `-` | `Minus` | arithmetic subtraction; geometry CSG difference is resolved after type checking |
| `**` | `StarStar` | exponentiation |

Longest-match tokenization is mandatory: `|>` and `||` are recognized before
the single `|`, and `&&` before `&`. A single `|` is never reinterpreted as a
pipeline token after lexing.

## Expression precedence

Rows are listed from lowest to highest precedence. Operators in one row are
left-associative unless explicitly marked otherwise.

| level | parser production | operators |
| ---: | --- | --- |
| 1 | `pipe` | `|>` (left-associative) |
| 2 | `csg` | `|`, `&` (left-associative) |
| 3 | `or` | `||` |
| 4 | `and` | `&&` |
| 5 | `comparison` | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| 6 | `additive` | `+`, `-` |
| 7 | `multiplicative` | `*`, `/`, `%` |
| 8 | `unary` | `+`, `-`, `!` |
| 9 | `power` | `**` (right-associative) |
| 10 | `postfix` | field access, calls, indexing |
| 11 | `primary` | literals, identifiers, constructors, arrays, parenthesized expressions |

CSG operands are parsed as expressions at the `or` level. This gives CSG
operators a stable boundary from boolean operators while leaving arithmetic
minus available to the type checker for scalar expressions.

The `..` and `..=` tokens are range delimiters owned by `for` clauses; they
are not expression-precedence operators.

## Parameterized geometry ports

A parameterized geometry can declare an explicit named result contract:

```meshlang
geometry Frame(width: Float) -> (body: Solid, inlet: Solid) {
    let body = Box(width, 1.0, 1.0)
    let inlet = Cylinder(r: 0.1, h: 0.5)
}
geometry model {
    let frame = Frame(2.0)
    let shifted = frame.inlet |> translate(1.0, 0.0, 0.0)
}
```

Each port must name a direct template binding and use a concrete geometry type
(`Point`, `Curve`/`EdgeSet`, `Surface`/`FaceSet`, or `Solid`). Ports cannot
have defaults. The first port remains the instance's primary entity for
compatibility; every declared output is accessed as `instance.port` and is
lowered into the canonical instance namespace before either backend emits it.

## Contract fixtures

Mesh declarations accept both named-target spellings, `mesh model { ... }`
and `mesh for model { ... }`; both populate the same `MeshBlock.target` AST
field. An empty `compound surface()` call is rejected as a typed parser
diagnostic rather than being treated as an implicit entity.

The Rust parser tests assert the longest-match token sequence and the AST shape
for mixed CSG/boolean/pipeline expressions. Any grammar change must update this
contract, the tree-sitter grammar, and the parser/compiler regression corpus in
the same change.
