# MeshLang grammar 1.0

This document is the normative concrete grammar for `version 1`. The AST JSON
schema is versioned independently; a parser accepts only language versions it
understands and emits the typed `unsupported_version` diagnostic otherwise.

The notation is EBNF: `{ X }` means repetition, `[ X ]` means optional, and
`X ("," X)* [","]` is a comma-separated list with an optional trailing comma.
Keywords introduced for modelling declarations are contextual: they keep their
identifier meaning outside the declaration position that selects a production.

## Lexical grammar

```ebnf
identifier   = ("_" | Unicode alphabetic), { "_" | Unicode alphanumeric } ;
type-ident   = identifier whose first scalar has Unicode uppercase property ;
integer      = ASCII digit, { ASCII digit } ;
float        = (digits, ".", [digits] | ".", digits | digits),
               ("e" | "E"), ["+" | "-"], digits
             | digits, ".", [digits]
             | ".", digits ;
string       = '"', { string-char | escape }, '"' ;
escape       = "\\\"" | "\\\\" | "\\n" | "\\r" | "\\t" | "\\0" ;
line-comment = "//", { scalar except newline } ;
doc-line     = "///", { scalar except newline } ;
block-comment = "/*", { block-comment | scalar }, "*/" ;
doc-block    = "/**", { block-comment | scalar }, "*/" ;
```

Source offsets and spans are UTF-8 byte offsets. All Unicode whitespace is
accepted. Raw newlines and unknown escapes in strings, unfinished nested block
comments, incomplete exponents, non-finite `f64` values, and overflowing
integers are errors. The parser input is Rust `str`; fuzzing arbitrary bytes
uses deterministic UTF-8-lossy conversion and must never split a scalar.

The spellings `|`, `||`, and `|>` have exactly one tokenization: CSG union,
logical OR, and pipe-arrow respectively. There is no parser-side reconstruction
of `|>` from two tokens.

## Source files and declarations

```ebnf
source-file = [ "version", integer, [";"] ], { declaration } ;

declaration = param-decl | const-decl | import-decl | function-decl
            | geometry-decl | mesh-decl | physics-decl | export-decl
            | struct-decl | design-decl | uncertainty-decl | objective-decl
            | constraint-decl | design-region-decl | physical-region-decl
            | boundary-decl | interface-decl | coordinate-system-decl
            | material-region-decl | geometry-field-decl ;

param-decl = "param", identifier, [":", type], "=", expression, [unit] ;
const-decl = "const", identifier, [":", type], "=", expression, [unit] ;
import-decl = "import", string, ["as", identifier]
            | "import", "{", identifier, {",", identifier}, [","], "}",
              "from", string ;
function-decl = ["pub"], "fn", identifier, "(", [parameters], ")",
                ["->", type], block ;
geometry-decl = ["pub"], "geometry", identifier,
                ["(", [parameters], ")"], block ;
parameters = parameter, {",", parameter}, [","] ;
parameter = identifier, ":", type, ["=", expression] ;
struct-decl = ["pub"], "struct", identifier, "{",
              { identifier, ":", type, [","] }, "}" ;
```

## Optimization and physical metadata

Metadata bodies contain key/value fields separated by whitespace or commas.
Required fields are shown without brackets. Unknown or duplicate fields are
diagnosed for closed typed records; boundary/interface/material settings remain
open extension maps.

```ebnf
fields = "{", { identifier, ":", expression, [unit], [","] }, "}" ;

design-decl = "design", identifier, "{",
  "type:", design-type, "initial:", expression,
  ["lower:", expression, "upper:", expression],
  ["unit:", (identifier | string)], ["distribution:", distribution], "}" ;
design-type = "continuous" | "integer" | "boolean" | "shape" | "topology"
            | "level_set" | "categorical", "(", array, ")"
            | "categorical" /* with a values field */ ;
distribution = "deterministic", "(", arguments, ")"
             | "uniform", "(", arguments, ")"
             | "normal", "(", arguments, ")"
             | "log_normal", "(", arguments, ")"
             | "discrete", "(", arguments, ")" ;

uncertainty-decl = "uncertainty", identifier,
                   "{", "distribution:", distribution, "}" ;
objective-decl = "objective", [identifier], "{",
                 "sense:", ("minimize" | "maximize"),
                 "expression:", expression, ["weight:", expression], "}" ;
constraint-decl = "constraint", [identifier], "{", "kind:",
  ( "equality", "left:", expression, "right:", expression,
      ["tolerance:", expression]
  | "inequality", "left:", expression,
      "relation:", ("less" | "less_equal" | "greater" | "greater_equal"),
      "right:", expression
  | "range", "expression:", expression, "lower:", expression,
      "upper:", expression
  | "volume_fraction", "region:", expression, "maximum:", expression
  | "minimum_feature", "region:", expression, "radius:", expression ), "}" ;

design-region-decl = "design_region", identifier, "{",
  "kind:", ("design" | "non_design" | "keep_out"),
  "region:", expression, ["annotations:", array], "}" ;
dimension = "point" | "curve" | "surface" | "volume" ;
physical-region-decl = "physical_region", dimension, identifier, "{",
  "entities:", expression, ["material:", identifier], "}" ;
boundary-decl = "boundary", dimension, identifier, "{",
  "entities:", expression, "kind:", identifier, { field }, "}" ;
interface-decl = "interface", dimension, identifier, "{",
  "primary:", expression, "secondary:", expression,
  "kind:", identifier, { field }, "}" ;
coordinate-system-decl = "coordinate_system", identifier, fields ;
material-region-decl = "material_region", identifier, "{",
  "region:", expression, "material:", identifier, { field }, "}" ;
geometry-field-decl = "field", identifier, fields ;
```

## Blocks, mesh, and types

```ebnf
block = "{", { statement }, "}" ;
statement = let-statement | for-statement | if-statement | group-statement
          | field | expression ;
let-statement = "let", identifier, [":", type], "=", expression ;
for-statement = "for", for-clause, {",", for-clause}, block ;
for-clause = identifier, "in", expression, (".." | "..="), expression ;
if-statement = "if", expression, block, ["else", (block | if-statement)] ;

mesh-decl = "mesh", ["for", identifier], "{", { mesh-statement }, "}" ;
mesh-statement = field | refine | boundary-layer | transfinite | embed
               | periodic | compound | metric ;
metric = "metric", expression, "{", "kind:",
         ("isotropic" | "anisotropic" | "hessian"), { field }, "}" ;

type = identifier | type-ident | "[", type, [";", integer], "]"
     | (identifier | type-ident), "<", type, {",", type}, ">" ;
unit = unit-symbol, { ("*" | "/"), unit-factor } ;
unit-factor = unit-symbol | "(", unit, ")" | unit-symbol, "^", ["+"|"-"], integer ;
```

Compound unit operators after the first symbol are contiguous (`kg/(m*s^2)`)
so recovery cannot consume a following expression. A space before the initial
unit remains valid (`1.0 m`).

## Expressions and precedence

From lowest to highest precedence:

| Level | Operators/forms | Associativity |
|---:|---|---|
| 1 | `value |> function(args)` | left |
| 2 | CSG union/intersection: `|`, `&` | left |
| 3 | logical OR: `||` | left |
| 4 | logical AND: `&&` | left |
| 5 | `== != < > <= >=` | non-associative |
| 6 | `+ -` (including type-resolved CSG difference) | left |
| 7 | `* / %` | left |
| 8 | unary `- !` | prefix |
| 9 | `**` | right |
| 10 | field/index postfix: `.name`, `[index]` | left |
| 11 | literals, identifiers, calls, constructors, arrays, parenthesized/if expressions | n/a |

```ebnf
expression = pipe ;
pipe = csg, { "|>", identifier, "(", [arguments], ")" } ;
csg = logical-or, { ("|" | "&"), logical-or } ;
logical-or = logical-and, { "||", logical-and } ;
logical-and = comparison, { "&&", comparison } ;
comparison = additive, [("=="|"!="|"<"|">"|"<="|">="), additive] ;
additive = multiplicative, {("+"|"-"), multiplicative} ;
multiplicative = unary, {("*"|"/"|"%"), unary} ;
unary = ("-"|"!"), unary | power ;
power = postfix, ["**", unary] ;
postfix = primary, {".", identifier | "[", expression, "]"} ;
arguments = argument, {",", argument}, [","] ;
argument = [identifier, ":"], expression ;
array = "[", [expression, {",", expression}, [","]], "]" ;
```

## Resource and recovery contract

Strict parsing returns the first typed diagnostic. Recovering parsing skips
forward to the next top-level declaration token, always consuming at least one
token after failure. It returns at most the configured diagnostic count. The
default maximum nesting is 128 and the default diagnostic count is 32. These
bounds, UTF-8 scalar advancement, and monotonic recovery are part of grammar
1.0's totality contract.
