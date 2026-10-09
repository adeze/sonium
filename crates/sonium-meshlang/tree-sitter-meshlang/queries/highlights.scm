; tree-sitter highlight queries for MeshLang

; ── Keywords ──────────────────────────────────────────────────
[
  "param"
  "const"
  "import"
  "fn"
  "pub"
  "geometry"
  "mesh"
  "physics"
  "export"
  "struct"
  "let"
  "for"
  "in"
  "if"
  "else"
  "as"
  "from"
  "group"
  "refine"
  "near"
  "on"
  "boundary_layer"
  "default"
  "true"
  "false"
] @keyword

; ── Types ─────────────────────────────────────────────────────
(type_identifier) @type

(constructor_expr
  type_name: (type_identifier) @type.builtin)

; ── Functions ─────────────────────────────────────────────────
(fn_decl
  name: (identifier) @function)

(pipe_expr
  fn: (identifier) @function)

(constructor_expr
  type_name: (type_identifier) @constructor)

; ── Parameters & Variables ────────────────────────────────────
(param_decl
  name: (identifier) @variable.parameter)

(const_decl
  name: (identifier) @constant)

(let_statement
  name: (identifier) @variable)

(param_def
  name: (identifier) @variable.parameter)

; ── Fields & Properties ───────────────────────────────────────
(postfix_expr
  field: (identifier) @property)

(kv_pair
  key: (identifier) @property)

(kv_arg
  key: (identifier) @property)

(argument
  name: (identifier) @property)

(struct_field
  name: (identifier) @property)

; ── Named blocks ──────────────────────────────────────────────
(geometry_block
  name: (identifier) @type.definition)

(mesh_block
  target: (identifier) @type)

(group_decl
  name: (identifier) @label)

(group_decl
  bc_type: (identifier) @attribute)

; ── Literals ──────────────────────────────────────────────────
(integer_literal) @number
(float_literal) @number.float
(string_literal) @string
(bool_literal) @boolean

; ── Units ─────────────────────────────────────────────────────
(simple_unit) @attribute
(compound_unit) @attribute

; ── Operators ─────────────────────────────────────────────────
[
  "+"  "-"  "*"  "/"  "%"  "**"
  "|"  "&"
  "|>"
  "==" "!=" "<" ">" "<=" ">="
  "&&" "||" "!"
  "="
  ".." "..="
] @operator

; ── Punctuation ───────────────────────────────────────────────
[ "(" ")" "{" "}" "[" "]" ] @punctuation.bracket
[ "," ":" "." "->" ] @punctuation.delimiter

; ── Comments ──────────────────────────────────────────────────
(line_comment) @comment
(block_comment) @comment
(doc_comment) @comment.documentation
