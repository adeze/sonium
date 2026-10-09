/// tree-sitter grammar for MeshLang (.mlg)
/// https://tree-sitter.github.io/tree-sitter/creating-parsers

module.exports = grammar({
  name: 'meshlang',

  extras: $ => [
    /\s/,
    $.line_comment,
    $.block_comment,
    $.doc_comment,
  ],

  word: $ => $.identifier,

  conflicts: $ => [
    [$.primary_expr, $.constructor_expr],
    [$.csg_expr, $.additive_expr],
  ],

  rules: {
    // ── Top Level ────────────────────────────────────────────────
    source_file: $ => repeat($.declaration),

    declaration: $ => choice(
      $.param_decl,
      $.const_decl,
      $.import_decl,
      $.fn_decl,
      $.geometry_block,
      $.mesh_block,
      $.physics_block,
      $.export_block,
      $.struct_decl,
    ),

    // ── Parameters & Constants ───────────────────────────────────
    param_decl: $ => seq(
      'param',
      field('name', $.identifier),
      optional(seq(':', field('type', $.type_expr))),
      '=',
      field('value', $.expression),
      optional(field('unit', $.unit)),
    ),

    const_decl: $ => seq(
      'const',
      field('name', $.identifier),
      optional(seq(':', field('type', $.type_expr))),
      '=',
      field('value', $.expression),
      optional(field('unit', $.unit)),
    ),

    // ── Imports ──────────────────────────────────────────────────
    import_decl: $ => choice(
      // import "path.mlg"
      seq('import', field('path', $.string_literal), optional(seq('as', field('alias', $.identifier)))),
      // import { A, B } from "path.mlg"
      seq('import', '{', commaSep1($.identifier), '}', 'from', field('path', $.string_literal)),
    ),

    // ── Functions ────────────────────────────────────────────────
    fn_decl: $ => seq(
      optional('pub'),
      'fn',
      field('name', $.identifier),
      '(',
      optional(field('params', $.param_list)),
      ')',
      optional(seq('->', field('return_type', $.type_expr))),
      field('body', $.block),
    ),

    param_list: $ => commaSep1($.param_def),

    param_def: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $.type_expr),
      optional(seq('=', field('default', $.expression))),
    ),

    // ── Geometry Block ───────────────────────────────────────────
    geometry_block: $ => seq(
      optional('pub'),
      'geometry',
      field('name', $.identifier),
      optional(seq('(', optional($.param_list), ')')),
      optional(seq('->', '(', optional(field('ports', $.param_list)), ')')),
      field('body', $.block),
    ),

    // ── Mesh Block ───────────────────────────────────────────────
    mesh_block: $ => seq(
      'mesh',
      optional(choice(
        seq('for', field('target', $.identifier)),
        field('target', $.identifier),
      )),
      '{',
      repeat($.mesh_statement),
      '}',
    ),

    mesh_statement: $ => choice(
      $.kv_pair,
      $.refine_statement,
      $.boundary_layer_statement,
    ),

    refine_statement: $ => seq(
      'refine',
      field('kind', $.refine_kind),
      field('body', $.block),
    ),

    refine_kind: $ => choice(
      seq('near', '(', $.expression, ',', $.kv_args, ')'),
      seq('on', '(', $.expression, ')'),
      seq('in', '(', $.expression, ')'),
      seq('field', '(', $.kv_args, ')'),
    ),

    boundary_layer_statement: $ => seq(
      'boundary_layer',
      'on',
      '(',
      field('target', $.expression),
      ')',
      field('body', $.block),
    ),

    // ── Physics Block ────────────────────────────────────────────
    physics_block: $ => seq(
      'physics',
      '{',
      repeat(choice($.group_decl, $.let_statement)),
      '}',
    ),

    group_decl: $ => seq(
      'group',
      field('name', $.identifier),
      ':',
      field('bc_type', $.identifier),
      '=',
      field('value', $.expression),
    ),

    // ── Export Block ─────────────────────────────────────────────
    export_block: $ => seq(
      'export',
      optional(seq('as', field('name', $.string_literal))),
      '{',
      repeat($.kv_pair),
      '}',
    ),

    // ── Struct ───────────────────────────────────────────────────
    struct_decl: $ => seq(
      optional('pub'),
      'struct',
      field('name', $.identifier),
      '{',
      commaSep1($.struct_field),
      optional(','),
      '}',
    ),

    struct_field: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $.type_expr),
    ),

    // ── Statements ───────────────────────────────────────────────
    block: $ => seq('{', repeat($.statement), '}'),

    statement: $ => choice(
      $.let_statement,
      $.expression_statement,
      $.for_statement,
      $.if_statement,
      $.kv_pair,
      $.group_decl,
    ),

    let_statement: $ => seq(
      'let',
      field('name', $.identifier),
      optional(seq(':', field('type', $.type_expr))),
      '=',
      field('value', $.expression),
    ),

    expression_statement: $ => $.expression,

    for_statement: $ => seq(
      'for',
      commaSep1($.for_clause),
      field('body', $.block),
    ),

    for_clause: $ => seq(
      field('var', $.identifier),
      'in',
      field('range', $.range_expr),
    ),

    range_expr: $ => choice(
      seq($.expression, '..', $.expression),
      seq($.expression, '..=', $.expression),
    ),

    if_statement: $ => seq(
      'if',
      field('condition', $.expression),
      field('then', $.block),
      optional(seq('else', field('else', choice($.block, $.if_statement)))),
    ),

    kv_pair: $ => seq(
      field('key', $.identifier),
      ':',
      field('value', $.expression),
      optional(field('unit', $.unit)),
    ),

    kv_args: $ => commaSep1($.kv_arg),

    kv_arg: $ => seq(
      field('key', $.identifier),
      ':',
      field('value', $.expression),
    ),

    // ── Expressions ──────────────────────────────────────────────
    expression: $ => $.pipe_expr,

    pipe_expr: $ => choice(
      seq($.pipe_expr, '|>', field('fn', $.identifier), '(', optional($.argument_list), ')'),
      $.csg_expr,
    ),

    // CSG operators: | (union), - (difference), & (intersection)
    csg_expr: $ => choice(
      prec.left(3, seq($.csg_expr, '|', $.or_expr)),
      prec.left(3, seq($.csg_expr, '-', $.or_expr)),
    prec.left(3, seq($.csg_expr, '&', $.or_expr)),
      $.or_expr,
    ),

    or_expr: $ => choice(
    prec.left(4, seq($.or_expr, '||', $.and_expr)),
      $.and_expr,
    ),

    and_expr: $ => choice(
    prec.left(5, seq($.and_expr, '&&', $.comparison_expr)),
      $.comparison_expr,
    ),

    comparison_expr: $ => choice(
    prec.left(6, seq($.additive_expr, $.comparison_op, $.additive_expr)),
      $.additive_expr,
    ),

    comparison_op: $ => choice('==', '!=', '<', '>', '<=', '>='),

    additive_expr: $ => choice(
    prec.left(7, seq($.additive_expr, '+', $.multiplicative_expr)),
    prec.left(7, seq($.additive_expr, '-', $.multiplicative_expr)),
      $.multiplicative_expr,
    ),

    multiplicative_expr: $ => choice(
    prec.left(8, seq($.multiplicative_expr, '*', $.unary_expr)),
    prec.left(8, seq($.multiplicative_expr, '/', $.unary_expr)),
    prec.left(8, seq($.multiplicative_expr, '%', $.unary_expr)),
      $.unary_expr,
    ),

    unary_expr: $ => choice(
    prec(9, seq('-', $.unary_expr)),
    prec(9, seq('!', $.unary_expr)),
      $.power_expr,
    ),

    power_expr: $ => choice(
    prec.right(10, seq($.postfix_expr, '**', $.unary_expr)),
      $.postfix_expr,
    ),

    postfix_expr: $ => choice(
      // Method-style: x.foo(args)  — but we'll handle this as dot + call
    prec.left(11, seq($.postfix_expr, '.', field('field', $.identifier))),
    prec.left(11, seq($.postfix_expr, '[', $.expression, ']')),
      $.primary_expr,
    ),

  primary_expr: $ => choice(
      $.identifier,
      $.integer_literal,
      $.float_literal,
      $.string_literal,
      $.bool_literal,
      $.constructor_expr,
      $.paren_expr,
      $.array_literal,
    ),

constructor_expr: $ => prec(12, seq(
      field('type_name', $.type_identifier),
      '(',
      optional($.argument_list),
      ')',
    )),

    paren_expr: $ => seq('(', $.expression, ')'),

    array_literal: $ => seq('[', commaSep($.expression), ']'),

    argument_list: $ => commaSep1($.argument),

    argument: $ => choice(
      // Named argument: key: value
      seq(field('name', $.identifier), ':', field('value', $.expression)),
      // Positional argument
      field('value', $.expression),
    ),

    // ── Types ────────────────────────────────────────────────────
    type_expr: $ => choice(
      $.type_identifier,
      $.array_type,
      $.generic_type,
    ),

    type_identifier: $ => /[A-Z][a-zA-Z0-9_]*/,

    array_type: $ => choice(
      seq('[', $.type_expr, ';', $.integer_literal, ']'),  // fixed size
      seq('[', $.type_expr, ']'),                           // dynamic
    ),

    generic_type: $ => seq(
      $.type_identifier,
      '<',
      commaSep1($.type_expr),
      '>',
    ),

    // ── Units ────────────────────────────────────────────────────
    unit: $ => prec(14, choice(
      $.simple_unit,
      $.compound_unit,
    )),

    simple_unit: $ => choice(
      'm', 'mm', 'cm', 'um', 'in',
      'deg', 'rad',
      'Hz', 'kHz',
      's', 'ms',
      'Pa', 'kPa',
    ),

    compound_unit: $ => seq(
      $.simple_unit,
      '/',
      $.simple_unit,
    ),

    // ── Literals ─────────────────────────────────────────────────
    identifier: $ => /[a-z_][a-zA-Z0-9_]*/,

    integer_literal: $ => /[0-9]+/,

    float_literal: $ => token(choice(
      /[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?/,
      /[0-9]+[eE][+-]?[0-9]+/,
    )),

    string_literal: $ => /"[^"]*"/,

    bool_literal: $ => choice('true', 'false'),

    // ── Comments ─────────────────────────────────────────────────
    line_comment: $ => /\/\/[^\n]*/,
    block_comment: $ => /\/\*[^*]*\*+([^/*][^*]*\*+)*\//,
    doc_comment: $ => /\/\/\/[^\n]*/,
  },
});

// ── Helpers ────────────────────────────────────────────────────
function commaSep(rule) {
  return optional(commaSep1(rule));
}

function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)), optional(','));
}
