//! CPU-only MeshLang AST visitor coverage contract.
//!
//! This is a structural traversal check. It constructs a bounded AST fixture;
//! it does not emit GEO/Python, invoke Gmsh, or start a simulation.

use meshlang_ast::span::Span;
use meshlang_ast::visitor::{Visitor, walk_expr};
use meshlang_ast::{
    Argument, ArrayExpr, BinaryExpr, BinaryOp, Block, CallExpr, ConstructorExpr, Expr,
    FieldAccessExpr, Ident, IfStatement, IndexExpr, KeyValue, Literal, MeshBlock, MeshMetricKind,
    MeshMetricStatement, MeshStatement, ParenExpr, PipeExpr, RefineKind, RefineStatement,
    Statement, StructInitExpr, StructInitField, TransfiniteKind, TransfiniteStatement, UnaryExpr,
    UnaryOp,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "meshlang.visitor-reference.v1";

struct CountingVisitor {
    expressions: usize,
}

impl Visitor for CountingVisitor {
    fn visit_expr(&mut self, expression: &Expr) {
        self.expressions += 1;
        walk_expr(self, expression);
    }
}

fn ident(name: &str) -> Ident {
    Ident {
        name: name.into(),
        span: Span::DUMMY,
    }
}

fn literal(value: i64) -> Expr {
    Expr::literal(Literal::Int(value), Span::DUMMY)
}

fn argument(value: i64) -> Argument {
    Argument {
        name: Some(ident("value")),
        value: literal(value),
        span: Span::DUMMY,
    }
}

fn setting(value: i64) -> KeyValue {
    KeyValue {
        key: ident("value"),
        value: literal(value),
        unit: None,
        span: Span::DUMMY,
    }
}

fn expression_fixture() -> Vec<Expr> {
    vec![
        Expr::Ident(ident("identifier")),
        literal(0),
        Expr::Binary(BinaryExpr {
            left: Box::new(literal(1)),
            op: BinaryOp::Add,
            right: Box::new(literal(2)),
            span: Span::DUMMY,
        }),
        Expr::Unary(UnaryExpr {
            op: UnaryOp::Neg,
            operand: Box::new(literal(3)),
            span: Span::DUMMY,
        }),
        Expr::Pipe(PipeExpr {
            value: Box::new(literal(4)),
            function: ident("translate"),
            args: vec![argument(5)],
            span: Span::DUMMY,
        }),
        Expr::FieldAccess(FieldAccessExpr {
            object: Box::new(literal(6)),
            field: ident("x"),
            span: Span::DUMMY,
        }),
        Expr::Index(IndexExpr {
            object: Box::new(literal(7)),
            index: Box::new(literal(8)),
            span: Span::DUMMY,
        }),
        Expr::Constructor(ConstructorExpr {
            type_name: ident("Point"),
            args: vec![argument(9)],
            span: Span::DUMMY,
        }),
        Expr::Call(CallExpr {
            function: ident("size"),
            args: vec![argument(10)],
            span: Span::DUMMY,
        }),
        Expr::Paren(ParenExpr::new(literal(11), Span::DUMMY)),
        Expr::Array(ArrayExpr {
            elements: vec![literal(12), literal(13)],
            span: Span::DUMMY,
        }),
        Expr::StructInit(StructInitExpr {
            type_name: ident("Frame"),
            fields: vec![StructInitField {
                name: ident("origin"),
                value: literal(14),
                span: Span::DUMMY,
            }],
            span: Span::DUMMY,
        }),
        Expr::If(Box::new(IfStatement {
            condition: literal(15),
            then_block: Block {
                statements: vec![Statement::Expr(literal(16))],
                span: Span::DUMMY,
            },
            else_block: None,
            span: Span::DUMMY,
        })),
    ]
}

fn run() -> Result<QaResult, String> {
    let expressions = expression_fixture();
    if Expr::VARIANT_NAMES.len() != expressions.len() {
        return Err(format!(
            "expression inventory has {} variants, fixture has {}",
            Expr::VARIANT_NAMES.len(),
            expressions.len()
        ));
    }
    let mut visitor = CountingVisitor { expressions: 0 };
    for expression in expressions {
        visitor.visit_expr(&expression);
    }
    if visitor.expressions != 29 {
        return Err(format!(
            "expression visitor reached {} nodes, expected 29",
            visitor.expressions
        ));
    }

    let mesh = MeshBlock {
        target: None,
        statements: vec![
            MeshStatement::Refine(RefineStatement {
                kind: RefineKind::On { entity: literal(1) },
                body: vec![setting(2)],
                span: Span::DUMMY,
            }),
            MeshStatement::Refine(RefineStatement {
                kind: RefineKind::In { region: literal(3) },
                body: vec![setting(4)],
                span: Span::DUMMY,
            }),
            MeshStatement::Refine(RefineStatement {
                kind: RefineKind::Field {
                    args: vec![setting(5)],
                },
                body: vec![setting(6)],
                span: Span::DUMMY,
            }),
            MeshStatement::Transfinite(TransfiniteStatement {
                kind: TransfiniteKind::Surface {
                    entity: literal(7),
                    settings: vec![setting(8)],
                },
                span: Span::DUMMY,
            }),
            MeshStatement::Transfinite(TransfiniteStatement {
                kind: TransfiniteKind::Volume {
                    entity: literal(9),
                    settings: vec![setting(10)],
                },
                span: Span::DUMMY,
            }),
            MeshStatement::Metric(MeshMetricStatement {
                target: literal(11),
                metric: MeshMetricKind::Anisotropic {
                    components: vec![literal(12), literal(13)],
                },
                coordinate_system: Some(ident("local")),
                span: Span::DUMMY,
            }),
            MeshStatement::Metric(MeshMetricStatement {
                target: literal(14),
                metric: MeshMetricKind::Hessian {
                    field: literal(15),
                    minimum_size: literal(16),
                    maximum_size: literal(17),
                },
                coordinate_system: None,
                span: Span::DUMMY,
            }),
        ],
        span: Span::DUMMY,
    };
    let mut mesh_visitor = CountingVisitor { expressions: 0 };
    mesh_visitor.visit_mesh_block(&mesh);
    if mesh_visitor.expressions != 17 {
        return Err(format!(
            "mesh visitor reached {} nodes, expected 17",
            mesh_visitor.expressions
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("expression_checks", 13.0)
        .minimum("expression_checks", 13.0)
        .metric("expression_visit_count", 29.0)
        .minimum("expression_visit_count", 29.0)
        .metric("mesh_branch_checks", 7.0)
        .minimum("mesh_branch_checks", 7.0)
        .metric("mesh_visit_count", 17.0)
        .minimum("mesh_visit_count", 17.0)
        .input_checksum("source:meshlang-visitor-reference:v1")
        .reference("AST visitor expression inventory and mesh-control branch fixture")
        .reference_checksum("analytic/deterministic-fixture:meshlang-visitor-v1")
        .dimension("allocation-free immutable AST traversal")
        .formulation("MeshLang visitor contract")
        .seed("meshlang-visitor-reference:cpu:v1"))
}

fn main() {
    let started = std::time::Instant::now();
    let result = match run() {
        Ok(result) => result.elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => {
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64)
        }
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
    if result.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
