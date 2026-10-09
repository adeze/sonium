//! Resolved nonlinear port flow: geometry, unsteady Navier–Stokes solve,
//! box/driver coupling, and THD/IMD analysis.
//!
//! `geometry` builds the deterministic 2D slot-port mesh; `geometry3d` builds
//! the current 3D round/flared port mesh. `coupled` advances the resolved
//! Taylor–Hood P2–P1 BDF flow in lockstep with the lumped box/driver ODE,
//! `analysis`/`measure`/`sweep` extract port/diaphragm volume flow, THD/IMD,
//! SPL, and compression sweeps, and `convergence`/`runtime` gate product
//! promotion on independent spatial and temporal ladders that agree at a
//! common fine state.
//!
//! Validated envelope: the strict QA gates cover the 2D path at a converged
//! timestep. The 3D pipe-flow and exact complex-reference gates are nightly
//! tier (`SPEAKER_PORT_3D=1` in `qa-speaker-port-nonlinear`) and must not be
//! reported passed from 2D evidence. Bounded linear straight-tube/port
//! references (`linear_reference`) do not establish nonlinear 3D port-flow
//! convergence at product scale.

pub mod analysis;
pub mod box_driver;
pub mod convergence;
pub mod coupled;
pub mod error;
pub mod geometry;
pub mod geometry3d;
pub mod linear_reference;
pub mod measure;
pub mod runtime;
pub mod sweep;

pub use convergence::{
    PORT_FLOW_PROMOTION_SCHEMA_VERSION, PortFlowConvergenceLevel, PortFlowPhasor,
    PortFlowPromotionEvidence, PortFlowPromotionGate, PortFlowPromotionReport,
    validate_port_flow_promotion,
};
pub use error::PortFlowError;
pub use geometry::{
    MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL, SlotPortGeometry, build_slot_port_mesh,
};
pub use geometry3d::{RoundPortGeometry, build_round_port_mesh, refine_round_port_mesh_projected};
pub use linear_reference::{
    ComplexPortTransfer, RoundPortLinearReferenceConfig, solve_round_port_linear_reference,
};
pub use runtime::{
    PortFlowRuntimeCase, PortFlowRuntimePlan, PortFlowRuntimePromotion,
    PortFlowRuntimePromotionArtifact, PortFlowRuntimeRefinement, replay_port_flow_promotion,
    run_port_flow_promotion,
};
