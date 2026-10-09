(* Published values from Wolfram verification case
   FluidDynamics-FEM-Stationary-2D-FluidDynamics-0001. *)
payload = <|
  "schema_version" -> 1,
  "channel_dimensions_m" -> {2.2, 0.41},
  "cylinder_center_m" -> {0.2, 0.2},
  "cylinder_radius_m" -> 0.05,
  "pressure_difference_pa" -> 0.11752016697,
  "drag_coefficient" -> 5.57953523384,
  "lift_coefficient" -> 0.010618948146
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
