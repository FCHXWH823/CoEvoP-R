# DREAMPlace Integration

CoEvoP&R keeps DREAMPlace as an external dependency. The integration patch binds a validated `typed_policy_v1` objective to the placement runtime and records the resulting optimization trajectory.

Apply the integration through the artifact command.

```bash
python3 -m coevop.cli dreamplace-status
python3 -m coevop.cli dreamplace-apply-patch
python3 -m coevop.cli dreamplace-status
```

The final status must report `placeobj_patch_applied: true`.

## Runtime boundary

`custom_objective_placeobj.patch` adds the following capabilities.

- Raw differentiable wirelength, density, routing, and pin-pressure terms
- Typed density, smoothing, routing, and pin schedules
- Density-state coupling to the objective and placer preconditioner
- Bounded smoothing control around the DREAMPlace base gamma
- Optional per-net weighting from timing criticality, span, and fanout
- Finite-value and gradient checks
- Named component, state, coefficient, HPWL, overflow, and gradient traces

Native DREAMPlace remains the matched reference when no objective program is supplied. The `dreamplace_controller_native_identity` preset verifies the typed objective execution path.

## Timing support

The timing patches adapt the bundled OpenTimer interface to flattened ChiPBench collateral and robust per-net updates.

```text
timing_name_normalization.patch
timing_pin_alias_enhancement.patch
timing_node_names_interface.patch
timing_net_weighting_robustness.patch
timing_driver_root.patch
timing_driver_pin_order.patch
timing_high_degree_guard.patch
```

The platform prepares run-local Verilog and SDC files and records every normalization in the timing run directory. Timing evidence is admitted only when WNS and TNS are finite and the configured net-coverage threshold is satisfied.

## Compatibility patches

`numpy2_compat.patch` supports NumPy 2.x installations. `upstream_4_0_modern_cuda.patch` and `upstream_4_0_movable_block_compat.patch` provide the tracked DREAMPlace 4.0 build and mixed-size compatibility changes.
