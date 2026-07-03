# DREAMPlace Patch Boundary

This repository will not vendor DREAMPlace.

The external patch targets:

```text
$DREAMPLACE_ROOT/dreamplace/PlaceObj.py
```

The patch must:

- preserve default DREAMPlace behavior when `custom_objective_spec` is unset,
- keep timing-driven net-weight updates compatible,
- log per-term objective values and gradient norms,
- reject non-finite objectives or gradients,
- expose only gradient-tested terms to new LLM-generated objectives.

Patch file:

```text
patches/dreamplace/custom_objective_placeobj.patch
```

Optional timing-proxy patch:

```text
patches/dreamplace/timing_name_normalization.patch
patches/dreamplace/timing_pin_alias_enhancement.patch
patches/dreamplace/timing_node_names_interface.patch
```

These patches target the DREAMPlace/OpenTimer timing-proxy path. The local
`bp_fe` timing collateral uses placement net names such as
`icache_1/lce/_02063_`, while the OpenTimer Verilog often contains
identifier-safe aliases such as `icache_1_lce__02063_`. The timing patches
resolve common net aliases, reconstruct full `instance:pin` names from
DREAMPlace `node_names + pin_names`, and skip still-unmatched nets/pins instead
of aborting the entire timing proxy run.

The platform-side timing proxy also copies SDC files into each run directory and
performs run-local collateral normalization. For `bp_fe`, this currently fixes
an unambiguous scalar port mismatch (`icache_id_i_0 -> icache_id_i`) and adds
conservative interface defaults when the source SDC lacks input transition or
output load constraints. These generated SDC edits are recorded in
`timing_collateral/sdc_normalization.json` under each timing-proxy cell.

Do not apply or rebuild this timing patch while a long non-timing DREAMPlace
experiment is active. Apply it only before a dedicated OpenTimer-feedback smoke
test, then rebuild DREAMPlace's timing extension and rerun the timing proxy
audit. Treat the resulting WNS/TNS as usable only if the audit produces
successful finite timing metrics and reports the skipped-net count.

The current patch changes:

- `dreamplace/PlaceObj.py`, to evaluate a validated CoEvoP&R `ObjectiveSpec`
  AST over differentiable DREAMPlace tensors.
- `dreamplace/ops/density_potential/density_potential.py`, to make the
  density-potential Gaussian filter compatible with the current PyTorch
  `conv2d` padding and dtype requirements.

New LLM-facing `dreamplace_replacement` objectives can use:

```text
wirelength
wirelength_wawl
wirelength_lse
density
density_electric
density_bell
soft_rudy_mean
soft_rudy_pnorm
route_pressure_long
pin_density_pnorm
pin_count_weighted_wl
```

`wirelength` and `density` are compatibility aliases. `native_objective`,
`density_pnorm`, and `timing_weighted_wirelength` are not exposed to new
LLM-generated objectives.

Tier-2 reports include two distinct controls:

- `default`: native DREAMPlace with `custom_objective_spec` disabled.
- `custom_default`: an internal native-identity custom objective using the
  hidden `native_objective` term. This is not LLM-facing; it only verifies that
  the custom objective hook executes and can reproduce native DREAMPlace.

Fixed symbolic baselines such as `dreamplace_explicit_wl_density` are separate
from `custom_default`.

Before using the patch for experiments, run `dreamplace-term-check` on the
target design family and require finite objective values, nonzero custom
objective calls, and finite nonzero gradient norms.
