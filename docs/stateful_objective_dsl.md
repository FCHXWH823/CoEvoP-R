# Trajectory-Aware Objective Interface

CoEvoP&R uses the `typed_policy_v1` interface to express differentiable objectives whose physical pressures and smoothing behavior adapt along the placement trajectory.

```python
def init_policy(obs):
    return {
        "density_weight": 0.00008 * obs("grad_ratio_component_density"),
        "gamma_scale": 1.0,
        "route_weight": 0.0,
    }

def update_policy(policy, obs):
    return {
        "density_weight": policy("density_weight") *
            (0.95 + 0.1 * sigmoid(-2000.0 * obs("hpwl_delta_rate"))),
        "gamma_scale": 0.1 + 0.9 * sigmoid(8.0 * (obs("overflow") - 0.5)),
        "route_weight": policy("route_weight") +
            0.001 * sigmoid(6.0 * (0.5 - obs("iter_frac"))),
    }

def objective(features, policy):
    wl = term("wirelength_wawl")
    density = term("density_electric")
    routing = log1p(term("soft_rudy_pnorm"))
    loss = wl + policy("density_weight") * density + \
        policy("route_weight") * routing
    return loss, {
        "wirelength": wl,
        "density": density,
        "routing": routing,
    }
```

## Program fields

`init_policy` creates named scalar state. `update_policy` updates the same state once per placement iteration from the previous state and current progress observables. `objective` returns a differentiable scalar loss and named components for execution tracing.

Required state variables are `density_weight` and `gamma_scale`. Optional variables include `route_weight` and `pin_weight`. An optional `update_net_weights` function may use normalized placement-time criticality, span, and fanout.

## Progress observables

| Observable | Meaning |
|---|---|
| `iter_frac` | completed fraction of the placement budget |
| `overflow` | current density overflow |
| `hpwl_delta_rate` | relative HPWL change since the previous update |
| `gamma_frac` | current smoothing scale relative to the base value |
| `grad_ratio_<term>` | initial wirelength to term gradient ratio |
| `init_value_<term>` | initial raw term value |
| `grad_ratio_component_<name>` | initial gradient ratio of a named component |
| `init_value_component_<name>` | initial value of a named component |

## Physical roles

Wirelength supplies the smooth analytical-placement anchor. `density_weight` controls the density component and DREAMPlace preconditioner. `gamma_scale` controls wirelength smoothing within configured bounds. `route_weight` and `pin_weight` scale their corresponding differentiable pressure families. Net-level timing features control bounded per-net multipliers when timing weighting is enabled.

## Validation and execution

The parser accepts constants, local variables, approved arithmetic, safe division, and smooth operators. It produces data rather than executable Python. Validation checks the program schema, expression grammar, physical roles, state consistency, bounds, and term availability. DREAMPlace then checks finite scalar values, finite gradients, state updates, component traces, and completion of the configured placement budget.

During execution, policy updates commit synchronously. DREAMPlace applies the density state to the objective and preconditioner, applies the smoothing state to the active wirelength operator, evaluates the scalar loss, and records objective components together with state and placement trajectories.
