# 2026-07-03 Report Verification

Computed from the packaged artifact CSV files only.

| Circuit | Objective | HPWL lower | Overflow lower | Post-GRT WL lower | GRT overflow lower | WNS delta | TNS delta | DRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bp_fe` | `obj_34cad18eb2fc5f98` | 13.56% | 72.81% | 45.88% | 83.47% | +0.7927 ns | +187.80 ns | 0->0 |
| `bp_be` | `obj_1c623761f3ce9552` | 53.86% | 63.48% | 68.34% | 91.39% | +0.9414 ns | +1549.32 ns | 0->0 |
| `ethernet` | `obj_705dbe679dcdb0f7` | 47.27% | 20.73% | 36.27% | 83.46% | +0.1901 ns | +15.80 ns | 0->0 |
| `swerv_wrapper` | `obj_4fd265b50589cb27` | 24.59% | 70.09% | 30.35% | 48.33% | +0.1782 ns | +2116.62 ns | 0->0 |

Verification status: PASS
Failed checks: 0
