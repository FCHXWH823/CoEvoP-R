# 2026-07-03 Report Placement Artifacts

This folder contains the compact placement evidence behind the 2026-07-03 progress report.

Included:

- compressed final DREAMPlace DEFs for the reported designs;
- `default` plus the selected reported objective for each design;
- seeds `1000`, `1001`, and `1002`;
- per-seed DREAMPlace config, run summary, and log text when available;
- Tier-2 placement metric tables and post-GRT metric tables;
- objective JSON specs where the referenced spec was available locally.

Raw `runs/` directories are intentionally not committed. To inspect a placement:

```bash
gzip -dc placements/<design>/<objective_id>/seed_<seed>/macro_placed.gp.def.gz > macro_placed.gp.def
```

Use `manifest.json` for SHA256 hashes and source artifact paths.
