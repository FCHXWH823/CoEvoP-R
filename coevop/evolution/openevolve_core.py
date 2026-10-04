"""OpenEvolve-style memory primitives for CoEvoP&R objective evolution.

This module adapts the architectural pattern of OpenEvolve's program database,
prompt memory, islands, and trace logging to CoEvoP&R's restricted
``ObjectiveSpec`` boundary. It intentionally does not execute LLM-generated
Python code; generated objective programs are parsed elsewhere into safe JSON
specifications before they can reach DREAMPlace.

OpenEvolve reference: https://github.com/algorithmicsuperintelligence/openevolve
License of referenced project: Apache-2.0.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from coevop.objectives.spec import ObjectiveSpec, objective_spec_to_dict
from coevop.prompt.sampler import ROUTED_EVIDENCE_KEYS, CoEvoPromptSampler

FEATURE_SCHEMA_VERSION = 2
DEFAULT_FEATURE_DIMENSIONS = (
    "complexity",
    "mechanism_family",
    "hpwl_delta_bucket",
    "overflow_delta_bucket",
    "opentimer_wns_delta_bucket",
)
TIMING_PROXY_WNS_DELTA_MIN_ABS_NS = 0.05


@dataclass
class ObjectiveProgram:
    id: str
    code: str
    objective_spec: dict[str, Any] | None = None
    parent_id: str | None = None
    generation: int = 0
    iteration_found: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    prompt: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    failure_reason: str | None = None
    status: str = "unknown"
    complexity: int = 0
    term_set: list[str] = field(default_factory=list)
    island: int = 0
    feature_coords: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_spec(
        cls,
        spec: ObjectiveSpec,
        *,
        program_id: str | None = None,
        code: str | None = None,
        parent_id: str | None = None,
        generation: int = 0,
        iteration_found: int = 0,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        prompt: dict[str, Any] | None = None,
        raw_response: dict[str, Any] | None = None,
        failure_reason: str | None = None,
        status: str = "accepted",
    ) -> "ObjectiveProgram":
        return cls(
            id=program_id or spec.id,
            code=code or objective_code_from_spec(spec),
            objective_spec=objective_spec_to_dict(spec),
            parent_id=parent_id,
            generation=generation,
            iteration_found=iteration_found,
            metrics=dict(metrics or {}),
            artifacts=dict(artifacts or {}),
            prompt=prompt,
            raw_response=raw_response,
            failure_reason=failure_reason,
            status=status,
            complexity=spec.complexity,
            term_set=list(spec.term_set),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectiveProgram":
        return cls(
            id=str(payload["id"]),
            code=str(payload.get("code", "")),
            objective_spec=payload.get("objective_spec"),
            parent_id=payload.get("parent_id"),
            generation=int(payload.get("generation", 0)),
            iteration_found=int(payload.get("iteration_found", 0)),
            metrics=dict(payload.get("metrics", {})),
            artifacts=dict(payload.get("artifacts", {})),
            prompt=payload.get("prompt"),
            raw_response=payload.get("raw_response"),
            failure_reason=payload.get("failure_reason"),
            status=str(payload.get("status", "unknown")),
            complexity=int(payload.get("complexity", 0)),
            term_set=[str(term) for term in payload.get("term_set", [])],
            island=int(payload.get("island", 0)),
            feature_coords=dict(payload.get("feature_coords", {})),
            timestamp=float(payload.get("timestamp", time.time())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def combined_score(self) -> float:
        return _finite_float(self.metrics.get("combined_score"), default=0.0)

    @property
    def is_parent_eligible(self) -> bool:
        if self.status != "accepted":
            return False
        if self.metrics.get("structural_failure_count", 0):
            return False
        if self.metrics.get("bootstrap_parent") or self.metrics.get("manual_safe_baseline"):
            return not bool(self.metrics.get("negative_memory_only"))
        if self.metrics.get("parent_eligible") is False:
            return False
        if self.metrics.get("negative_memory_only"):
            return False
        if self.metrics.get("hpwl_gate_passed") is False:
            return False
        if self.metrics.get("overflow_gate_passed") is False:
            return False
        if self.combined_score > 0.0:
            return True
        return bool(self.metrics.get("manual_safe_baseline"))

    @property
    def is_elite_eligible(self) -> bool:
        if self.status != "accepted":
            return False
        if self.metrics.get("exploration_parent_only"):
            return False
        if (
            not self.metrics.get("manuscript_multimetric_selection")
            and
            self.metrics.get("baseline_portfolio_compared")
            and not self.metrics.get("beats_baseline_portfolio")
        ):
            return False
        if self.metrics.get("elite_gate_passed") is False:
            return False
        if self.metrics.get("negative_memory_only"):
            return False
        if self.metrics.get("structural_failure_count", 0):
            return False
        if self.metrics.get("severe_regression_count", 0):
            return False
        return self.combined_score > 0.0


@dataclass(frozen=True)
class ObjectiveDatabaseConfig:
    population_size: int = 100
    archive_size: int = 20
    num_islands: int = 5
    map_elites_enabled: bool = True
    feature_bins: int = 8
    feature_dimensions: tuple[str, ...] = DEFAULT_FEATURE_DIMENSIONS
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    timing_proxy_wns_delta_min_abs_ns: float = TIMING_PROXY_WNS_DELTA_MIN_ABS_NS
    migration_interval: int = 20
    migration_rate: float = 0.10
    migration_topology: str = "ring"
    # "generation": every migration_interval evolution generations, all
    # populated islands exchange elites (Algorithm 1). "island_generation":
    # an island migrates after migration_interval of its own generations.
    migration_clock: str = "generation"


class ObjectiveProgramDatabase:
    def __init__(self, root: str | Path, config: ObjectiveDatabaseConfig | None = None) -> None:
        self.root = Path(root)
        self.config = config or ObjectiveDatabaseConfig()
        self.programs: dict[str, ObjectiveProgram] = {}
        self.islands: list[set[str]] = [set() for _ in range(max(1, self.config.num_islands))]
        self.island_feature_maps: list[dict[str, str]] = [
            {} for _ in range(max(1, self.config.num_islands))
        ]
        self.island_generations: list[int] = [0 for _ in range(max(1, self.config.num_islands))]
        self.island_last_migrated_generations: list[int] = [
            0 for _ in range(max(1, self.config.num_islands))
        ]
        self.archive: set[str] = set()
        self.best_program_id: str | None = None
        self.last_iteration: int = 0
        self.last_migration_iteration: int = 0
        self.migration_events: list[dict[str, Any]] = []
        self.feature_recode_summary: dict[str, Any] = {}

    @property
    def is_empty(self) -> bool:
        return not self.programs

    def add(self, program: ObjectiveProgram, *, target_island: int | None = None) -> ObjectiveProgram:
        island = self._assign_island(program, target_island)
        program.island = island
        program.feature_coords = self._feature_coords(program)
        self.programs[program.id] = program
        self.islands[island].add(program.id)
        self.last_iteration = max(self.last_iteration, program.iteration_found)
        self._update_feature_map(program)
        self._update_archive(program)
        self._update_best(program)
        self._enforce_population_limit()
        return program

    def rebuild_indexes(self) -> None:
        """Rebuild quality-dependent indexes after population-level rescoring."""

        self.archive.clear()
        self.best_program_id = None
        self.island_feature_maps = [{} for _ in range(max(1, self.config.num_islands))]
        for program in self.programs.values():
            program.feature_coords = self._feature_coords(program)
            self._update_feature_map(program)
            self._update_best(program)
        self._rebuild_archive_from_feature_maps()

    def get(self, program_id: str | None) -> ObjectiveProgram | None:
        if not program_id:
            return None
        return self.programs.get(program_id)

    def sample_parent(
        self,
        *,
        rng: random.Random,
        island_id: int | None = None,
        parent_policy: str = "score_weighted",
    ) -> ObjectiveProgram:
        if not self.programs:
            raise ValueError("cannot sample parent from empty objective database")
        if island_id is None:
            non_empty = [index for index, island in enumerate(self.islands) if island]
            island_id = rng.choice(non_empty) if non_empty else 0
        candidates = self._island_parent_pool(island_id)
        if not candidates:
            candidates = list(self.programs.values())
        eligible_parent_policies = {
            "hpwl_safe_only",
            "manuscript_multiobjective",
            "pareto_multiobjective",
        }
        if parent_policy in eligible_parent_policies:
            candidates = [candidate for candidate in candidates if candidate.is_parent_eligible]
            candidates = _prefer_discovered_candidates(candidates)
            if not candidates:
                fallback = [
                    candidate for candidate in self.programs.values() if candidate.is_parent_eligible
                ]
                candidates = _prefer_discovered_candidates(fallback)
            if not candidates:
                raise ValueError(
                    "no evaluator-approved parent-eligible objectives are available"
                )
        elif parent_policy != "score_weighted":
            raise ValueError(f"unknown parent policy: {parent_policy}")
        novelty_aware = parent_policy in eligible_parent_policies
        weights = _parent_sample_weights(candidates, novelty_aware=novelty_aware)
        return rng.choices(candidates, weights=weights, k=1)[0]

    def top_programs(self, limit: int = 3, *, eligible_only: bool = False) -> list[ObjectiveProgram]:
        pool = list(self.programs.values())
        if eligible_only:
            pool = [program for program in pool if program.is_parent_eligible]
        candidates = sorted(
            pool,
            key=lambda program: (
                program.combined_score,
                -int(program.metrics.get("structural_failure_count", 0) or 0),
                -int(program.metrics.get("severe_regression_count", 0) or 0),
            ),
            reverse=True,
        )
        return candidates[: max(0, limit)]

    def diverse_programs(
        self,
        *,
        parent_id: str | None = None,
        limit: int = 2,
        rng: random.Random,
        include_negative: bool = False,
    ) -> list[ObjectiveProgram]:
        seen: set[tuple[Any, ...]] = set()
        candidates = self._mapped_program_pool()
        if not candidates:
            candidates = list(self.programs.values())
        if not include_negative:
            candidates = [program for program in candidates if program.is_parent_eligible]
        rng.shuffle(candidates)
        candidates.sort(key=lambda program: program.combined_score, reverse=True)
        selected = []
        for program in candidates:
            if program.id == parent_id:
                continue
            coords = program.feature_coords or self._feature_coords(program)
            signature = (
                program.island,
                coords.get("mechanism_family"),
                coords.get("hpwl_delta_bucket"),
                coords.get("overflow_delta_bucket"),
                coords.get("opentimer_wns_delta_bucket"),
            )
            if signature in seen:
                continue
            selected.append(program)
            seen.add(signature)
            if len(selected) >= limit:
                break
        return selected

    def increment_island_generation(self, island_id: int) -> int:
        island = int(island_id) % len(self.islands)
        self.island_generations[island] += 1
        return self.island_generations[island]

    def maybe_migrate(self, *, iteration: int, rng: random.Random) -> list[ObjectiveProgram]:
        interval = int(self.config.migration_interval)
        if interval <= 0 or len(self.islands) <= 1:
            return []
        if self.config.migration_clock == "generation":
            if (
                iteration <= 0
                or iteration % interval != 0
                or self.last_migration_iteration == iteration
            ):
                return []
            active = list(range(len(self.islands)))
            self.last_migration_iteration = iteration
        elif self.config.migration_clock == "island_generation":
            active = [
                index
                for index, count in enumerate(self.island_generations)
                if count > 0 and count % interval == 0
                and self.island_last_migrated_generations[index] != count
            ]
        else:
            raise ValueError(f"unknown migration clock: {self.config.migration_clock}")
        # Snapshot every source island first so an elite received in this
        # round is not forwarded again within the same round.
        elites_by_source = {
            source: sorted(
                (
                    self.programs[pid]
                    for pid in self.islands[source]
                    if pid in self.programs and self.programs[pid].is_parent_eligible
                ),
                key=lambda program: program.combined_score,
                reverse=True,
            )
            for source in active
        }
        migrants: list[ObjectiveProgram] = []
        for source in active:
            targets = self._migration_targets(source)
            source_programs = elites_by_source[source]
            if not targets or not source_programs:
                continue
            count = max(1, int(math.ceil(len(source_programs) * float(self.config.migration_rate))))
            for target in targets:
                copied = 0
                target_signatures = {
                    _program_signature(self.programs[pid])
                    for pid in self.islands[target]
                    if pid in self.programs
                }
                for program in source_programs:
                    if copied >= count:
                        break
                    signature = _program_signature(program)
                    if signature in target_signatures:
                        continue
                    migrant = self._copy_for_migration(
                        program,
                        source_island=source,
                        target_island=target,
                        iteration=iteration,
                        ordinal=copied,
                    )
                    self.add(migrant, target_island=target)
                    target_signatures.add(signature)
                    migrants.append(migrant)
                    copied += 1
                self.migration_events.append(
                    {
                        "iteration": iteration,
                        "source_island": source,
                        "target_island": target,
                        "requested": count,
                        "migrated": copied,
                    }
                )
            self.island_last_migrated_generations[source] = self.island_generations[source]
        if migrants:
            rng.shuffle(migrants)
        return migrants

    def map_elites_summary(self) -> dict[str, Any]:
        occupied = [len(feature_map) for feature_map in self.island_feature_maps]
        return {
            "enabled": bool(self.config.map_elites_enabled),
            "feature_schema_version": self.config.feature_schema_version,
            "feature_dimensions": list(self.config.feature_dimensions),
            "feature_recode_summary": dict(self.feature_recode_summary),
            "feature_bins": self.config.feature_bins,
            "occupied_cells_by_island": occupied,
            "occupied_cells_total": sum(occupied),
            "archive_program_ids": sorted(self.archive),
            "migration_events": list(self.migration_events[-20:]),
        }

    def recent_failures(self, limit: int = 3) -> list[ObjectiveProgram]:
        failures = [
            program
            for program in self.programs.values()
            if program.status != "accepted"
            or program.metrics.get("structural_failure_count")
            or program.metrics.get("severe_regression_count")
        ]
        failures.sort(key=lambda program: (program.iteration_found, program.timestamp), reverse=True)
        return failures[: max(0, limit)]

    def recent_negative_memory(self, limit: int = 3) -> list[ObjectiveProgram]:
        negatives = [
            program
            for program in self.programs.values()
            if program.metrics.get("negative_memory_only")
            or program.status != "accepted"
            or program.metrics.get("structural_failure_count")
            or program.metrics.get("severe_regression_count")
        ]
        negatives.sort(key=lambda program: (program.iteration_found, program.timestamp), reverse=True)
        return negatives[: max(0, limit)]

    def near_miss_programs(self, limit: int = 5) -> list[ObjectiveProgram]:
        """Programs with useful partial evidence that are not parent-eligible."""
        candidates = []
        for program in self.programs.values():
            metrics = program.metrics
            if program.status != "accepted" or program.is_parent_eligible:
                continue
            if metrics.get("structural_failure_count"):
                continue
            has_partial_signal = any(
                bool(metrics.get(key))
                for key in (
                    "hpwl_gate_passed",
                    "overflow_gate_passed",
                    "custom_default_hpwl_gate_passed",
                    "custom_default_overflow_gate_passed",
                    "custom_default_effect_gate_passed",
                )
            )
            has_design_signal = _finite_float(
                metrics.get("design_both_improvement_fraction"),
                default=0.0,
            ) > 0.0
            if has_partial_signal or has_design_signal:
                candidates.append(program)
        candidates.sort(
            key=lambda program: (
                bool(program.metrics.get("custom_default_gate_passed")),
                _finite_float(program.metrics.get("custom_default_effect_pct"), default=0.0),
                _finite_float(program.metrics.get("combined_score"), default=0.0),
                program.iteration_found,
                program.timestamp,
            ),
            reverse=True,
        )
        return candidates[: max(0, limit)]

    def save(self, iteration: int | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        programs_dir = self.root / "programs"
        programs_dir.mkdir(parents=True, exist_ok=True)
        for program in self.programs.values():
            _write_json(program.to_dict(), programs_dir / f"{program.id}.json")
        metadata = {
            "archive": sorted(self.archive),
            "best_program_id": self.best_program_id,
            "islands": [sorted(island) for island in self.islands],
            "island_feature_maps": [
                dict(sorted(feature_map.items()))
                for feature_map in self.island_feature_maps
            ],
            "island_generations": list(self.island_generations),
            "island_last_migrated_generations": list(self.island_last_migrated_generations),
            "last_migration_iteration": self.last_migration_iteration,
            "migration_events": list(self.migration_events),
            "last_iteration": self.last_iteration if iteration is None else iteration,
            "config": asdict(self.config),
            "feature_schema_version": self.config.feature_schema_version,
            "feature_dimensions": list(self.config.feature_dimensions),
            "feature_recode_summary": dict(self.feature_recode_summary),
        }
        _write_json(metadata, self.root / "metadata.json")

    def load(self) -> None:
        metadata_path = self.root / "metadata.json"
        programs_dir = self.root / "programs"
        if not metadata_path.is_file() or not programs_dir.is_dir():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_recode_summary = dict(metadata.get("feature_recode_summary", {}))
        self.programs = {}
        for path in programs_dir.glob("*.json"):
            program = ObjectiveProgram.from_dict(json.loads(path.read_text(encoding="utf-8")))
            self.programs[program.id] = program
        self.archive = {str(pid) for pid in metadata.get("archive", []) if pid in self.programs}
        self.best_program_id = metadata.get("best_program_id")
        if self.best_program_id not in self.programs:
            self.best_program_id = None
        saved_islands = metadata.get("islands", [])
        self.islands = [set() for _ in range(max(1, self.config.num_islands))]
        self.island_feature_maps = [{} for _ in range(max(1, self.config.num_islands))]
        self.island_generations = [0 for _ in range(max(1, self.config.num_islands))]
        self.island_last_migrated_generations = [
            0 for _ in range(max(1, self.config.num_islands))
        ]
        for index, island in enumerate(saved_islands[: len(self.islands)]):
            for program_id in island:
                if program_id in self.programs:
                    self.islands[index].add(program_id)
                    self.programs[program_id].island = index
        for program in self.programs.values():
            if not any(program.id in island for island in self.islands):
                self.islands[program.island % len(self.islands)].add(program.id)
        saved_feature_maps = metadata.get("island_feature_maps", [])
        old_feature_count = sum(
            len(feature_map) for feature_map in saved_feature_maps if isinstance(feature_map, dict)
        )
        needs_recode = (
            int(metadata.get("feature_schema_version", 0) or 0) != self.config.feature_schema_version
            or tuple(metadata.get("feature_dimensions", [])) != tuple(self.config.feature_dimensions)
        )
        if needs_recode:
            self._recode_loaded_feature_maps(occupied_cells_before=old_feature_count)
        else:
            for index, feature_map in enumerate(saved_feature_maps[: len(self.island_feature_maps)]):
                if not isinstance(feature_map, dict):
                    continue
                self.island_feature_maps[index] = {
                    str(cell): str(program_id)
                    for cell, program_id in feature_map.items()
                    if program_id in self.programs
                }
            if not any(self.island_feature_maps):
                self._recode_loaded_feature_maps(occupied_cells_before=old_feature_count)
        saved_generations = metadata.get("island_generations", [])
        for index, value in enumerate(saved_generations[: len(self.island_generations)]):
            try:
                self.island_generations[index] = int(value)
            except (TypeError, ValueError):
                self.island_generations[index] = 0
        saved_migrated_generations = metadata.get("island_last_migrated_generations", [])
        for index, value in enumerate(
            saved_migrated_generations[: len(self.island_last_migrated_generations)]
        ):
            try:
                self.island_last_migrated_generations[index] = int(value)
            except (TypeError, ValueError):
                self.island_last_migrated_generations[index] = 0
        self.migration_events = [
            dict(event)
            for event in metadata.get("migration_events", [])
            if isinstance(event, dict)
        ]
        self.last_iteration = int(metadata.get("last_iteration", 0))
        self.last_migration_iteration = int(
            metadata.get("last_migration_iteration", 0) or 0
        )

    def _recode_loaded_feature_maps(self, *, occupied_cells_before: int) -> None:
        self.archive.clear()
        self.best_program_id = None
        self.island_feature_maps = [{} for _ in range(max(1, self.config.num_islands))]
        recoded = 0
        excluded = 0
        timing_unknown = 0
        for program in self.programs.values():
            terms = _program_terms_for_features(program)
            if not terms:
                program.metrics["feature_recode_failed"] = True
                program.metrics["parent_eligible"] = False
                program.metrics["negative_memory_only"] = True
                excluded += 1
                continue
            if not program.term_set:
                program.term_set = sorted(terms)
            program.metrics.pop("feature_recode_failed", None)
            program.feature_coords = self._feature_coords(program)
            if program.feature_coords.get("opentimer_wns_delta_bucket") == "timing_unknown":
                timing_unknown += 1
            recoded += 1
            self._update_feature_map(program)
            self._update_archive(program)
            self._update_best(program)
        occupied_after = sum(len(feature_map) for feature_map in self.island_feature_maps)
        self.feature_recode_summary = {
            "feature_schema_version": self.config.feature_schema_version,
            "feature_dimensions": list(self.config.feature_dimensions),
            "programs_recoded": recoded,
            "programs_excluded_from_feature_map": excluded,
            "programs_timing_unknown": timing_unknown,
            "occupied_cells_before": occupied_cells_before,
            "occupied_cells_after": occupied_after,
        }

    def checkpoint(self, checkpoint_root: str | Path, iteration: int) -> Path:
        checkpoint_dir = Path(checkpoint_root) / f"checkpoint_{iteration:04d}"
        saved_root = self.root
        try:
            self.root = checkpoint_dir / "program_db"
            self.save(iteration=iteration)
        finally:
            self.root = saved_root
        return checkpoint_dir

    def _assign_island(self, program: ObjectiveProgram, target_island: int | None) -> int:
        if target_island is not None:
            return int(target_island) % len(self.islands)
        parent = self.get(program.parent_id)
        if parent is not None:
            return parent.island % len(self.islands)
        return _stable_hash(program.id) % len(self.islands)

    def _island_parent_pool(self, island_id: int) -> list[ObjectiveProgram]:
        island = int(island_id) % len(self.islands)
        mapped = [
            self.programs[pid]
            for pid in self.island_feature_maps[island].values()
            if pid in self.programs
        ]
        if mapped:
            return mapped
        return [self.programs[pid] for pid in self.islands[island] if pid in self.programs]

    def _mapped_program_pool(self) -> list[ObjectiveProgram]:
        seen: set[str] = set()
        programs = []
        for feature_map in self.island_feature_maps:
            for program_id in feature_map.values():
                if program_id in seen or program_id not in self.programs:
                    continue
                seen.add(program_id)
                programs.append(self.programs[program_id])
        return programs

    def _feature_coords(self, program: ObjectiveProgram) -> dict[str, Any]:
        mechanism_signature = str(
            program.metrics.get("mechanism_signature")
            or program.metrics.get("signature")
            or "+".join(program.term_set)
            or "none"
        )
        terms = _program_terms_for_features(program)
        mechanism_family = _mechanism_family(terms, program=program)
        hpwl_bucket = _hpwl_delta_bucket(program.metrics.get("hpwl_delta_pct"))
        overflow_bucket = _overflow_delta_bucket(program.metrics.get("overflow_delta_pct"))
        timing_bucket = _opentimer_wns_delta_bucket(
            program.metrics,
            min_abs_ns=self.config.timing_proxy_wns_delta_min_abs_ns,
        )
        program.metrics["mechanism_family"] = mechanism_family
        program.metrics["hpwl_delta_bucket"] = hpwl_bucket
        program.metrics["overflow_delta_bucket"] = overflow_bucket
        program.metrics["opentimer_wns_delta_bucket"] = timing_bucket
        return {
            "complexity": int(program.complexity),
            "term_signature": mechanism_signature,
            "mechanism_signature": mechanism_signature,
            "mechanism_family": mechanism_family,
            "hpwl_delta_pct": _bucket(program.metrics.get("hpwl_delta_pct")),
            "overflow_delta_pct": _bucket(program.metrics.get("overflow_delta_pct")),
            "hpwl_delta_bucket": hpwl_bucket,
            "overflow_delta_bucket": overflow_bucket,
            "gradient_stability": _gradient_bucket(program.metrics.get("custom_grad_norm")),
            "opentimer_wns_delta_bucket": timing_bucket,
        }

    def _update_archive(self, program: ObjectiveProgram) -> None:
        if self.config.map_elites_enabled:
            self._rebuild_archive_from_feature_maps()
            return
        if not program.is_parent_eligible:
            return
        self.archive.add(program.id)
        if len(self.archive) <= self.config.archive_size:
            return
        self.archive = {
            program.id
            for program in _diversity_trim(
                [self.programs[pid] for pid in self.archive if pid in self.programs],
                limit=self.config.archive_size,
            )
        }

    def _update_feature_map(self, program: ObjectiveProgram) -> None:
        if not self.config.map_elites_enabled or not program.is_parent_eligible:
            return
        island = program.island % len(self.island_feature_maps)
        cell_key = self._cell_key(program)
        current_id = self.island_feature_maps[island].get(cell_key)
        current = self.get(current_id)
        if current is None or _program_fitness_key(program) > _program_fitness_key(current):
            self.island_feature_maps[island][cell_key] = program.id

    def _rebuild_archive_from_feature_maps(self) -> None:
        occupants = {
            program_id
            for feature_map in self.island_feature_maps
            for program_id in feature_map.values()
            if program_id in self.programs
        }
        if len(occupants) <= self.config.archive_size:
            self.archive = occupants
            return
        ranked = sorted(
            (self.programs[pid] for pid in occupants),
            key=_program_fitness_key,
            reverse=True,
        )
        self.archive = {program.id for program in ranked[: self.config.archive_size]}

    def _cell_key(self, program: ObjectiveProgram) -> str:
        coords = program.feature_coords or self._feature_coords(program)
        values = []
        for dimension in self.config.feature_dimensions:
            value = coords.get(dimension)
            if dimension == "complexity":
                values.append(f"complexity:{self._numeric_bucket(value)}")
            elif dimension in {"hpwl_delta_pct", "overflow_delta_pct"}:
                values.append(f"{dimension}:{value}")
            elif dimension == "gradient_stability":
                values.append(f"gradient:{value}")
            elif dimension in {
                "mechanism_family",
                "hpwl_delta_bucket",
                "overflow_delta_bucket",
                "opentimer_wns_delta_bucket",
            }:
                values.append(f"{dimension}:{value or 'unknown'}")
            else:
                values.append(f"{dimension}:{value or 'none'}")
        return "|".join(values)

    def _numeric_bucket(self, value: Any) -> str:
        numeric = _finite_float(value, default=math.nan)
        if not math.isfinite(numeric):
            return "unknown"
        bins = max(1, int(self.config.feature_bins))
        if numeric <= 0:
            return "0"
        return str(min(bins - 1, int(numeric) % bins))

    def _migration_targets(self, source_island: int) -> list[int]:
        if self.config.migration_topology != "ring":
            raise ValueError(f"unknown migration topology: {self.config.migration_topology}")
        count = len(self.islands)
        if count <= 1:
            return []
        targets = [(source_island + 1) % count]
        previous = (source_island - 1) % count
        if previous not in targets:
            targets.append(previous)
        return targets

    def _copy_for_migration(
        self,
        program: ObjectiveProgram,
        *,
        source_island: int,
        target_island: int,
        iteration: int,
        ordinal: int,
    ) -> ObjectiveProgram:
        payload = program.to_dict()
        payload["id"] = (
            f"{program.id}__migrant_i{iteration:04d}_"
            f"{source_island}_to_{target_island}_{ordinal:02d}"
        )
        payload["parent_id"] = program.id
        payload["island"] = target_island
        payload["iteration_found"] = iteration
        payload["timestamp"] = time.time()
        payload["artifacts"] = {
            **dict(program.artifacts),
            "migration": {
                "source_program_id": program.id,
                "source_island": source_island,
                "target_island": target_island,
                "iteration": iteration,
            },
        }
        payload["metrics"] = {
            **dict(program.metrics),
            "migrant": True,
            "source_program_id": program.id,
            "source_island": source_island,
            "target_island": target_island,
        }
        return ObjectiveProgram.from_dict(payload)

    def _update_best(self, program: ObjectiveProgram) -> None:
        if not program.is_parent_eligible:
            return
        current = self.get(self.best_program_id)
        if current is None or program.combined_score > current.combined_score:
            self.best_program_id = program.id

    def _enforce_population_limit(self) -> None:
        if len(self.programs) <= self.config.population_size:
            return
        protected = set(self.archive)
        if self.best_program_id:
            protected.add(self.best_program_id)
        removable = [
            program
            for program in self.programs.values()
            if program.id not in protected
        ]
        removable.sort(key=lambda program: (program.combined_score, program.timestamp))
        while len(self.programs) > self.config.population_size and removable:
            victim = removable.pop(0)
            self.programs.pop(victim.id, None)
            for island in self.islands:
                island.discard(victim.id)
            for feature_map in self.island_feature_maps:
                for cell, program_id in list(feature_map.items()):
                    if program_id == victim.id:
                        feature_map.pop(cell, None)


class ObjectivePromptSampler:
    def __init__(
        self,
        *,
        term_scope: str,
        num_top_programs: int = 3,
        num_diverse_programs: int = 2,
        objective_mode: str = "replacement",
        mutation_mode: str = "full",
        policy: dict[str, Any] | None = None,
        router_background: str | None = None,
    ) -> None:
        self.term_scope = term_scope
        self.num_top_programs = num_top_programs
        self.num_diverse_programs = num_diverse_programs
        self.objective_mode = objective_mode
        self.mutation_mode = mutation_mode
        self.policy = dict(policy or {})
        self.router_background = router_background
        self.prompt_sampler = CoEvoPromptSampler()

    def build_context(
        self,
        *,
        parent: ObjectiveProgram,
        database: ObjectiveProgramDatabase,
        iteration: int,
        rng: random.Random,
        run_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        eligible = database.top_programs(max(self.num_top_programs * 4, self.num_top_programs), eligible_only=True)
        top = [
            program
            for program in eligible
            if not _is_baseline_reference(program)
        ][: self.num_top_programs]
        baseline_refs = [
            program
            for program in eligible
            if _is_baseline_reference(program)
        ][: self.num_top_programs]
        raw_diverse = database.diverse_programs(
            parent_id=parent.id,
            limit=max(self.num_diverse_programs * 3, self.num_diverse_programs),
            rng=rng,
            include_negative=False,
        )
        diverse = [
            program
            for program in raw_diverse
            if not _is_baseline_reference(program)
        ][: self.num_diverse_programs]
        failures = database.recent_negative_memory(5)
        near_misses = database.near_miss_programs(5)
        if self.objective_mode == "native_residual":
            near_misses = [
                program
                for program in near_misses
                if not program.metrics.get("diagnostic_baseline_only")
            ]
        prompt_memory_programs = [
            program
            for program in database.programs.values()
            if not (
                self.objective_mode == "native_residual"
                and program.metrics.get("diagnostic_baseline_only")
            )
        ]
        recent = sorted(
            prompt_memory_programs,
            key=lambda program: (program.iteration_found, program.timestamp),
        )[-5:]
        feedback_table = [
            _feedback_row(program)
            for program in [*top, *diverse, *baseline_refs, *near_misses, *failures]
        ]
        visible_policy = _prompt_visible_policy(self.policy)
        top_summaries = [_program_summary(program, include_code=True) for program in top]
        baseline_ref_summaries = [
            _program_summary(program, include_code=False) for program in baseline_refs
        ]
        diverse_summaries = [
            _program_summary(program, include_code=True) for program in diverse
        ]
        negative_summaries = [
            _program_summary(program, include_code=True) for program in failures
        ]
        near_miss_summaries = [
            _program_summary(program, include_code=True) for program in near_misses
        ]
        recent_summaries = [
            _program_summary(program, include_code=False) for program in recent
        ]
        mechanism_memory = _mechanism_memory(
            database,
            exclude_diagnostic_baselines=self.objective_mode == "native_residual",
        )
        mechanism_cluster_memory = _mechanism_cluster_memory(
            database,
            exclude_diagnostic_baselines=self.objective_mode == "native_residual",
        )
        mechanism_archetype_memory = _mechanism_archetype_memory(
            database,
            exclude_diagnostic_baselines=self.objective_mode == "native_residual",
        )
        component_feedback_memory = _component_feedback_memory(database)
        map_elites_memory = database.map_elites_summary()
        artifacts = {
            "router_objective_background": self.router_background or "",
            "component_feedback_memory": component_feedback_memory,
            "map_elites_memory": map_elites_memory,
            "mechanism_archetype_memory": mechanism_archetype_memory,
            "mechanism_cluster_memory": mechanism_cluster_memory,
            "structured_feedback_table": feedback_table,
            "mechanism_memory": mechanism_memory,
            "baseline_reference_programs": baseline_ref_summaries,
            "baseline_reference_note": (
                "These are evaluated fixed baselines or bootstrap parents. They are "
                "provided as reference behavior, not as discovered objectives to copy."
            ),
            "near_miss_programs": near_miss_summaries,
            "prior_run_feedback": run_feedback or {},
            "evaluator_note": (
                "The evaluator computes fitness after DREAMPlace execution. "
                "Metrics such as HPWL, overflow, gradient norm, runtime, and "
                "failure status are feedback, not objective inputs."
            ),
        }
        prompt = self.prompt_sampler.build_prompt(
            current_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=recent_summaries,
            top_programs=top_summaries,
            inspirations=[*diverse_summaries, *negative_summaries],
            term_scope=self.term_scope,
            objective_mode=self.objective_mode,
            mutation_mode=self.mutation_mode,
            artifacts=artifacts,
            feature_dimensions=list(database.config.feature_dimensions),
            extra_context={"search_policy": visible_policy},
        )
        prompt_messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        return {
            "mode": "archive_conditioned_objective_proposal",
            "memory_source": "explicit_archive_context",
            "iteration": iteration,
            "term_scope": self.term_scope,
            "objective_mode": self.objective_mode,
            "mutation_mode": self.mutation_mode,
            "prompt_style": "restricted_program_evolution",
            "prompt_messages": prompt_messages,
            "prompt_system": prompt["system"],
            "prompt_user": prompt["user"],
            "router_objective_background": self.router_background or "",
            "search_policy": visible_policy,
            "task": (
                "Mutate the parent DREAMPlace objective into one new restricted "
                "objective program for downstream routed-PPA improvement. The "
                "evaluator decides the quality of wirelength, congestion, timing, "
                "runtime, and stability tradeoffs after execution."
            ),
            "parent_program": _program_summary(parent, include_code=True),
            "top_safe_programs": top_summaries,
            "top_programs": top_summaries,
            "baseline_reference_programs": baseline_ref_summaries,
            "diverse_inspirations": diverse_summaries,
            "near_miss_programs": near_miss_summaries,
            "negative_memory": negative_summaries,
            "mechanism_memory": mechanism_memory,
            "mechanism_cluster_memory": mechanism_cluster_memory,
            "mechanism_archetype_memory": mechanism_archetype_memory,
            "structured_feedback_table": feedback_table,
            "run_feedback": run_feedback or {},
            "output_guidance": {
                "use_restricted_objective_program": True,
                "avoid_single_term_objectives": True,
                "residual_mode_requires_wirelength_and_density": self.objective_mode == "residual",
                "require_nonzero_routing_term": self.policy.get("require_nonzero_routing_term"),
                "active_routing_terms": self.policy.get("active_routing_terms"),
                "coefficient_calibration": {
                    "note": (
                        "Initialization observables provide measured component values "
                        "and gradient ratios. Use them to construct a physically "
                        "meaningful trajectory-aware objective."
                    ),
                },
                "composition_exploration": {
                    "note": (
                        "The DSL supports smooth transforms and multiplicative "
                        "interactions in addition to simple additive sums. Use "
                        "the archetype memory to choose a different mechanism "
                        "family when recent attempts collapse to the same shape."
                    ),
                },
                "robustness_rule": {
                    "note": (
                        "The evaluator records per-design behavior and may reject "
                        "solutions that only win by sacrificing other designs."
                    ),
                },
                "prefer_physically_meaningful_mechanisms": True,
                "downstream_metrics_are_feedback_only": True,
            },
        }


def _prompt_visible_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Return the evaluator policy subset that should be shown to the LLM.

    Exact thresholds and coefficient grids remain in the resolved run config and
    evaluator code. They are deliberately not inserted into prompts, so the model
    sees the environment and measured feedback rather than a target-threshold
    script to imitate.
    """

    visible: dict[str, Any] = {}
    for key in (
        "active_routing_terms",
        "target_design_context",
        "chip_design_profiles",
        "term_scale_audit",
        "term_scale_blocked_terms",
    ):
        value = policy.get(key)
        if value not in (None, [], {}):
            visible[key] = value
    if policy.get("require_nonzero_routing_term"):
        visible["routing_mechanism_policy"] = (
            "The search is intended to discover objectives that use routing, "
            "pin-access, or alternative smoothing observables as real mechanisms, "
            "not objectives that are only explicit wirelength plus density."
        )
    if policy.get("min_replacement_nonbaseline_terms") or policy.get(
        "reject_baseline_mechanism_clones"
    ):
        visible["baseline_similarity"] = (
            "The validator may reject objectives that are indistinguishable from "
            "fixed baselines or previous failed mechanisms. Use a meaningful "
            "mechanism change rather than coefficient-only edits."
        )
    if policy.get("objective_mode") == "controller":
        visible["physical_anchor_contract"] = (
            "A trajectory-aware objective defines density and smoothing schedules "
            "through typed policy slots. Its scalar loss retains smooth wirelength "
            "and density anchors, with optional routing and pin-access pressures."
        )
        visible["net_weight_policy"] = (
            "update_net_weights is available: timing criticalities are refreshed "
            "during placement on every search design."
            if (policy.get("timing_controller") or {}).get("available")
            else "update_net_weights is unavailable in this run because no in-loop "
            "timing collateral is configured; do not define it."
        )
    elif policy.get("objective_mode") == "native_residual":
        visible["physical_anchor_contract"] = (
            "A native-residual objective should preserve term(\"native_objective\") "
            "as the base optimizer objective, then add a compact routing or "
            "pin-access correction. Do not replace the base with explicit "
            "wirelength/density terms in this mode."
        )
    else:
        visible["physical_anchor_contract"] = (
            "A DREAMPlace replacement objective should include a canonical wire-span "
            "observable and a density/utilization observable as optimizer anchors. "
            "Routing and pin-access observables are additional mechanisms; "
            "pin_count_weighted_wl is not a standalone substitute for the normal "
            "wire-span term."
        )
    visible["coefficient_calibration"] = (
        "Component values and gradient ratios are measured at initialization and "
        "available to the objective schedule as calibration observables."
    )
    visible["composition_mechanisms"] = (
        "The objective DSL can express additive combinations, smooth transforms, "
        "and compact multiplicative interactions among deployable observables. "
        "A useful proposal may change the composition family, not only the "
        "coefficient on an existing additive term."
    )
    visible["selection_feedback"] = (
        "Selection is based on measured DREAMPlace behavior across the configured "
        "panel. Parent candidates are chosen from objectives with useful measured "
        "behavior rather than from raw baseline copies. External metrics are "
        "feedback after execution, not objective inputs."
    )
    if policy.get("samples_per_iteration") or policy.get("map_elites_enabled"):
        visible["evolution_process"] = (
            "Each evolution step may request multiple independent LLM samples "
            "from one parent/context. Successful programs are stored in island "
            "memory and a MAP-Elites-style cell archive; periodic migration "
            "shares strong programs across islands."
        )
    if policy.get("allow_near_miss_parents"):
        visible["evolution_memory"] = (
            "The local memory may mutate measured near-miss tradeoffs as stepping "
            "stones. Such programs are not final winners; they are examples of "
            "mechanisms whose measured behavior can be revised in later rounds."
        )
    return visible


class ObjectiveEvolutionTrace:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        iteration: int,
        parent: ObjectiveProgram | None,
        child: ObjectiveProgram,
    ) -> None:
        payload = {
            "iteration": iteration,
            "timestamp": time.time(),
            "parent_id": parent.id if parent else None,
            "child_id": child.id,
            "parent_metrics": parent.metrics if parent else {},
            "child_metrics": child.metrics,
            "improvement_delta": _metric_delta(parent.metrics if parent else {}, child.metrics),
            "status": child.status,
            "failure_reason": child.failure_reason,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def objective_code_from_spec(spec: ObjectiveSpec) -> str:
    if spec.source_program:
        return _with_evolve_markers(spec.source_program.strip())
    if spec.is_typed_policy:
        return _with_evolve_markers(_typed_policy_code_from_spec(spec))
    expression = _ast_to_expr(spec.ast)
    component_entries = ", ".join(f'"{term}": {term}' for term in spec.term_set)
    lines = ["def objective(features):"]
    for term in spec.term_set:
        lines.append(f'    {term} = term("{term}")')
    lines.append(f"    score = {expression}")
    lines.append(f"    return score, {{{component_entries}}}")
    return _with_evolve_markers("\n".join(lines))


def _typed_policy_code_from_spec(spec: ObjectiveSpec) -> str:
    registers = dict((spec.state or {}).get("registers", {}))
    lines = ["def init_policy(obs):", "    return {"]
    for name in sorted(registers):
        expression = _ast_to_expr(registers[name]["init"], state_accessor="policy")
        lines.append(f'        "{name}": {expression},')
    lines.extend(["    }", ""])

    if any("update" in register for register in registers.values()):
        lines.extend(["def update_policy(policy, obs):", "    return {"])
        for name in sorted(registers):
            update = registers[name].get(
                "update", {"op": "state", "name": name}
            )
            expression = _ast_to_expr(update, state_accessor="policy")
            lines.append(f'        "{name}": {expression},')
        lines.extend(["    }", ""])

    if spec.net_weight_policy and isinstance(spec.net_weight_policy.get("update"), dict):
        net_update = _ast_to_expr(
            spec.net_weight_policy["update"], state_accessor="policy"
        )
        lines.extend(
            [
                "def update_net_weights(net, policy, obs):",
                f"    return {net_update}",
                "",
            ]
        )

    lines.append("def objective(features, policy):")
    for term in spec.term_set:
        lines.append(f'    {term} = term("{term}")')
    expression = _ast_to_expr(spec.ast, state_accessor="policy")
    lines.append(f"    score = {expression}")
    components = spec.components or {
        term: {"op": "term", "name": term} for term in spec.term_set
    }
    component_entries = ", ".join(
        f'"{name}": {_ast_to_expr(component, state_accessor="policy")}'
        for name, component in sorted(components.items())
    )
    lines.append(f"    return score, {{{component_entries}}}")
    return "\n".join(lines)


def _ast_to_expr(node: dict[str, Any], *, state_accessor: str = "state") -> str:
    op = node.get("op")
    if op == "term":
        return str(node["name"])
    if op == "state":
        return f'{state_accessor}("{node["name"]}")'
    if op == "obs":
        return f'obs("{node["name"]}")'
    if op == "net":
        return f'net("{node["name"]}")'
    if op == "const":
        return repr(float(node["value"]))
    args = [
        _ast_to_expr(arg, state_accessor=state_accessor)
        for arg in node.get("args", [])
    ]
    if op == "add":
        return "(" + " + ".join(args) + ")"
    if op == "sub":
        return f"({args[0]} - {args[1]})"
    if op == "mul":
        return "(" + " * ".join(args) + ")"
    if op == "safe_div":
        return f"safe_div({args[0]}, {args[1]})"
    if op in {"log1p", "sqrt", "square", "softplus", "sigmoid"}:
        return f"{op}({args[0]})"
    return "0.0"


def _with_evolve_markers(code: str) -> str:
    if "EVOLVE-BLOCK-START" in code:
        return code.rstrip() + "\n"
    return "\n".join(["# EVOLVE-BLOCK-START", code.rstrip(), "# EVOLVE-BLOCK-END", ""])


def _is_baseline_reference(program: ObjectiveProgram) -> bool:
    return bool(
        program.metrics.get("seed_baseline")
        or program.metrics.get("manual_safe_baseline")
        or program.metrics.get("bootstrap_parent")
    )


def _prefer_discovered_candidates(candidates: list[ObjectiveProgram]) -> list[ObjectiveProgram]:
    discovered = [
        candidate for candidate in candidates if not _is_baseline_reference(candidate)
    ]
    return discovered or candidates


def _parent_sample_weights(
    candidates: list[ObjectiveProgram],
    *,
    novelty_aware: bool,
) -> list[float]:
    if not novelty_aware:
        return [max(candidate.combined_score, 1e-3) for candidate in candidates]
    signature_counts: dict[str, int] = {}
    for candidate in candidates:
        signature = _candidate_mechanism_signature(candidate)
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
    weights = []
    for candidate in candidates:
        score = max(candidate.combined_score, 0.0)
        score_factor = max(math.log1p(score), 1e-3)
        distance = _clamped_float(
            candidate.metrics.get("baseline_mechanism_distance"),
            default=0.5,
            low=0.0,
            high=1.0,
        )
        novelty_factor = 0.5 + distance
        signature_count = max(1, signature_counts.get(_candidate_mechanism_signature(candidate), 1))
        rarity_factor = 1.0 / math.sqrt(signature_count)
        weights.append(max(score_factor * novelty_factor * rarity_factor, 1e-6))
    return weights


def _candidate_mechanism_signature(candidate: ObjectiveProgram) -> str:
    return str(
        candidate.metrics.get("mechanism_signature")
        or candidate.metrics.get("signature")
        or "+".join(candidate.term_set)
        or candidate.id
    )


def _clamped_float(value: Any, *, default: float, low: float, high: float) -> float:
    numeric = _finite_float(value, default=default)
    return min(high, max(low, numeric))


def _diversity_trim(candidates: list[ObjectiveProgram], *, limit: int) -> list[ObjectiveProgram]:
    if limit <= 0:
        return []
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.combined_score,
            -int(candidate.metrics.get("structural_failure_count", 0) or 0),
            -int(candidate.metrics.get("severe_regression_count", 0) or 0),
            candidate.timestamp,
        ),
        reverse=True,
    )
    selected: list[ObjectiveProgram] = []
    selected_ids: set[str] = set()
    occupied_cells: set[tuple[Any, ...]] = set()
    for candidate in ranked:
        cell = _archive_cell(candidate)
        if cell in occupied_cells:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.id)
        occupied_cells.add(cell)
        if len(selected) >= limit:
            return selected
    for candidate in ranked:
        if candidate.id in selected_ids:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            return selected
    return selected


def _archive_cell(program: ObjectiveProgram) -> tuple[Any, ...]:
    coords = program.feature_coords or {}
    return (
        coords.get("mechanism_family"),
        coords.get("hpwl_delta_bucket"),
        coords.get("overflow_delta_bucket"),
        coords.get("opentimer_wns_delta_bucket"),
        coords.get("mechanism_signature") or program.metrics.get("mechanism_signature"),
        coords.get("term_signature") or "+".join(program.term_set),
    )


def _program_role(program: ObjectiveProgram) -> str:
    if _is_baseline_reference(program):
        return "baseline_reference"
    if program.metrics.get("negative_memory_only") or program.status != "accepted":
        return "negative_memory"
    if program.is_parent_eligible:
        return "discovered_candidate"
    return "evaluated_candidate"


def _program_summary(program: ObjectiveProgram, *, include_code: bool) -> dict[str, Any]:
    payload = {
        "id": program.id,
        "program_role": _program_role(program),
        "parent_id": program.parent_id,
        "generation": program.generation,
        "status": program.status,
        "failure_reason": _sanitize_prompt_text(program.failure_reason),
        "metrics": _compact_metrics(program.metrics),
        "term_set": program.term_set,
        "complexity": program.complexity,
        "island": program.island,
        "feature_coords": program.feature_coords,
        "artifacts": _compact_artifacts(program.artifacts),
    }
    if include_code:
        payload["code"] = program.code
    return payload


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    # Prompt-visible metrics should describe measured behavior, not expose the
    # evaluator's exact gate bookkeeping. The full metrics remain persisted in
    # the program database and trace files for reproducibility.
    keep = {
        "combined_score",
        "constrained_score",
        "tier2_average_rank",
        "hpwl_delta_pct",
        "overflow_delta_pct",
        "mechanism_family",
        "hpwl_delta_bucket",
        "overflow_delta_bucket",
        "opentimer_wns_delta_bucket",
        "timing_proxy_status",
        "timing_proxy_mode",
        "timing_proxy_wns",
        "timing_proxy_tns",
        "timing_proxy_wns_delta",
        "timing_proxy_tns_delta",
        "timing_proxy_tns_delta_pct",
        "timing_proxy_quality",
        "timing_proxy_parent_signal",
        "timing_proxy_net_coverage",
        "timing_proxy_source_placement_budget_satisfied",
        "final_def_hpwl_delta_pct",
        "final_def_overflow_delta_pct",
        "selection_metric_stage",
        *ROUTED_EVIDENCE_KEYS,
        "post_route_evidence_iteration",
        "post_route_evidence_status",
        "tier_b_evaluated",
        "tier_c_evaluated",
        "timing_proxy_admission",
        "pareto_front",
        "custom_grad_norm",
        "native_default_hpwl_delta_pct",
        "native_default_overflow_delta_pct",
        "custom_default_hpwl_delta_pct",
        "custom_default_overflow_delta_pct",
        "custom_default_effect_pct",
        "baseline_portfolio_compared",
        "baseline_portfolio_size",
        "baseline_portfolio_candidate_rank",
        "baseline_portfolio_candidate_average_rank",
        "baseline_portfolio_best_objective_id",
        "baseline_portfolio_best_average_rank",
        "baseline_portfolio_best_hpwl_delta_pct",
        "baseline_portfolio_best_overflow_delta_pct",
        "baseline_portfolio_hpwl_delta_vs_best_pct",
        "baseline_portfolio_overflow_delta_vs_best_pct",
        "baseline_portfolio_pareto_label_vs_best",
        "baseline_portfolio_pareto_dominates_best",
        "baseline_portfolio_pareto_tradeoff_vs_best",
        "baseline_portfolio_rank_beats_best",
        "baseline_portfolio_effect_pct",
        "baseline_portfolio_closest_behavior_objective_id",
        "baseline_portfolio_behavior_distance_pct",
        "beats_baseline_portfolio",
        "structural_failure_count",
        "severe_regression_count",
        "runtime_seconds",
        "outcome_label",
        "worst_design_hpwl_delta_pct",
        "worst_design_overflow_delta_pct",
        "design_hpwl_improvement_fraction",
        "design_overflow_improvement_fraction",
        "design_both_improvement_fraction",
        "cell_both_improvement_fraction",
        "per_design_deltas",
        "primary_baseline",
        "routing_term_required",
        "routing_aware_tier2_candidate",
        "active_routing_terms",
        "active_routing_coefficients",
        "mechanism_signature",
        "mechanism_terms",
        "aggregate_tier2_winner",
        "all_design_robust_tier2_winner",
        "exploration_parent_only",
        "near_miss_parent_score",
        "near_miss_parent_reason",
        "feedback_lesson",
        "component_summary",
        "component_feedback",
        "seed_baseline",
    }
    return {
        key: _sanitize_prompt_value(metrics[key])
        for key in sorted(keep)
        if key in metrics
    }


def _feedback_row(program: ObjectiveProgram) -> dict[str, Any]:
    metrics = program.metrics
    return {
        "objective_id": program.id,
        "program_role": _program_role(program),
        "terms": "+".join(program.term_set) or "none",
        "hpwl_delta_pct": metrics.get("hpwl_delta_pct"),
        "overflow_delta_pct": metrics.get("overflow_delta_pct"),
        "mechanism_family": metrics.get("mechanism_family"),
        "hpwl_delta_bucket": metrics.get("hpwl_delta_bucket"),
        "overflow_delta_bucket": metrics.get("overflow_delta_bucket"),
        "opentimer_wns_delta_bucket": metrics.get("opentimer_wns_delta_bucket"),
        "timing_proxy_status": metrics.get("timing_proxy_status"),
        "timing_proxy_wns_delta": metrics.get("timing_proxy_wns_delta"),
        "timing_proxy_tns_delta": metrics.get("timing_proxy_tns_delta"),
        "timing_proxy_tns_delta_pct": metrics.get("timing_proxy_tns_delta_pct"),
        "timing_proxy_parent_signal": metrics.get("timing_proxy_parent_signal"),
        "timing_proxy_net_coverage": metrics.get("timing_proxy_net_coverage"),
        **{key: metrics.get(key) for key in ROUTED_EVIDENCE_KEYS},
        "post_route_evidence_iteration": metrics.get("post_route_evidence_iteration"),
        "native_default_hpwl_delta_pct": metrics.get("native_default_hpwl_delta_pct"),
        "native_default_overflow_delta_pct": metrics.get("native_default_overflow_delta_pct"),
        "custom_default_hpwl_delta_pct": metrics.get("custom_default_hpwl_delta_pct"),
        "custom_default_overflow_delta_pct": metrics.get("custom_default_overflow_delta_pct"),
        "candidate_rank_against_fixed_baselines": metrics.get(
            "baseline_portfolio_candidate_rank"
        ),
        "rank_beats_best_fixed_baseline": metrics.get(
            "baseline_portfolio_rank_beats_best"
        ),
        "promotion_beats_fixed_baselines": metrics.get("beats_baseline_portfolio"),
        "best_fixed_baseline": metrics.get("baseline_portfolio_best_objective_id"),
        "hpwl_delta_vs_best_fixed_baseline_pct": metrics.get(
            "baseline_portfolio_hpwl_delta_vs_best_pct"
        ),
        "overflow_delta_vs_best_fixed_baseline_pct": metrics.get(
            "baseline_portfolio_overflow_delta_vs_best_pct"
        ),
        "pareto_label_vs_best_fixed_baseline": metrics.get(
            "baseline_portfolio_pareto_label_vs_best"
        ),
        "pareto_dominates_best_fixed_baseline": metrics.get(
            "baseline_portfolio_pareto_dominates_best"
        ),
        "per_design_deltas": _compact_per_design_deltas(metrics.get("per_design_deltas")),
        "worst_hpwl_design": _worst_design(metrics.get("per_design_deltas"), metric="hpwl_delta_pct"),
        "outcome": metrics.get("outcome_label") or program.status,
        "active_routing_terms": metrics.get("active_routing_terms"),
        "component_feedback": _compact_component_summary(metrics.get("component_summary")),
        "lesson": _sanitize_prompt_text(metrics.get("feedback_lesson") or program.failure_reason),
    }


def _component_feedback_memory(
    database: ObjectiveProgramDatabase,
    limit: int = 8,
) -> dict[str, Any]:
    entries = []
    for program in sorted(
        database.programs.values(),
        key=lambda item: (item.iteration_found, item.timestamp),
        reverse=True,
    ):
        summary = _compact_component_summary(program.metrics.get("component_summary"))
        if not summary:
            continue
        entries.append(
            {
                "program_id": program.id,
                "program_role": _program_role(program),
                "terms": "+".join(program.term_set) or "none",
                "hpwl_delta_pct": program.metrics.get("hpwl_delta_pct"),
                "overflow_delta_pct": program.metrics.get("overflow_delta_pct"),
                "outcome": program.metrics.get("outcome_label") or program.status,
                "components": summary,
            }
        )
        if len(entries) >= limit:
            break
    return {
        "purpose": (
            "Named objective components measured during DREAMPlace execution "
            "and summarized over the placement trajectory."
        ),
        "recent_component_summaries": entries,
    }


def _compact_component_summary(value: Any, *, limit: int = 6) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {}
    for name in sorted(value)[:limit]:
        item = value.get(name)
        if not isinstance(item, dict):
            continue
        compact[str(name)] = {
            "trajectory": _compact_trajectory(item.get("trajectory"), limit=10),
            "start": item.get("start"),
            "mid": item.get("mid"),
            "end": item.get("end"),
            "min": item.get("min"),
            "max": item.get("max"),
            "mean": item.get("mean"),
            "trend": item.get("trend"),
            "flat_or_saturated": item.get("flat_or_saturated"),
        }
    return compact


def _compact_trajectory(value: Any, *, limit: int) -> list[float]:
    if not isinstance(value, list):
        return []
    values = []
    for item in value[: max(0, limit)]:
        numeric = _finite_float(item, default=math.nan)
        if math.isfinite(numeric):
            values.append(float(f"{numeric:.6g}"))
    return values


def _compact_per_design_deltas(value: Any, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "design": item.get("design"),
                "hpwl_delta_pct": item.get("hpwl_delta_pct"),
                "overflow_delta_pct": item.get("overflow_delta_pct"),
            }
        )
    return rows


def _worst_design(value: Any, *, metric: str) -> dict[str, Any] | None:
    rows = _compact_per_design_deltas(value, limit=20)
    finite_rows = []
    for row in rows:
        try:
            score = float(row.get(metric))
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            finite_rows.append((score, row))
    if not finite_rows:
        return None
    score, row = max(finite_rows, key=lambda item: item[0])
    return {"design": row.get("design"), metric: score}


def _mechanism_memory(
    database: ObjectiveProgramDatabase,
    limit: int = 8,
    *,
    exclude_diagnostic_baselines: bool = False,
) -> dict[str, Any]:
    negative = []
    safe = []
    recent = []
    for program in sorted(
        database.programs.values(),
        key=lambda item: (item.iteration_found, item.timestamp),
        reverse=True,
    ):
        if exclude_diagnostic_baselines and program.metrics.get("diagnostic_baseline_only"):
            continue
        signature = program.metrics.get("mechanism_signature")
        if not signature:
            continue
        item = {
            "program_id": program.id,
            "signature": signature,
            "terms": program.metrics.get("mechanism_terms") or program.term_set,
            "hpwl_delta_pct": program.metrics.get("hpwl_delta_pct"),
            "overflow_delta_pct": program.metrics.get("overflow_delta_pct"),
            **{
                key: program.metrics[key]
                for key in ROUTED_EVIDENCE_KEYS
                if program.metrics.get(key) is not None
            },
            "native_default_hpwl_delta_pct": program.metrics.get(
                "native_default_hpwl_delta_pct"
            ),
            "native_default_overflow_delta_pct": program.metrics.get(
                "native_default_overflow_delta_pct"
            ),
            "custom_default_hpwl_delta_pct": program.metrics.get(
                "custom_default_hpwl_delta_pct"
            ),
            "custom_default_overflow_delta_pct": program.metrics.get(
                "custom_default_overflow_delta_pct"
            ),
            "candidate_rank_against_fixed_baselines": program.metrics.get(
                "baseline_portfolio_candidate_rank"
            ),
            "rank_beats_best_fixed_baseline": program.metrics.get(
                "baseline_portfolio_rank_beats_best"
            ),
            "promotion_beats_fixed_baselines": program.metrics.get(
                "beats_baseline_portfolio"
            ),
            "baseline_portfolio_best_objective_id": program.metrics.get(
                "baseline_portfolio_best_objective_id"
            ),
            "baseline_portfolio_hpwl_delta_vs_best_pct": program.metrics.get(
                "baseline_portfolio_hpwl_delta_vs_best_pct"
            ),
            "baseline_portfolio_overflow_delta_vs_best_pct": program.metrics.get(
                "baseline_portfolio_overflow_delta_vs_best_pct"
            ),
            "baseline_portfolio_pareto_label_vs_best": program.metrics.get(
                "baseline_portfolio_pareto_label_vs_best"
            ),
            "per_design_deltas": _compact_per_design_deltas(
                program.metrics.get("per_design_deltas")
            ),
            "worst_hpwl_design": _worst_design(
                program.metrics.get("per_design_deltas"),
                metric="hpwl_delta_pct",
            ),
            "outcome": program.metrics.get("outcome_label") or program.status,
            "lesson": _sanitize_prompt_text(program.metrics.get("feedback_lesson")),
        }
        recent.append(item)
        if program.metrics.get("negative_memory_only") or program.metrics.get(
            "severe_regression_count", 0
        ):
            negative.append(item)
        elif program.is_parent_eligible:
            safe.append(item)
    return {
        "purpose": (
            "Empirical memory of mechanism structures. Negative mechanisms should "
            "not be repeated by coefficient-only edits; safe mechanisms can be "
            "used as baselines or inspirations. This is evaluator feedback, not "
            "a formula template."
        ),
        "negative_mechanisms": negative[:limit],
        "safe_mechanisms": safe[:limit],
        "recent_mechanisms": recent[:limit],
    }


def _mechanism_cluster_memory(
    database: ObjectiveProgramDatabase,
    *,
    limit: int = 8,
    exclude_diagnostic_baselines: bool = False,
) -> dict[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    for program in sorted(
        database.programs.values(),
        key=lambda item: (item.iteration_found, item.timestamp),
        reverse=True,
    ):
        if exclude_diagnostic_baselines and program.metrics.get("diagnostic_baseline_only"):
            continue
        terms = _mechanism_terms(program)
        if not terms:
            continue
        key = "+".join(terms)
        cluster = clusters.setdefault(
            key,
            {
                "terms": terms,
                "program_count": 0,
                "representative_program_ids": [],
                "outcomes": {},
                "lessons": [],
                "native_default_hpwl_delta_pct": [],
                "native_default_overflow_delta_pct": [],
                "baseline_portfolio_hpwl_delta_vs_best_pct": [],
                "baseline_portfolio_overflow_delta_vs_best_pct": [],
                "baseline_portfolio_pareto_labels": {},
                "per_design_deltas": [],
                "has_parent_candidate": False,
                "has_negative_or_near_miss": False,
            },
        )
        cluster["program_count"] += 1
        if len(cluster["representative_program_ids"]) < 5:
            cluster["representative_program_ids"].append(program.id)
        outcome = str(program.metrics.get("outcome_label") or program.status)
        cluster["outcomes"][outcome] = int(cluster["outcomes"].get(outcome, 0)) + 1
        lesson = _sanitize_prompt_text(program.metrics.get("feedback_lesson"))
        if lesson and lesson not in cluster["lessons"] and len(cluster["lessons"]) < 3:
            cluster["lessons"].append(lesson)
        for metric_name in (
            "native_default_hpwl_delta_pct",
            "native_default_overflow_delta_pct",
            "baseline_portfolio_hpwl_delta_vs_best_pct",
            "baseline_portfolio_overflow_delta_vs_best_pct",
        ):
            numeric = _finite_float(program.metrics.get(metric_name), default=math.nan)
            if math.isfinite(numeric):
                cluster[metric_name].append(numeric)
        pareto_label = program.metrics.get("baseline_portfolio_pareto_label_vs_best")
        if pareto_label:
            labels = cluster["baseline_portfolio_pareto_labels"]
            labels[str(pareto_label)] = int(labels.get(str(pareto_label), 0)) + 1
        cluster["per_design_deltas"].extend(
            _compact_per_design_deltas(program.metrics.get("per_design_deltas"))
        )
        if program.is_parent_eligible:
            cluster["has_parent_candidate"] = True
        if (
            program.metrics.get("negative_memory_only")
            or program.metrics.get("severe_regression_count", 0)
            or (
                program.metrics.get("worst_design_hpwl_delta_pct") is not None
                and _finite_float(program.metrics.get("worst_design_hpwl_delta_pct"), default=0.0)
                > 0.0
            )
        ):
            cluster["has_negative_or_near_miss"] = True

    summarized = []
    for key, cluster in clusters.items():
        hpwl_values = cluster.pop("native_default_hpwl_delta_pct")
        overflow_values = cluster.pop("native_default_overflow_delta_pct")
        hpwl_vs_best_values = cluster.pop("baseline_portfolio_hpwl_delta_vs_best_pct")
        overflow_vs_best_values = cluster.pop(
            "baseline_portfolio_overflow_delta_vs_best_pct"
        )
        pareto_labels = cluster.pop("baseline_portfolio_pareto_labels")
        per_design = cluster.pop("per_design_deltas")
        item = {
            "mechanism_key": key,
            "terms": cluster["terms"],
            "program_count": cluster["program_count"],
            "representative_program_ids": cluster["representative_program_ids"],
            "observed_outcomes": cluster["outcomes"],
            "best_native_default_hpwl_delta_pct": min(hpwl_values) if hpwl_values else None,
            "best_native_default_overflow_delta_pct": (
                min(overflow_values) if overflow_values else None
            ),
            "best_hpwl_delta_vs_fixed_baseline_pct": (
                min(hpwl_vs_best_values) if hpwl_vs_best_values else None
            ),
            "best_overflow_delta_vs_fixed_baseline_pct": (
                min(overflow_vs_best_values) if overflow_vs_best_values else None
            ),
            "pareto_labels_vs_fixed_baseline": pareto_labels,
            "worst_hpwl_design": _worst_design(per_design, metric="hpwl_delta_pct"),
            "lessons": cluster["lessons"],
            "cluster_role": (
                "mixed_or_negative"
                if cluster["has_negative_or_near_miss"]
                else "candidate_parent_memory"
                if cluster["has_parent_candidate"]
                else "observed_memory"
            ),
        }
        summarized.append(item)

    summarized.sort(
        key=lambda item: (
            0 if item["cluster_role"] == "mixed_or_negative" else 1,
            -int(item["program_count"]),
            str(item["mechanism_key"]),
        )
    )
    return {
        "purpose": (
            "Compact behavior memory grouped by mechanism term families. Use this "
            "to avoid repeatedly proposing the same failed family and to identify "
            "which observables need a mechanism-level change."
        ),
        "clusters": summarized[:limit],
    }


_MECHANISM_ARCHETYPES: tuple[dict[str, Any], ...] = (
    {
        "name": "alternative_wire_smoothing",
        "purpose": "Compare a sharper smooth wire model with a density anchor.",
        "required_any": [["wirelength_lse"]],
        "avoid_with_any": [["wirelength_wawl", "wirelength"]],
    },
    {
        "name": "bell_density_anchor",
        "purpose": "Use bell-shaped density potential instead of electric density.",
        "required_any": [["density_bell"]],
        "avoid_with_any": [["density_electric", "density"]],
    },
    {
        "name": "long_net_route_pressure",
        "purpose": "Target route pressure caused by long-span nets.",
        "required_any": [["route_pressure_long"]],
    },
    {
        "name": "pin_access_pressure",
        "purpose": "Represent local pin-access pressure and high-pin-count nets.",
        "required_any": [["pin_density_pnorm", "pin_count_weighted_wl"]],
    },
    {
        "name": "soft_rudy_hotspot_pressure",
        "purpose": "Use differentiable RUDY-style demand observables.",
        "required_any": [["soft_rudy_mean", "soft_rudy_pnorm"]],
    },
    {
        "name": "route_density_interaction",
        "purpose": (
            "Let route pressure modulate a density/utilization term through a "
            "smooth transform or product, rather than only adding a route term."
        ),
        "required_features": ["route_density_mul"],
    },
    {
        "name": "route_wire_interaction",
        "purpose": (
            "Let route or pin pressure modulate a wire-span term, including "
            "pin-count-weighted wirelength."
        ),
        "required_features": ["route_wire_mul"],
    },
    {
        "name": "bounded_route_transform",
        "purpose": (
            "Use a bounded or saturating smooth transform of route pressure so "
            "the optimizer can test non-linear route response."
        ),
        "required_any": [["soft_rudy_mean", "soft_rudy_pnorm", "route_pressure_long", "pin_density_pnorm"]],
        "required_ops": ["sigmoid", "softplus", "log1p", "sqrt"],
    },
)

_PROMPT_ROUTE_TERMS = {
    "soft_rudy_mean",
    "soft_rudy_pnorm",
    "route_pressure_long",
    "pin_density_pnorm",
    "pin_count_weighted_wl",
}
_PROMPT_DENSITY_TERMS = {"density", "density_electric", "density_bell"}
_PROMPT_WIRE_TERMS = {"wirelength", "wirelength_wawl", "wirelength_lse"}


def _mechanism_archetype_memory(
    database: ObjectiveProgramDatabase,
    *,
    limit: int = 8,
    exclude_diagnostic_baselines: bool = False,
) -> dict[str, Any]:
    """Summarize structural search coverage for prompt-visible memory.

    This is intentionally descriptive rather than prescriptive. It tells the
    model which mechanism families have already been tried and whether they
    produced useful or negative measured behavior, without exposing hard gates or
    target thresholds.
    """

    observed: list[tuple[ObjectiveProgram, set[str], set[str], set[str]]] = []
    for program in database.programs.values():
        if exclude_diagnostic_baselines and program.metrics.get("diagnostic_baseline_only"):
            continue
        ast = (program.objective_spec or {}).get("ast")
        if not ast:
            continue
        observed.append(
            (
                program,
                {_canonical_prompt_term(term) for term in _terms_from_prompt_ast(ast)},
                _ops_from_prompt_ast(ast),
                _interaction_features_from_prompt_ast(ast),
            )
        )

    items = []
    for archetype in _MECHANISM_ARCHETYPES:
        matches = [
            program
            for program, terms, ops, features in observed
            if _archetype_matches(archetype, terms, ops, features)
        ]
        negative = [
            program
            for program in matches
            if program.metrics.get("negative_memory_only")
            or program.metrics.get("severe_regression_count", 0)
        ]
        parent_candidates = [
            program
            for program in matches
            if program.is_parent_eligible and not _is_baseline_reference(program)
        ]
        fixed_baselines = [
            program for program in matches if _is_baseline_reference(program)
        ]
        best_match = _best_measured_program(matches)
        items.append(
            {
                "name": archetype["name"],
                "purpose": archetype["purpose"],
                "observed_count": len(matches),
                "negative_count": len(negative),
                "parent_candidate_count": len(parent_candidates),
                "fixed_baseline_count": len(fixed_baselines),
                "status": _archetype_status(matches, negative, parent_candidates),
                "best_measured_program": (
                    _program_summary(best_match, include_code=False)
                    if best_match is not None
                    else None
                ),
                "selection_hint": _archetype_hint(matches, negative, parent_candidates),
            }
        )

    items.sort(
        key=lambda item: (
            0 if item["status"] == "unexplored" else 1,
            0 if item["status"] == "mostly_negative" else 1,
            int(item["observed_count"]),
            str(item["name"]),
        )
    )
    return {
        "purpose": (
            "Archive summary of structural objective families already measured "
            "by DREAMPlace. It guides later proposals toward mechanism-level "
            "changes and underexplored behavior cells."
        ),
        "archetypes": items[:limit],
    }


def _archetype_matches(
    archetype: dict[str, Any],
    terms: set[str],
    ops: set[str],
    features: set[str],
) -> bool:
    for group in archetype.get("required_any", []):
        canonical = {_canonical_prompt_term(str(term)) for term in group}
        if not (terms & canonical):
            return False
    for group in archetype.get("avoid_with_any", []):
        canonical = {_canonical_prompt_term(str(term)) for term in group}
        if terms & canonical:
            return False
    required_ops = {str(op) for op in archetype.get("required_ops", [])}
    if not required_ops <= ops:
        return False
    required_features = {str(feature) for feature in archetype.get("required_features", [])}
    return required_features <= features


def _archetype_status(
    matches: list[ObjectiveProgram],
    negative: list[ObjectiveProgram],
    parent_candidates: list[ObjectiveProgram],
) -> str:
    if not matches:
        return "unexplored"
    if parent_candidates:
        return "has_parent_candidate"
    if len(negative) >= max(1, len(matches) // 2):
        return "mostly_negative"
    return "measured_no_winner"


def _archetype_hint(
    matches: list[ObjectiveProgram],
    negative: list[ObjectiveProgram],
    parent_candidates: list[ObjectiveProgram],
) -> str:
    if not matches:
        return "No measured program in this family yet; it is available for exploration."
    if parent_candidates:
        return "This family has at least one parent-eligible measured program."
    if negative:
        return "This family has negative measurements; use a different composition if revisiting it."
    return "Measured, but no promotion-worthy program yet."


def _best_measured_program(programs: list[ObjectiveProgram]) -> ObjectiveProgram | None:
    if not programs:
        return None
    return max(
        programs,
        key=lambda program: (
            float(program.metrics.get("combined_score") or 0.0),
            -int(program.metrics.get("structural_failure_count", 0) or 0),
            -int(program.metrics.get("severe_regression_count", 0) or 0),
        ),
    )


def _terms_from_prompt_ast(node: Any) -> set[str]:
    terms: set[str] = set()
    if not isinstance(node, dict):
        return terms
    if node.get("op") == "term":
        terms.add(str(node.get("name")))
    for child in node.get("args", []):
        terms.update(_terms_from_prompt_ast(child))
    return terms


def _ops_from_prompt_ast(node: Any) -> set[str]:
    ops: set[str] = set()
    if not isinstance(node, dict):
        return ops
    op = node.get("op")
    if op:
        ops.add(str(op))
    for child in node.get("args", []):
        ops.update(_ops_from_prompt_ast(child))
    return ops


def _interaction_features_from_prompt_ast(node: Any) -> set[str]:
    features: set[str] = set()
    if not isinstance(node, dict):
        return features
    if node.get("op") == "mul":
        child_terms = [
            {_canonical_prompt_term(term) for term in _terms_from_prompt_ast(child)}
            for child in node.get("args", [])
            if isinstance(child, dict)
        ]
        has_route_factor = any(terms & _PROMPT_ROUTE_TERMS for terms in child_terms)
        has_density_factor = any(terms & _PROMPT_DENSITY_TERMS for terms in child_terms)
        has_wire_factor = any(terms & _PROMPT_WIRE_TERMS for terms in child_terms)
        if has_route_factor and has_density_factor:
            features.add("route_density_mul")
        if has_route_factor and has_wire_factor:
            features.add("route_wire_mul")
    for child in node.get("args", []):
        features.update(_interaction_features_from_prompt_ast(child))
    return features


def _mechanism_terms(program: ObjectiveProgram) -> list[str]:
    terms = program.metrics.get("mechanism_terms") or program.term_set
    if not isinstance(terms, list):
        return []
    return sorted({_canonical_prompt_term(str(term)) for term in terms if term})


def _canonical_prompt_term(term: str) -> str:
    if term in {"wirelength", "wirelength_wawl"}:
        return "wirelength_wawl"
    if term in {"density", "density_electric"}:
        return "density_electric"
    return term


_PROMPT_ARTIFACT_ALLOWLIST = {
    "baseline_seed_mode",
    "calibration_batch",
    "component_feedback_memory",
    "dreamplace",
    "error",
    "map_elites_memory",
    "objective_path",
    "preset",
    "prompt_context_path",
    "prompt_id",
    "provider_metadata",
    "provider_usage",
    "raw_response_path",
    "summary",
}

_PROMPT_ARTIFACT_FORBIDDEN_FRAGMENTS = (
    "baseline_portfolio_gate_passed",
    "require_baseline_portfolio_pareto",
    "baseline_portfolio_effect_gate_passed",
    "min_baseline_portfolio_effect_pct",
    "baseline_portfolio_behavior_distance_gate_passed",
    "min_baseline_behavior_distance_pct",
    "baseline_portfolio_min_behavior_distance_pct",
    "custom_default_gate_passed",
    "custom_default_hpwl_gate_passed",
    "density_coeff_grid",
    "elite_gate_passed",
    "hpwl_gate_passed",
    "hpwl_gate_search_pct",
    "hpwl_gate_elite_pct",
    "per_design_hpwl_gate_search_pct",
    "custom_default_hpwl_gate_pct",
    "min_abs_routing_coeff",
    "min_replacement_nonbaseline_terms",
    "negative_memory_only",
    "overflow_gate_passed",
    "parent_eligible",
    "route_coeff_cap",
    "route_coeff_grid",
    "wirelength_coeff_grid",
    "coefficient exceeds cap",
    "route coefficient exceeds cap",
    "replacement route coefficient exceeds cap",
    "residual route coefficient exceeds cap",
    "native-residual route coefficient exceeds cap",
    "Use at least 2 non-default",
    "use at least 2 non-baseline",
    "abs(c)",
)


def _compact_artifacts(artifacts: dict[str, Any], *, limit: int = 2000) -> dict[str, Any]:
    compact = {}
    for key, value in artifacts.items():
        if key not in _PROMPT_ARTIFACT_ALLOWLIST:
            continue
        text = str(value)
        if any(fragment in text for fragment in _PROMPT_ARTIFACT_FORBIDDEN_FRAGMENTS):
            continue
        text = _sanitize_prompt_text(text)
        if len(text) > limit:
            text = text[:limit] + "... (truncated)"
        compact[key] = text
    return compact


def _sanitize_prompt_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_prompt_text(value)
    if isinstance(value, list):
        return [_sanitize_prompt_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_prompt_value(item)
            for key, item in value.items()
            if not any(fragment in str(key) for fragment in _PROMPT_ARTIFACT_FORBIDDEN_FRAGMENTS)
        }
    return value


def _sanitize_prompt_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    for fragment in _PROMPT_ARTIFACT_FORBIDDEN_FRAGMENTS:
        text = text.replace(fragment, "[internal_evaluator_key]")
    return text


def _metric_delta(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, float]:
    delta = {}
    for key, value in child.items():
        if key not in parent:
            continue
        child_value = _finite_float(value, default=math.nan)
        parent_value = _finite_float(parent[key], default=math.nan)
        if math.isfinite(child_value) and math.isfinite(parent_value):
            delta[key] = child_value - parent_value
    return delta


def _bucket(value: Any) -> str:
    numeric = _finite_float(value, default=math.nan)
    if not math.isfinite(numeric):
        return "nan"
    if numeric <= -10:
        return "improve_large"
    if numeric < 0:
        return "improve"
    if numeric == 0:
        return "flat"
    if numeric <= 10:
        return "regress"
    return "regress_large"


def _hpwl_delta_bucket(value: Any) -> str:
    numeric = _finite_float(value, default=math.nan)
    if not math.isfinite(numeric):
        return "hpwl_neutral"
    if numeric <= -1.0:
        return "hpwl_improved"
    if numeric >= 1.0:
        return "hpwl_regressed"
    return "hpwl_neutral"


def _overflow_delta_bucket(value: Any) -> str:
    numeric = _finite_float(value, default=math.nan)
    if not math.isfinite(numeric):
        return "overflow_neutral"
    if numeric <= -5.0:
        return "overflow_improved"
    if numeric >= 5.0:
        return "overflow_regressed"
    return "overflow_neutral"


def _opentimer_wns_delta_bucket(metrics: dict[str, Any], *, min_abs_ns: float) -> str:
    status = _normalize_timing_proxy_status(metrics.get("timing_proxy_status"))
    delta = _finite_float(metrics.get("timing_proxy_wns_delta"), default=math.nan)
    if status not in {"success", "partial"} or not math.isfinite(delta):
        return "timing_unknown"
    threshold = abs(_finite_float(min_abs_ns, default=TIMING_PROXY_WNS_DELTA_MIN_ABS_NS))
    if delta >= threshold:
        return "wns_improved"
    if delta <= -threshold:
        return "wns_regressed"
    return "wns_neutral"


def _normalize_timing_proxy_status(value: Any) -> str:
    status = str(value or "unavailable").strip()
    if status in {"success", "partial", "failed", "missing_timing_collateral", "not_run"}:
        return status
    return "unavailable"


def _program_terms_for_features(program: ObjectiveProgram) -> set[str]:
    raw_terms: set[str] = set()
    if program.term_set:
        raw_terms.update(str(term) for term in program.term_set if term)
    spec = program.objective_spec if isinstance(program.objective_spec, dict) else {}
    if not raw_terms:
        spec_terms = spec.get("term_set")
        if isinstance(spec_terms, list):
            raw_terms.update(str(term) for term in spec_terms if term)
    if not raw_terms:
        raw_terms.update(_terms_from_prompt_ast(spec.get("ast")))
    return {_canonical_prompt_term(term) for term in raw_terms if term}


def _mechanism_family(
    terms: set[str],
    *,
    program: ObjectiveProgram | None = None,
) -> str:
    schedule_family = _controller_schedule_family(program)
    if schedule_family is not None:
        return schedule_family
    if not terms:
        return "unknown"
    route_terms = {"soft_rudy_mean", "soft_rudy_pnorm", "route_pressure_long"}
    has_route = bool(terms & route_terms)
    has_pin = "pin_density_pnorm" in terms
    if has_route and has_pin:
        return "route_pin_hybrid"
    if "route_pressure_long" in terms:
        return "long_route_pressure"
    if "soft_rudy_pnorm" in terms:
        return "route_hotspot"
    if "soft_rudy_mean" in terms:
        return "route_mean"
    if has_pin:
        return "pin_pressure"
    if "pin_count_weighted_wl" in terms:
        return "fanout_weighted_wl"
    if "wirelength_lse" in terms or "density_bell" in terms:
        return "wl_density_variant"
    wl_density_terms = {"wirelength_wawl", "density_electric", "native_objective"}
    if terms and terms <= wl_density_terms:
        return "wl_density"
    return "unknown"


def _controller_schedule_family(program: ObjectiveProgram | None) -> str | None:
    if program is None or not isinstance(program.objective_spec, dict):
        return None
    spec = program.objective_spec
    state = spec.get("state") if isinstance(spec.get("state"), dict) else {}
    if state.get("interface") != "typed_policy_v1":
        return None
    registers = state.get("registers") if isinstance(state.get("registers"), dict) else {}
    observables: set[str] = set()
    for entry in registers.values():
        if not isinstance(entry, dict):
            continue
        observables.update(_observable_names_from_ast(entry.get("init")))
        observables.update(_observable_names_from_ast(entry.get("update")))
    drivers: list[str] = []
    if spec.get("net_weight_policy"):
        drivers.append("timing")
    if "hpwl_delta_rate" in observables:
        drivers.append("hpwl_trend")
    if "overflow" in observables:
        drivers.append("overflow")
    if "iter_frac" in observables:
        drivers.append("progress")
    if {"route_weight", "pin_weight"} & set(registers):
        drivers.append("routing")
    drivers = list(dict.fromkeys(drivers))
    if len(drivers) > 1:
        return "hybrid_schedule"
    if not drivers:
        return "static_policy"
    return {
        "timing": "timing_driven",
        "hpwl_trend": "hpwl_trend_driven",
        "overflow": "overflow_driven",
        "progress": "progress_driven",
        "routing": "routing_driven",
    }[drivers[0]]


def _observable_names_from_ast(node: Any) -> set[str]:
    if not isinstance(node, dict):
        return set()
    names = {str(node.get("name"))} if node.get("op") == "obs" else set()
    for child in node.get("args", []):
        names.update(_observable_names_from_ast(child))
    return names


def _gradient_bucket(value: Any) -> str:
    numeric = _finite_float(value, default=math.nan)
    if not math.isfinite(numeric):
        return "unknown"
    if numeric <= 1e-9:
        return "zero"
    if numeric < 1e3:
        return "finite"
    return "large"


def _program_fitness_key(program: ObjectiveProgram) -> tuple[Any, ...]:
    return (
        _finite_float(program.metrics.get("combined_score"), default=0.0),
        -int(program.metrics.get("structural_failure_count", 0) or 0),
        -int(program.metrics.get("severe_regression_count", 0) or 0),
        bool(program.metrics.get("parent_eligible")),
        program.timestamp,
    )


def _program_signature(program: ObjectiveProgram) -> tuple[Any, ...]:
    return (
        tuple(sorted(program.term_set)),
        str(program.metrics.get("mechanism_signature") or ""),
        str(program.objective_spec.get("ast") if isinstance(program.objective_spec, dict) else ""),
    )


def _stable_hash(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value))


def _finite_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
