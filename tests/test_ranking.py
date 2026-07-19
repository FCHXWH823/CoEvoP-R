from coevop.eval.ranking import aggregate_rank_scores, classify_result


def test_rank_aggregation_handles_failures_and_regressions() -> None:
    rows = [
        {
            "design": "d1",
            "seed": 1,
            "objective_id": "default",
            "status": "success",
            "hpwl_delta_pct": 0.0,
            "overflow_delta_pct": 0.0,
            "runtime_seconds": 10.0,
        },
        {
            "design": "d1",
            "seed": 1,
            "objective_id": "good",
            "status": "success",
            "hpwl_delta_pct": -1.0,
            "overflow_delta_pct": -5.0,
            "runtime_seconds": 12.0,
        },
        {
            "design": "d1",
            "seed": 1,
            "objective_id": "failed",
            "status": "failed",
            "failure_stage": "output_artifact",
            "hpwl_delta_pct": "",
            "overflow_delta_pct": "",
            "runtime_seconds": 1.0,
        },
    ]

    rankings = aggregate_rank_scores(rows)

    assert [summary.objective_id for summary in rankings] == ["good", "default", "failed"]
    assert rankings[-1].structural_failure_count == 1
    assert classify_result(rows[2]) == "structural_failure"


def test_classify_result_keeps_metric_regression_separate_from_failure() -> None:
    row = {
        "status": "success",
        "failure_stage": "",
        "hpwl_delta_pct": 2.0,
        "overflow_delta_pct": -1.0,
    }

    assert classify_result(row, severe_threshold_pct=10.0) == "metric_regression"
    assert classify_result({**row, "hpwl_delta_pct": 12.0}) == "severe_regression"
