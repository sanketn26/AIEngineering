"""Metric regressions run without ML dependencies; optional real forward passes."""

import importlib.util
import json
import subprocess
import sys

import pytest

from src.inference_bench import paired_comparison, percentile, summarize


def test_throughput_weights_tokens_by_total_time():
    # Mean of per-request rates would incorrectly give 62.5 instead of 40.
    rows = [
        {"elapsed_ms": 40, "output_tokens": 4, "cuda_peak_allocated_bytes": None},
        {"elapsed_ms": 160, "output_tokens": 4, "cuda_peak_allocated_bytes": None},
    ]
    result = summarize(rows)
    assert result["output_tokens_per_second"] == 40
    assert result["cuda_peak_allocated_bytes"] is None
    assert result["p95_ms"] == pytest.approx(154)
    assert percentile([], 0.95) is None


def test_ttft_and_tpot_summarize_only_measured_rows():
    rows = [
        {
            "elapsed_ms": 100,
            "output_tokens": 5,
            "local_ttft_ms": 60,
            "local_tpot_ms": 10,
            "cuda_peak_allocated_bytes": None,
        },
        # A single-token response has no inter-token interval to report.
        {
            "elapsed_ms": 50,
            "output_tokens": 1,
            "local_ttft_ms": 50,
            "local_tpot_ms": None,
            "cuda_peak_allocated_bytes": None,
        },
    ]
    result = summarize(rows)
    assert result["median_local_ttft_ms"] == 55
    assert result["median_local_tpot_ms"] == 10
    assert result["tpot_sample_count"] == 1


def test_measured_zero_peak_is_not_reported_as_unmeasured():
    rows = [
        {"elapsed_ms": 10, "output_tokens": 1, "cuda_peak_allocated_bytes": 0},
        {"elapsed_ms": 10, "output_tokens": 1, "cuda_peak_allocated_bytes": None},
    ]
    assert summarize(rows)["cuda_peak_allocated_bytes"] == 0


def test_parity_pairs_requests_instead_of_run_order():
    rows = [
        {"repeat": 0, "prompt_index": 1, "variant": "candidate", "output_ids": [3]},
        {"repeat": 0, "prompt_index": 0, "variant": "baseline", "output_ids": [1]},
        {"repeat": 0, "prompt_index": 1, "variant": "baseline", "output_ids": [2]},
        {"repeat": 0, "prompt_index": 0, "variant": "candidate", "output_ids": [1]},
    ]
    assert paired_comparison(rows)["paired_requests"] == 2
    assert paired_comparison(rows)["identical_greedy_outputs"] == 1


@pytest.mark.skipif(
    not all(importlib.util.find_spec(n) for n in ("torch", "transformers")),
    reason="Install examples/inference/requirements.txt for real-model smoke tests",
)
@pytest.mark.parametrize("experiment", ["cache", "attention", "speculative"])
def test_offline_real_generation(tmp_path, experiment):
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference_bench",
            "--smoke",
            "--experiment",
            experiment,
            "--new-tokens",
            "4",
            "--prompt-repeats",
            "2",
            "--repeats",
            "1",
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(report_path.read_text())
    assert report["metadata"]["smoke_random_weights"] is True
    if experiment == "speculative":
        # The first logits call can precede any committed token under assisted
        # generation, so neither derived metric may be reported at all.
        assert all(r["local_ttft_ms"] is None for r in report["rows"])
        assert all(r["local_tpot_ms"] is None for r in report["rows"])
        assert report["summary"]["candidate"]["median_local_ttft_ms"] is None
        assert report["summary"]["candidate"]["tpot_sample_count"] == 0
    else:
        assert all(
            0 < r["local_ttft_ms"] < r["elapsed_ms"] for r in report["rows"]
        ), "first-token time must fall inside the measured generation window"
    assert report["parity"]["paired_requests"] == 2
    if experiment == "attention":
        # Reassociated attention math may flip a greedy token; the lesson says so.
        # Record it rather than failing the suite on a floating-point difference.
        assert report["parity"]["identical_greedy_outputs"] in (1, 2)
    else:
        # use_cache and assisted generation must reproduce target-only greedy tokens.
        assert report["parity"]["identical_greedy_outputs"] == 2
    assert all(r["elapsed_ms"] > 0 for r in report["rows"])
    assert all(0 < r["output_tokens"] <= 4 for r in report["rows"])
    assert all(r["cuda_peak_allocated_bytes"] is None for r in report["rows"])
