from examples.retrieval.compare import claims_supported, run


def test_known_citation_does_not_prove_a_claim():
    docs = [{"id": "policy", "claims": ["window_30_days"]}]
    assert not claims_supported(
        [{"cite": "policy", "claim": "window_365_days"}], docs, ["policy"]
    )
    assert not claims_supported([], docs, ["policy"])
    assert not claims_supported(
        [{"cite": "policy", "claim": "window_30_days"}], docs, []
    )
    assert claims_supported(
        [{"cite": "policy", "claim": "window_30_days"}], docs, ["policy"]
    )


def test_comparison_keeps_unanswerable_queries_visible():
    report = run()
    assert report["n_queries"] == 12
    assert len(report["rows"]) == 48
    assert report["summary"]["reranked"]["source_support"] == 1
    assert report["summary"]["reranked"]["answers_question"] < 1
    assert all(
        row["recall_at_2"] is None for row in report["rows"] if row["query"] == "q12"
    )
