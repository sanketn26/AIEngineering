"""A score increase can be real, noisy, or simply too small to resolve."""

import json
from src.evals import paired_release_gate


def experiments():
    # Deliberately constructed fixtures, NOT observed model runs.
    baseline = [[1.0, 1.0, 1.0, 1.0]] * 16 + [[0.0, 0.0, 0.0, 0.0]] * 4
    candidate = (
        [[1.0, 1.0, 1.0, 1.0]] * 15
        + [[0.0, 0.0, 0.0, 0.0]]
        + [[1.0, 1.0, 1.0, 1.0]] * 2
        + [[0.0, 0.0, 0.0, 0.0]] * 2
    )
    return {
        "one_more_win": paired_release_gate(baseline, candidate),
        "same_two_tickets_100_times": paired_release_gate(
            [[1.0] * 100, [0.0] * 100], [[1.0] * 100, [1.0] * 100]
        ),
        "clear_regression": paired_release_gate(
            [[1.0, 1.0]] * 30 + [[0.0, 0.0]] * 10, [[0.0, 0.0]] * 30 + [[1.0, 1.0]] * 10
        ),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {"data": "constructed teaching fixtures", "experiments": experiments()},
            indent=2,
        )
    )
