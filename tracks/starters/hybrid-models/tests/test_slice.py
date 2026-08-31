import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_slice import (  # noqa: E402
    attention,
    load_tiny_data,
    predict_label,
    softmax,
)


def test_mlp_labels_xor_corners():
    data = load_tiny_data()
    corners = [row for row in data["rows"] if row["x"] in ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0])]
    assert len(corners) == 4
    for row in corners:
        assert predict_label(row["x"]) == row["y"]


def test_attention_puts_mass_on_matching_key():
    query = [1.0, 0.0]
    keys = [[1.0, 0.0], [0.0, 1.0]]
    values = [[10.0], [0.0]]
    out = attention(query, keys, values)
    assert len(out) == 1
    assert 5.0 < out[0] < 10.0  # majority mass on the matching key's value
    weights_sum = sum(softmax([1.0, 0.0]))
    assert abs(weights_sum - 1.0) < 1e-9
