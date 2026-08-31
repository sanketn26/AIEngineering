# Starter — hybrid models (tiny MLP + attention)

**Not the 90-day dual-path trainer.** A stdlib forward pass: one hidden-layer MLP on XOR-like points, plus scaled-dot attention you can unit test. No GPU, no PyTorch required.

Full track: [docs/tracks/hybrid-models.md](../../../docs/tracks/hybrid-models.md).

```bash
cd tracks/starters/hybrid-models
python3 -c "from model_slice import predict_label; print(predict_label([1.0, 0.0]))"
python3 -m pytest tests/test_slice.py -v
```

Baselines before hybrids: this slice *is* the MLP toy. Do not claim fusion wins until you have MLP-only and Transformer-only numbers on a leakage-safe split.
