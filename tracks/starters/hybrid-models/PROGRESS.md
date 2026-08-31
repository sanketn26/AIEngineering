# Hybrid starter — milestone TODOs

- [x] Day 1: stdlib MLP forward + attention; XOR corners test (`tests/test_slice.py`)
- [ ] **Data card** — entity, `x_tab` vs `x_seq`, horizon, leakage risks, license
- [ ] **Time / entity split** — scalers fit on train only; no shuffle on autocorrelated series
- [ ] **MLP-only baseline** that actually trains (PyTorch) with a pinned seed
- [ ] **Transformer-only baseline** on the sequence path, same split
- [ ] **Fusion** — concat or gated; ablate it; complexity must earn the MAE/accuracy
- [ ] **Shape tests + tiny serve** — CLI or FastAPI predict; export optional

Track rubric: [docs/reference/assessment.md](../../../docs/reference/assessment.md).
