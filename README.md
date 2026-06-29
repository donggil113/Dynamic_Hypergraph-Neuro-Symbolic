# Neuro-Symbolic Open-Set / Few-Shot RF Emitter Recognition (Paper P2)

Interpretable RF emitter recognition that grounds **attribute predicates** with a neural
front-end and reasons over them with a **symbolic knowledge base** (Logic Tensor Network),
enabling **open-set rejection** and **few-shot family addition** without retraining the
front-end. ML / signal-processing framing (target: IEEE TCCN / Expert Systems with Applications).

## Ground rules (non-negotiable)
1. **No fabricated results.** Every number traces to a logged run of real code on real data.
2. **Reproducibility.** Each run logs config + git commit + seed + raw metrics + logfile.
   Figures/tables are script-generated from logs, never hand-typed.
3. **Flag uncertainty.** Design choices needing domain knowledge STOP and ask.
4. **No unimplemented contributions.** No paper claim without backing code + measured results.

## Gate status
| Gate | Description | Status |
|------|-------------|--------|
| G0 | OpenEW-SA data audit | ⏳ blocked — awaiting dataset path |
| G1 | Method & KB spec | 📝 draft for sign-off → [`docs/G1_method_and_kb_spec.md`](docs/G1_method_and_kb_spec.md) |
| G2 | Pipeline + closed-set smoke test | not started |
| G3 | Full experiments + ablations | not started |
| G4 | Paper draft (LaTeX from logs) | not started |

## Layout
```
docs/    design docs (G1 spec; G0 audit pending data)
configs/ run configs (yaml)        — to be populated at G2
src/     method + pipeline code    — to be populated at G2
experiments/ experiment runners    — to be populated at G3
data/    placeholder; real data lives at an author-provided path, never committed
```

> **Current state:** repository skeleton + G1 design doc. No experiments run, no results exist
> yet. Next action is author sign-off on `docs/G1_method_and_kb_spec.md` and provision of the
> OpenEW-SA dataset path to begin the G0 audit.
