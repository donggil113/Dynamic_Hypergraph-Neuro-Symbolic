# G0 — OpenEW-SA Data Audit Plan

**Status:** ⏳ BLOCKED — awaiting dataset path from author. **This is a plan, not findings.** No numbers below are real until the audit runs on the actual data. Nothing here assumes dataset semantics.

**Goal:** determine exactly what the OpenEW-SA data *actually* contains, then reconcile every `[VERIFY@G0]` item in `G1_method_and_kb_spec.md`. Report what exists vs. what is missing. **Do not proceed past gaps — list them.**

## How the audit runs
1. Author provides the dataset path → set `data.root`.
2. Run the read-only inspector: `python src/audit_openew_sa.py --root <PATH> --out runs/g0_audit/`.
   It only **reads and reports** (file tree, formats, array shapes, tabular columns, unique values of candidate label fields, basic stats). It imposes **no taxonomy** and writes nothing back to the data.
3. Fill the checklist below from the inspector's `audit_report.json` + manual inspection.
4. Write `docs/G0_data_audit.md` (the findings doc) and STOP for author review.

## Checklist (to be filled from real data — currently all UNKNOWN)

### A. Per-attribute labels (G1 item 1 — pivotal)
- [ ] RF band / carrier frequency present? as numeric value or categorical band? — **UNKNOWN**
- [ ] PRI value present? units? per-pulse or per-train? — **UNKNOWN**
- [ ] PRI-modulation **type** labelled? value set? — **UNKNOWN**
- [ ] PW present? units? — **UNKNOWN**
- [ ] Scan type labelled? value set? — **UNKNOWN**
- [ ] Intra-pulse modulation labelled? value set? — **UNKNOWN**
- [ ] **Verdict:** per-attribute labels exist / family-only / partial → drives §4 axiom 1. — **UNKNOWN**

### B. Emitter-family labels (G1)
- [ ] Family/class label field present? how many classes? class frequencies? — **UNKNOWN**
- [ ] Multifunction emitters labelled at **mode** or **emitter** level (item 4)? — **UNKNOWN**

### C. Open-set hold-out splits (G1)
- [ ] Predefined open-set splits present, or must we construct them? — **UNKNOWN**
- [ ] Do splits separate **unseen-mode / unseen-recombination / true-OOD** (item 6, §3.2)? — **UNKNOWN**

### D. Cross-domain / node splits (G1)
- [ ] Domain / collection-node / SNR / channel metadata present for cross-domain splits? — **UNKNOWN**

### E. Input representation (item 3)
- [ ] Available formats: PDW sequences? raw IQ? time–frequency images? other? — **UNKNOWN**
- [ ] Author preference is **PDW if available** — confirm availability. — **UNKNOWN**

### F. Taxonomy reconciliation (item 2)
- [ ] For each labelled categorical attribute, list the dataset's actual value set and diff it against the §1 convention. **Report mismatches; do not coerce.** — **UNKNOWN**

### G. Integrity / reproducibility
- [ ] Record file count, total size, checksums, any train/val/test partition, label completeness (missing/NaN), class balance. — **UNKNOWN**

## Outputs of G0
- `runs/g0_audit/audit_report.json` (machine-readable inventory; logged, reproducible).
- `docs/G0_data_audit.md` (human findings: exists-vs-missing table + every `[VERIFY@G0]` resolved or flagged as a gap).
- A go/no-go recommendation per downstream method component (esp. whether concept-bottleneck / attribute supervision is viable).
