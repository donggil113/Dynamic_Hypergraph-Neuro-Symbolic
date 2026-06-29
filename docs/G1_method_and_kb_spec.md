# G1 — Method & Knowledge-Base Specification (Design Doc)

**Project:** Neuro-Symbolic Open-Set / Few-Shot RF Emitter Recognition (Paper P2)
**Status:** DRAFT — awaiting author (Eagle) sign-off. **No code is written against this spec until approved.**
**Gate:** G1 (Method & KB spec). Precedes G2 (pipeline + smoke test).
**Framing:** ML / signal-processing (target IEEE TCCN or Expert Systems with Applications). We deliberately use generic *RF emitter recognition / spectrum-awareness* language and avoid threat/EW framing.

> ⚠️ **Dependency on G0.** This spec proposes attribute definitions and a knowledge base *before* the OpenEW-SA data audit (G0) has been run, because the author asked for the method design first. Every claim about what attributes/labels exist is therefore a **proposal to be validated against the real data**. All such places are marked **[VERIFY@G0]**. If G0 shows a label is absent, the corresponding part of this spec must be revised before coding.

---

## 0. Notation and scope

We classify an RF emitter observation `x` (an IQ segment, a pulse-descriptor-word (PDW) sequence, or a time–frequency representation — the exact input is **[VERIFY@G0]**) into one of `K` known emitter families, **or** reject it as *unknown* (open-set). We additionally support adding a brand-new family from `k ∈ {1,5,10}` labelled samples without retraining the neural front-end (few-shot).

The architecture is a **two-layer neuro-symbolic stack**:

```
            interpretable attribute predicates                  symbolic KB (FOL / fuzzy logic)
  x  ──►  [ frozen-able neural front-end ]  ──►  groundings g(x) ∈ [0,1]^A  ──►  [ LTN rule layer ]  ──►  family decision  +  reject  + explanation
            (multi-head attribute estimator)        (per attribute)                (family definitions)
```

The neural front-end grounds **attribute predicates**, *not* family labels. Family identity is *derived* by the symbolic layer from the attributes. This is the core design commitment and the source of both the open-set and few-shot capabilities.

---

## 1. Attribute set (the interpretable layer)

We propose six attribute groups, matching the labels the brief expects OpenEW-SA to carry (RF band, PRI value, PRI-modulation type, PW, scan, intra-pulse modulation). Each group is grounded by a dedicated head of the neural front-end.

| # | Attribute | Type | Proposed values / range | Grounding head output | **[VERIFY@G0]** |
|---|-----------|------|--------------------------|------------------------|-----------------|
| A1 | **Carrier frequency / RF band** | continuous → fuzzified | f_c in Hz, fuzzified into bands | regressed f̂_c + per-band membership | exact band edges & whether label is numeric or categorical |
| A2 | **PRI value** | continuous → fuzzified | PRI in µs, fuzzified into ranges | regressed P̂RI + per-range membership | units, range, whether single-value or interval per pulse train |
| A3 | **PRI-modulation type** | categorical | {constant/stable, jittered, staggered, dwell-&-switch, sliding, sinusoidal} | softmax over types | exact type taxonomy used in the dataset |
| A4 | **Pulse width (PW)** | continuous → fuzzified | PW in µs, fuzzified into ranges | regressed P̂W + per-range membership | units, range |
| A5 | **Scan type** | categorical | {non-scanning/steady, circular, sector, raster, conical, lobe-switching, electronic-steer} | softmax over types | which scan classes are actually labelled |
| A6 | **Intra-pulse modulation** | categorical | {unmodulated/CW, LFM (chirp), NLFM, phase-coded (Barker/poly-phase), FSK} | softmax over types | which IPM classes are labelled |

**Fuzzification of continuous attributes (A1, A2, A4).** The head regresses a value and we convert it to fuzzy membership over a set of named bands/ranges using trapezoidal (or Gaussian) membership functions with overlapping supports. This keeps the predicate **interpretable** ("S-band", "short PW") while remaining differentiable. Band/range edges are hyperparameters fixed from the **training-set** empirical distribution (logged), never tuned on test.

**Atomic predicates.** For each attribute group we define a family of atomic fuzzy predicates over `x`:

- `InBand(x, b)` for each band `b` (A1)
- `PRIin(x, r)` for each PRI range `r` (A2)
- `PRImod(x, m)` for each modulation type `m` (A3)
- `PWin(x, p)` for each PW range `p` (A4)
- `Scan(x, s)` for each scan type `s` (A5)
- `IPM(x, i)` for each intra-pulse type `i` (A6)

The truth value of each atomic predicate is the corresponding neural-head output in `[0,1]` (softmax component or membership value). Collect them into the grounding vector `g(x) ∈ [0,1]^A`, where `A = Σ |values per attribute|`.

> ⚠️ **Open question for the author.** Does OpenEW-SA provide *per-attribute* ground-truth labels (so the heads can be trained with direct supervision / concept-bottleneck style), or only *family* labels? This is the single most important G0 finding. If only family labels exist, attribute supervision must be weak/derived and Ablation B vs. the full method changes character. **[VERIFY@G0]**

---

## 2. Symbolic layer — first-order / fuzzy logic over predicates

We use a **real-valued (fuzzy) logic** so the whole stack is differentiable end-to-end, instantiated as a **Logic Tensor Network (LTN)** (primary). DeepProbLog and a plain concept-bottleneck classifier are the two ablation alternatives (see §6).

### 2.1 Connectives (t-norms)

- Conjunction `∧`: **product t-norm** `T(a,b)=a·b` (use the log-sum stabilised form to avoid vanishing gradients over a 6-way conjunction).
- Disjunction `∨`: product t-conorm `S(a,b)=a+b−a·b`.
- Negation `¬a = 1−a`.
- Universal `∀`: smooth p-mean-error aggregation; Existential `∃`: p-mean. (Standard LTN aggregators, Badreddine et al. 2022.)

The exact t-norm is a hyperparameter; product is the default because the 6-way family conjunction needs *all* attributes to agree, and product penalises any single low factor. (Gödel/min is an alternative ablation if product gradients prove unstable — logged if changed.)

### 2.2 Emitter-family definition = conjunction of attribute predicates

Each known family `F` is defined by a **conjunction** binding one (or a small disjunctive set of) value(s) per attribute:

```
Fdef(x) :=  InBand(x, b_F) ∧ PRIin(x, r_F) ∧ PRImod(x, m_F) ∧ PWin(x, p_F) ∧ Scan(x, s_F) ∧ IPM(x, i_F)
```

where `(b_F, r_F, m_F, p_F, s_F, i_F)` is the **attribute signature** of family `F`, stored in the knowledge base. Where a family admits several values for an attribute (e.g. two PW ranges), that slot becomes a small intra-attribute disjunction.

The **fired strength** of family `F` on `x` is `T_F(x) = Fdef(x) ∈ [0,1]`.

### 2.3 Multifunction emitters = disjunction over modes

A multifunction / multi-mode emitter `M` is the **disjunction** of its modes, each mode being its own conjunction:

```
Mdef(x) :=  mode_1(x)  ∨  mode_2(x)  ∨  …  ∨  mode_J(x),     mode_j(x) = conjunction as in §2.2
```

so `T_M(x) = S_j ( mode_j(x) )`. A single-mode family is the `J=1` special case. **[VERIFY@G0]:** whether OpenEW-SA labels multifunction emitters at the mode level or only at the emitter level — this determines whether modes are supervised or latent.

### 2.4 Knowledge base contents

The KB (a versioned YAML, see §7) holds, per family:
- attribute signature (the predicate value bindings), with intra-attribute disjunctions;
- mode list (for multifunction);
- provenance: whether the signature was **author-specified** (domain rule) or **data-induced** (estimated from training samples) — logged separately so the paper can report each.

> ⚠️ **Domain-knowledge gate.** The actual family→attribute signatures are a *threat-library-style* artifact that needs author input or must be induced from labelled training data. I will **not** invent these. Two acceptable sources: (a) the author supplies a rule table; (b) we induce signatures from the training split and you review them. Decide at sign-off. **[VERIFY@G0]**

---

## 3. Two-condition open-set rejection

An observation is **accepted** as family `F* = argmax_F T_F(x)` only if **both** conditions hold; otherwise it is **rejected** as *unknown*.

- **(C1) Rule satisfaction.** Some family fires above a threshold: `max_F T_F(x) ≥ τ_rule`.
  *Interpretation:* the attribute combination matches a known family definition.

- **(C2) In-distribution attribute grounding.** Every attribute the winning family relies on is grounded **in-distribution**: `min_{a ∈ sig(F*)} d_a(x) ≥ τ_id`, where `d_a(x)` is an attribute-level ID score (e.g. max-softmax / fitted density / Mahalanobis on the head's pre-softmax features — choice logged).
  *Interpretation:* the attributes themselves were confidently recognised, not a low-confidence guess that happened to maximise a rule.

```
ACCEPT(x, F*)   ⇔   ( max_F T_F(x) ≥ τ_rule )   ∧   ( min_{a∈sig(F*)} d_a(x) ≥ τ_id )
REJECT(x)       ⇔   ¬C1  ∨  ¬C2
```

Both thresholds are calibrated on a **validation** split (never test), logged with the run.

### 3.1 Explainability (required by the spec)

Every rejection carries a reason:
- `¬C1` → "no known family definition matched" + the closest family and which attribute slot(s) broke the conjunction (lowest factor in the product).
- `¬C2` → "attribute grounding anomalous" + the specific attribute `a* = argmin_a d_a(x)` (e.g. *"intra-pulse modulation out-of-distribution"*).

This gives a human-readable attribution per decision and feeds the **explanation-quality** metric (attribute attribution vs. ground-truth attribute labels) at G3.

### 3.2 Mapping to the required novelty-type breakdown

The two-condition design *predicts* the novelty taxonomy the brief asks us to report, which is a nice testable consequence:

| Novelty type | Expected trigger | Meaning |
|--------------|------------------|---------|
| **unseen-mode** | ¬C1, all attributes ID | known family's attributes, novel *combination/mode* — no conjunction fires |
| **unseen-recombination** | ¬C1, all attributes ID | attributes from different known families recombined |
| **true-OOD** | ¬C2 | an attribute itself is out-of-distribution (genuinely new signal physics) |

At G3 we measure whether rejections actually fall into these buckets as predicted — this is an evaluation, not an assumption.

---

## 4. Training objective (LTN satisfiability)

Maximise satisfiability of a knowledge base of axioms over the training data (gradient ascent on aggregated truth):

1. **Grounding axioms (if per-attribute labels exist [VERIFY@G0]):** for each labelled attribute, `∀x ( head predicts the labelled value )` — i.e. the head groundings match ground-truth attribute labels.
2. **Definitional axioms:** for each known family `F`, `∀x ( familyLabel(x)=F → Fdef(x) )` and `∀x ( Fdef(x) → familyLabel(x)=F )` softly (the second is weakened to avoid forcing exclusivity where families share attributes).
3. **Regularisation:** standard LTN smoothness; optional sparsity on which attributes define a family.

If per-attribute labels are absent, axiom (1) is dropped and attributes are learned only through (2) (weak supervision); we will report this honestly as it weakens interpretability claims.

The neural front-end can be **frozen** after this training so few-shot families add only symbolic rules (§5).

---

## 5. Few-shot family addition (no neural retraining)

To add a new family `F_new` from `k ∈ {1,5,10}` labelled samples:

1. Run the **frozen** front-end on the `k` samples → `k` grounding vectors.
2. **Induce** the attribute signature: for each categorical attribute take the modal value (with a support set if tied); for each continuous attribute take the empirical `[min,max]` interval widened by a tolerance (for `k=1`, a point ± fixed tolerance).
3. Write `F_new`'s conjunction (§2.2) into the KB. **No gradient steps.**

`F_new` then participates in the same two-condition decision (§3). This tests whether the attribute space is rich enough that a new class is just a new *definition*, not new *features* — the central few-shot claim. Acc-vs-shot curves (k=1,5,10) at G3.

---

## 6. Ablation mapping (must trace to a switch in code)

| Ablation | What it removes | Tests | Concrete change |
|----------|-----------------|-------|-----------------|
| **A** | symbolic logic layer | does logic add value? | keep attribute heads, replace LTN rules with a learned classifier / threshold on `g(x)` |
| **B** | attribute decomposition | does decomposition add value? | train neural net directly on family labels (no attributes); standard closed-set + OSR baseline |
| **C** | attribute-confidence term (C2) | is the two-condition rejection justified? | rejection uses **only** C1 (`τ_id = −∞`) |

Each is a config flag, not a separate codebase. Ablation B doubles as the "atomic-label" reference point.

### 6.1 Baselines (for completeness, G3)
- OSR: softmax-threshold/MSP, ODIN/energy, OpenMax.
- FSL: Prototypical Networks, Matching Networks.
- Symbolic alternatives (ablation of the *symbolic engine*, not the idea): DeepProbLog, concept-bottleneck model.

---

## 7. Reproducibility artifacts (locks G2+)

Every run writes: resolved `config.yaml`, git commit hash, RNG seed, raw metrics (`metrics.json` + `*.csv`), and a logfile. The **KB is a versioned `kb.yaml`** so family definitions are auditable and diffable. Figures/tables at G4 are generated by scripts from these logs — never hand-typed (non-negotiable rule #2).

Proposed config skeleton (illustrative, not yet wired):

```yaml
seed: 0
data: { root: <path-from-author>, split: open_set_v1 }
frontend: { input: <pdw|iq|tfr>, heads: [rf, pri, primod, pw, scan, ipm], freeze_after_pretrain: true }
logic: { engine: ltn, tnorm: product, p_forall: 2 }
reject: { tau_rule: <calib>, tau_id: <calib>, id_score: msp }
kb: kb.yaml
```

---

## 8. Open questions requiring author sign-off

1. **[VERIFY@G0]** Does OpenEW-SA carry **per-attribute** labels, or only family labels? (Determines whether attribute supervision / concept-bottleneck is even possible.)
2. **[VERIFY@G0]** Exact taxonomies for A3 (PRI-mod), A5 (scan), A6 (intra-pulse) as labelled in the data — my proposed value lists are conventions, not the dataset's.
3. **[VERIFY@G0]** Input representation: PDW sequences, raw IQ, or time–frequency images?
4. **[VERIFY@G0]** Are multifunction emitters labelled at the **mode** level or emitter level?
5. **Family→attribute signatures:** author-supplied rule table, or data-induced + your review? (I will not invent these.)
6. **Novelty splits:** do the OpenEW-SA open-set hold-outs already separate unseen-mode / unseen-recombination / true-OOD, or must we construct these splits? (Needed for §3.2 breakdown.)
7. **Repo-name discrepancy:** the repository is named *Dynamic_Hypergraph*-Neuro-Symbolic, but the P2 brief specifies an **LTN** symbolic layer with no hypergraph component. Is a dynamic-hypergraph representation an intended part of this method (e.g. for the symbolic layer or emitter relations), or is the name a carry-over? This materially affects the architecture and must be resolved before coding.

---

## 9. What is intentionally NOT decided here
- Exact neural backbone (set at G2 once input representation is confirmed).
- Threshold values (calibrated on real validation data, logged).
- Whether to freeze vs. fine-tune the front-end (empirical, G3) — design *supports* freezing for few-shot.

**Next action after sign-off:** proceed to **G0 data audit** the moment the dataset path is provided, reconcile every **[VERIFY@G0]** item, then G2 (pipeline + closed-set smoke test). No coding against this spec until items 1–7 above are answered.
