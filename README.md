# CGA Paper 1 — Numerical Companion

Reproducibility artifact for:

> **Defining Wavefunction Branches: Coherence Graph Fragmentation and the Monitoring Structure**
> Brian St. Laurent (2026)

This repository contains the paper (PDF) and the complete, self-contained test suite backing every numerical claim it makes.

## Contents

| File | Description |
|------|-------------|
| `CGA_Paper1_3.0.pdf` | The paper (27 pages). |
| `paper1_3.0_companion.py` | 18-test suite. Runs end-to-end with one command. |

## Quick start

```bash
python paper1_3.0_companion.py
```

Runs all 18 tests sequentially, prints PASS/FAIL for each, and exits 0 on success.

**Expected runtime:** ~15 s on a modern laptop (single CPU core, no GPU).
**No arguments, no input files, no network access.**

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.20
- scipy ≥ 1.7

Tested on Python 3.14.3, numpy 2.4.2, scipy 1.17.1 (Windows 10). No other dependencies.

## Method

All decoherence **emerges** from unitary evolution of a physical Hamiltonian (system + apparatus + environment) followed by partial trace. Nothing is put in by hand — no Lindblad equation, no Born–Markov approximation, no pre-selected pointer basis. The analytic Markov predictions stated in the paper are compared against this exact unitary dynamics.

The pipeline (build *G*<sub>H</sub> → Fiedler partition → check monitoring conditions M1/M2/M3 → predict formation time *t*\* → compare to simulated ρ(*t*)) is run end-to-end wherever the paper claims it applies.

## Test → paper map

Tests walk the paper in section order. Tests 1–15 are cited throughout the paper; Tests 16–18 are appendix tests supporting the §7.2 discussion of the Fiedler regime boundary.

| Test | Paper § | Claim |
|-----:|---------|-------|
| 1  | §2.4    | Fiedler robustness vs coupling ratio |
| 2  | §2.5    | Multi-branch *k* = 3 (sequential bisection) |
| 3  | §2.6    | Threshold robustness (Thm 2.6a) |
| 4  | §2.6    | Initial-state universality (Thm 2.6b) |
| 5  | §3.2    | Selective dephasing rates (Thm 3.1) |
| 6  | §3.3    | Formation time / Grönwall bound (Thm 3.2) |
| 7  | §4.1    | Tree structure / no re-merger (Thm 4.1) |
| 8  | §4.2    | Pointer variance bound (Prop 4.2) |
| 9  | §4.3    | Effective collapse (Prop 4.3) |
| 10 | §4.4    | Area-law / coherence decay (Lemma 4.4) |
| 11 | §5.1    | Stern–Gerlach |
| 12 | §5.2    | Double-slit |
| 13 | §5.3    | Bell correlations |
| 14 | §7.1    | Environment scaling *d*<sub>E</sub> = 4…256 |
| 15 | §7.2    | Monitoring stress (A4 necessity) |
| 16 | §7.2    | Fiedler gap degeneracy (λ₂/λ₃ diagnostic) |
| 17 | §7.2    | Random unstructured Hamiltonians (honest failure) |
| 18 | §3.2    | Null monitoring false-positive check |

## Citation

If you use this code or reproduce results from the paper, please cite:

> St. Laurent, B. (2026). *Defining Wavefunction Branches: Coherence Graph Fragmentation and the Monitoring Structure.* Companion code: `paper1_3.0_companion.py`.

## License

MIT.

## Contact

Brian St. Laurent — <bnstlaurent@gmail.com>
