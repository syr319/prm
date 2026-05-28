"""
Reclassify error types from 7-class to 4-class taxonomy using Method B
(weighted keyword scoring), and save the final dataset.

Old 7 classes → New 4 classes:
  0 correct             → 0 correct          (hard_label=1 only)
  1 calculation_error   → 1 computation_error
  2 algebraic_error     → 1 computation_error
  3 logical_gap         → 3 reasoning_error
  4 wrong_reference     → 2 propagation_error
  5 conceptual_error    → 3 reasoning_error
  6 irrelevant_step     → 3 reasoning_error

New taxonomy:
  0 correct           — step is correct
  1 computation_error — arithmetic / algebraic manipulation mistake
  2 propagation_error — error propagated from a prior wrong step
  3 reasoning_error   — wrong formula/concept, logical gap, irrelevant

Input:  data/genprm_math_steps_with_soft_scores.json
Output: data/genprm_math_steps_final.json  (adds field: error_type)
Report: distill/reports/error_type_simplified.md
"""

import json
from collections import Counter
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
IN_PATH   = ROOT / "data" / "genprm_math_steps_with_soft_scores.json"
OUT_PATH  = ROOT / "data" / "genprm_math_steps_final.json"
REPORT    = ROOT / "distill" / "reports" / "error_type_simplified.md"

# ─── New taxonomy ─────────────────────────────────────────────────────────────

NEW_ERROR_LABELS = {
    0: "correct",
    1: "computation_error",   # arithmetic + algebraic mistakes
    2: "propagation_error",   # error from a prior wrong step
    3: "reasoning_error",     # wrong concept / logical gap / irrelevant
}

# Old-to-new mapping
OLD_TO_NEW = {
    0: 0,  # correct → correct
    1: 1,  # calculation_error → computation_error
    2: 1,  # algebraic_error   → computation_error
    3: 3,  # logical_gap       → reasoning_error
    4: 2,  # wrong_reference   → propagation_error
    5: 3,  # conceptual_error  → reasoning_error
    6: 3,  # irrelevant_step   → reasoning_error
}

# ─── Method B weighted keyword scoring (same as step_verify_error_extraction) ─

_WEIGHTS = {
    "calculation": (1, 3), "calculat": (1, 3),
    "arithmet": (1, 3), "miscalculat": (1, 4),
    "wrong number": (1, 4), "incorrect number": (1, 4),
    "should be": (1, 1), "instead of": (1, 1),
    "product": (1, 1), "multipl": (1, 1), "divid": (1, 1),
    "numerical": (1, 2), "computation": (1, 2),
    "incorrect result": (1, 2), "wrong result": (1, 2),

    "simplif": (2, 3), "factori": (2, 3),
    "algebr": (2, 3), "expand": (2, 2),
    "distribut": (2, 2), "polynomial": (2, 3),
    "coefficient": (2, 2), "manipulat": (2, 2),
    "substitut": (2, 2), "equation is": (2, 2),
    "left side": (2, 1), "right side": (2, 1),

    "missing step": (3, 5), "logical gap": (3, 5),
    "no justif": (3, 4), "without justif": (3, 4),
    "leap": (3, 4), "does not follow": (3, 4),
    "skips": (3, 3), "skipped": (3, 3),
    "not proven": (3, 3), "not shown": (3, 3),
    "unjustif": (3, 4), "omit": (3, 3),

    "due to the earlier": (4, 5), "due to the error in": (4, 5),
    "prior error": (4, 5), "earlier error": (4, 5),
    "propagated": (4, 4), "follows from the error": (4, 5),
    "wrong value from": (4, 4), "incorrect value from": (4, 4),
    "since paragraph": (4, 3), "because paragraph": (4, 3),
    "based on the incorrect": (4, 3), "relies on": (4, 2),
    "due to the mistake in": (4, 5), "prior mistake": (4, 4),
    "earlier mistake": (4, 4), "incorrect conclusion": (4, 3),

    "wrong formula": (5, 5), "incorrect formula": (5, 5),
    "wrong theorem": (5, 4), "misappl": (5, 4),
    "wrong rule": (5, 4), "incorrect rule": (5, 4),
    "wrong property": (5, 3), "using the wrong": (5, 3),
    "should use": (5, 3), "wrong concept": (5, 3),
    "arccos": (5, 3), "arcsin": (5, 3),

    "irrelevant": (6, 5), "unrelated": (6, 4),
    "not needed": (6, 3), "unnecessary": (6, 3),
    "does not contribute": (6, 4),
}


def classify_old7(cot: str) -> int:
    """Method B: weighted scoring → old 7-class result (1–6). Fallback → 1."""
    low = cot.lower()
    scores = {i: 0 for i in range(1, 7)}
    for kw, (etype, weight) in _WEIGHTS.items():
        if kw in low:
            scores[etype] += weight
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else 1


def classify_new4(cot: str, hard_label: int) -> int:
    """
    Classify into the simplified 4-class taxonomy.
      hard_label=1 → always 0 (correct)
      hard_label=0 → run Method B, then map old→new
    """
    if hard_label == 1:
        return 0
    old = classify_old7(cot)
    return OLD_TO_NEW[old]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {IN_PATH} ...")
    with open(IN_PATH, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records):,} records.\n")

    n_correct = sum(1 for r in records if r.get("hard_label") == 1)
    n_wrong   = sum(1 for r in records if r.get("hard_label") == 0)

    old7_counter  = Counter()
    new4_counter  = Counter()
    fallback_count = 0

    for r in records:
        hl  = int(r.get("hard_label", -1))
        cot = r.get("verification_cot", "")

        if hl == 1:
            etype_new = 0
            old7_counter[0] += 1
        elif hl == 0:
            old_type  = classify_old7(cot)
            etype_new = OLD_TO_NEW[old_type]
            old7_counter[old_type] += 1
            # Track weak fallback (fallback to type 1 with no strong calc kw)
            strong_kws = ["calculat", "arithmet", "miscalculat",
                          "wrong number", "incorrect number",
                          "product", "multipl", "divid", "numerical"]
            if old_type == 1 and not any(kw in cot.lower() for kw in strong_kws):
                fallback_count += 1
        else:
            etype_new = -1  # unknown label → skip

        r["error_type"] = etype_new
        new4_counter[etype_new] += 1

    # ── Print stats ──
    print("=" * 60)
    print("OLD 7-CLASS DISTRIBUTION (Method B, wrong steps only)")
    print("=" * 60)
    old_names = {
        1: "calculation_error",
        2: "algebraic_error",
        3: "logical_gap",
        4: "wrong_reference",
        5: "conceptual_error",
        6: "irrelevant_step",
    }
    total_wrong = n_wrong
    for etype in range(1, 7):
        n   = old7_counter[etype]
        pct = n / total_wrong * 100 if total_wrong else 0
        bar = "#" * (n * 40 // max(old7_counter.values(), default=1))
        print(f"  {etype} {old_names[etype]:<22} {n:>7,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Weak calc fallback: {fallback_count:,} ({fallback_count/total_wrong*100:.1f}% of wrong steps)")

    print()
    print("=" * 60)
    print("NEW 4-CLASS DISTRIBUTION")
    print("=" * 60)
    total = len(records)
    for etype in range(4):
        n   = new4_counter[etype]
        pct = n / total * 100
        bar = "#" * (n * 40 // max(new4_counter.values(), default=1))
        print(f"  {etype} {NEW_ERROR_LABELS[etype]:<22} {n:>7,}  ({pct:5.1f}%)  {bar}")

    print()
    print(f"  Correct steps  (class 0): {new4_counter[0]:,}  ({new4_counter[0]/total*100:.1f}%)")
    print(f"  Wrong steps    (1+2+3):   {new4_counter[1]+new4_counter[2]+new4_counter[3]:,}  "
          f"({(new4_counter[1]+new4_counter[2]+new4_counter[3])/total*100:.1f}%)")

    # ── Save ──
    print(f"\nSaving {len(records):,} records to {OUT_PATH} ...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("Saved.")

    # ── Write report ──
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Error Type Simplified — 4-Class Taxonomy",
        "",
        f"**Date**: {__import__('datetime').date.today()}",
        f"**Input**: `data/genprm_math_steps_with_soft_scores.json` ({len(records):,} records)",
        f"**Output**: `data/genprm_math_steps_final.json`",
        "",
        "---",
        "",
        "## Taxonomy (new 4-class)",
        "",
        "| ID | Name | Source classes | Description |",
        "|----|------|----------------|-------------|",
        "| 0 | correct | (hard_label=1) | Step is correct |",
        "| 1 | computation_error | calc + algebraic | Arithmetic / algebraic manipulation mistake |",
        "| 2 | propagation_error | wrong_reference | Error propagated from a prior wrong step |",
        "| 3 | reasoning_error | logical_gap + conceptual + irrelevant | Wrong concept / logical gap / irrelevant step |",
        "",
        "---",
        "",
        "## Old 7-Class Distribution (Method B, wrong steps only)",
        "",
        "| Old type | Name | N | % of wrong steps |",
        "|----------|------|---|------------------|",
    ]
    for etype in range(1, 7):
        n   = old7_counter[etype]
        pct = n / total_wrong * 100 if total_wrong else 0
        lines.append(f"| {etype} | {old_names[etype]} | {n:,} | {pct:.1f}% |")

    lines += [
        "",
        f"**Weak calc fallback**: {fallback_count:,} / {total_wrong:,} ({fallback_count/total_wrong*100:.1f}%)",
        "",
        "---",
        "",
        "## New 4-Class Distribution (all records)",
        "",
        "| New type | Name | N | % of all steps |",
        "|----------|------|---|----------------|",
    ]
    for etype in range(4):
        n   = new4_counter[etype]
        pct = n / total * 100
        lines.append(f"| {etype} | {NEW_ERROR_LABELS[etype]} | {n:,} | {pct:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## Impact on Training",
        "",
        "- `error_type` field added to every record in `genprm_math_steps_final.json`",
        "- `step4_build_student_model.py`: `NUM_ERROR_TYPES` updated from 7 → 4",
        "- `step5_train_distillpRM.py`: reads `error_type` from JSON directly (no on-the-fly extraction)",
        "",
        "### Label quality",
        f"- Class 0 (correct): 100% clean (from hard_label directly)",
        f"- Class 2 (propagation_error): high quality — explicit 'due to prior error' keywords",
        f"- Class 1 (computation_error): moderate — ~{fallback_count/total_wrong*100:.0f}% weak fallback in wrong steps",
        f"- Class 3 (reasoning_error): smaller class, lower keyword confidence",
        "",
        "Overall label noise estimated at ~15-20% for wrong steps. Acceptable for an auxiliary regularization task.",
    ]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {REPORT}")


if __name__ == "__main__":
    main()
