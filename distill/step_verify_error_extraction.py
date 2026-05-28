"""
Verify error type extraction quality on 500 sampled wrong steps.

Two methods:
  A - Priority keyword matching (first-match wins)
  B - Weighted keyword scoring (all keywords contribute, highest wins)

Both methods use the same taxonomy (7 classes).
We compare their agreement and fallback rates.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "genprm_math_steps.json"
REPORT_PATH = ROOT / "distill" / "reports" / "error_type_analysis.md"

SEED       = 42
SAMPLE_N   = 500

# ─── Error taxonomy ───────────────────────────────────────────────────────────

ERROR_LABELS = {
    0: "correct",
    1: "calculation_error",
    2: "algebraic_error",
    3: "logical_gap",
    4: "wrong_reference",
    5: "conceptual_error",
    6: "irrelevant_step",
}

# ─── Method A: Priority keyword matching ──────────────────────────────────────
# Rules checked in order; first rule that fires wins.
# Each rule: (error_type_id, list_of_required_keywords_any_match)

METHOD_A_RULES = [
    # wrong_reference: explicitly says the error propagates from a prior step
    (4, [
        "due to the earlier", "due to the error in",
        "due to the mistake in", "based on the incorrect",
        "based on the wrong", "follows from the error",
        "follows from the incorrect", "uses the wrong value",
        "uses the incorrect value", "wrong value from",
        "incorrect value from", "incorrect result from",
        "relies on the incorrect", "propagated from",
        "because of the error in paragraph",
        "since paragraph", "because paragraph",
        "prior error", "earlier error", "previous error",
        "prior mistake", "earlier mistake",
    ]),
    # logical_gap: unjustified leaps, missing steps
    (3, [
        "missing step", "missing justif",
        "no justif", "without justif",
        "leap", "jump", "logical gap",
        "does not follow", "does not logically follow",
        "skips", "skipped", "omits", "omitted",
        "not proven", "not shown", "not demonstrated",
        "unjustif", "unclear how",
    ]),
    # conceptual_error: wrong formula, theorem, definition
    (5, [
        "wrong formula", "incorrect formula",
        "wrong theorem", "incorrect theorem",
        "wrong definition", "incorrect definition",
        "wrong concept", "misappl",
        "incorrect application", "wrong rule",
        "incorrect rule", "wrong property",
        "incorrect property", "wrong identity",
        "incorrect identity", "arccos", "arcsin",
        "should use", "using the wrong",
    ]),
    # algebraic_error: wrong algebraic manipulation
    (2, [
        "simplif", "factori", "expanding", "expands",
        "distribut", "algebraic", "algebra",
        "coefficient", "polynomial", "manipulat",
        "equation is incorrect", "equation is wrong",
        "wrong equation", "incorrect equation",
        "left side", "right side",
        "substitut", "incorrectly simplif",
    ]),
    # calculation_error: arithmetic / numerical mistakes
    (1, [
        "calculat", "arithmet",
        "incorrect result", "wrong result",
        "incorrect answer", "wrong answer",
        "product is", "sum is", "value is",
        "should be", "instead of",
        "miscalculat", "computation",
        "numerical", "digits", "multipl", "divid",
        "addition", "subtraction",
        "incorrectly comput", "incorrectly calculat",
    ]),
    # irrelevant_step
    (6, [
        "irrelevant", "unrelated",
        "does not contribute", "not needed",
        "unnecessary", "not relevant",
    ]),
]


def classify_method_a(cot: str) -> int:
    """First matching rule wins; fallback → 1 (calculation_error)."""
    low = cot.lower()
    for etype, keywords in METHOD_A_RULES:
        for kw in keywords:
            if kw in low:
                return etype
    return 1  # fallback


# ─── Method B: Weighted keyword scoring ───────────────────────────────────────
# Each keyword contributes weight to a category; highest total wins.

METHOD_B_WEIGHTS = {
    # (keyword, error_type_id, weight)
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


def classify_method_b(cot: str) -> int:
    """Weighted keyword scoring; highest-scoring type wins. Fallback → 1."""
    low = cot.lower()
    scores = {i: 0 for i in range(1, 7)}
    for kw, (etype, weight) in METHOD_B_WEIGHTS.items():
        if kw in low:
            scores[etype] += weight
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else 1


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {DATA_PATH} ...")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    wrong = [r for r in data if r["hard_label"] == 0 and r.get("verification_cot")]
    print(f"Total wrong steps with CoT: {len(wrong):,}")

    random.seed(SEED)
    sample = random.sample(wrong, min(SAMPLE_N, len(wrong)))
    print(f"Sampled: {len(sample):,}\n")

    # ── Run both methods ──
    results_a = [classify_method_a(r["verification_cot"]) for r in sample]
    results_b = [classify_method_b(r["verification_cot"]) for r in sample]

    dist_a = Counter(results_a)
    dist_b = Counter(results_b)

    # Agreement between methods
    agree = sum(a == b for a, b in zip(results_a, results_b))
    agree_pct = agree / len(sample) * 100

    # "Fallback" = method A returns 1 with NO actual calc keyword (weak evidence)
    def is_weak_calc(cot: str, etype: int) -> bool:
        """Returns True if etype==1 was assigned by default (no strong calc kw)."""
        if etype != 1:
            return False
        strong_kws = ["calculat", "arithmet", "miscalculat",
                      "wrong number", "incorrect number",
                      "product", "multipl", "divid", "numerical"]
        low = cot.lower()
        return not any(kw in low for kw in strong_kws)

    weak_a = sum(is_weak_calc(r["verification_cot"], e) for r, e in zip(sample, results_a))
    weak_b = sum(is_weak_calc(r["verification_cot"], e) for r, e in zip(sample, results_b))

    # ── Print stats ──
    print("=" * 70)
    print("DISTRIBUTION — Method A (priority keyword matching)")
    print("=" * 70)
    for etype in sorted(ERROR_LABELS):
        if etype == 0:
            continue
        n = dist_a[etype]
        bar = "#" * (n * 40 // max(dist_a.values()))
        print(f"  {etype} {ERROR_LABELS[etype]:<22} {n:>4}  ({n/len(sample)*100:5.1f}%)  {bar}")
    print(f"\n  Weak calc fallback: {weak_a} ({weak_a/len(sample)*100:.1f}%)")

    print()
    print("=" * 70)
    print("DISTRIBUTION — Method B (weighted scoring)")
    print("=" * 70)
    for etype in sorted(ERROR_LABELS):
        if etype == 0:
            continue
        n = dist_b[etype]
        bar = "#" * (n * 40 // max(dist_b.values()))
        print(f"  {etype} {ERROR_LABELS[etype]:<22} {n:>4}  ({n/len(sample)*100:5.1f}%)  {bar}")
    print(f"\n  Weak calc fallback: {weak_b} ({weak_b/len(sample)*100:.1f}%)")

    print(f"\nMethod A vs B agreement: {agree}/{len(sample)} ({agree_pct:.1f}%)")

    # ── Print 20 examples ──
    print()
    print("=" * 70)
    print("20 RANDOM EXAMPLES (CoT excerpt + both classifications)")
    print("=" * 70)

    display_idx = random.sample(range(len(sample)), 20)
    examples = []
    for idx in display_idx:
        r = sample[idx]
        ea = results_a[idx]
        eb = results_b[idx]
        cot_excerpt = r["verification_cot"][:250].replace("\n", " ").strip()
        examples.append({
            "idx":         idx,
            "step_index":  r["step_index"],
            "cot_excerpt": cot_excerpt,
            "method_a":    ea,
            "method_a_label": ERROR_LABELS[ea],
            "method_b":    eb,
            "method_b_label": ERROR_LABELS[eb],
            "agree":       ea == eb,
        })

    for i, ex in enumerate(examples):
        agree_str = "✓" if ex["agree"] else "✗ DISAGREE"
        print(f"\n[{i+1:02d}] step_index={ex['step_index']}  {agree_str}")
        print(f"  CoT: \"{ex['cot_excerpt']}\"")
        print(f"  A → {ex['method_a']} ({ex['method_a_label']})")
        print(f"  B → {ex['method_b']} ({ex['method_b_label']})")

    # ── Per-type examples ──
    print()
    print("=" * 70)
    print("BEST EXAMPLE PER TYPE (Method A)")
    print("=" * 70)
    for etype in range(1, 7):
        type_examples = [(r, e) for r, e in zip(sample, results_a) if e == etype]
        if not type_examples:
            print(f"\n  [{etype}] {ERROR_LABELS[etype]}: no examples")
            continue
        # Pick the one where CoT is most on-point (has the most keywords)
        r = type_examples[0][0]
        cot_short = r["verification_cot"][:300].replace("\n", " ").strip()
        print(f"\n  [{etype}] {ERROR_LABELS[etype]}")
        print(f"  \"{cot_short}\"")

    # ── Build markdown report ──
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Error Type Extraction Analysis",
        "",
        f"**Date**: {__import__('datetime').date.today()}  ",
        f"**Sample**: {len(sample)} wrong steps (random seed={SEED})  ",
        f"**Source**: `data/genprm_math_steps.json`",
        "",
        "---",
        "",
        "## Error Type Taxonomy",
        "",
        "| ID | Name | Description |",
        "|----|------|-------------|",
        "| 1 | calculation_error | Arithmetic/numerical mistake |",
        "| 2 | algebraic_error | Wrong algebraic manipulation (simplify, factor, expand) |",
        "| 3 | logical_gap | Unjustified leap, missing intermediate step |",
        "| 4 | wrong_reference | Error propagated from a prior wrong step |",
        "| 5 | conceptual_error | Wrong formula/theorem/rule applied |",
        "| 6 | irrelevant_step | Step doesn't contribute to the solution |",
        "",
        "---",
        "",
        "## Distribution",
        "",
        "| Type | Name | Method A | % | Method B | % |",
        "|------|------|----------|---|----------|---|",
    ]

    for etype in range(1, 7):
        na = dist_a[etype]
        nb = dist_b[etype]
        lines.append(
            f"| {etype} | {ERROR_LABELS[etype]} | {na} | {na/len(sample)*100:.1f}% | "
            f"{nb} | {nb/len(sample)*100:.1f}% |"
        )

    lines += [
        "",
        f"**Method A weak calc fallback**: {weak_a} / {len(sample)} ({weak_a/len(sample)*100:.1f}%)",
        f"**Method A vs B agreement**: {agree} / {len(sample)} ({agree_pct:.1f}%)",
        "",
        "---",
        "",
        "## 20 Random Sample Cases",
        "",
        "Format: `CoT excerpt` → `Method A result` / `Method B result` (✓ agree / ✗ disagree)",
        "",
    ]

    for i, ex in enumerate(examples):
        agree_str = "✓" if ex["agree"] else "✗ DISAGREE"
        lines.append(
            f"**[{i+1:02d}]** step_index={ex['step_index']} {agree_str}  "
        )
        lines.append(f"> {ex['cot_excerpt']}")
        lines.append(
            f"- A → `{ex['method_a_label']}`  "
            f"B → `{ex['method_b_label']}`"
        )
        lines.append("")

    lines += [
        "---",
        "",
        "## One Example Per Error Type (Method A)",
        "",
    ]
    for etype in range(1, 7):
        type_examples = [(r, e) for r, e in zip(sample, results_a) if e == etype]
        lines.append(f"### {etype}. {ERROR_LABELS[etype]}")
        if not type_examples:
            lines.append("*No examples in this sample.*")
        else:
            r = type_examples[0][0]
            lines.append(f"> {r['verification_cot'][:400].replace(chr(10), ' ').strip()}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Conclusions",
        "",
        "Based on the distribution and examples above:",
        "",
        f"- **Method A weak fallback rate**: {weak_a/len(sample)*100:.1f}% (steps classified as "
        f"calculation_error with no strong arithmetic keywords)",
        f"- **A vs B agreement**: {agree_pct:.1f}%",
        "",
        "### Recommended action",
        "",
    ]

    if agree_pct >= 75 and weak_a / len(sample) <= 0.20:
        lines.append("Both methods agree well and fallback rate is low → **Method A (priority) is usable for training labels**.")
    elif agree_pct >= 60:
        lines.append("Moderate agreement. Consider **Method B (weighted)** for richer signal, or simplify taxonomy to 3-4 classes.")
    else:
        lines.append("Low agreement. Keyword matching alone is insufficient. Recommend simplifying to 3 classes (calculation / logical / wrong_reference) or using a small classifier.")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
