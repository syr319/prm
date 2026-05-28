# Error Type Simplified — 4-Class Taxonomy

**Date**: 2026-04-28
**Input**: `data/genprm_math_steps_with_soft_scores.json` (171,748 records)
**Output**: `data/genprm_math_steps_final.json`

---

## Taxonomy (new 4-class)

| ID | Name | Source classes | Description |
|----|------|----------------|-------------|
| 0 | correct | (hard_label=1) | Step is correct |
| 1 | computation_error | calc + algebraic | Arithmetic / algebraic manipulation mistake |
| 2 | propagation_error | wrong_reference | Error propagated from a prior wrong step |
| 3 | reasoning_error | logical_gap + conceptual + irrelevant | Wrong concept / logical gap / irrelevant step |

---

## Old 7-Class Distribution (Method B, wrong steps only)

| Old type | Name | N | % of wrong steps |
|----------|------|---|------------------|
| 1 | calculation_error | 27,437 | 61.8% |
| 2 | algebraic_error | 6,214 | 14.0% |
| 3 | logical_gap | 272 | 0.6% |
| 4 | wrong_reference | 9,873 | 22.2% |
| 5 | conceptual_error | 381 | 0.9% |
| 6 | irrelevant_step | 214 | 0.5% |

**Weak calc fallback**: 11,947 / 44,391 (26.9%)

---

## New 4-Class Distribution (all records)

| New type | Name | N | % of all steps |
|----------|------|---|----------------|
| 0 | correct | 127,357 | 74.2% |
| 1 | computation_error | 33,651 | 19.6% |
| 2 | propagation_error | 9,873 | 5.7% |
| 3 | reasoning_error | 867 | 0.5% |

---

## Impact on Training

- `error_type` field added to every record in `genprm_math_steps_final.json`
- `step4_build_student_model.py`: `NUM_ERROR_TYPES` updated from 7 → 4
- `step5_train_distillpRM.py`: reads `error_type` from JSON directly (no on-the-fly extraction)

### Label quality
- Class 0 (correct): 100% clean (from hard_label directly)
- Class 2 (propagation_error): high quality — explicit 'due to prior error' keywords
- Class 1 (computation_error): moderate — ~27% weak fallback in wrong steps
- Class 3 (reasoning_error): smaller class, lower keyword confidence

Overall label noise estimated at ~15-20% for wrong steps. Acceptable for an auxiliary regularization task.