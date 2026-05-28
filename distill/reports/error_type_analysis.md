# Error Type Extraction Analysis

**Date**: 2026-04-28  
**Sample**: 500 wrong steps (random seed=42)  
**Source**: `data/genprm_math_steps.json`

---

## Error Type Taxonomy

| ID | Name | Description |
|----|------|-------------|
| 1 | calculation_error | Arithmetic/numerical mistake |
| 2 | algebraic_error | Wrong algebraic manipulation (simplify, factor, expand) |
| 3 | logical_gap | Unjustified leap, missing intermediate step |
| 4 | wrong_reference | Error propagated from a prior wrong step |
| 5 | conceptual_error | Wrong formula/theorem/rule applied |
| 6 | irrelevant_step | Step doesn't contribute to the solution |

---

## Distribution

| Type | Name | Method A | % | Method B | % |
|------|------|----------|---|----------|---|
| 1 | calculation_error | 253 | 50.6% | 335 | 67.0% |
| 2 | algebraic_error | 80 | 16.0% | 63 | 12.6% |
| 3 | logical_gap | 3 | 0.6% | 3 | 0.6% |
| 4 | wrong_reference | 152 | 30.4% | 94 | 18.8% |
| 5 | conceptual_error | 11 | 2.2% | 2 | 0.4% |
| 6 | irrelevant_step | 1 | 0.2% | 3 | 0.6% |

**Method A weak calc fallback**: 133 / 500 (26.6%)
**Method A vs B agreement**: 399 / 500 (79.8%)

---

## 20 Random Sample Cases

Format: `CoT excerpt` → `Method A result` / `Method B result` (✓ agree / ✗ disagree)

**[01]** step_index=7 ✓  
> Paragraph 8 states the total favorable is 28, but this is incorrect as shown in paragraph 7's verification. The correct value should be 55. Hence, this paragraph is wrong due to the earlier error.
- A → `wrong_reference`  B → `wrong_reference`

**[02]** step_index=13 ✓  
> The paragraph correctly points out the contradiction in the system (0=2 from x² term and c needing to be both 1 and 0). However, since this path was based on an incorrect assumption (starting from p(x)=x+c after an earlier flawed approach), the concl
- A → `calculation_error`  B → `calculation_error`

**[03]** step_index=20 ✓  
> Paragraph 21 calculates revenue with 10 workers as 10*$124 = $1240. Correct.
- A → `calculation_error`  B → `calculation_error`

**[04]** step_index=3 ✗ DISAGREE  
> Paragraph 4 incorrectly uses the exponent 3/2 to derive sqrt(3^3). However, since the correct exponent from Paragraph 2 is 3, this step is based on an earlier mistake. The proper simplification of 3^3 would be 27, but the approach here is flawed beca
- A → `wrong_reference`  B → `algebraic_error`

**[05]** step_index=7 ✓  
> Paragraph 8 subtracts 9 from 28 (incorrect Figure 1 total) to get 19, which coincidentally matches the correct Figure 2 total from the code. However, this is due to the initial error in paragraph 3 and 4 balancing out (30-11=19, but they did 28-9=19)
- A → `calculation_error`  B → `calculation_error`

**[06]** step_index=8 ✓  
> The conclusion restates the final answer derived from the flawed earlier steps. The correct answer is different, so this paragraph is also incorrect.
- A → `calculation_error`  B → `calculation_error`

**[07]** step_index=6 ✓  
> Paragraph7 is incorrect because it results from the prior errors. The correct value of e^{i(α+β)} should have real part -56/65 and imaginary part -33/65, leading to sin(α+β) being -33/65. However, the paragraph claims -84/65 +35/65i, which is wrong.
- A → `wrong_reference`  B → `wrong_reference`

**[08]** step_index=6 ✓  
> Substituting \( y = 30 - x \) into the points equation is correct in method, but since the original points equation was incorrect (from paragraph 4/5), the substituted equation here is also wrong. The correct substituted equation should be \(0.6x + 0
- A → `algebraic_error`  B → `algebraic_error`

**[09]** step_index=9 ✓  
> Paragraph 10 solves the quadratic equation x² +x +1=0 using the quadratic formula, arriving at complex roots. The calculations are correct: discriminant b²−4ac = 1−4= -3, leading to sqrt(-3). The steps are accurate.
- A → `calculation_error`  B → `calculation_error`

**[10]** step_index=8 ✓  
> Calculating the difference between the final total ($1500) and initial total ($1000) as $500 is correct. This part is accurate.
- A → `calculation_error`  B → `calculation_error`

**[11]** step_index=3 ✓  
> The fourth paragraph states that \(25 - 2k\) must be a perfect square. Here, the approach is to list all perfect squares ≤25 and set \(25 - 2k\) equal to them. However, there's confusion here because \(25 - 2k\) is actually \(x^2\). The paragraph cor
- A → `algebraic_error`  B → `algebraic_error`

**[12]** step_index=2 ✗ DISAGREE  
> Paragraph 3 claims that simplifying the terms inside the parentheses gives \(27 \times 20 = 540\). However, according to the previous analysis in paragraph 2, the terms were \(34\) and \(20\), not \(27\) and \(20\). The error arises because the first
- A → `algebraic_error`  B → `calculation_error`

**[13]** step_index=3 ✓  
> Paragraph 4 substitutes the expression for c into the dot product with b. The substitution is correctly done by replacing c with the expression from paragraph 3. This step is valid.
- A → `algebraic_error`  B → `algebraic_error`

**[14]** step_index=4 ✓  
> Paragraph 5 claims that 0.17 is closer to 0.2 (1/5) than to the other options. To verify this, we must calculate the absolute differences between 0.17 and each option: - |0.17 - 0.25| = 0.08 - |0.17 - 0.2| = 0.03 - |0.17 - 0.1667| ≈ 0.0033 - |0.17 -
- A → `calculation_error`  B → `calculation_error`

**[15]** step_index=6 ✓  
> Paragraph 7 states the range of sinx is [-1,1) because sinx ≠1. This is correct because the sine function normally ranges between -1 and 1, and excluding 1 as per the problem's condition. So this part is correct.
- A → `calculation_error`  B → `calculation_error`

**[16]** step_index=2 ✓  
> Paragraph 3 attempts to model the quadratic as a cubic equation, which is incorrect. The original problem involves a quadratic inequality, so introducing a cubic term is a mistake. This error likely stems from the flawed premise in paragraph 2 that a
- A → `calculation_error`  B → `calculation_error`

**[17]** step_index=1 ✓  
> Paragraph 2 calculates Jay's speed by noting he walks 0.75 miles every 15 minutes. Since there are 4 intervals of 15 minutes in an hour, multiplying by 4 would give his hourly rate. However, the paragraph incorrectly states "3 times that distance" (u
- A → `calculation_error`  B → `calculation_error`

**[18]** step_index=2 ✗ DISAGREE  
> Paragraph 3 counts the frequency of each score. However, there's an error here. Looking at the stem row 8 | 2 6 7 9 9 9 9: - The leaves are 2,6,7,9,9,9,9 → so 89 appears 4 times (not 1). The paragraph incorrectly states that 89 appears 1 time.   Addi
- A → `wrong_reference`  B → `calculation_error`

**[19]** step_index=10 ✓  
> Paragraph 11 references the exponential series \(e = \sum_{n=0}^\infty 1/n!\). This foundational knowledge is correct.
- A → `calculation_error`  B → `calculation_error`

**[20]** step_index=6 ✗ DISAGREE  
> Paragraph 7 is incorrect due to the prior errors. Since the second bracket was miscalculated as 10 + 8/3 instead of (10+8)/3, adding 10 + 8/3 gives 38/3, but the correct value should be 6. Hence, this paragraph's result is wrong.
- A → `wrong_reference`  B → `calculation_error`

---

## One Example Per Error Type (Method A)

### 1. calculation_error
> The final answer presented is 2,903,040, which is incorrect because the problem was miscalculated from the start due to the wrong number of groups. The correct answer should be 3! (for arranging the three teams) multiplied by each team's internal arrangements: 3!×3!×3!×2!? Wait, wait, let me check again. Wait, the three teams are Cubs (3), Red Sox (3), Yankees (2). So the groups are 3 in total. So

### 2. algebraic_error
> Paragraph 5 states that the greatest power of 3 dividing 144 is \(3^4\), so \(y = 4\). This is incorrect because the correct prime factorization (from paragraph 2's code) shows that the exponent of 3 is 2, not 4. The error arises from the earlier incorrect factorization in paragraphs 2 and 3, leading to an incorrect value for \(y\).

### 3. logical_gap
> Paragraph 3 claims that the numbers must be 49, 50, 51. However, this is incorrect. The problem states that one of the three numbers must be divisible by 49, but the triplet could be any three consecutive numbers where one is a multiple of 49. Starting at 49 is the first possible triplet, but there's no justification given for why it must be exactly 49, 50, 51. The solution seems to assume the fir

### 4. wrong_reference
> Adding the base area (100π) and the incorrect curved SA (100π) gives 200π. But the correct curved SA should be 200π, so total would be 300π. The error here is due to the mistake in paragraph 4. Thus, this paragraph's conclusion is wrong because it uses the wrong curved surface area.

### 5. conceptual_error
> Paragraph 5 addresses the case with 2 stools. It uses combinations to choose 2 positions out of 9, yielding 36. This is correct because when arranging n objects with k indistinct and m indistinct, the formula is (n choose k) where n is total positions. Here, total seats are 10 (8+2), but the stools are placed among the chairs. Wait, actually, when arranging 8 chairs and 2 stools in a row, the tota

### 6. irrelevant_step
> Paragraph 5 introduces an alternative approach using the altitude from C to AB. However, this is unnecessary because the median to the hypotenuse is already known to be half the hypotenuse. The paragraph incorrectly states that the distance from C to M is related to the altitude, which may confuse the two concepts (median vs. altitude). The altitude is different from the median, so this part is mi

---

## Conclusions

Based on the distribution and examples above:

- **Method A weak fallback rate**: 26.6% (steps classified as calculation_error with no strong arithmetic keywords)
- **A vs B agreement**: 79.8%

### Recommended action

Moderate agreement. Consider **Method B (weighted)** for richer signal, or simplify taxonomy to 3-4 classes.