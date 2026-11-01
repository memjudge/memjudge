# 0005 — Abstention and LoCoMo category 5

Status: accepted (2026-10-01)

**Context.** LoCoMo category 5 (446 adversarial questions) has empty gold answers and the
reference scorer marks every one wrong, so a system that correctly says nothing scores the
same as one that makes something up. LongMemEval marks abstention questions with `_abs`.

**Decision.** Both become `kind: abstention`. The context is correct when it does not support
a confident answer. LoCoMo's `adversarial_answer` is passed to the judge as a distractor the
context must not support. Abstention accuracy is reported separately.

**Consequences.** LoCoMo totals include category 5 (1,986 questions), unlike most published
numbers (1,540). Reports always show per-category results so both views are available.
LongMemEval abstention questions keep their question type as category, as LongMemEval's own
reporting does; abstention accuracy still covers them through `kind`.
