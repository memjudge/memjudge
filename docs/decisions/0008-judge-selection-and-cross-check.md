# 0008 — Judge selection rule and cross-judge check

Status: accepted (2026-10-03)

**Context.** MemJudge is maintained by the founder of Recalld, a memory system that is itself
on the leaderboard. Recalld uses Google Gemini models internally. LLM judges are known to
slightly prefer text written by their own model family, and the memories a system returns
are largely LLM-written (extracted facts, summaries). A judge from the same family as one
tested system could favour it, and a judge hand-picked by a competitor would not be trusted
even if it did not. No judge is neutral to everyone: most other systems default to OpenAI
models.

**Decision.**

1. **Calibrate every candidate the same way.** Each candidate judge is run with
   `memjudge calibrate labels.jsonl --judge A --judge B …` on the same ~200 hand-labelled
   examples, with the same prompt and pinned reasoning effort. Agreement, false-accept rate,
   false-reject rate and cost are published for every candidate, not only the winner.
2. **Selection.** The candidate with the highest agreement wins. Candidates within
   2 percentage points of the best agreement, whose false-accept rate is not higher than the
   best candidate's by more than 2 points, count as tied.
3. **Conflict-of-interest tie-break.** Among tied candidates, any model from a family used
   inside a system the maintainer is affiliated with (currently: Gemini, because of Recalld)
   is excluded. Among the rest, the cheaper wins. Such a model can therefore only become the
   judge by beating every unaffiliated candidate by more than the tie margin.
4. **Cross-judge check every season.** A seeded sample (default 300 questions) of every
   published run is re-judged with a second judge from a different vendor
   (`memjudge crosscheck`). Per connector, the disagreement rate and the accuracy under both
   judges are published. A system whose accuracy shifts much more than the others' is
   flagged in the season notes.
5. **Disclosure.** Season notes state the judge, the calibration table, the cross-check
   table, and which model families each seed connector's system uses, where known.

**Consequences.** The judge is chosen by a public rule rather than by preference, and the
effect of that choice on each system is a published number. The 2-point margin reflects
the precision of ~200 examples (roughly ±3 points); it should be tightened if the
calibration set grows. The cross-check costs about one extra judge call per sampled
question per run. When the season-1 judge is chosen, the outcome is appended below.

**Outcome.** _Pending calibration._
