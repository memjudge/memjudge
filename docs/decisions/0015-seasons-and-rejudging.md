# 0015 — Seasons and re-judging

Status: accepted (2026-10-28)

**Context.** Scores depend on how answers are judged: the judge model and its settings, the
prompt's rules, how replies are read, what the judge is shown and how metrics are counted.
These will change after launch, through fixes and contributions. Results judged under
different rules are not comparable: the clearer trick-question rule of prompt v3, for
example, moves accuracy by roughly half a point to a point. A season label kept apart from
the judging rules would not stop results judged under different rules from sharing a table.

**Decision.**

- A **season** is one fixed set of judging rules. `src/memjudge/judge/seasons.json` lists
  every season, oldest first, and the last entry is the current one. Each entry has an id, a
  name, its start date, a one-line summary, a details link (the pull request that started
  it), the judge (routes, effort, output limit, prompt version and prompt hash) and an
  `upgrade` flag.
- Any change to how answers are judged starts a new season. That means the judge, its
  settings, the prompt, how replies are read or retried, what the judge sees (budget,
  tokenizer, truncation) or how metrics are computed. Infrastructure, speed, connectors and
  the site do not. Judging changes are batched into planned seasons, except fixes for wrong
  verdicts. The checklist is in [docs/seasons.md](../seasons.md).
- Past seasons are fixed:
  - Tests check that every season's prompt hash matches its prompt file, so a used prompt
    file can never change.
  - Tests check that the current season uses the current prompt.
  - A CI step rejects any change to a past entry except its summary and details wording.
  - A judge counts as official only with the season's prompt.
- `upgrade` says how runs from earlier seasons get into a season:
  - `re-judge`: only the judge changed, so their stored contexts can be judged again.
  - `re-run`: what the judge sees changed, so they need a new run. Stored contexts are
    already cut to the old budget.
  - A run can move across several seasons only if every one of them is `re-judge`.
- The leaderboard ranks current-season runs. Older runs stay listed below a line with their
  season's badge, until they are re-judged or, when they can't be, until their system has
  a current-season run. Each past season keeps its own table.
- **Hosted re-judge**:
  - Anyone signed in can move an eligible published run into the current season, paying
    with their own judge key.
  - The finish job judges the stored contexts. It takes them from the published artifacts
    and checks them against the signed hashes.
  - The result is a new signed result with `rejudged_from` (run and season). It keeps the
    original's connector record, dataset, affiliation and resumes, and takes this release's
    tool version.
  - One re-judge per run per season, with a lock while it runs. A failed or stopped attempt
    frees it. The normal run limits apply.
  - The person who clicks only pays: the judging and the signature are MemJudge's.

**Consequences.**

- A new season starts with every earlier run below the line. Re-judging refills it at the
  judge's cost only (about $15 for a full LoCoMo run), with no new memory-system calls.
- "Once per run per season" stops re-judging until the judge's small randomness gives a
  better score. Repeat runs stay the way to measure variance.
- The site keeps a copy of `seasons.json`, as it does for the result schema. Its build fails
  when a result names a season it doesn't know, and skips a result whose judge settings
  differ from its season's.
- Season 1 is the first season. Prompts v1 and v2 were development versions and belong to
  no season.
- Season 1 is still provisional: the calibration in ADR 0008 has not run. If it picks a
  different judge, that is Season 2.
