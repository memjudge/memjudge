# Seasons

A season is one fixed set of judging rules. Every result records the season it was judged
in, and the leaderboard only ranks results from the same season together. Seasons are listed
in [`src/memjudge/judge/seasons.json`](../src/memjudge/judge/seasons.json); the last entry is
the current season. Background: [ADR 0015](decisions/0015-seasons-and-rejudging.md).

## What starts a new season

| Change | New season? |
| --- | --- |
| Judge model, route, effort or output limit | Yes |
| Judge prompt (its rules or wording) | Yes |
| How a judge reply is read or retried, or how a failed verdict counts | Yes |
| What the judge sees: token budget, tokenizer, truncation | Yes (`upgrade: re-run`) |
| How accuracy or another metric is computed | Yes |
| Runner infrastructure, speed, logging, the site | No |
| A connector fix | No: it is recorded in the result, and that system is run again |
| A new dataset version | No: datasets are versioned and get their own table |

Batch judging changes into a planned season, every month or two. A fix for verdicts that
were wrong is the exception and gets a new season straight away.

## The `upgrade` flag

Each season says how runs from earlier seasons get into it:

- `re-judge`: only the judge changed. A run's stored contexts can be judged again into this
  season, with no new memory-system calls. On the site this is the **Re-judge** button.
- `re-run`: what the judge sees changed (budget, tokenizer, truncation). Stored contexts are
  already cut to the old budget, so the run needs a new run. On the site this shows as
  **Needs a new run**.

A run can be re-judged into the current season only if **every** season after its own is
`re-judge`. When unsure, choose `re-run`: it costs more but cannot give wrong results.

## Checklist for a new season

1. Make the judging change. A prompt change is a new file (`prompts/v4.md`) and a new
   `PROMPT_VERSION` in `judge/judge.py`. A used prompt file is never edited.
2. Add an entry at the end of `seasons.json`:
   - `id`: the next number, as a string.
   - `name`: "Season N".
   - `started`: the release date.
   - `provisional`: true until a calibration (ADR 0008) backs the judge.
   - `summary`: one plain line on what changed.
   - `details`: the pull request's URL.
   - `upgrade`: as above.
   - `judge`: the full judge settings, with `prompt_hash` from
     `python -c "from memjudge.judge.judge import load_prompt; print(load_prompt().hash)"`.
3. Open the pull request. `pytest` checks that the new entry matches the prompt and that
   every season's prompt file is unchanged. CI checks that no past season changed.
4. After merging, deploy the runner. New runs and re-judges are judged in the new season.
5. Copy `seasons.json` into the leaderboard repo (`leaderboard/seasons.json`). The site's build
   fails on a result from a season it doesn't know, so a missed copy is caught.
6. Note the season in `CHANGELOG.md`, marked **[scoring]**.

## Re-judging

Anyone signed in to the site can re-judge a published run whose season is older than the
current one, when the `upgrade` flags allow it. They enter their own judge key and see the
estimated cost first. The hosted finish job judges the stored contexts, after checking them
against the signed hashes. It publishes a new signed result with `rejudged_from` set; the
original stays in its season. Each run can be re-judged once per season, and only one
re-judge of a run can run at a time.
