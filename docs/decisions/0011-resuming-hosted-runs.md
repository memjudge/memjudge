# 0011 — Stopping and resuming runs

Status: accepted (2026-10-13)

**Context.** A full LongMemEval-S or LoCoMo run takes hours, and the submitter pays for
every memory-system and judge call. It can fail part-way: the memory system goes down or
runs out of credit, the judge key runs out, or the job crashes or reaches the 24 h task
limit. Without resume, everything spent so far is lost. Worse, a memory-system outage
used to be recorded as errors for the rest of the dataset and published as a low score.

**Decision.**

- A run stops instead of recording an outage: any ingestion failure, or 5 retrieval
  failures in a row (`MAX_FAILURES_IN_ROW`), raises `RunStoppedError`. A single failed
  retrieval is still recorded as an error. Local runs behave the same.
- The conversation is the unit of progress. A conversation is complete only when all its
  questions are recorded and the run was not stopped meanwhile. A resume wipes (`reset`)
  and redoes the others. It never continues inside a conversation: the half-stored data
  lives in the vendor's system, where MemJudge cannot check it.
- The retrieve job uploads its records every 5 minutes and when it stops, and a resumed
  retrieve job starts from them. Finish saves its verdicts every minute while judging and
  when it stops, so a resumed finish judges only the questions not yet judged (for
  unchanged contexts). The judge spend of failed attempts is carried into the retried
  verdict, so the published judge cost counts every billed call.
- A run stops (resumable) on a memory-system or judge failure, a crash, the time limit,
  an internal error, or the owner's Stop. Only invalid retrieval output or an undeclared
  endpoint fails it for good.
- The owner can resume a stopped run within 48 hours, at most 3 times, by entering every
  key again: keys are never kept between attempts, and finish needs the connector keys
  to redact them. Each attempt gets a new run token. A resume needs a free active-run
  slot but no daily run.
- Same version: each release deploys its own jobs (`retrieve-<connector>-<version>`,
  `finish-<version>`), and every attempt of a run starts the jobs of the release it
  started on. Cloud Run cannot change a job's image for one execution. A release's jobs
  are deleted 14 days after it is replaced: idle jobs cost nothing, and 14 days covers a
  run with 3 resumes. The API always runs the latest release, so its internal endpoints
  stay compatible with the jobs of earlier releases. Redeploying the same release (before
  launch, every deploy of 0.1.0 from `main`) replaces its jobs in place.
- The result records `resumes`, and the site labels the run "Resumed N times". The count
  comes from the runner's own records. How many conversations were redone is not
  recorded, because only the untrusted retrieve job could report it. Since ADR 0016 the
  result also records when each stop and resume happened and why the run stopped
  (`history`).

**Consequences.**

- A vendor could force a stop to redo a conversation that went badly, since its servers
  see their own answers. The label, the cap of 3 and the 48-hour window limit this; whole
  re-runs were always possible, and every one is listed.
- A hosted memory system's own version can change between attempts. It can also change
  during one long run; v1 does not track it.
- Infrastructure: the staging bucket keeps objects 14 days; finish can write to staging;
  the API can sign GET URLs on staging, delete objects in the secrets bucket and cancel
  executions.
- Recalld cannot find an agent left by an interrupted attempt (already true for local
  resume); it stays in the user's account.
