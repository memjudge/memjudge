# 0016 — The run's log and history

Status: accepted (2026-10-28)

**Context.** The /run page showed a hosted run's steps, a progress bar and one error line.
When a run was slow, retrying or stuck, its owner could not see what it was doing, while
the maintainer could read the jobs' logs in Cloud Logging. Readers of a published result
saw "Resumed N times" but not when the run stopped or why.

**Decision.**

- **Owner: the job log.** `GET /runs/{run_id}/log` returns the newest 200 lines that the
  run's job executions wrote, oldest first, to the run's owner only (the same check as the
  run's status). The /run page shows them in a collapsed "Log" section that loads when
  opened and refreshes while the run is active.
  - The run record lists every execution that worked on the run (`executions`), so the
    log covers every attempt. The API reads Cloud Logging by execution name, which needs
    `roles/logging.viewer` on its service account.
  - Reads are cached 15 seconds per run: Cloud Logging allows 60 reads a minute per
    project. When a read fails, the last lines read are shown with a note.
  - The jobs remove the users' keys from everything they write (`redact.py`), so the lines
    hold none. They log each start-up step (job started, loading the dataset, waiting for
    the network and how long it took, starting the connector), each conversation's
    ingestion and completion, and when judging and publishing start.
  - The log shows MemJudge's own lines, SDK retries (the `<sdk>._base_client` loggers) and
    every library's warnings and errors. Libraries' lines per HTTP request or per stored
    item are left out, as they would bury the rest.
  - Until the current job reports in (`job_started`: its first progress report, or
    finish's first update), the form shows "Starting the job" rather than "Storing the
    conversations": a job's container start and set-up take one to two minutes, and
    nothing is logged while the container starts.
  - Cloud Logging keeps the lines 30 days; the run record stays 90.
- **Public: the run history.** The runner records each stop (with its cause) and each
  resume in the run record (`history`), and the published result carries it (`history`;
  the schema stays version 1, as nothing was published before). The cause is one the
  runner sets at each stop: `submitter` (Stop pressed), `retrieval` (the retrieval job
  reported a failure: the memory system or the job), `judge`, `crash` (the job ended
  without reporting, including the time limit) or `runner` (an internal error). A
  re-judge carries the original run's history, as it does `resumes`. The result page
  shows it under the run's tags.

**Not published: the raw log, and the stop messages.** The job log stays with the owner:

- `redact.py` removes exact copies of a key. A key that a vendor's message cut or encoded
  would get through, and a published result is permanent.
- The retrieval job runs the connector and is untrusted (ADR 0009). Its log is its own
  account of what happened and proves nothing. What a reader can check is already
  published per question: the context, the verdict, timings and errors.
- Stop messages can come from that job and carry vendor text, so the history publishes
  only the cause the runner set.

**Consequences.**

- An owner can follow a run's ingestion conversation by conversation and see vendor
  retries and failures without asking the maintainer.
- The history counts stops and resumes; `resumes` stays as before. A resume that could not
  start leaves the history as it was, as it does `resumes`.
- How many conversations a resume redid is still not recorded (ADR 0011).
