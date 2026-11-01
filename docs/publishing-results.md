# Publishing results

How MemJudge runs and publishes results for systems whose makers are not involved, so the
results are fair and anyone can check them.

## Fairness rules

- **Pinned versions.** Each connector pins its SDK exactly; the result records the connector's
  code hash, the MemJudge version and, for hosted runs, the image digest.
- **Open-source versions where terms forbid benchmarking.** A hosted service whose terms forbid
  publishing results is run locally only; its open-source version can be published.
- **Same LLM.** Every system that lets you choose its LLM runs GPT-6 Luna, so the memory design
  is compared, not the model. Embedders are the system's own choice and are shown with each
  result (ADR 0014).
- **Conversation dates are passed.** Every connector gives the system the date of each message,
  through the system's own API. Where the API has no date field, the connector says how it
  passes the date instead.
- **Same retrieval sizes.** Every run scores Top 10 and Top 30: each connector is asked for up
  to 10 and up to 30 items per question (ADR 0018), and the harness cuts every context to
  the same token budget.
- **No per-dataset tuning.** Connectors use the vendor's documented defaults or one fixed
  setting for all datasets.
- **Errors count.** A question with a failed call counts as incorrect; nothing is dropped.
- **Repeat runs.** Run a system 3 times before drawing conclusions; every run stays listed.

## After publishing

Publish first; do not wait for the vendor's permission. The same day, open an issue on the
project's GitHub repository:

> **MemJudge result for <system>**
>
> We published an independent benchmark result for <system> (<version>):
> <run page link>
>
> The connector code is open: <connector folder link>. It records every setting used, and the
> per-question retrieved context is published with the result.
>
> If the connector uses <system> in a way you would not recommend, please tell us in this issue
> or at https://github.com/memjudge/memjudge/issues. We will fix it, re-run, and keep both
> results listed.

## When a problem is reported

1. Check the report against the vendor's documentation.
2. If it is valid, fix the connector and re-run. The new result has a new code hash; the old
   one stays listed on the run page under "Other runs of this connector".
3. Record the change in the [corrections log](https://github.com/memjudge/results/blob/main/CORRECTIONS.md).

See also the [corrections policy](https://memjudge.org/corrections/).
