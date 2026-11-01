"""Google Cloud implementations: KMS, Cloud Storage, Cloud Run jobs and Firestore.

Thin wrappers only; the logic they serve is tested with in-memory fakes. Credentials come
from the service account each service or job runs as.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from google.cloud import firestore, kms, run_v2, storage  # type: ignore[attr-defined]
from google.cloud.firestore_v1.base_query import FieldFilter
from memjudge.result import StopCause
from pydantic import BaseModel

from memjudge_runner.config import QUOTA_DAYS, RECORD_DAYS, Limits
from memjudge_runner.logs import LogLine, log_filter, log_line
from memjudge_runner.runs import (
    ACTIVE_STATES,
    Ending,
    RejudgeState,
    RunRecord,
    State,
    ended,
    limit_error,
    lock_doc,
    locks_by_original,
    newest,
    now,
    rejudge_error,
    resume_error,
    resumed,
)

CONTENT_TYPE = "application/octet-stream"


class KmsCipher:
    def __init__(self) -> None:
        self._client = kms.KeyManagementServiceClient()

    def encrypt(self, key: str, plaintext: bytes, aad: bytes) -> bytes:
        resp = self._client.encrypt(
            request={"name": key, "plaintext": plaintext, "additional_authenticated_data": aad}
        )
        return bytes(resp.ciphertext)

    def decrypt(self, key: str, ciphertext: bytes, aad: bytes) -> bytes:
        resp = self._client.decrypt(
            request={"name": key, "ciphertext": ciphertext, "additional_authenticated_data": aad}
        )
        return bytes(resp.plaintext)


class GcsBlobs:
    def __init__(self) -> None:
        self._client = storage.Client()

    def _blob(self, bucket: str, name: str) -> storage.Blob:
        return self._client.bucket(bucket).blob(name)

    def put(self, bucket: str, name: str, data: bytes) -> None:
        self._blob(bucket, name).upload_from_string(data, content_type=CONTENT_TYPE)

    def get(self, bucket: str, name: str) -> bytes:
        data: bytes = self._blob(bucket, name).download_as_bytes()
        return data

    def delete(self, bucket: str, name: str) -> None:
        self._blob(bucket, name).delete()

    def download(self, bucket: str, name: str, dest: Path) -> None:
        tmp = dest.with_suffix(".part")
        self._blob(bucket, name).download_to_filename(str(tmp))
        tmp.replace(dest)

    def exists(self, bucket: str, name: str) -> bool:
        return bool(self._blob(bucket, name).exists())

    def signed_put_url(self, bucket: str, name: str, minutes: int) -> str:
        return self._signed_url(bucket, name, minutes, "PUT")

    def signed_get_url(self, bucket: str, name: str, minutes: int) -> str:
        return self._signed_url(bucket, name, minutes, "GET")

    def _signed_url(self, bucket: str, name: str, minutes: int, method: str) -> str:
        # On Cloud Run there is no key file: sign through IAM with the service's own token.
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())  # type: ignore[no-untyped-call]
        url: str = self._blob(bucket, name).generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=minutes),
            method=method,
            content_type=CONTENT_TYPE if method == "PUT" else None,
            service_account_email=creds.service_account_email,  # type: ignore[attr-defined]
            access_token=creds.token,
        )
        return url


class CloudRunJobs:
    def __init__(self, project: str, region: str) -> None:
        self._parent = f"projects/{project}/locations/{region}"
        self._jobs = run_v2.JobsClient()
        self._executions = run_v2.ExecutionsClient()

    def start(self, job: str, args: list[str]) -> str:
        overrides = run_v2.RunJobRequest.Overrides(
            container_overrides=[run_v2.RunJobRequest.Overrides.ContainerOverride(args=args)]
        )
        operation = self._jobs.run_job(
            request=run_v2.RunJobRequest(name=f"{self._parent}/jobs/{job}", overrides=overrides)
        )
        name: str = operation.metadata.name
        return name

    def finished(self, execution: str) -> bool:
        ex = self._executions.get_execution(name=execution)
        return ex.completion_time is not None and ex.completion_time.timestamp() > 0

    def cancel(self, execution: str) -> None:
        self._executions.cancel_execution(name=execution)


class CloudLogs:
    """Job log lines from Cloud Logging's REST API (``entries.list``)."""

    URL = "https://logging.googleapis.com/v2/entries:list"

    def __init__(self, project: str) -> None:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/logging.read"])
        self._session = AuthorizedSession(creds)  # type: ignore[no-untyped-call]
        self._project = project

    def read(self, executions: list[str], since: datetime, limit: int) -> list[LogLine]:
        resp = self._session.post(
            self.URL,
            json={
                "resourceNames": [f"projects/{self._project}"],
                "filter": log_filter(executions, since),
                "orderBy": "timestamp desc",
                "pageSize": limit,
            },
            timeout=30,
        )
        resp.raise_for_status()
        entries = resp.json().get("entries", [])
        return [line for e in reversed(entries) if (line := log_line(e))]


def _plain(fields: dict[str, Any]) -> dict[str, Any]:
    """Fields as the record stores them: datetimes as ISO text, models (``history``) as JSON."""

    def plain(v: Any) -> Any:
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, BaseModel):
            return v.model_dump(mode="json")
        if isinstance(v, list):
            return [plain(x) for x in v]
        return v

    return {k: plain(v) for k, v in fields.items()}


class FirestoreRuns:
    """RunStore on Firestore: ``runs/{run_id}``, ``quota/{user}_{day}`` and the re-judge
    locks ``rejudges/{published run}_{season}`` (kept: no TTL, so a run is re-judged once
    per season)."""

    def __init__(self, client: firestore.Client) -> None:
        self._db = client
        self._runs = client.collection("runs")
        self._quota = client.collection("quota")
        self._rejudges = client.collection("rejudges")

    def _active_query(self) -> Any:
        return self._runs.where(filter=FieldFilter("state", "in", sorted(ACTIVE_STATES)))

    def create(self, record: RunRecord, limits: Limits) -> str | None:
        @firestore.transactional
        def txn(tx: firestore.Transaction) -> str | None:
            active = [
                RunRecord.model_validate(d.to_dict())
                for d in self._active_query().stream(transaction=tx)
            ]
            quota = self._quota.document(f"{record.user}_{record.day}")
            snap = quota.get(transaction=tx)
            today = int(snap.get("count")) if snap.exists else 0
            lock = self._rejudges.document(record.rejudge_lock) if record.rejudge_lock else None
            held = lock.get(transaction=tx) if lock else None
            error = rejudge_error(held.to_dict() if held and held.exists else None) or limit_error(
                record.user, active, today, limits
            )
            if error:
                return error
            if lock:
                tx.set(lock, {**lock_doc(record, "active"), "updated_at": now()})
            # expire_at is a native timestamp, the field Firestore's TTL policies delete by
            day = datetime.strptime(record.day, "%Y%m%d").replace(tzinfo=UTC)
            tx.set(quota, {"count": today + 1, "expire_at": day + timedelta(days=QUOTA_DAYS)})
            expire = record.created_at + timedelta(days=RECORD_DAYS)
            tx.set(
                self._runs.document(record.run_id),
                {**record.model_dump(mode="json"), "expire_at": expire},
            )
            return None

        result: str | None = txn(self._db.transaction())
        return result

    def get(self, run_id: str) -> RunRecord | None:
        snap = self._runs.document(run_id).get()
        return RunRecord.model_validate(snap.to_dict()) if snap.exists else None

    def update(self, run_id: str, **fields: Any) -> None:
        self._runs.document(run_id).update(_plain({**fields, "updated_at": now()}))

    def transition(self, run_id: str, expect: State, **fields: Any) -> bool:
        @firestore.transactional
        def txn(tx: firestore.Transaction) -> bool:
            ref = self._runs.document(run_id)
            snap = ref.get(transaction=tx)
            if not snap.exists or snap.get("state") != expect:
                return False
            tx.update(ref, _plain({**fields, "updated_at": now()}))
            return True

        changed: bool = txn(self._db.transaction())
        return changed

    def active(self) -> list[RunRecord]:
        return [RunRecord.model_validate(d.to_dict()) for d in self._active_query().stream()]

    def for_user(self, user: str) -> list[RunRecord]:
        query = self._runs.where(filter=FieldFilter("user", "==", user))
        return newest(RunRecord.model_validate(d.to_dict()) for d in query.stream())

    def fail(self, run_id: str, error: str, *, counts: bool, execution: str | None = None) -> bool:
        return self._end(run_id, "failed", error, counts, execution, None)

    def stop(
        self,
        run_id: str,
        error: str,
        *,
        counts: bool,
        cause: StopCause,
        execution: str | None = None,
    ) -> bool:
        return self._end(run_id, "stopped", error, counts, execution, cause)

    def _end(
        self,
        run_id: str,
        state: Ending,
        error: str,
        counts: bool,
        execution: str | None,
        cause: StopCause | None,
    ) -> bool:
        @firestore.transactional
        def txn(tx: firestore.Transaction) -> bool:
            ref = self._runs.document(run_id)
            snap = ref.get(transaction=tx)
            if not snap.exists:
                return False
            run = RunRecord.model_validate(snap.to_dict())
            if not run.active or (execution is not None and run.execution != execution):
                return False
            # Firestore wants every read of a transaction before its writes.
            lock = self._rejudges.document(run.rejudge_lock) if run.rejudge_lock else None
            held = lock.get(transaction=tx) if lock else None
            if not counts and run.counts:
                tx.update(
                    self._quota.document(f"{run.user}_{run.day}"),
                    {"count": firestore.Increment(-1)},
                )
            tx.update(ref, _plain(ended(run, state, error, counts, cause)))
            if lock and held and held.exists and held.get("run_id") == run.run_id:
                tx.set(lock, {**lock_doc(run, "failed"), "updated_at": now()})
            return True

        changed: bool = txn(self._db.transaction())
        return changed

    def set_rejudge(self, run: RunRecord, state: RejudgeState) -> None:
        if not run.rejudge_lock:
            return
        lock = self._rejudges.document(run.rejudge_lock)

        @firestore.transactional
        def txn(tx: firestore.Transaction) -> None:
            held = lock.get(transaction=tx)
            if held.exists and held.get("run_id") == run.run_id:
                tx.set(lock, {**lock_doc(run, state), "updated_at": now()})

        txn(self._db.transaction())

    def rejudges(self, season: str) -> dict[str, dict[str, str]]:
        query = self._rejudges.where(filter=FieldFilter("season", "==", season))
        return locks_by_original((d.to_dict() or {} for d in query.stream()), season)

    def resume(self, run_id: str, limits: Limits, **fields: Any) -> str | None:
        @firestore.transactional
        def txn(tx: firestore.Transaction) -> str | None:
            active = [
                RunRecord.model_validate(d.to_dict())
                for d in self._active_query().stream(transaction=tx)
            ]
            ref = self._runs.document(run_id)
            snap = ref.get(transaction=tx)
            if not snap.exists:
                return "No such run."
            run = RunRecord.model_validate(snap.to_dict())
            error = resume_error(run) or limit_error(run.user, active, 0, limits)
            if error:
                return error
            tx.update(ref, _plain(resumed(run, fields)))
            return None

        result: str | None = txn(self._db.transaction())
        return result
