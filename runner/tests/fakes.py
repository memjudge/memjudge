"""In-memory stand-ins for KMS, Cloud Storage, Cloud Run jobs and their logs, publishing
and the judge."""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from pathlib import Path

from memjudge.harness import JudgeSetup
from memjudge.judge.client import Completion
from memjudge.judge.judge import load_prompt
from memjudge.result import JudgeInfo

from memjudge_runner.logs import LogLine

TINY = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny.json"


class FakeCipher:
    def encrypt(self, key: str, plaintext: bytes, aad: bytes) -> bytes:
        return json.dumps(
            {"key": key, "aad": aad.decode(), "data": base64.b64encode(plaintext).decode()}
        ).encode()

    def decrypt(self, key: str, ciphertext: bytes, aad: bytes) -> bytes:
        box = json.loads(ciphertext)
        if box["key"] != key or box["aad"] != aad.decode():
            raise PermissionError("wrong key or additional data")
        return base64.b64decode(box["data"])


class FakeBlobs:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put(self, bucket: str, name: str, data: bytes) -> None:
        self.objects[bucket, name] = data

    def get(self, bucket: str, name: str) -> bytes:
        return self.objects[bucket, name]

    def delete(self, bucket: str, name: str) -> None:
        del self.objects[bucket, name]

    def download(self, bucket: str, name: str, dest: Path) -> None:
        dest.write_bytes(self.objects[bucket, name])

    def exists(self, bucket: str, name: str) -> bool:
        return (bucket, name) in self.objects

    def signed_put_url(self, bucket: str, name: str, minutes: int) -> str:
        return f"http://testserver/__fake_gcs/{bucket}/{name}"

    def signed_get_url(self, bucket: str, name: str, minutes: int) -> str:
        return f"http://testserver/__fake_gcs/{bucket}/{name}"


class FakeJobs:
    def __init__(self) -> None:
        self.started: list[tuple[str, list[str]]] = []
        self.ended: set[str] = set()

    def start(self, job: str, args: list[str]) -> str:
        self.started.append((job, args))
        return f"executions/{job}-{len(self.started)}"

    def finished(self, execution: str) -> bool:
        return execution in self.ended

    def cancel(self, execution: str) -> None:
        self.ended.add(execution)


class FakeLogs:
    """Lines per execution name; ``fail`` makes reads raise."""

    def __init__(self) -> None:
        self.lines: dict[str, list[LogLine]] = {}
        self.reads = 0
        self.fail = False

    def read(self, executions: list[str], since: datetime, limit: int) -> list[LogLine]:
        self.reads += 1
        if self.fail:
            raise RuntimeError("quota exceeded")
        found = [line for e in executions for line in self.lines.get(e, [])]
        return sorted(found, key=lambda line: line.at)[-limit:]


class FakePublisher:
    def __init__(self) -> None:
        self.artifacts: dict[str, bytes] = {}
        self.commits: dict[str, str] = {}

    def upload_artifact(self, key: str, data: bytes) -> None:
        self.artifacts[key] = data

    def commit_result(self, path: str, text: str, message: str) -> None:
        self.commits[path] = text

    def download_artifact(self, key: str) -> bytes:
        return self.artifacts[key]

    def read_result(self, path: str) -> str:
        return self.commits[path]


class GoldCompleter:
    """Correct when the gold answer appears in the context."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def complete(self, system: str, user: str) -> Completion:
        if self.fail:
            from memjudge.judge.client import JudgeError

            raise JudgeError("credit balance too low")
        gold = re.search(r"Gold answer: (.*)", user).group(1)  # type: ignore[union-attr]
        context = user.split("<<<CONTEXT\n", 1)[1]
        verdict = "correct" if gold.lower() in context.lower() else "incorrect"
        return Completion(
            text=json.dumps({"verdict": verdict, "reason": "fake", "stale": None}),
            input_tokens=100,
            output_tokens=10,
            served_by="fake-model",
        )


def official_judge(completer: GoldCompleter | None = None) -> JudgeSetup:
    prompt = load_prompt()

    def info(served_by: list[str]) -> JudgeInfo:
        return JudgeInfo(
            route="anthropic",
            model="fake",
            upstream_provider=None,
            effort="medium",
            max_output_tokens=4000,
            prompt_version=prompt.version,
            prompt_hash=prompt.hash,
            served_by=served_by,
            official=True,
            season="test",
        )

    return JudgeSetup(
        client=completer or GoldCompleter(), prompt=prompt, info=info, price=lambda i, o: 0.0
    )
