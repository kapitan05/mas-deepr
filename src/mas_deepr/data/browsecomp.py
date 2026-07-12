"""BrowseComp loader (eval-only). Rows are XOR-encrypted per-row to keep the
answer set out of web-crawled training corpora; decrypt scheme and CSV URL
match OpenAI's published eval (github.com/openai/simple-evals/browsecomp_eval.py).
"""

import base64
import hashlib

import httpx
import polars as pl

from mas_deepr.config import Settings
from mas_deepr.data.schema import Question

_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)


def _derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key, strict=True)).decode()


def _cached_csv_path(settings: Settings) -> str:
    dest = settings.data_dir / "browsecomp" / "browse_comp_test_set.csv"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(_CSV_URL, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return str(dest)


def load_browsecomp(settings: Settings, *, limit: int | None = None) -> list[Question]:
    path = _cached_csv_path(settings)
    df = pl.read_csv(path)
    if limit is not None:
        df = df.head(limit)

    questions = []
    for i, row in enumerate(df.iter_rows(named=True)):
        canary = row["canary"]
        problem = _decrypt(row["problem"], canary)
        answer = _decrypt(row["answer"], canary)
        stable_suffix = hashlib.sha256(problem.encode()).hexdigest()[:10]
        questions.append(
            Question(
                question_id=f"browsecomp-{i}-{stable_suffix}",
                source="browsecomp",
                split="test",
                prompt=problem,
                answer=answer,
                metadata={"topic": row.get("problem_topic", "")},
            )
        )
    return questions
