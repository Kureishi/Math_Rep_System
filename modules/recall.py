"""
"Have I solved this before?" -- search the history of solved problems by meaning, wording and equation shape. Streamlit-free.

Three signals, each optional, each shown in the explanation so a match can be judged rather than trusted:
  * MEANING   embeddings from LM Studio (when an embedding model is loaded): "a car speeds up" finds "an automobile
              accelerates". Cached in data/embeddings.db by (model, hash of the text), so each past problem is embedded once.
  * WORDING   TF-IDF over the problem text with every number replaced by a placeholder, so the same problem with different
              numbers ("8 m/s ... 6 s" against "12 m/s ... 9 s") matches on its wording, not on its data.
  * SHAPE     the equation-shape fingerprint already used by "Find similar past problems" (modules/similarity.py), when the
              question comes with a model (the problem currently open).
The score is the weighted mean of the signals that are AVAILABLE (weights renormalised), so a missing embedding model
or a text-only question degrades the search rather than breaking it. If the embedding call fails mid-search the results
come back without it and say so.
"""
import hashlib
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from modules.app_logging import logger
from modules.db_migrations import apply_migrations
from modules.db_util import ClosingConnection
from modules.similarity import jaccard_similarity

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings.db"
WEIGHTS = {"meaning": 0.5, "wording": 0.3, "shape": 0.2}
SAME, VERY_SIMILAR, RELATED = 0.85, 0.65, 0.40
EMBED_BATCH = 32

_STOP = frozenset("a an the of to in on at is are was were be been it its and or but if then than that this these those with for from "
                  "by as so do does did how what which who when where why can could would should will shall has have had i we you "
                  "my our your their them his her him she he they there here also not no yes".split())
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?", re.I)
_WORD = re.compile(r"[a-z][a-z0-9_]*|NUM")


# ------------------------------------------------------------------------------------------ wording

def tokens(text: str) -> list[str]:
    """Lower-cased content words; every number becomes the single token NUM; plurals are folded; stop words are dropped."""
    out = []
    for w in _WORD.findall(_NUM.sub(" NUM ", text.lower())):
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def normalised(text: str) -> str:
    return " ".join(tokens(text))


def tfidf_cosine(query: str, docs: list[str]) -> list[float]:
    """Cosine similarity of `query` to each doc under TF-IDF weights (sublinear tf, smoothed idf from `docs` plus the query)."""
    bags = [Counter(tokens(d)) for d in docs]
    qbag = Counter(tokens(query))
    n = len(docs) + 1
    df: Counter = Counter()
    for bag in bags + [qbag]:
        df.update(bag.keys())

    def vec(bag: Counter) -> dict[str, float]:
        return {t: (1 + math.log(c)) * (math.log((1 + n) / (1 + df[t])) + 1.0) for t, c in bag.items()}

    q = vec(qbag)
    qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
    out = []
    for bag in bags:
        d = vec(bag)
        dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
        out.append(sum(v * d.get(t, 0.0) for t, v in q.items()) / (qn * dn))
    return out


# ------------------------------------------------------------------------------------------ the embedding cache

def _m001(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (model TEXT NOT NULL, text_hash TEXT NOT NULL, dim INTEGER NOT NULL, "
                 "vec BLOB NOT NULL, PRIMARY KEY (model, text_hash))")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, factory=ClosingConnection)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    apply_migrations(conn, [_m001])
    conn.commit()
    return conn


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def cached_vectors(model: str, texts: list[str]) -> dict[str, np.ndarray]:
    """{text_hash: vector} for the texts already embedded under `model`."""
    hashes = [text_hash(t) for t in texts]
    found: dict[str, np.ndarray] = {}
    with _connect() as conn:
        for i in range(0, len(hashes), 500):
            chunk = hashes[i:i + 500]
            rows = conn.execute(f"SELECT text_hash, vec FROM embeddings WHERE model = ? AND text_hash IN ({','.join('?' * len(chunk))})",
                                (model, *chunk)).fetchall()
            found.update({h: np.frombuffer(v, dtype=np.float32) for h, v in rows})
    return found


def store_vectors(model: str, items: dict[str, np.ndarray]) -> None:
    with _connect() as conn:
        conn.executemany("INSERT OR REPLACE INTO embeddings (model, text_hash, dim, vec) VALUES (?, ?, ?, ?)",
                         [(model, h, len(v), np.asarray(v, dtype=np.float32).tobytes()) for h, v in items.items()])


def vectors_for(texts: list[str], model: str, embed: Callable[[list[str]], list[list[float]]],
                on_progress: Callable[[int, int], None] | None = None) -> list[np.ndarray]:
    """Vectors for `texts`, embedding only those not cached (in batches) and caching what it computes."""
    have = cached_vectors(model, texts)
    missing = list(dict.fromkeys(t for t in texts if text_hash(t) not in have))
    for i in range(0, len(missing), EMBED_BATCH):
        batch = missing[i:i + EMBED_BATCH]
        vecs = embed(batch)
        if len(vecs) != len(batch):
            raise ValueError(f"the embedding model returned {len(vecs)} vectors for {len(batch)} texts")
        new = {text_hash(t): np.asarray(v, dtype=np.float32) for t, v in zip(batch, vecs)}
        store_vectors(model, new)
        have.update(new)
        if on_progress:
            on_progress(min(i + EMBED_BATCH, len(missing)), len(missing))
    return [have[text_hash(t)] for t in texts]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ------------------------------------------------------------------------------------------ the search

@dataclass
class Match:
    id: int
    score: float
    parts: dict[str, float]                  # the signals that were available: meaning / wording / shape
    verdict: str                             # same | very similar | related
    why: str
    problem_text: str
    domain: str | None
    timestamp: str
    same_problem_new_numbers: bool = False


@dataclass
class SearchReport:
    matches: list[Match] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)       # which of meaning / wording / shape were used
    notes: list[str] = field(default_factory=list)
    searched: int = 0

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.matches else None


def verdict_for(score: float) -> str | None:
    return "same" if score >= SAME else "very similar" if score >= VERY_SIMILAR else "related" if score >= RELATED else None


def _why(parts: dict[str, float], renumbered: bool) -> str:
    bits = []
    if renumbered:
        bits.append("the same wording with different numbers")
    elif "wording" in parts:
        bits.append(f"{parts['wording']:.0%} of the wording")
    if "meaning" in parts:
        bits.append(f"{max(parts['meaning'], 0):.0%} similar in meaning")
    if "shape" in parts:
        bits.append("the same equation shape" if parts["shape"] >= 0.999 else f"{parts['shape']:.0%} of the equation shape")
    return "; ".join(bits)


def search(query: str, corpus: list[dict], *, embed: Callable[[list[str]], list[list[float]]] | None = None,
           embedding_model: str | None = None, query_shape: frozenset[str] | None = None, limit: int = 5,
           min_score: float = RELATED, exclude_id: int | None = None,
           on_progress: Callable[[int, int], None] | None = None) -> SearchReport:
    """Rank `corpus` (dicts with id, problem_text, domain, timestamp, shape) against `query` by the available signals."""
    report = SearchReport()
    items = [c for c in corpus if c["id"] != exclude_id and c["problem_text"].strip()]
    report.searched = len(items)
    if not query.strip() or not items:
        return report
    docs = [c["problem_text"] for c in items]
    parts: list[dict[str, float]] = [{} for _ in items]

    for p, s in zip(parts, tfidf_cosine(query, docs)):
        p["wording"] = s
    report.signals.append("wording")

    if embed is not None and embedding_model:
        try:
            vecs = vectors_for([query] + docs, embedding_model, embed, on_progress)
            for p, v in zip(parts, vecs[1:]):
                p["meaning"] = max(0.0, cosine(vecs[0], v))
            report.signals.insert(0, "meaning")
        except Exception as exc:  # noqa: BLE001 -- embeddings are an extra; the other signals still answer
            logger.warning("semantic search unavailable: %s: %s", type(exc).__name__, exc)
            report.notes.append(f"Meaning-based matching was skipped ({type(exc).__name__}); results use wording and equation shape.")
    elif embed is None:
        report.notes.append("No embedding model is loaded, so matching uses wording and equation shape (load one such as "
                            "nomic-embed-text in LM Studio for meaning-based matching).")

    if query_shape:
        for p, c in zip(parts, items):
            if c.get("shape"):
                p["shape"] = jaccard_similarity(query_shape, c["shape"])
        if any("shape" in p for p in parts):
            report.signals.append("shape")

    qnorm = normalised(query)
    ranked = []
    for c, p in zip(items, parts):
        used = {k: w for k, w in WEIGHTS.items() if k in p}
        score = sum(p[k] * w for k, w in used.items()) / sum(used.values())
        renumbered = bool(qnorm) and qnorm == normalised(c["problem_text"]) and query.strip() != c["problem_text"].strip()
        if renumbered or qnorm == normalised(c["problem_text"]):
            score = max(score, SAME)                           # identical once the numbers are set aside
        verdict = verdict_for(score)
        if verdict is None or score < min_score:
            continue
        ranked.append(Match(c["id"], round(score, 4), {k: round(x, 4) for k, x in p.items()}, verdict, _why(p, renumbered),
                            c["problem_text"], c.get("domain"), c.get("timestamp", ""), renumbered))
    ranked.sort(key=lambda m: -m.score)
    report.matches = ranked[:limit]
    return report


def search_history(query: str, client=None, query_shape: frozenset[str] | None = None, limit: int = 5,
                   exclude_id: int | None = None, on_progress: Callable[[int, int], None] | None = None) -> SearchReport:
    """`search` over the saved history, using LM Studio's embedding model when one is loaded."""
    from modules import history
    from modules.llm_client import pick_embedding_model
    embed, model = None, None
    if client is not None:
        try:
            model = pick_embedding_model(client.list_models())
            embed = (lambda texts: client.embed(texts, model)) if model else None
        except Exception:  # noqa: BLE001 -- an unreachable server just means no embeddings
            embed, model = None, None
    return search(query, history.list_corpus(), embed=embed, embedding_model=model, query_shape=query_shape, limit=limit,
                  exclude_id=exclude_id, on_progress=on_progress)
