"""
Embedding + cache utilities for Item Master duplicate detection.
Kept separate to keep `Item_Master_Duplicate_Engine.py` small and focused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from Config import EMBED_APPROVAL_CACHE_FILE as _DEFAULT_APPROVAL_EMBED_CACHE_PATH
from Config import EMBED_CACHE_FILE as _DEFAULT_EMBED_CACHE_PATH

# ─────────────────────────────────────────────
#  EMBEDDINGS CONFIG — SentenceTransformers + FAISS
# ─────────────────────────────────────────────

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model id (downloads once)
EMBED_BATCH = 256  # encoding batch size (CPU)

# Default cache lives next to the deployed app (Config.py path).
EMBED_CACHE_FILE = str(_DEFAULT_EMBED_CACHE_PATH)
EMBED_APPROVAL_CACHE_FILE = str(_DEFAULT_APPROVAL_EMBED_CACHE_PATH)

# Global singleton embedder (loaded once per process)
_EMBEDDER: SentenceTransformer | None = None


def get_embedder(model_id: str = EMBED_MODEL) -> SentenceTransformer:
    """
    Loads the HF embedding model once and reuses it.
    On first run, this downloads the model to the HuggingFace cache on disk.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        # Tune CPU threading for typical on-prem servers (adjust if needed).
        try:
            import torch

            torch.set_num_threads(6)
            torch.set_num_interop_threads(2)
        except Exception:
            pass
        _EMBEDDER = SentenceTransformer(model_id, device="cpu")
    return _EMBEDDER


def build_embedding_text(row: dict[str, Any]) -> str:
    """
    Flatten one minimized JSON row into a single string for embedding.
    The input is expected to have: item_type, main_group, sub_group, text, numeric.
    """
    parts = [
        row.get("item_type") or "",
        row.get("main_group") or "",
        row.get("sub_group") or "",
        row.get("text") or "",
        row.get("numeric") or "",
    ]
    return " ".join(p.strip() for p in parts if p.strip())


def _texts_digest_records(records: list[dict[str, Any]]) -> str:
    """Stable hash of embedding input texts for cache validation (streaming, low-RAM)."""
    h = hashlib.sha256()
    for r in records:
        t = build_embedding_text(r)
        b = (t or "").encode("utf-8", errors="ignore")
        h.update(len(b).to_bytes(8, byteorder="little", signed=False))
        h.update(b)
    return h.hexdigest()


def embed_texts_local(
    texts: list[str],
    *,
    model_id: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH,
) -> np.ndarray:
    """
    Compute embeddings locally (CPU) using a HuggingFace SentenceTransformer model.

    - normalize_embeddings=True makes cosine similarity easy (dot product == cosine).
    - Returns float32 matrix of shape (N, D).
    """
    embedder = get_embedder(model_id)
    try:
        vecs = embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
    except Exception as e:
        # On constrained machines, a large batch may blow up memory.
        msg = str(e).lower()
        oomish = any(s in msg for s in ("out of memory", "cannot allocate memory", "alloc", "memoryerror"))
        if oomish and batch_size > 1 and len(texts) > 1:
            new_bs = max(1, batch_size // 2)
            mid = len(texts) // 2
            left = embed_texts_local(texts[:mid], model_id=model_id, batch_size=new_bs)
            right = embed_texts_local(texts[mid:], model_id=model_id, batch_size=new_bs)
            return np.vstack([left, right]).astype(np.float32, copy=False)
        raise


def load_embedding_cache(cache_path: str | Path = EMBED_CACHE_FILE) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load the persisted embedding matrix + metadata.

    The cache is expected to be produced by `build_faiss_index` (normalize_embeddings=True).
    """
    cache_npy = Path(cache_path)
    cache_meta = cache_npy.with_suffix(cache_npy.suffix + ".meta.json")
    if not cache_npy.exists() or not cache_meta.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_npy} (+ {cache_meta})")
    meta = json.loads(cache_meta.read_text(encoding="utf-8"))
    mat = np.load(cache_npy).astype(np.float32, copy=False)
    if mat.ndim != 2:
        raise ValueError("Embedding cache must be a 2D matrix")
    return mat, meta if isinstance(meta, dict) else {}


def build_faiss_index(
    records: list[dict[str, Any]],
    *,
    model: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH,
    cache_path: str | Path | None = EMBED_CACHE_FILE,
    reuse_only: bool = False,
    force_recompute: bool = False,
) -> tuple[Any, np.ndarray]:
    """
    Embed all records locally and build a FAISS flat inner-product index.

    If ``force_recompute`` is True, any existing cache is ignored and embeddings are recomputed.
    """
    try:
        import faiss  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu")

    if reuse_only and force_recompute:
        raise ValueError("reuse_only and force_recompute cannot both be true")

    total = len(records)
    digest = _texts_digest_records(records)

    cache_npy: Path | None = Path(cache_path) if cache_path else None
    cache_meta: Path | None = None
    if cache_npy is not None:
        cache_meta = cache_npy.with_suffix(cache_npy.suffix + ".meta.json")
        if not force_recompute:
            if cache_npy.exists() and cache_meta.exists():
                try:
                    meta = json.loads(cache_meta.read_text(encoding="utf-8"))
                    mat_cached = np.load(cache_npy)
                    ok = (
                        isinstance(meta, dict)
                        and meta.get("model") == model
                        and int(meta.get("rows", -1)) == total
                        and str(meta.get("text_digest", "")) == digest
                        and mat_cached.ndim == 2
                        and int(mat_cached.shape[0]) == total
                    )
                    if ok:
                        print(f"  Reusing cached embeddings: {cache_npy}")
                        mat_cached = mat_cached.astype(np.float32)
                        dim = mat_cached.shape[1]
                        index = faiss.IndexFlatIP(dim)
                        index.add(mat_cached)
                        return index, mat_cached
                    if reuse_only:
                        raise RuntimeError(
                            "Embedding cache present but does not match current rows/model/content. Refusing to recompute (reuse_only)."
                        )
                    print("  Cache present but does not match rows/model/content. Recomputing embeddings...")
                except Exception as e:
                    # Preserve intentional reuse_only failures
                    if reuse_only and isinstance(e, RuntimeError):
                        raise
                    if reuse_only:
                        raise RuntimeError("Embedding cache unreadable. Refusing to recompute (reuse_only).") from e
                    print("  Cache unreadable. Recomputing embeddings...")
            elif reuse_only:
                raise RuntimeError("Embedding cache not found. Refusing to compute embeddings (reuse_only).")

    # Local embedding in batches, written directly to cache (memory-mapped) to avoid high RAM usage.
    cache_npy_out: Path | None = cache_npy
    if total == 0:
        mat = np.zeros((0, 0), dtype=np.float32)
    else:
        # If caching disabled, still avoid large RAM by using a temp memmap in the cache location.
        if cache_npy_out is None:
            cache_npy_out = Path("embeddings_tmp.npy")

        # Embed first batch to discover vector dimension, then create a memmap .npy and fill it.
        first_end = min(batch_size, total)
        first_texts = [build_embedding_text(r) for r in records[:first_end]]
        first_vecs = embed_texts_local(first_texts, model_id=model, batch_size=min(first_end, batch_size))
        dim = int(first_vecs.shape[1])

        mm = np.lib.format.open_memmap(
            cache_npy_out,
            mode="w+",
            dtype=np.float32,
            shape=(total, dim),
        )
        mm[:first_end] = first_vecs
        print(f"  Embedded {first_end}/{total} rows...")

        for start in range(first_end, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = [build_embedding_text(r) for r in records[start:end]]
            vecs = embed_texts_local(batch_texts, model_id=model, batch_size=min(len(batch_texts), batch_size))
            mm[start:end] = vecs
            print(f"  Embedded {end}/{total} rows...")

        mm.flush()
        mat = mm

    dim = int(mat.shape[1])
    index = faiss.IndexFlatIP(dim)  # IP = Inner Product (cosine on normalized vectors)
    index.add(np.asarray(mat, dtype=np.float32))

    if cache_npy is not None:
        try:
            cache_npy.parent.mkdir(parents=True, exist_ok=True)
            # If we already wrote directly to cache_npy via memmap, avoid rewriting.
            if cache_npy_out is not None and Path(cache_npy_out).resolve() != Path(cache_npy).resolve():
                np.save(cache_npy, np.asarray(mat, dtype=np.float32))
            if cache_meta is not None:
                cache_meta.write_text(
                    json.dumps(
                        {"model": model, "rows": int(total), "dim": int(dim), "text_digest": digest},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            print(f"  Saved embedding cache: {cache_npy}")
        except Exception as e:
            print(f"  Warning: could not save cache: {e}")

    return index, mat


def find_exact_duplicate_groups(
    mat: np.ndarray,
    *,
    round_decimals: int = 6,
) -> list[list[int]]:
    """Find exact-duplicate groups by hashing embedding vectors (no FAISS top_k limits)."""
    if mat.size == 0:
        return []

    rounded = np.round(mat, decimals=round_decimals).astype(np.float32, copy=False)
    from collections import defaultdict

    buckets: dict[bytes, list[int]] = defaultdict(list)
    for i in range(rounded.shape[0]):
        buckets[rounded[i].tobytes()].append(i)

    groups = [members for members in buckets.values() if len(members) >= 2]
    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: g[0])
    return groups

