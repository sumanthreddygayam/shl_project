"""Build a validated FAISS index from the normalized SHL catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.schemas import Assessment

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CATALOG = Path(__file__).parents[1] / "scraper" / "catalog.json"
DEFAULT_DESTINATION = Path(__file__).parents[1] / "vectorstore"
INDEX_FILENAME = "index.faiss"
MANIFEST_FILENAME = "manifest.json"


class EmbeddingBuildError(RuntimeError):
    """Raised when catalog or vector artifacts cannot be built safely."""


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


class VectorstoreManifest(BaseModel):
    """Versioned contract binding a FAISS index to its source catalog."""

    model_config = ConfigDict(extra="forbid")

    format_version: int = Field(default=1, ge=1)
    embedding_model: str = Field(min_length=1)
    distance: str = Field(default="cosine", pattern=r"^cosine$")
    dimension: int = Field(gt=0)
    assessment_count: int = Field(gt=0)
    catalog_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assessment_ids: list[str] = Field(min_length=1)


def catalog_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path: Path) -> list[Assessment]:
    if not path.is_file():
        raise EmbeddingBuildError(f"normalized catalog not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assessments = TypeAdapter(list[Assessment]).validate_python(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise EmbeddingBuildError(f"invalid normalized catalog: {path}") from error
    if not assessments:
        raise EmbeddingBuildError("normalized catalog is empty")
    ids = [item.id for item in assessments]
    if len(ids) != len(set(ids)):
        raise EmbeddingBuildError("normalized catalog contains duplicate ids")
    return assessments


def _load_encoder(model_name: str) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise EmbeddingBuildError(
            "sentence-transformers is required to build embeddings"
        ) from error
    return SentenceTransformer(model_name)


def _load_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        raise EmbeddingBuildError("faiss-cpu is required to build the index") from error
    return faiss


def _embed(encoder: Encoder, assessments: Sequence[Assessment]) -> np.ndarray:
    texts = [assessment.embedding_text() for assessment in assessments]
    vectors = np.asarray(
        encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2:
        raise EmbeddingBuildError("encoder must return a two-dimensional matrix")
    if vectors.shape[0] != len(assessments) or vectors.shape[1] < 1:
        raise EmbeddingBuildError("embedding matrix shape does not match the catalog")
    if not np.isfinite(vectors).all():
        raise EmbeddingBuildError("embedding matrix contains non-finite values")

    # Defensively normalize even if an encoder ignores normalize_embeddings.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise EmbeddingBuildError("embedding matrix contains a zero vector")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def build(
    catalog_path: Path = DEFAULT_CATALOG,
    destination: Path = DEFAULT_DESTINATION,
    model_name: str = DEFAULT_MODEL,
    *,
    encoder: Encoder | None = None,
    faiss_module: Any | None = None,
) -> VectorstoreManifest:
    """Create FAISS and manifest artifacts, replacing each file atomically."""
    assessments = load_catalog(catalog_path)
    active_encoder = encoder or _load_encoder(model_name)
    active_faiss = faiss_module or _load_faiss()
    vectors = _embed(active_encoder, assessments)

    index = active_faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    if int(index.ntotal) != len(assessments):
        raise EmbeddingBuildError("FAISS index count does not match the catalog")

    manifest = VectorstoreManifest(
        embedding_model=model_name,
        dimension=int(vectors.shape[1]),
        assessment_count=len(assessments),
        catalog_sha256=catalog_sha256(catalog_path),
        assessment_ids=[item.id for item in assessments],
    )

    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / INDEX_FILENAME
    index_temporary = destination / f"{INDEX_FILENAME}.tmp"
    manifest_path = destination / MANIFEST_FILENAME
    manifest_temporary = destination / f"{MANIFEST_FILENAME}.tmp"

    try:
        active_faiss.write_index(index, str(index_temporary))
        manifest_temporary.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        if not index_temporary.is_file() or index_temporary.stat().st_size == 0:
            raise EmbeddingBuildError("FAISS did not produce a valid index file")
        index_temporary.replace(index_path)
        manifest_temporary.replace(manifest_path)
    finally:
        index_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)

    return manifest


def validate_artifacts(
    catalog_path: Path = DEFAULT_CATALOG,
    destination: Path = DEFAULT_DESTINATION,
    *,
    faiss_module: Any | None = None,
) -> VectorstoreManifest:
    """Fail fast when an index is absent, corrupt, or stale."""
    active_faiss = faiss_module or _load_faiss()
    index_path = destination / INDEX_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    if not index_path.is_file() or not manifest_path.is_file():
        raise EmbeddingBuildError("vectorstore index or manifest is missing")
    try:
        manifest = VectorstoreManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        index = active_faiss.read_index(str(index_path))
    except (OSError, ValueError, RuntimeError) as error:
        raise EmbeddingBuildError("vectorstore artifact validation failed") from error

    if manifest.catalog_sha256 != catalog_sha256(catalog_path):
        raise EmbeddingBuildError("vectorstore is stale for the current catalog")
    if int(index.ntotal) != manifest.assessment_count:
        raise EmbeddingBuildError("FAISS count does not match its manifest")
    if int(index.d) != manifest.dimension:
        raise EmbeddingBuildError("FAISS dimension does not match its manifest")
    return manifest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(os.getenv("CATALOG_PATH", str(DEFAULT_CATALOG))),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(os.getenv("VECTORSTORE_PATH", str(DEFAULT_DESTINATION))),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing artifacts without loading the embedding model.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _arguments()
    if args.validate_only:
        manifest = validate_artifacts(args.catalog, args.destination)
        LOGGER.info(
            "Validated %d vectors (%d dimensions)",
            manifest.assessment_count,
            manifest.dimension,
        )
    else:
        manifest = build(args.catalog, args.destination, args.model)
        LOGGER.info(
            "Built %d vectors (%d dimensions) in %s",
            manifest.assessment_count,
            manifest.dimension,
            args.destination,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
