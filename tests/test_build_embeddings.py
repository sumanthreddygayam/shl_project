"""Tests for deterministic FAISS artifact construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.schemas import Assessment
from embeddings.build_embeddings import (
    EmbeddingBuildError,
    build,
    load_catalog,
    validate_artifacts,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.received: list[str] = []
        self.options: dict[str, object] = {}

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.received = list(sentences)
        self.options = kwargs
        return np.array(
            [[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]][: len(sentences)],
            dtype=np.float32,
        )


class FakeIndex:
    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.ntotal = 0
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ntotal = len(vectors)


class FakeFaiss:
    IndexFlatIP = FakeIndex

    @staticmethod
    def write_index(index: FakeIndex, path: str) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "d": index.d,
                    "ntotal": index.ntotal,
                    "vectors": index.vectors.tolist(),
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def read_index(path: str) -> FakeIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = FakeIndex(payload["d"])
        index.add(np.asarray(payload["vectors"], dtype=np.float32))
        return index


def _assessment(identifier: str, name: str) -> Assessment:
    return Assessment(
        id=identifier,
        name=name,
        description=f"{name} description",
        url=f"https://example.test/{identifier}",
        categories=["Ability"],
        skills=["Reasoning"],
        job_levels=["Graduate"],
        cognitive=True,
    )


class EmbeddingBuilderTests(unittest.TestCase):
    def test_builds_normalized_index_and_valid_manifest(self) -> None:
        encoder = FakeEncoder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            destination = root / "vectorstore"
            catalog.write_text(
                json.dumps(
                    [
                        item.model_dump(mode="json")
                        for item in (_assessment("1", "Alpha"), _assessment("2", "Beta"))
                    ]
                ),
                encoding="utf-8",
            )

            manifest = build(
                catalog,
                destination,
                "test/model",
                encoder=encoder,
                faiss_module=FakeFaiss,
            )
            validated = validate_artifacts(
                catalog, destination, faiss_module=FakeFaiss
            )
            index = FakeFaiss.read_index(str(destination / "index.faiss"))

        self.assertEqual(manifest, validated)
        self.assertEqual(manifest.assessment_ids, ["1", "2"])
        self.assertEqual(manifest.dimension, 3)
        self.assertEqual(manifest.assessment_count, 2)
        np.testing.assert_allclose(
            np.linalg.norm(index.vectors, axis=1), np.ones(2), atol=1e-6
        )
        self.assertIn("Alpha description", encoder.received[0])
        self.assertTrue(encoder.options["normalize_embeddings"])

    def test_validation_detects_catalog_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            destination = root / "vectorstore"
            catalog.write_text(
                json.dumps([_assessment("1", "Alpha").model_dump(mode="json")]),
                encoding="utf-8",
            )
            build(
                catalog,
                destination,
                encoder=FakeEncoder(),
                faiss_module=FakeFaiss,
            )
            catalog.write_text(catalog.read_text() + " ", encoding="utf-8")

            with self.assertRaisesRegex(EmbeddingBuildError, "stale"):
                validate_artifacts(catalog, destination, faiss_module=FakeFaiss)

    def test_rejects_empty_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(EmbeddingBuildError, "empty"):
                load_catalog(catalog)

    def test_rejects_zero_vectors(self) -> None:
        class ZeroEncoder:
            def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
                return np.zeros((len(sentences), 3), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps([_assessment("1", "Alpha").model_dump(mode="json")]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EmbeddingBuildError, "zero vector"):
                build(
                    catalog,
                    root / "vectors",
                    encoder=ZeroEncoder(),
                    faiss_module=FakeFaiss,
                )


if __name__ == "__main__":
    unittest.main()
