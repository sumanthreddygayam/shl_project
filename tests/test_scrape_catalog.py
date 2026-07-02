"""Tests for catalog filtering, normalization, and persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import TypeAdapter

from app.schemas import Assessment
from scraper.scrape_catalog import (
    CatalogDownloadError,
    CatalogNormalizationError,
    SHLCatalogSource,
    normalize_record,
)


class CatalogNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SHLCatalogSource(timeout_seconds=1, retries=1)

    def test_keeps_only_individual_test_solutions(self) -> None:
        payload = {
            "Individual Test Solutions": [
                {
                    "productId": "java-1",
                    "title": "Java Programming",
                    "shortDescription": "<p>Measures Java coding ability.</p>",
                    "productUrl": "https://example.test/java",
                    "jobLevels": ["Professional", "Mid-Level"],
                    "testTypes": ["Technical", "Knowledge & Skills"],
                    "skills": ["Java", "Programming"],
                    "remote": "Yes",
                }
            ],
            "Job Solutions": [
                {
                    "id": "job-1",
                    "name": "Software Engineer Job Solution",
                    "url": "https://example.test/job-solution",
                }
            ],
        }

        assessments = self.source.normalize(payload)

        self.assertEqual([item.id for item in assessments], ["java-1"])
        assessment = assessments[0]
        self.assertEqual(assessment.description, "Measures Java coding ability.")
        self.assertEqual(assessment.skills, ["Java", "Programming"])
        self.assertTrue(assessment.technical)
        self.assertTrue(assessment.cognitive)
        self.assertTrue(assessment.remote)

    def test_explicit_false_overrides_keyword_inference(self) -> None:
        record = {
            "type": "Individual Test Solution",
            "name": "Adaptive Reasoning",
            "description": "An adaptive cognitive ability assessment",
            "url": "https://example.test/reasoning",
            "adaptive": False,
            "cognitive": False,
        }

        assessment = normalize_record(record)

        self.assertIsNotNone(assessment)
        assert assessment is not None
        self.assertFalse(assessment.adaptive)
        self.assertFalse(assessment.cognitive)

    def test_accepts_curated_flat_shl_export_signature(self) -> None:
        payload = [
            {
                "entity_id": "4302",
                "name": "Example Ability",
                "link": "https://www.shl.com/products/product-catalog/view/example/",
                "status": "ok",
                "keys": ["Ability & Aptitude"],
                "description": "Measures numerical reasoning.",
                "job_levels": ["Graduate"],
                "remote": "yes",
                "adaptive": "no",
            }
        ]

        assessments = self.source.normalize(payload)

        self.assertEqual(assessments[0].id, "4302")
        self.assertEqual(assessments[0].categories, ["Ability & Aptitude"])
        self.assertTrue(assessments[0].cognitive)

    def test_deduplicates_by_url_and_sorts_by_name(self) -> None:
        payload = {
            "individual test solutions": [
                {
                    "id": "z",
                    "name": "Zulu",
                    "url": "https://example.test/z",
                },
                {
                    "id": "a",
                    "name": "Alpha",
                    "url": "https://example.test/a",
                },
                {
                    "id": "duplicate",
                    "name": "Alpha duplicate",
                    "url": "https://example.test/a",
                },
            ]
        }

        assessments = self.source.normalize(payload)

        self.assertEqual([item.name for item in assessments], ["Alpha", "Zulu"])

    def test_rejects_payload_without_individual_solutions(self) -> None:
        with self.assertRaises(CatalogNormalizationError):
            self.source.normalize(
                {
                    "Job Solutions": [
                        {
                            "name": "Graduate Job Solution",
                            "url": "https://example.test/job",
                        }
                    ]
                }
            )

    def test_generates_stable_id_when_source_id_is_missing(self) -> None:
        record = {
            "productType": "Individual Test Solution",
            "name": "General Ability",
            "url": "https://example.test/general-ability",
        }

        first = normalize_record(record)
        second = normalize_record(record)

        self.assertIsNotNone(first)
        self.assertEqual(first.id, second.id)  # type: ignore[union-attr]
        self.assertEqual(len(first.id), 20)  # type: ignore[union-attr]


class CatalogPersistenceTests(unittest.TestCase):
    def test_writes_canonical_json_array_that_round_trips(self) -> None:
        assessment = Assessment(
            id="one",
            name="One",
            description="Description",
            url="https://example.test/one",
            skills=["Python"],
            technical=True,
        )
        source = SHLCatalogSource(timeout_seconds=1, retries=1)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "catalog.json"
            source.write([assessment], destination)
            raw = json.loads(destination.read_text(encoding="utf-8"))
            restored = TypeAdapter(list[Assessment]).validate_python(raw)

        self.assertIsInstance(raw, list)
        self.assertEqual(restored, [assessment])
        self.assertFalse(destination.with_suffix(".json.tmp").exists())


class CatalogDownloadValidationTests(unittest.TestCase):
    def test_rejects_non_http_url_before_network_access(self) -> None:
        source = SHLCatalogSource(timeout_seconds=1, retries=1)
        with self.assertRaises(CatalogDownloadError):
            source.download("file:///tmp/catalog.json")


if __name__ == "__main__":
    unittest.main()
