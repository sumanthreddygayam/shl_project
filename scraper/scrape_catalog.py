"""Download and normalize the SHL Individual Test Solutions catalog.

The source URL is accepted only through ``--catalog-url`` or ``CATALOG_URL``.
The normalizer tolerates common JSON naming variants while enforcing the
canonical :class:`app.schemas.Assessment` contract at the output boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from pydantic import TypeAdapter, ValidationError

from app.schemas import Assessment

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT = Path(__file__).with_name("catalog.json")
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_RETRIES = 3
USER_AGENT = "shl-assessment-recommender/1.0"

_ALIASES: dict[str, tuple[str, ...]] = {
    "id": (
        "id",
        "entity_id",
        "product_id",
        "productId",
        "assessment_id",
        "assessmentId",
        "sku",
    ),
    "name": ("name", "title", "product_name", "productName", "assessment_name"),
    "description": (
        "description",
        "short_description",
        "shortDescription",
        "summary",
        "details",
    ),
    "url": ("url", "product_url", "productUrl", "link", "href"),
    "job_levels": ("job_levels", "jobLevels", "job_level", "jobLevel", "levels"),
    "categories": (
        "categories",
        "category",
        "test_types",
        "testTypes",
        "test_type",
        "testType",
        "keys",
    ),
    "skills": ("skills", "skill", "competencies", "knowledge_skills"),
    "product_type": (
        "product_type",
        "productType",
        "solution_type",
        "solutionType",
        "type",
    ),
}

_FLAG_TERMS: dict[str, tuple[str, ...]] = {
    "technical": (
        "technical",
        "programming",
        "coding",
        "software",
        "it skill",
        "computer science",
    ),
    "cognitive": (
        "cognitive",
        "ability",
        "reasoning",
        "numerical",
        "verbal",
        "inductive",
        "deductive",
    ),
    "behavioral": ("behavioral", "behavioural", "situational judgement", "sjt"),
    "personality": ("personality", "occupational personality", "opq"),
    "leadership": ("leadership", "leader", "managerial"),
    "communication": ("communication", "spoken", "written", "language"),
    "stakeholder": ("stakeholder", "influencing", "negotiation"),
}


class CatalogDownloadError(RuntimeError):
    """Raised when the configured catalog cannot be downloaded or decoded."""


class CatalogNormalizationError(RuntimeError):
    """Raised when a payload contains no valid Individual Test Solutions."""


def _key_text(value: object) -> str:
    return str(value).strip()


def _first(record: Mapping[str, Any], field: str) -> Any:
    for key in _ALIASES[field]:
        if key in record and record[key] not in (None, "", [], {}):
            return record[key]
    return None


def _plain_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(_plain_text(item) for item in value)
    elif isinstance(value, Mapping):
        value = " ".join(_plain_text(item) for item in value.values())
    text = str(value)
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return " ".join(text.split())


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        values: Iterable[object] = value.values()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = _plain_text(value)
        values = text.replace("|", ",").replace(";", ",").split(",")

    unique: dict[str, str] = {}
    for item in values:
        if isinstance(item, Mapping):
            item = item.get("name") or item.get("label") or item.get("value") or ""
        text = _plain_text(item)
        if text:
            unique.setdefault(text.casefold(), text)
    return list(unique.values())


def _boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "y", "1", "available", "supported"}:
            return True
        if normalized in {"false", "no", "n", "0", "unavailable", "not supported"}:
            return False
    return None


def _explicit_flag(record: Mapping[str, Any], flag: str) -> bool | None:
    aliases = (
        flag,
        f"is_{flag}",
        f"is{flag.title()}",
        f"{flag}_available",
        f"{flag}Available",
    )
    for alias in aliases:
        if alias in record:
            parsed = _boolean(record[alias])
            if parsed is not None:
                return parsed
    return None


def _solution_context(text: str) -> str | None:
    normalized = " ".join(text.casefold().replace("_", " ").replace("-", " ").split())
    if "job solution" in normalized:
        return "job"
    if "individual test solution" in normalized or "individual assessment" in normalized:
        return "individual"
    return None


def _walk_records(node: object, context: str | None = None) -> Iterable[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        explicit_type = _plain_text(_first(node, "product_type"))
        local_context = _solution_context(explicit_type) or context

        name = _first(node, "name")
        url = _first(node, "url")
        if name and url:
            yield {**node, "__solution_context__": local_context}

        for key, value in node.items():
            child_context = _solution_context(_key_text(key)) or local_context
            yield from _walk_records(value, child_context)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_records(item, context)


def _is_individual(record: Mapping[str, Any]) -> bool:
    context = record.get("__solution_context__")
    if context == "job":
        return False
    if context == "individual":
        return True

    searchable = " ".join(
        (
            _plain_text(_first(record, "product_type")),
            _plain_text(record.get("solution")),
            _plain_text(record.get("product_family")),
        )
    )
    resolved = _solution_context(searchable)
    if resolved is not None:
        return resolved == "individual"

    # Adapter for SHL's curated flat product-catalog export. It does not expose
    # a solution-type field, but every Individual Test Solution record has this
    # stable signature and canonical detail route. Avoid accepting arbitrary
    # unlabeled records merely because they contain a name and URL.
    source_url = _valid_http_url(record.get("link"))
    path = urlparse(source_url).path.casefold() if source_url else ""
    return (
        "entity_id" in record
        and "keys" in record
        and "status" in record
        and "/products/product-catalog/view/" in path
        and "job solution" not in _plain_text(_first(record, "name")).casefold()
    )


def _valid_http_url(value: object) -> str | None:
    text = _plain_text(value)
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return text
    return None


def _stable_id(record: Mapping[str, Any], name: str, url: str) -> str:
    raw_id = _plain_text(_first(record, "id"))
    if raw_id:
        return raw_id
    return hashlib.sha256(f"{url}\0{name}".encode("utf-8")).hexdigest()[:20]


def _normalized_flag(
    record: Mapping[str, Any], flag: str, searchable_text: str
) -> bool:
    explicit = _explicit_flag(record, flag)
    if explicit is not None:
        return explicit
    return any(term in searchable_text for term in _FLAG_TERMS.get(flag, ()))


def normalize_record(record: Mapping[str, Any]) -> Assessment | None:
    """Normalize one source record, returning ``None`` for unusable records."""
    if not _is_individual(record):
        return None

    name = _plain_text(_first(record, "name"))
    url = _valid_http_url(_first(record, "url"))
    if not name or not url:
        return None

    description = _plain_text(_first(record, "description"))
    job_levels = _string_list(_first(record, "job_levels"))
    categories = _string_list(_first(record, "categories"))
    skills = _string_list(_first(record, "skills"))
    searchable = " ".join(
        [name, description, *job_levels, *categories, *skills]
    ).casefold()

    flags = {
        flag: _normalized_flag(record, flag, searchable)
        for flag in _FLAG_TERMS
    }
    flags["remote"] = bool(
        _explicit_flag(record, "remote")
        if _explicit_flag(record, "remote") is not None
        else any(term in searchable for term in ("remote", "remotely", "online"))
    )
    flags["adaptive"] = bool(
        _explicit_flag(record, "adaptive")
        if _explicit_flag(record, "adaptive") is not None
        else "adaptive" in searchable
    )

    return Assessment(
        id=_stable_id(record, name, url),
        name=name,
        description=description,
        url=url,
        job_levels=job_levels,
        categories=categories,
        skills=skills,
        **flags,
    )


class SHLCatalogSource:
    """HTTP adapter implementing the catalog source interface."""

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retries < 1:
            raise ValueError("retries must be at least 1")
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def download(self, source_url: str) -> Any:
        parsed = urlparse(source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CatalogDownloadError("CATALOG_URL must be an absolute HTTP(S) URL")

        request = Request(
            source_url,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    payload = response.read().decode(charset)
                # The upstream export has historically contained literal control
                # characters inside strings. Accept those at the transport
                # boundary; canonical output is re-encoded by ``json.dumps``.
                return json.loads(payload, strict=False)
            except (HTTPError, URLError, TimeoutError, socket.timeout, UnicodeError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(0.25 * (2 ** (attempt - 1)))

        raise CatalogDownloadError(
            f"catalog download failed after {self.retries} attempts"
        ) from last_error

    def normalize(self, payload: Any) -> list[Assessment]:
        normalized: dict[str, Assessment] = {}
        invalid_count = 0
        for record in _walk_records(payload):
            try:
                assessment = normalize_record(record)
            except ValidationError:
                invalid_count += 1
                continue
            if assessment is not None:
                # URL is the safest cross-version identity for duplicate records.
                normalized.setdefault(str(assessment.url), assessment)

        assessments = sorted(
            normalized.values(), key=lambda item: (item.name.casefold(), item.id)
        )
        if not assessments:
            raise CatalogNormalizationError(
                "catalog contains no valid Individual Test Solutions"
            )
        if invalid_count:
            LOGGER.warning("Skipped %d invalid Individual Test Solution records", invalid_count)
        return assessments

    def write(
        self, assessments: Sequence[Assessment], destination: Path
    ) -> None:
        adapter = TypeAdapter(list[Assessment])
        validated = adapter.validate_python(list(assessments))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in validated],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)


def run(catalog_url: str, output: Path = DEFAULT_OUTPUT) -> list[Assessment]:
    """Download, normalize, validate, and atomically persist the catalog."""
    source = SHLCatalogSource()
    payload = source.download(catalog_url)
    assessments = source.normalize(payload)
    source.write(assessments, output)
    return assessments


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-url",
        default=os.getenv("CATALOG_URL"),
        help="Catalog JSON URL; defaults to CATALOG_URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("CATALOG_PATH", str(DEFAULT_OUTPUT))),
        help="Normalized JSON destination.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _arguments()
    if not args.catalog_url:
        raise SystemExit("CATALOG_URL is required (or pass --catalog-url)")
    assessments = run(args.catalog_url, args.output)
    LOGGER.info("Wrote %d assessments to %s", len(assessments), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
