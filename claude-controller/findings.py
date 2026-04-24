"""Finding file writer.

Findings are written from the orchestrator's structured `file_finding` tool
call. The legacy text-based parser has been removed — the orchestrator
produces well-formed fields directly.
"""

import os
import re

from tools import FindingCandidate, FindingFiled


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def _canonical_endpoint(endpoint: str) -> str:
    """Normalize an endpoint string for dedup comparison."""
    if not endpoint:
        return ""
    # Strip method prefix if present
    parts = endpoint.strip().split(None, 1)
    path = parts[1] if len(parts) == 2 and parts[0].upper() in {
        "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
    } else endpoint
    path = path.strip().lower()
    # Drop query string and trailing slash
    path = path.split("?", 1)[0].rstrip("/")
    return path


def _titles_similar(a: str, b: str) -> bool:
    sa, sb = slugify(a), slugify(b)
    if not sa or not sb:
        return False
    if sa == sb or sa in sb or sb in sa:
        return True
    wa, wb = set(sa.split("-")), set(sb.split("-"))
    if not wa or not wb:
        return False
    overlap = len(wa & wb) / max(len(wa), len(wb))
    return overlap > 0.8


def match_pending_candidates(
    filed: FindingFiled, pending: list[FindingCandidate],
) -> list[str]:
    """Return candidate_ids from `pending` whose endpoint and title match `filed`.

    Used when the verifier files a finding without `supersedes_candidate_ids`
    so the controller can still mark the originating candidate(s) resolved.
    Ambiguous cases (e.g. endpoint-only hits) are deliberately NOT matched
    here — those candidates stay pending and the verifier must resolve them
    explicitly on the next substep.
    """
    filed_ep = _canonical_endpoint(filed.endpoint)
    if not filed_ep or not filed.title:
        return []
    matched: list[str] = []
    for c in pending:
        if _canonical_endpoint(c.endpoint) != filed_ep:
            continue
        if not _titles_similar(c.title, filed.title):
            continue
        matched.append(c.candidate_id)
    return matched


_MARKDOWN_TEMPLATE = """\
# {title}

- **Severity**: {severity}
- **Affected Endpoint**: {endpoint}

## Description

{description}

## Reproduction Steps

{reproduction_steps}

## Evidence

{evidence}

## Impact

{impact}

## Verification

{verification_notes}
"""


class FindingWriter:
    """Persists verified findings from `FindingFiled` records."""

    def __init__(self, findings_dir: str) -> None:
        self.findings_dir = findings_dir
        self.count = 0
        self.paths: list[str] = []
        self._index: list[dict] = []

    def is_duplicate(self, filed: FindingFiled) -> bool:
        title_slug = slugify(filed.title)
        endpoint = _canonical_endpoint(filed.endpoint)
        for entry in self._index:
            if title_slug and title_slug == entry["title_slug"]:
                return True
            if endpoint and entry["endpoint"] == endpoint and _titles_similar(filed.title, entry["title"]):
                return True
        return False

    def summary_for_orchestrator(self) -> str:
        if not self._index:
            return "No findings filed yet."
        lines = []
        for i, entry in enumerate(self._index, 1):
            sev = entry["severity"] or "unknown"
            ep = entry["endpoint"] or "N/A"
            lines.append(f"{i}. [{sev}] {entry['title']} — {ep}")
        return "**Findings filed so far:**\n" + "\n".join(lines)

    def summary_for_worker(self) -> str:
        """Worker-facing roster: title + endpoint only, no severity.

        Severity and verifier reasoning are intentionally omitted — workers
        might argue with the verifier's judgement rather than do new work.
        Returns an empty string when nothing has been filed so the caller
        can suppress the whole block.
        """
        if not self._index:
            return ""
        lines = []
        for entry in self._index:
            ep = entry["endpoint"] or "N/A"
            lines.append(f"- {entry['title']} — {ep}")
        return "Findings filed so far — do not re-file:\n" + "\n".join(lines)

    def write(self, filed: FindingFiled) -> str:
        os.makedirs(self.findings_dir, exist_ok=True)
        self.count += 1

        slug = slugify(filed.title) or "untitled"
        if len(slug) > 60:
            slug = slug[:60].rstrip("-")
        filename = f"finding-{self.count:02d}-{slug}.md"
        filepath = os.path.join(self.findings_dir, filename)

        body = _MARKDOWN_TEMPLATE.format(
            title=filed.title,
            severity=filed.severity,
            endpoint=filed.endpoint or "N/A",
            description=filed.description or "(none)",
            reproduction_steps=filed.reproduction_steps or "(none)",
            evidence=filed.evidence or "(none)",
            impact=filed.impact or "(none)",
            verification_notes=filed.verification_notes or "(none)",
        )
        with open(filepath, "w") as f:
            f.write(body)

        self.paths.append(filepath)
        self._index.append({
            "title": filed.title,
            "title_slug": slugify(filed.title),
            "endpoint": _canonical_endpoint(filed.endpoint),
            "severity": filed.severity,
            "path": filepath,
        })
        return filepath
