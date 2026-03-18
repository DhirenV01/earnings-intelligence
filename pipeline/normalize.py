"""
normalize.py

Converts raw earnings call transcript text (Motley Fool format) into a
structured Silver-layer JSON record.

Usage:
    with open("aapl_q1_2026.txt") as f:
        raw_text = f.read()
    result = normalize_transcript(raw_text, ticker="AAPL")
"""

import re
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def normalize_transcript(raw_text: str, ticker: str) -> dict:
    """
    Parameters
    ----------
    raw_text : str
        Full page text copied from Motley Fool (or equivalent source).
    ticker : str
        Equity ticker symbol, e.g. "AAPL".  Passed explicitly because the
        ticker isn't always unambiguous in the body text.

    Returns
    -------
    dict
        Silver-layer record matching the schema defined in the architecture.
    """
    metadata   = _extract_metadata(raw_text, ticker)
    role_map   = _build_role_map(raw_text, metadata["company"])
    transcript = _isolate_transcript(raw_text)
    segments   = _parse_segments(transcript, role_map)
    glossary   = _extract_glossary(raw_text)

    return {
        "transcript_id": _build_transcript_id(ticker, metadata),
        "ticker":        ticker.upper(),
        "company":       metadata["company"],
        "quarter":       metadata["quarter"],
        "fiscal_year":   metadata["fiscal_year"],
        "call_date":     metadata["call_date"],
        "source":        "motley_fool",
        "participants":  role_map,
        "segments":      segments,
        "glossary":      glossary,
    }


# ---------------------------------------------------------------------------
# Step 1 — Metadata extraction
# ---------------------------------------------------------------------------

# Maps short names extracted from inline mentions to formal registered names.
# Keyed by lowercase for case-insensitive lookup.
_FORMAL_COMPANY_NAMES: dict[str, str] = {
    "apple":     "Apple Inc.",
    "microsoft": "Microsoft Corporation",
    "amazon":    "Amazon.com, Inc.",
    "alphabet":  "Alphabet Inc.",
    "google":    "Alphabet Inc.",
    "meta":      "Meta Platforms, Inc.",
    "nvidia":    "NVIDIA Corporation",
    "tesla":     "Tesla, Inc.",
}


def _normalize_company_name(name: str) -> str:
    """Expand a short extracted name to its formal registered name if known."""
    return _FORMAL_COMPANY_NAMES.get(name.lower().strip(), name)


def _extract_metadata(text: str, ticker: str) -> dict:
    """Pull date, quarter, fiscal year from the transcript header."""

    # --- Date ---
    # Matches: "Thursday, January 29, 2026 at 5 p.m. ET"
    date_pattern = re.compile(
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
        r"(\w+ \d{1,2},\s+\d{4})"
    )
    date_match = date_pattern.search(text)
    call_date = None
    fiscal_year = None
    if date_match:
        try:
            parsed = datetime.strptime(date_match.group(1).strip(), "%B %d, %Y")
            call_date   = parsed.strftime("%Y-%m-%d")
            fiscal_year = parsed.year
        except ValueError:
            pass

    # --- Quarter ---
    # Look for explicit quarter mentions like "Q1 2026", "first quarter fiscal 2026",
    # "fourth quarter fiscal year 2024", or "fiscal Q1"
    quarter = _infer_quarter(text, call_date)

    # --- Company name ---
    # Primary: title line "Company Name (TICKER) Qn YYYY Earnings Call"
    # Fallback: inline mention like "all-time high for Apple (AAPL +1.05%)"
    # After extraction, normalize short names to their formal registered names.
    company_pattern = re.compile(
        r"\b([A-Za-z][A-Za-z0-9 .,&'-]{1,60}?)\s*\(\s*" + re.escape(ticker.upper()) + r"[^)]*\)",
        re.IGNORECASE
    )
    company_match = company_pattern.search(text)
    raw_company = company_match.group(1).strip() if company_match else ticker.upper()
    company = _normalize_company_name(raw_company)

    return {
        "company":     company,
        "call_date":   call_date,
        "fiscal_year": fiscal_year,
        "quarter":     quarter,
    }


def _infer_quarter(text: str, call_date: Optional[str]) -> str:
    """
    Infer the fiscal quarter label (e.g. 'Q1 2026') from the transcript.

    Strategy (in priority order):
    1. Named quarter in title: "Q1 2026 Earnings Call"
    2. Named month in body: "December quarter" → Q1 (for Oct-Dec quarters)
    3. Explicit "QX YEAR" only where the word before it suggests it's the
       *current* quarter (reported/our/Q1), NOT a prior-quarter reference.
    4. Fall back to calendar quarter from call date.
    """
    # Map reported month names → fiscal quarter labels
    # (works for Apple Oct-Dec=Q1, but labelled generically)
    month_to_q = {
        "december": "Q1", "january": "Q1", "february": "Q1",
        "march":    "Q2", "april":   "Q2", "may":      "Q2",
        "june":     "Q3", "july":    "Q3", "august":   "Q3",
        "september":"Q4", "october": "Q4", "november": "Q4",
    }

    year = call_date[:4] if call_date else "UNKNOWN"

    # Pattern 1: explicit "Q1 2026" in the title line (first 3 lines of text)
    title_block = "\n".join(text.splitlines()[:5])
    title_q = re.search(r"\b(Q[1-4])\s+(\d{4})\b", title_block)
    if title_q:
        return f"{title_q.group(1).upper()} {title_q.group(2)}"

    # Pattern 2: named quarter — "December quarter", "September quarter results"
    # Scope to transcript body to avoid Takeaways guidance ("March quarter") false matches
    body_match = re.search(r"Full Conference Call Transcript\s*\n", text, re.IGNORECASE)
    body_text = text[body_match.end():] if body_match else text
    named_q = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+quarter\b",
        body_text, re.IGNORECASE
    )
    if named_q:
        month = named_q.group(1).lower()
        q_label = month_to_q.get(month, "Q?")
        return f"{q_label} {year}"

    # Pattern 3: "our Q1" / "reported Q2" — current quarter language only
    current_q = re.search(
        r"(?:our|reported|reporting|results? for|fiscal)\s+(Q[1-4])\s+(\d{4})",
        text, re.IGNORECASE
    )
    if current_q:
        return f"{current_q.group(1).upper()} {current_q.group(2)}"

    # Pattern 4: call date calendar fallback
    if call_date:
        month_num = int(call_date[5:7])
        q_num = (month_num - 1) // 3 + 1
        return f"Q{q_num} {year}"

    return "UNKNOWN"

# ---------------------------------------------------------------------------
# Step 2 — Role map
# ---------------------------------------------------------------------------

# Motley Fool participant blocks look like:
#   "Chief Executive Officer — Timothy D. Cook"
#   "Shannon Cross -- Cross Research -- Analyst"
# The separator can be em-dash, en-dash, or double hyphen.

_PARTICIPANT_SECTION_RE = re.compile(
    r"Call participants\s*\n(.*?)(?=\nTakeaways|\nSummary|\nRisks|\n\n|\Z)",
    re.DOTALL | re.IGNORECASE
)

_ROLE_LINE_RE = re.compile(
    r"^(.+?)\s*(?:—|--|–)\s*(.+)$"
)

# Canonical internal roles we want to preserve exactly
_INTERNAL_ROLE_KEYWORDS = {
    "chief executive officer":    "CEO",
    "chief financial officer":    "CFO",
    "chief operating officer":    "COO",
    "chief technology officer":   "CTO",
    "director, investor relations": "IR Director",
    "investor relations":         "IR Director",
    "vice president":             "VP",
}


def _build_role_map(text: str, company: str) -> dict:
    """
    Returns:
        {
          "Timothy D. Cook": {"role": "CEO",      "firm": "Apple Inc."},
          "Shannon Cross":   {"role": "Analyst",  "firm": "Cross Research"},
          ...
        }

    Two passes:
    1. Parse the explicit 'Call participants' block (internal speakers + some analysts)
    2. Extract analyst name + firm from operator introduction lines in the transcript
       e.g. "Our next question is from Eric Woodring of Morgan Stanley"
    """
    role_map = {}

    # --- Pass 1: participants block ---
    section_match = _PARTICIPANT_SECTION_RE.search(text)
    if section_match:
        block = section_match.group(1).strip()
        for line in block.splitlines():
            line = line.strip()
            line = re.sub(r"^[\t\s•\-\*]+", "", line)  # strip •\t prefix
            if not line:
                continue
            # Strip leading bullet/tab chars (Motley Fool uses \t•\t prefixes)
            line = re.sub(r'^[\t\s•\-\*]+', '', line)
            m = _ROLE_LINE_RE.match(line)
            if not m:
                continue
            left, right = m.group(1).strip(), m.group(2).strip()
            role_label = _match_internal_role(left)
            if role_label:
                role_map[right.strip()] = {"role": role_label, "firm": company}
            else:
                parts = re.split(r"\s*(?:—|--|–)\s*", line)
                if len(parts) >= 3:
                    name, firm, role = parts[0].strip(), parts[1].strip(), parts[2].strip()
                elif len(parts) == 2:
                    name, firm, role = parts[0].strip(), parts[1].strip(), "Analyst"
                else:
                    continue
                role_map[name] = {"role": role, "firm": firm}

    # --- Pass 2: operator intros in the transcript body ---
    # Covers: "question from NAME of FIRM", "go to NAME of FIRM",
    #         "question is from NAME", "question will be coming from NAME"
    # IMPORTANT: no re.IGNORECASE — [A-Z] must stay case-sensitive so that
    # lowercase "of" doesn't get absorbed into the name capture group.
    analyst_intro_re = re.compile(
        r"(?:question\s+(?:will\s+be\s+(?:coming\s+)?from|is\s+from|from)"
        r"|go\s+to)\s+"
        r"((?:[A-Z][a-z]+\.?\s+){1,3}[A-Z][a-z]+)"
        r"(?:\s+(?:of|with|calling\s+from)\s+"
        r"([A-Z][A-Za-z0-9\s&.]+?)(?=\s*[.,]|\s+Please|\s+Your|\s*$))?"
    )
    for m in analyst_intro_re.finditer(text):
        name = m.group(1).strip()
        firm = m.group(2).strip() if m.group(2) else ""
        if name not in role_map:
            role_map[name] = {"role": "Analyst", "firm": firm}

    return role_map


def _match_internal_role(text: str) -> Optional[str]:
    """Return canonical role label if text matches a known internal role."""
    lowered = text.lower()
    for keyword, label in _INTERNAL_ROLE_KEYWORDS.items():
        if keyword in lowered:
            return label
    return None


# ---------------------------------------------------------------------------
# Step 3 — Isolate the actual transcript
# ---------------------------------------------------------------------------

_TRANSCRIPT_START_RE = re.compile(
    r"Full Conference Call Transcript\s*\n",
    re.IGNORECASE
)


def _isolate_transcript(text: str) -> str:
    """
    Strip everything before 'Full Conference Call Transcript'.
    Also strip trailing replay / contact info boilerplate.
    """
    match = _TRANSCRIPT_START_RE.search(text)
    transcript = text[match.end():].strip() if match else text.strip()

    # Strip trailing boilerplate that starts with replay instructions
    boilerplate_markers = [
        "A replay of today's call",
        "a replay of today's call",
        "This concludes",
        "this concludes",
    ]
    for marker in boilerplate_markers:
        idx = transcript.find(marker)
        if idx != -1:
            transcript = transcript[:idx].strip()
            break

    return transcript


# ---------------------------------------------------------------------------
# Step 4 — Parse into segments
# ---------------------------------------------------------------------------

def _parse_segments(transcript: str, role_map: dict) -> list:
    """
    Split transcript into per-speaker segments and tag each with:
      - speaker (normalized name)
      - role
      - firm
      - section (prepared_remarks | qa)
      - segment_index
      - text
    """
    # Build a regex that matches any known speaker name followed by a colon.
    # Also match "Operator:" as a special case.
    # We use a greedy split on lines that start with "Name:" patterns.
    raw_segments = _split_by_speaker(transcript)

    segments   = []
    in_qa      = False
    seg_index  = 0

    for speaker_raw, text in raw_segments:
        text = text.strip()
        if not text:
            continue

        speaker_norm = _normalize_speaker_name(speaker_raw)
        info         = role_map.get(speaker_norm, {})
        role         = info.get("role", _infer_role(speaker_norm))
        firm         = info.get("firm", "")

        # Detect transition into Q&A: operator says "open the call for questions"
        if speaker_norm.lower() == "operator" and "question" in text.lower():
            in_qa = True

        # Also detect Q&A by presence of analyst questions
        if role == "Analyst" and not in_qa:
            in_qa = True

        segment = {
            "segment_index": seg_index,
            "speaker":       speaker_norm,
            "role":          role,
            "firm":          firm,
            "section":       "qa" if in_qa else "prepared_remarks",
            "text":          _clean_text(text),
        }
        segments.append(segment)
        seg_index += 1

    return segments


# Speaker line pattern: "Firstname [Middle] Lastname: " at start of a block.
# Also handles "Operator:" and bare names.
_SPEAKER_LINE_RE = re.compile(
    r"^([A-Z][a-zA-Z\-'\.]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zA-Z\-']+)+|Operator)\s*:",
    re.MULTILINE
)


def _split_by_speaker(transcript: str) -> list[tuple[str, str]]:
    """
    Returns a list of (speaker_name, utterance_text) tuples.
    """
    matches  = list(_SPEAKER_LINE_RE.finditer(transcript))
    segments = []

    for i, match in enumerate(matches):
        speaker = match.group(1).strip()
        start   = match.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(transcript)
        text    = transcript[start:end].strip()
        segments.append((speaker, text))

    return segments


def _normalize_speaker_name(name: str) -> str:
    """Normalize spacing and strip trailing punctuation."""
    return re.sub(r"\s+", " ", name).strip(" :")


def _infer_role(name: str) -> str:
    """
    Fallback role inference for speakers not in the participants block.
    """
    if name.lower() == "operator":
        return "Operator"
    return "Unknown"


def _clean_text(text: str) -> str:
    """Remove freestar ad markers and normalize whitespace."""
    text = re.sub(r"\s*freestar\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 5 — Glossary extraction
# ---------------------------------------------------------------------------

_GLOSSARY_RE = re.compile(
    r"Industry glossary\s*\n(.*?)(?=Full Conference Call Transcript)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_glossary(text: str) -> list:
    """
    Extract term/definition pairs from the 'Industry glossary' preamble block.

    Each line in the block has the Motley Fool bullet format:
        \t•\tTerm: Definition text here.

    Returns a list of {"term": str, "definition": str} dicts.
    Returns an empty list if no glossary section is found.
    """
    m = _GLOSSARY_RE.search(text)
    if not m:
        return []

    items = []
    for line in m.group(1).splitlines():
        # Strip bullet chars, tabs, and leading whitespace
        line = re.sub(r"^[\t\s•\-\*]+", "", line).strip()
        if not line:
            continue
        # Split on first ": " — term is everything before, definition after
        parts = line.split(": ", 1)
        if len(parts) == 2:
            items.append({"term": parts[0].strip(), "definition": parts[1].strip()})

    return items


# ---------------------------------------------------------------------------
# Step 6 — Transcript ID
# ---------------------------------------------------------------------------

def _build_transcript_id(ticker: str, metadata: dict) -> str:
    """e.g. AAPL_Q1_2026"""
    q   = metadata.get("quarter", "UNKNOWN").replace(" ", "_")
    return f"{ticker.upper()}_{q}"


# ---------------------------------------------------------------------------
# Quick smoke test — run with: python normalize.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python normalize.py <transcript_file.txt>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        raw = f.read()

    result = normalize_transcript(raw, ticker="AAPL")

    # Print summary instead of the full JSON blob
    print(f"\n=== Transcript ID : {result['transcript_id']}")
    print(f"    Company        : {result['company']}")
    print(f"    Quarter        : {result['quarter']}")
    print(f"    Fiscal Year    : {result['fiscal_year']}")
    print(f"    Call Date      : {result['call_date']}")
    print(f"    Participants   : {len(result['participants'])}")
    print(f"    Segments       : {len(result['segments'])}")
    print(f"\n--- Participants ---")
    for name, info in result["participants"].items():
        print(f"  {name:<30} {info['role']:<15} {info['firm']}")
    print(f"\n--- First 3 segments ---")
    for seg in result["segments"][:3]:
        preview = seg["text"][:120].replace("\n", " ")
        print(f"  [{seg['section']:<17}] {seg['speaker']:<25} ({seg['role']})")
        print(f"    {preview}...")
    print(f"\n--- Full JSON written to normalized_output.json ---")

    with open("normalized_output.json", "w") as f:
        json.dump(result, f, indent=2)
