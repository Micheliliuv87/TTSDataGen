#!/usr/bin/env python3
"""
Site-specific cleaning utilities for HappyScribe podcast transcripts.

This module is intentionally conservative. It removes obvious transcript boilerplate,
ad/promotional blocks, and repeated footer text, while preserving timestamped content.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

TIMESTAMP_PREFIX_RE = re.compile(r"^\[\d{1,2}:\d{2}:\d{2}\]\s*")
WHITESPACE_RE = re.compile(r"[ \t]+")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# High-confidence phrases that often indicate ad / boilerplate blocks.
AD_START_RE = re.compile(
    r"\b("
    r"this episode is brought to you by|"
    r"this podcast is brought to you by|"
    r"support for .* comes from|"
    r"sponsored by|"
    r"paid for by|"
    r"presented by|"
    r"brought to you by"
    r")\b",
    re.IGNORECASE,
)

PROMO_RE = re.compile(
    r"\b("
    r"promo code|use code|coupon code|free shipping|"
    r"percent off|% off|\d+ percent off|"
    r"terms apply|member fdic|"
    r"get your quote|download .* app|"
    r"go to .* dot com|visit .* dot com|"
    r"available wherever books are sold|"
    r"now streaming|premieres tonight|premieres .* on"
    r")\b",
    re.IGNORECASE,
)

PODCAST_FOOTER_RE = re.compile(
    r"\b("
    r"for more podcasts from|"
    r"wherever you get your podcasts|"
    r"wherever else you listen|"
    r"apple podcasts or wherever|"
    r"iheartradio app|i heart radio app|"
    r"subscribe to (our|the) show|"
    r"you can find us on social media|"
    r"write to us at"
    r")\b",
    re.IGNORECASE,
)

# Brand list is only used as weak evidence. A brand mention alone does not remove a block.
WEAK_BRAND_RE = re.compile(
    r"\b("
    r"capital one|geico|progressive|clorox|madison reed|lord jones|"
    r"general motors|best fiends|best beans|abc|hulu|audible|ibm|"
    r"gm dot com|dunkin|state farm|allstate|stitcher premium"
    r")\b",
    re.IGNORECASE,
)

INLINE_REMOVE_PATTERNS = [
    # Often appended at the very end of HappyScribe transcript text.
    re.compile(r"\s*Episode description\s+.*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\s*Learn more about your ad[- ]choices? at\s+\S+", re.IGNORECASE),
    re.compile(r"\s*Learn more about your ad choices at\s+\S+", re.IGNORECASE),
]

STANDALONE_NOISE_RE = re.compile(
    r"^("
    r"copy link to transcript|"
    r"audio transcription by|"
    r"transcribed from audio to text by|"
    r"about us|contact us|privacy|terms|request podcast"
    r")$",
    re.IGNORECASE,
)


@dataclass
class CleanResult:
    text: str
    stats: dict = field(default_factory=dict)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = []
    for line in text.splitlines():
        line = WHITESPACE_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def strip_inline_boilerplate(block: str) -> tuple[str, int]:
    removed = 0
    cleaned = block
    for pattern in INLINE_REMOVE_PATTERNS:
        new_cleaned = pattern.sub("", cleaned).strip()
        if new_cleaned != cleaned:
            removed += 1
            cleaned = new_cleaned
    return cleaned.strip(), removed


def block_without_timestamp(block: str) -> str:
    return TIMESTAMP_PREFIX_RE.sub("", block).strip()


def is_ad_or_boilerplate_block(block: str) -> tuple[bool, str]:
    """Return (should_remove, reason). Conservative score-based filter."""
    text = block_without_timestamp(block)
    lower = text.lower().strip()

    if not lower:
        return True, "empty_block"

    if STANDALONE_NOISE_RE.match(lower):
        return True, "standalone_noise"

    score = 0
    reasons: list[str] = []

    if AD_START_RE.search(text):
        score += 3
        reasons.append("ad_start_phrase")

    if PROMO_RE.search(text):
        score += 2
        reasons.append("promo_phrase")

    if PODCAST_FOOTER_RE.search(text):
        score += 2
        reasons.append("podcast_footer")

    if WEAK_BRAND_RE.search(text):
        score += 1
        reasons.append("weak_brand")

    # Very URL-heavy blocks are usually boilerplate or promo text.
    if len(URL_RE.findall(text)) >= 2:
        score += 2
        reasons.append("url_heavy")

    # Remove short-to-medium blocks when enough evidence suggests ad/boilerplate.
    # Longer blocks are kept unless the evidence is very strong, to avoid deleting real content.
    if score >= 3 and len(text) <= 2500:
        return True, "+".join(reasons)

    if score >= 5:
        return True, "+".join(reasons)

    return False, ""


def clean_happyscribe_transcript(text: str) -> CleanResult:
    """Clean one HappyScribe transcript while preserving timestamped segments."""
    original_text = text or ""
    normalized = normalize_text(original_text)

    kept_blocks: list[str] = []
    removed_blocks = 0
    removed_inline_phrases = 0
    removed_duplicate_blocks = 0
    remove_reasons: dict[str, int] = {}
    seen_consecutive_body: str | None = None

    for raw_block in normalized.splitlines():
        block, inline_removed = strip_inline_boilerplate(raw_block)
        removed_inline_phrases += inline_removed

        if not block:
            removed_blocks += 1
            remove_reasons["empty_after_inline_clean"] = remove_reasons.get("empty_after_inline_clean", 0) + 1
            continue

        should_remove, reason = is_ad_or_boilerplate_block(block)
        if should_remove:
            removed_blocks += 1
            remove_reasons[reason] = remove_reasons.get(reason, 0) + 1
            continue

        # Drop exact consecutive duplicate blocks after removing timestamp.
        body = block_without_timestamp(block)
        body_key = re.sub(r"\s+", " ", body.lower()).strip()
        if body_key and body_key == seen_consecutive_body:
            removed_duplicate_blocks += 1
            continue

        seen_consecutive_body = body_key
        kept_blocks.append(block)

    cleaned = "\n".join(kept_blocks).strip()

    stats = {
        "cleaning_version": "happyscribe_clean_v0_1",
        "original_chars": len(original_text),
        "normalized_chars": len(normalized),
        "cleaned_chars": len(cleaned),
        "removed_chars": max(0, len(normalized) - len(cleaned)),
        "removed_blocks": removed_blocks,
        "removed_inline_phrases": removed_inline_phrases,
        "removed_consecutive_duplicate_blocks": removed_duplicate_blocks,
        "remove_reasons": remove_reasons,
    }

    return CleanResult(text=cleaned, stats=stats)


def clean_transcript_text(text: str) -> str:
    """Backward-compatible helper: return only cleaned text."""
    return clean_happyscribe_transcript(text).text
