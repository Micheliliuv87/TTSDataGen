#!/usr/bin/env python3
"""
HappyScribe podcast transcript scraper for Content2Dialogue V0.

Supports two modes:

1. Single podcast mode
   python scrapers/happyscribe/scrape.py \
     --podcast-url "https://podcasts.happyscribe.com/stuff-you-missed-in-history-class?page=3" \
     --output data/interim/podcasts/happyscribe/stuff-you-missed-in-history-class/source_documents.jsonl \
     --sleep 1.0 \
     --validate

2. Full HappyScribe podcast catalog mode
   python scrapers/happyscribe/scrape.py \
     --all-podcasts \
     --output-dir data/interim/podcasts/happyscribe \
     --sleep 1.0 \
     --validate

Data lifecycle recommendation:
  data/raw/       = raw HTML caches, only if --save-raw-html is used
  data/interim/   = parsed full source documents, one JSONL row per episode
  data/processed/ = cleaned/chunked RAG-ready data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://podcasts.happyscribe.com"
DEFAULT_CATALOG_URL = "https://podcasts.happyscribe.com/podcasts"
DEFAULT_PODCAST_URL = (
    "https://podcasts.happyscribe.com/stuff-you-missed-in-history-class?page=3"
)
DEFAULT_SINGLE_OUTPUT_PATH = (
    "data/interim/podcasts/happyscribe/"
    "stuff-you-missed-in-history-class/source_documents.jsonl"
)
DEFAULT_ALL_OUTPUT_DIR = "data/interim/podcasts/happyscribe"
DEFAULT_RAW_HTML_DIR = "data/raw/podcasts/happyscribe"
DEFAULT_USER_AGENT = (
    "Content2DialogueBot/0.1 "
    "(+local research prototype; contact: replace-with-your-email@example.com)"
)
TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
WHITESPACE_RE = re.compile(r"\s+")
EPISODE_COUNT_RE = re.compile(r"(\d+)\s+episodes?", re.IGNORECASE)


@dataclass(frozen=True)
class PodcastSeries:
    title: str
    url: str
    slug: str
    catalog_page_url: str
    episode_count: Optional[int] = None
    views_text: Optional[str] = None


@dataclass(frozen=True)
class EpisodeLink:
    title: str
    url: str
    list_page_url: str
    description: Optional[str] = None
    duration: Optional[str] = None
    published_text: Optional[str] = None


@dataclass(frozen=True)
class TranscriptSegment:
    timestamp: str
    text: str


@dataclass
class SourceDocument:
    doc_id: str
    source_site: str
    source_type: str
    podcast_title: Optional[str]
    title: str
    url: str
    list_page_url: str
    retrieved_at: str
    language: str
    clean_text: str
    content_hash: str
    extraction_method: str
    duration: Optional[str] = None
    published_date: Optional[str] = None
    published_text: Optional[str] = None
    description: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def setup_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("happyscribe_scraper")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def compact_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if parts:
        return parts[-1]
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"unknown-{digest}"


def stable_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        return "happyscribe__" + "__".join(parts[-2:])
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"happyscribe__{digest}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_read_prefix(label: str) -> str:
    return re.sub(r"^\s*Read\s+", "", label or "", flags=re.IGNORECASE).strip()


def strip_visit_prefix(label: str) -> str:
    return re.sub(r"^\s*Visit\s+", "", label or "", flags=re.IGNORECASE).strip()


def remove_noise_tags(soup: BeautifulSoup) -> None:
    for tag in soup.select(
        "script, style, noscript, svg, header, footer, nav, aside, form, button"
    ):
        tag.decompose()


def load_existing_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()

    urls: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            url = obj.get("url")
            if isinstance(url, str):
                urls.add(url)

    return urls


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


class HappyScribePodcastScraper:
    """Scraper adapter for podcasts.happyscribe.com podcast transcript pages."""

    def __init__(
        self,
        podcast_url: str = DEFAULT_PODCAST_URL,
        catalog_url: str = DEFAULT_CATALOG_URL,
        sleep_seconds: float = 1.0,
        timeout_seconds: int = 30,
        user_agent: str = DEFAULT_USER_AGENT,
        raw_html_dir: Optional[Path] = None,
        save_raw_html: bool = False,
        verbose: bool = False,
    ) -> None:
        self.podcast_url = podcast_url
        self.catalog_url = catalog_url
        self.sleep_seconds = sleep_seconds
        self.timeout_seconds = timeout_seconds
        self.raw_html_dir = raw_html_dir
        self.save_raw_html = save_raw_html
        self.log = setup_logger(verbose)
        self.session = self._make_session(user_agent)

    @staticmethod
    def _make_session(user_agent: str) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch(self, url: str, raw_kind: Optional[str] = None, raw_slug: Optional[str] = None) -> str:
        self.log.debug("Fetching %s", url)
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        html = response.text

        if self.save_raw_html and self.raw_html_dir and raw_kind and raw_slug:
            self.save_raw_response(url=url, html=html, raw_kind=raw_kind, raw_slug=raw_slug)

        return html

    def save_raw_response(self, url: str, html: str, raw_kind: str, raw_slug: str) -> None:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        page = qs.get("page", ["1"])[0]

        if raw_kind == "catalog_page":
            path = self.raw_html_dir / "catalog" / f"page_{int(page):03d}.html"
        elif raw_kind == "podcast_list_page":
            path = self.raw_html_dir / raw_slug / "list_pages" / f"page_{int(page):03d}.html"
        elif raw_kind == "episode_page":
            episode_slug = slug_from_url(url)
            path = self.raw_html_dir / raw_slug / "episode_pages" / f"{episode_slug}.html"
        else:
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            path = self.raw_html_dir / "misc" / f"{digest}.html"

        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(html, encoding="utf-8")

    def polite_sleep(self) -> None:
        if self.sleep_seconds <= 0:
            return
        time.sleep(self.sleep_seconds + random.uniform(0, min(0.5, self.sleep_seconds)))

    def normalize_url(self, href: str) -> str:
        return urljoin(BASE_URL, href)

    def page_url(self, base_url: str, page_num: int) -> str:
        parsed = urlparse(base_url)
        path = parsed.path.rstrip("/") or "/"
        query = {} if page_num == 1 else {"page": str(page_num)}
        return urlunparse(
            (
                parsed.scheme or "https",
                parsed.netloc or urlparse(BASE_URL).netloc,
                path,
                "",
                urlencode(query),
                "",
            )
        )

    def discover_total_pages(self, soup: BeautifulSoup) -> int:
        page_info = soup.select_one(".podcast-page-info")
        if page_info:
            found = re.search(r"Page\s+\d+\s+of\s+(\d+)", page_info.get_text(" "))
            if found:
                return int(found.group(1))

        page_numbers: set[int] = set()
        for a in soup.select("a[href*='page=']"):
            href = a.get("href") or ""
            qs = parse_qs(urlparse(href).query)
            for value in qs.get("page", []):
                if value.isdigit():
                    page_numbers.add(int(value))

        if soup.select_one(".pagination"):
            page_numbers.add(1)

        return max(page_numbers) if page_numbers else 1

    def extract_podcast_title(self, soup: BeautifulSoup) -> Optional[str]:
        h1 = soup.select_one("h1.podcast-title") or soup.select_one("h1")
        if not h1:
            return None
        title = compact_text(h1.get_text(" "))
        title = re.sub(r"\s+-\s+Page\s+\d+\s*$", "", title, flags=re.IGNORECASE)
        return title or None

    # -----------------------------
    # Catalog-level parsing
    # -----------------------------

    def parse_podcast_cards(self, html: str, catalog_page_url: str) -> list[PodcastSeries]:
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select("a.podcast-card[href]")
        podcasts: list[PodcastSeries] = []

        for card in cards:
            href = card.get("href")
            if not href:
                continue

            url = self.normalize_url(href)
            title_node = card.select_one(".podcast-card-title")
            visible_title = compact_text(title_node.get_text(" ")) if title_node else ""
            label_title = strip_visit_prefix(card.get("aria-label") or "")
            title = visible_title or label_title or slug_from_url(url)

            episode_count: Optional[int] = None
            views_text: Optional[str] = None
            tags = [compact_text(tag.get_text(" ")) for tag in card.select(".tag")]
            for tag_text in tags:
                episode_match = EPISODE_COUNT_RE.search(tag_text)
                if episode_match:
                    episode_count = int(episode_match.group(1))
                elif "view" in tag_text.lower():
                    views_text = tag_text

            podcasts.append(
                PodcastSeries(
                    title=title,
                    url=url,
                    slug=slug_from_url(url),
                    catalog_page_url=catalog_page_url,
                    episode_count=episode_count,
                    views_text=views_text,
                )
            )

        return podcasts

    def discover_podcast_series(
        self,
        max_catalog_pages: Optional[int] = None,
        max_podcasts: Optional[int] = None,
    ) -> list[PodcastSeries]:
        first_html = self.fetch(
            self.catalog_url,
            raw_kind="catalog_page",
            raw_slug="catalog",
        )
        first_soup = BeautifulSoup(first_html, "html.parser")
        total_pages = self.discover_total_pages(first_soup)

        if max_catalog_pages is not None:
            total_pages = min(total_pages, max_catalog_pages)

        self.log.info("Discovered %d catalog page(s)", total_pages)

        all_podcasts: list[PodcastSeries] = []
        seen_urls: set[str] = set()

        for page_num in range(1, total_pages + 1):
            catalog_page_url = self.page_url(self.catalog_url, page_num)
            if page_num == 1:
                html = first_html
            else:
                html = self.fetch(
                    catalog_page_url,
                    raw_kind="catalog_page",
                    raw_slug="catalog",
                )

            podcasts = self.parse_podcast_cards(html, catalog_page_url)
            self.log.info(
                "Catalog page %d/%d: found %d podcast(s)",
                page_num,
                total_pages,
                len(podcasts),
            )

            for podcast in podcasts:
                if podcast.url in seen_urls:
                    continue
                seen_urls.add(podcast.url)
                all_podcasts.append(podcast)
                if max_podcasts is not None and len(all_podcasts) >= max_podcasts:
                    self.log.info("Reached max_podcasts=%d", max_podcasts)
                    return all_podcasts

            if page_num < total_pages:
                self.polite_sleep()

        self.log.info("Total unique podcast series: %d", len(all_podcasts))
        return all_podcasts

    # -----------------------------
    # Podcast-level parsing
    # -----------------------------

    def parse_episode_cards(self, html: str, list_page_url: str) -> list[EpisodeLink]:
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select("a.podcast-episode-card[href]")
        episodes: list[EpisodeLink] = []

        for card in cards:
            href = card.get("href")
            if not href:
                continue

            label_title = strip_read_prefix(card.get("aria-label") or "")
            h3 = card.select_one(".podcast-episode-title")
            visible_title = compact_text(h3.get_text(" ")) if h3 else ""
            title = label_title or visible_title
            if not title:
                continue

            desc_node = card.select_one(".podcast-episode-description")
            description = compact_text(desc_node.get_text(" ")) if desc_node else None

            stats_text = [compact_text(s.get_text(" ")) for s in card.select(".tag")]
            duration = next((s for s in stats_text if re.match(r"^\d{1,2}:\d{2}$", s)), None)
            published_text = next(
                (s for s in stats_text if s.lower().startswith("published")), None
            )

            episodes.append(
                EpisodeLink(
                    title=title,
                    url=self.normalize_url(href),
                    list_page_url=list_page_url,
                    description=description,
                    duration=duration,
                    published_text=published_text,
                )
            )

        return episodes

    def discover_episode_links_for_podcast(
        self,
        podcast_url: str,
        max_pages: Optional[int] = None,
    ) -> tuple[Optional[str], list[EpisodeLink]]:
        podcast_slug = slug_from_url(podcast_url)
        first_page_url = self.page_url(podcast_url, 1)
        first_html = self.fetch(
            first_page_url,
            raw_kind="podcast_list_page",
            raw_slug=podcast_slug,
        )
        first_soup = BeautifulSoup(first_html, "html.parser")
        podcast_title = self.extract_podcast_title(first_soup)
        total_pages = self.discover_total_pages(first_soup)

        if max_pages is not None:
            total_pages = min(total_pages, max_pages)

        self.log.info("Podcast: %s", podcast_title or podcast_slug)
        self.log.info("Podcast pages: %d", total_pages)

        all_links: list[EpisodeLink] = []
        seen_urls: set[str] = set()

        for page_num in range(1, total_pages + 1):
            list_url = self.page_url(podcast_url, page_num)
            if page_num == 1:
                html = first_html
            else:
                html = self.fetch(
                    list_url,
                    raw_kind="podcast_list_page",
                    raw_slug=podcast_slug,
                )
            links = self.parse_episode_cards(html, list_url)
            self.log.info(
                "Podcast page %d/%d: found %d episode link(s)",
                page_num,
                total_pages,
                len(links),
            )

            for link in links:
                if link.url in seen_urls:
                    continue
                seen_urls.add(link.url)
                all_links.append(link)

            if page_num < total_pages:
                self.polite_sleep()

        self.log.info("Total unique episode links: %d", len(all_links))
        return podcast_title, all_links

    def discover_episode_links(
        self,
        max_pages: Optional[int] = None,
    ) -> tuple[Optional[str], list[EpisodeLink]]:
        return self.discover_episode_links_for_podcast(
            podcast_url=self.podcast_url,
            max_pages=max_pages,
        )

    # -----------------------------
    # Episode transcript parsing
    # -----------------------------

    def parse_transcript_segments(self, soup: BeautifulSoup) -> list[TranscriptSegment]:
        """
        HappyScribe transcript pages expose timestamp lines followed by text.
        This parser works from visible text, so it is less brittle than relying on one class.
        """
        remove_noise_tags(soup)
        main = soup.select_one("main") or soup.body or soup
        raw_lines = [compact_text(line) for line in main.get_text("\n").splitlines()]
        lines = [line for line in raw_lines if line]

        segments: list[TranscriptSegment] = []
        current_ts: Optional[str] = None
        current_parts: list[str] = []
        started = False

        for line in lines:
            if TIMESTAMP_RE.match(line):
                started = True
                if current_ts and current_parts:
                    segments.append(
                        TranscriptSegment(
                            timestamp=current_ts,
                            text=compact_text(" ".join(current_parts)),
                        )
                    )
                current_ts = line
                current_parts = []
                continue

            if not started:
                continue

            lowered = line.lower()
            if lowered.startswith("description of ") or lowered in {
                "about us",
                "contact us",
                "privacy",
                "terms",
            }:
                break

            if lowered in {
                "copy link to transcript",
                "audio transcription by",
                "transcribed from audio to text by",
            }:
                continue

            current_parts.append(line)

        if current_ts and current_parts:
            segments.append(
                TranscriptSegment(
                    timestamp=current_ts,
                    text=compact_text(" ".join(current_parts)),
                )
            )

        return segments

    def fallback_transcript_text(self, soup: BeautifulSoup) -> str:
        remove_noise_tags(soup)
        main = soup.select_one("main") or soup.body or soup
        lines = [compact_text(line) for line in main.get_text("\n").splitlines()]
        lines = [line for line in lines if line]

        filtered: list[str] = []
        skip_terms = {
            "home",
            "podcasts",
            "youtube channels",
            "about us",
            "request podcast",
            "copy link to transcript",
            "audio transcription by",
            "privacy",
            "terms",
        }

        for line in lines:
            if line.lower() in skip_terms:
                continue
            if line.startswith("Transcript of "):
                continue
            filtered.append(line)

        return clean_text("\n".join(filtered))

    def parse_episode_page(
        self,
        episode: EpisodeLink,
        podcast_title: Optional[str],
        podcast_slug: Optional[str] = None,
        keep_timestamps: bool = True,
    ) -> SourceDocument:
        raw_slug = podcast_slug or slug_from_url(episode.list_page_url)
        html = self.fetch(
            episode.url,
            raw_kind="episode_page",
            raw_slug=raw_slug,
        )
        soup = BeautifulSoup(html, "html.parser")

        h1 = soup.select_one("h1")
        page_title = compact_text(h1.get_text(" ")) if h1 else episode.title
        page_title = re.sub(
            r"^Transcript of\s+", "", page_title, flags=re.IGNORECASE
        ).strip()
        title = episode.title or page_title

        segments = self.parse_transcript_segments(soup)
        if segments:
            if keep_timestamps:
                body = "\n\n".join(f"[{seg.timestamp}] {seg.text}" for seg in segments)
            else:
                body = "\n\n".join(seg.text for seg in segments)
        else:
            body = self.fallback_transcript_text(soup)

        body = clean_text(body)
        content_hash = sha256_text(body)
        retrieved_at = datetime.now(timezone.utc).isoformat()

        return SourceDocument(
            doc_id=stable_id_from_url(episode.url),
            source_site="happyscribe_podcasts",
            source_type="podcast_transcript",
            podcast_title=podcast_title,
            title=title,
            url=episode.url,
            list_page_url=episode.list_page_url,
            retrieved_at=retrieved_at,
            language="en",
            clean_text=body,
            content_hash=content_hash,
            extraction_method="happyscribe_podcasts_adapter_v2",
            duration=episode.duration,
            published_date=None,
            published_text=episode.published_text,
            description=episode.description,
            metadata={
                "segment_count": len(segments),
                "text_char_count": len(body),
                "podcast_slug": raw_slug,
                "podcast_path": urlparse(episode.list_page_url).path,
            },
        )

    # -----------------------------
    # Scrape modes
    # -----------------------------

    def scrape_single_podcast(
        self,
        output_path: Path,
        podcast_url: Optional[str] = None,
        max_pages: Optional[int] = None,
        max_episodes: Optional[int] = None,
        keep_timestamps: bool = True,
        resume: bool = True,
    ) -> dict:
        target_url = podcast_url or self.podcast_url
        podcast_slug = slug_from_url(target_url)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_urls = load_existing_urls(output_path) if resume else set()

        podcast_title, episode_links = self.discover_episode_links_for_podcast(
            podcast_url=target_url,
            max_pages=max_pages,
        )

        if max_episodes is not None:
            episode_links = episode_links[:max_episodes]

        write_mode = "a" if resume and output_path.exists() else "w"
        written = 0
        skipped = 0
        failed = 0

        with output_path.open(write_mode, encoding="utf-8") as f:
            for idx, episode in enumerate(episode_links, start=1):
                if episode.url in existing_urls:
                    skipped += 1
                    self.log.info("[%d/%d] Skip existing: %s", idx, len(episode_links), episode.title)
                    continue

                try:
                    self.log.info("[%d/%d] Scrape: %s", idx, len(episode_links), episode.title)
                    doc = self.parse_episode_page(
                        episode=episode,
                        podcast_title=podcast_title,
                        podcast_slug=podcast_slug,
                        keep_timestamps=keep_timestamps,
                    )

                    if not doc.clean_text or len(doc.clean_text) < 500:
                        self.log.warning(
                            "Very short transcript (%d chars): %s",
                            len(doc.clean_text),
                            episode.url,
                        )

                    f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
                    f.flush()
                    written += 1
                    self.polite_sleep()

                except Exception as exc:
                    failed += 1
                    self.log.exception("Failed to scrape %s: %s", episode.url, exc)

        summary = {
            "podcast_slug": podcast_slug,
            "podcast_title": podcast_title,
            "podcast_url": target_url,
            "output_path": str(output_path),
            "episode_links": len(episode_links),
            "written": written,
            "skipped": skipped,
            "failed": failed,
        }
        self.log.info("Podcast done: %s", summary)
        return summary

    # Backward-compatible alias for old code.
    def scrape(
        self,
        output_path: Path,
        max_pages: Optional[int] = None,
        max_episodes: Optional[int] = None,
        keep_timestamps: bool = True,
        resume: bool = True,
    ) -> None:
        self.scrape_single_podcast(
            output_path=output_path,
            podcast_url=self.podcast_url,
            max_pages=max_pages,
            max_episodes=max_episodes,
            keep_timestamps=keep_timestamps,
            resume=resume,
        )

    def scrape_all_podcasts(
        self,
        output_dir: Path,
        max_catalog_pages: Optional[int] = None,
        max_podcasts: Optional[int] = None,
        max_pages_per_podcast: Optional[int] = None,
        max_episodes_per_podcast: Optional[int] = None,
        keep_timestamps: bool = True,
        resume: bool = True,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        catalog_dir = output_dir / "catalog"
        catalog_dir.mkdir(parents=True, exist_ok=True)

        podcasts = self.discover_podcast_series(
            max_catalog_pages=max_catalog_pages,
            max_podcasts=max_podcasts,
        )

        catalog_path = catalog_dir / "podcasts.jsonl"
        write_jsonl(catalog_path, (asdict(p) for p in podcasts), append=False)
        self.log.info("Saved podcast catalog: %s", catalog_path)

        summaries: list[dict] = []
        total_written = 0
        total_skipped = 0
        total_failed = 0

        for idx, podcast in enumerate(podcasts, start=1):
            self.log.info("=== Podcast %d/%d: %s ===", idx, len(podcasts), podcast.title)
            podcast_output = output_dir / podcast.slug / "source_documents.jsonl"

            try:
                summary = self.scrape_single_podcast(
                    output_path=podcast_output,
                    podcast_url=podcast.url,
                    max_pages=max_pages_per_podcast,
                    max_episodes=max_episodes_per_podcast,
                    keep_timestamps=keep_timestamps,
                    resume=resume,
                )
            except Exception as exc:
                self.log.exception("Podcast failed completely: %s | %s", podcast.url, exc)
                summary = {
                    "podcast_slug": podcast.slug,
                    "podcast_title": podcast.title,
                    "podcast_url": podcast.url,
                    "output_path": str(podcast_output),
                    "episode_links": 0,
                    "written": 0,
                    "skipped": 0,
                    "failed": 1,
                    "error": repr(exc),
                }

            summaries.append(summary)
            total_written += int(summary.get("written", 0))
            total_skipped += int(summary.get("skipped", 0))
            total_failed += int(summary.get("failed", 0))

            write_jsonl(catalog_dir / "run_summary.jsonl", [summary], append=True)
            self.polite_sleep()

        run_summary = {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "catalog_url": self.catalog_url,
            "podcast_count": len(podcasts),
            "total_written": total_written,
            "total_skipped": total_skipped,
            "total_failed": total_failed,
            "output_dir": str(output_dir),
            "catalog_path": str(catalog_path),
            "summary_path": str(catalog_dir / "run_summary.jsonl"),
        }

        (catalog_dir / "run_summary.json").write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.log.info("All-podcast scrape done: %s", run_summary)
        return run_summary


def validate_output(path: Path) -> None:
    if not path.exists():
        print(f"Output file does not exist: {path}", file=sys.stderr)
        return

    docs = list(iter_jsonl(path))
    if not docs:
        print("No documents written.", file=sys.stderr)
        return

    lengths = [len(doc.get("clean_text", "")) for doc in docs]
    print(
        json.dumps(
            {
                "documents": len(docs),
                "min_chars": min(lengths),
                "max_chars": max(lengths),
                "avg_chars": round(sum(lengths) / len(lengths), 1),
                "sample": {
                    "title": docs[0].get("title"),
                    "url": docs[0].get("url"),
                    "text_preview": docs[0].get("clean_text", "")[:300],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def validate_all_output(output_dir: Path) -> None:
    source_files = sorted(output_dir.glob("*/source_documents.jsonl"))
    total_docs = 0
    total_chars = 0
    nonempty_files = 0

    for path in source_files:
        file_docs = 0
        file_chars = 0
        for doc in iter_jsonl(path):
            file_docs += 1
            file_chars += len(doc.get("clean_text", ""))
        if file_docs:
            nonempty_files += 1
        total_docs += file_docs
        total_chars += file_chars

    print(
        json.dumps(
            {
                "podcast_files": len(source_files),
                "nonempty_podcast_files": nonempty_files,
                "documents": total_docs,
                "avg_chars": round(total_chars / total_docs, 1) if total_docs else 0,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape HappyScribe podcast transcripts into standardized JSONL."
    )
    parser.add_argument("--podcast-url", default=DEFAULT_PODCAST_URL)
    parser.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL)
    parser.add_argument("--output", default=DEFAULT_SINGLE_OUTPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_ALL_OUTPUT_DIR)
    parser.add_argument("--raw-html-dir", default=DEFAULT_RAW_HTML_DIR)
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between requests.")
    parser.add_argument("--timeout", type=int, default=30)

    parser.add_argument(
        "--all-podcasts",
        action="store_true",
        help="Scrape every podcast series listed on the HappyScribe /podcasts catalog.",
    )
    parser.add_argument("--max-catalog-pages", type=int, default=None)
    parser.add_argument("--max-podcasts", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None, help="Single-podcast mode: max listing pages.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Single-podcast mode: max episodes.")
    parser.add_argument("--max-pages-per-podcast", type=int, default=None)
    parser.add_argument("--max-episodes-per-podcast", type=int, default=None)

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite output instead of appending and skipping existing URLs.",
    )
    parser.add_argument(
        "--drop-timestamps",
        action="store_true",
        help="Store transcript text without [HH:MM:SS] prefixes.",
    )
    parser.add_argument(
        "--save-raw-html",
        action="store_true",
        help="Also cache raw HTML under data/raw/podcasts/happyscribe.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print a small summary after scraping.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scraper = HappyScribePodcastScraper(
        podcast_url=args.podcast_url,
        catalog_url=args.catalog_url,
        sleep_seconds=args.sleep,
        timeout_seconds=args.timeout,
        raw_html_dir=Path(args.raw_html_dir),
        save_raw_html=args.save_raw_html,
        verbose=args.verbose,
    )

    if args.all_podcasts:
        output_dir = Path(args.output_dir)
        scraper.scrape_all_podcasts(
            output_dir=output_dir,
            max_catalog_pages=args.max_catalog_pages,
            max_podcasts=args.max_podcasts,
            max_pages_per_podcast=args.max_pages_per_podcast,
            max_episodes_per_podcast=args.max_episodes_per_podcast,
            keep_timestamps=not args.drop_timestamps,
            resume=not args.no_resume,
        )
        if args.validate:
            validate_all_output(output_dir)
    else:
        output_path = Path(args.output)
        scraper.scrape_single_podcast(
            output_path=output_path,
            podcast_url=args.podcast_url,
            max_pages=args.max_pages,
            max_episodes=args.max_episodes,
            keep_timestamps=not args.drop_timestamps,
            resume=not args.no_resume,
        )
        if args.validate:
            validate_output(output_path)


if __name__ == "__main__":
    main()
