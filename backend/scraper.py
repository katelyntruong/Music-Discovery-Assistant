"""
Scrapes Rolling Stone album reviews and ingests them into ChromaDB.

Usage:
    python scraper.py                # scrape 3 pages (~30 reviews)
    python scraper.py --pages 10     # scrape more
    python scraper.py --status       # show how many docs are indexed
"""

import argparse
import hashlib
import time
import sys

import httpx
from bs4 import BeautifulSoup

from rag import ingest_review, collection_count

BASE_URL = "https://www.rollingstone.com/music/music-album-reviews/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.5  # seconds between requests — be polite


def _get(url: str, timeout: int = 15) -> httpx.Response | None:
    try:
        r = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"  [warn] GET failed for {url}: {e}")
        return None


def _extract_article_links(list_page_html: str) -> list[tuple[str, str]]:
    """Return [(title, url), …] from a review listing page."""
    soup = BeautifulSoup(list_page_html, "html.parser")
    seen: set[str] = set()
    results = []

    for a in soup.select("a[href]"):
        href = a["href"]
        # must be a review article URL (not the category page itself)
        if "/music/music-album-reviews/" not in href:
            continue
        if not href.startswith("http"):
            href = "https://www.rollingstone.com" + href
        # skip the listing page itself and duplicates
        if href.rstrip("/") in {
            "https://www.rollingstone.com/music/music-album-reviews",
            "https://www.rollingstone.com/music/music-album-reviews/",
        }:
            continue
        title = a.get_text(strip=True)
        if len(title) < 10:
            continue  # skip image links and nav labels

        if href in seen:
            continue
        seen.add(href)

        results.append((title, href))

    return results


def _extract_review_text(article_html: str) -> tuple[str, list[str]]:
    """
    Return (body_text, detected_artists).
    Pulls the main review paragraphs and tries to extract artist names
    from structured metadata on the page.
    """
    soup = BeautifulSoup(article_html, "html.parser")

    # remove nav / footer noise
    for tag in soup.select("nav, footer, aside, script, style"):
        tag.decompose()

    # grab main article body paragraphs
    body_paragraphs = soup.select("article p, .article-content p, .content p")
    if not body_paragraphs:
        body_paragraphs = soup.select("p")

    text = " ".join(p.get_text(strip=True) for p in body_paragraphs[:30])

    # try to pick up artist names from structured tags or byline hints
    artists: list[str] = []
    for tag in soup.select("[class*='artist'], [class*='artist-name'], .c-taglist a"):
        name = tag.get_text(strip=True)
        if name:
            artists.append(name)

    return text, artists


def scrape_page(page: int) -> list[tuple[str, str]]:
    """Fetch one listing page and return article (title, url) pairs."""
    url = BASE_URL if page == 1 else f"{BASE_URL}page/{page}/"
    print(f"\n[page {page}] {url}")
    r = _get(url)
    if not r:
        return []
    links = _extract_article_links(r.text)
    print(f"  found {len(links)} article links")
    return links


def scrape_and_ingest(pages: int = 3, delay: float = REQUEST_DELAY) -> int:
    """
    Scrape `pages` listing pages, fetch each article, and ingest into ChromaDB.
    Returns the number of successfully ingested documents.
    """
    ingested = 0
    seen_urls: set[str] = set()

    for page in range(1, pages + 1):
        links = scrape_page(page)
        time.sleep(delay)

        for title, url in links:
            if url in seen_urls:
                continue
            seen_urls.add(url)

            print(f"  fetching: {title[:60]}")
            r = _get(url)
            if not r:
                continue

            body, artists = _extract_review_text(r.text)
            if len(body) < 100:
                print("    [skip] too short")
                continue

            # stable ID = sha1 of URL so re-runs upsert rather than duplicate
            doc_id = hashlib.sha1(url.encode()).hexdigest()

            metadata = {"source": "rolling_stone", "url": url}
            if artists:
                metadata["artists"] = ", ".join(artists[:5])

            ingest_review(doc_id=doc_id, title=title, content=body, metadata=metadata)
            ingested += 1
            print(f"    [ok] ingested ({ingested} total so far)")

            time.sleep(delay)

    return ingested


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling Stone review scraper / ingester")
    parser.add_argument("--pages", type=int, default=3, help="Listing pages to scrape (default 3)")
    parser.add_argument("--status", action="store_true", help="Show indexed doc count and exit")
    args = parser.parse_args()

    if args.status:
        count = collection_count()
        print(f"ChromaDB collection contains {count} document(s).")
        sys.exit(0)

    print(f"Scraping {args.pages} page(s) from Rolling Stone…")
    total = scrape_and_ingest(pages=args.pages)
    final_count = collection_count()
    print(f"\nDone. Ingested {total} new/updated reviews. Total in DB: {final_count}")


if __name__ == "__main__":
    main()
