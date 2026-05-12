import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urlunparse

FEEDS = [
    "https://bhuvanchennoju.com/rss.xml",
    "https://medium.com/feed/@bhuvanchennoju",
]
NOW_API = "https://bcwebsite.onrender.com/site-content/home_now"
MAX_POSTS = 5
DUPE_THRESHOLD = 0.6
README = "README.md"
START = "<!-- BLOG-POST-LIST:START -->"
END = "<!-- BLOG-POST-LIST:END -->"
NOW_START = "<!-- NOW:START -->"
NOW_END = "<!-- NOW:END -->"

STOPWORDS = {"a", "an", "the", "how", "do", "you", "to", "for", "of",
             "in", "on", "at", "is", "it", "vs", "and", "or", "with"}


def clean_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query="", fragment=""))


def key_words(title: str) -> str:
    words = re.sub(r"\W+", " ", title.lower()).split()
    return " ".join(w for w in words if w not in STOPWORDS)


def is_duplicate(title: str, seen_titles: list[str]) -> bool:
    kw = key_words(title)
    return any(
        SequenceMatcher(None, kw, key_words(t)).ratio() >= DUPE_THRESHOLD
        for t in seen_titles
    )


def fetch_now() -> str | None:
    req = urllib.request.Request(NOW_API, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("content", "").replace("\r\n", "\n").strip()
    except Exception as exc:
        print(f"warning: could not fetch now content: {exc}")
        return None


def fetch_posts(feed_url: str) -> list[dict]:
    req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    channel = root.find("channel")
    posts = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = clean_url((item.findtext("link") or "").strip())
        pub = item.findtext("pubDate") or ""
        try:
            dt = parsedate_to_datetime(pub)
        except Exception:
            dt = datetime.min.replace(tzinfo=timezone.utc)
        if title and link:
            posts.append({"title": title, "link": link, "date": dt})
    return posts


all_posts: list[dict] = []
seen_titles: list[str] = []

for url in FEEDS:
    try:
        for post in fetch_posts(url):
            if not is_duplicate(post["title"], seen_titles):
                seen_titles.append(post["title"])
                all_posts.append(post)
    except Exception as exc:
        print(f"warning: could not fetch {url}: {exc}")

all_posts.sort(key=lambda p: p["date"], reverse=True)
lines = [f'- [{p["title"]}]({p["link"]})' for p in all_posts[:MAX_POSTS]]

with open(README) as f:
    content = f.read()

updated = re.sub(
    rf"{re.escape(START)}.*?{re.escape(END)}",
    f"{START}\n" + "\n".join(lines) + f"\n{END}",
    content,
    flags=re.DOTALL,
)

now_text = fetch_now()
if now_text:
    updated = re.sub(
        rf"{re.escape(NOW_START)}.*?{re.escape(NOW_END)}",
        f"{NOW_START}\n{now_text}\n{NOW_END}",
        updated,
        flags=re.DOTALL,
    )
    print("updated now section")
else:
    print("skipped now section (api unavailable)")

with open(README, "w") as f:
    f.write(updated)

print(f"wrote {len(lines)} posts to README")
