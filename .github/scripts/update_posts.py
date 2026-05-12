import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urlunparse

FEEDS = [
    "https://bhuvanchennoju.com/rss.xml",
    "https://medium.com/feed/@bhuvanchennoju",
]
MAX_POSTS = 5
README = "README.md"
START = "<!-- BLOG-POST-LIST:START -->"
END = "<!-- BLOG-POST-LIST:END -->"


def clean_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query="", fragment=""))


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


def normalize(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())


all_posts: list[dict] = []
seen: set[str] = set()

for url in FEEDS:
    try:
        for post in fetch_posts(url):
            key = normalize(post["title"])
            if key not in seen:
                seen.add(key)
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

with open(README, "w") as f:
    f.write(updated)

print(f"wrote {len(lines)} posts to README")
