import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urlunparse

GITHUB_USER = "bhuvanchennoju"
FEEDS = [
    "https://bhuvanchennoju.com/rss.xml",
    "https://medium.com/feed/@bhuvanchennoju",
]
NOW_API = "https://bcwebsite.onrender.com/site-content/home_now"
MAX_POSTS = 5
MAX_ACTIVITY = 5
DUPE_THRESHOLD = 0.6
README = "README.md"

START       = "<!-- BLOG-POST-LIST:START -->"
END         = "<!-- BLOG-POST-LIST:END -->"
NOW_START   = "<!-- NOW:START -->"
NOW_END     = "<!-- NOW:END -->"
REC_START   = "<!-- RECENTLY:START -->"
REC_END     = "<!-- RECENTLY:END -->"

STOPWORDS = {"a", "an", "the", "how", "do", "you", "to", "for", "of",
             "in", "on", "at", "is", "it", "vs", "and", "or", "with"}

SKIP_EVENTS = {"WatchEvent", "ForkEvent", "IssueCommentEvent",
               "MemberEvent", "PublicEvent", "DeleteEvent"}


# ── helpers ──────────────────────────────────────────────────────────────────

def clean_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query="", fragment=""))


def key_words(title: str) -> str:
    words = re.sub(r"\W+", " ", title.lower()).split()
    return " ".join(w for w in words if w not in STOPWORDS)


def is_duplicate(title: str, seen: list[str]) -> bool:
    kw = key_words(title)
    return any(
        SequenceMatcher(None, kw, key_words(t)).ratio() >= DUPE_THRESHOLD
        for t in seen
    )


def gh_request(url: str, token: str | None) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


# ── now ───────────────────────────────────────────────────────────────────────

def fetch_now() -> str | None:
    req = urllib.request.Request(NOW_API, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("content", "").replace("\r\n", "\n").strip()
    except Exception as exc:
        print(f"warning: could not fetch now content: {exc}")
        return None


# ── github activity ───────────────────────────────────────────────────────────

def fetch_github_activity(token: str | None) -> list[str]:
    try:
        data = gh_request(
            f"https://api.github.com/users/{GITHUB_USER}/events/public?per_page=100",
            token,
        )
        events = json.loads(data)
    except Exception as exc:
        print(f"warning: could not fetch GitHub activity: {exc}")
        return []

    profile_repo = f"{GITHUB_USER}/{GITHUB_USER}"
    items: list[str] = []
    seen_repos: set[str] = set()

    for event in events:
        if len(items) >= MAX_ACTIVITY:
            break

        etype = event.get("type", "")
        if etype in SKIP_EVENTS:
            continue

        repo_full = event.get("repo", {}).get("name", "")
        if repo_full == profile_repo:
            continue

        repo = repo_full.split("/")[-1]
        repo_url = f"https://github.com/{repo_full}"
        payload = event.get("payload", {})
        item: str | None = None

        if etype == "PushEvent" and repo_full not in seen_repos:
            ref = payload.get("ref", "").split("/")[-1]
            branch = f" on `{ref}`" if ref and ref not in ("master", "main") else ""
            commits = payload.get("commits", [])
            msg = f" — {commits[0]['message'].split(chr(10))[0][:60]}" if commits else ""
            item = f"Pushed to [{repo}]({repo_url}){branch}{msg}"

        elif etype == "PullRequestEvent":
            pr = payload.get("pull_request", {})
            if payload.get("action") == "closed" and pr.get("merged"):
                title = (pr.get("title") or "")[:72]
                item = f"Merged [{title}]({pr.get('html_url', repo_url)}) in `{repo}`"

        elif etype == "CreateEvent" and repo_full not in seen_repos:
            ref_type = payload.get("ref_type", "")
            ref = payload.get("ref") or ""
            if ref_type == "repository" or (ref_type == "branch" and ref in ("main", "master")):
                desc = payload.get("description") or ""
                suffix = f" — {desc[:60]}" if desc else ""
                item = f"Created [{repo}]({repo_url}){suffix}"

        elif etype == "ReleaseEvent":
            tag = (payload.get("release") or {}).get("tag_name", "")
            item = f"Released `{tag}` in [{repo}]({repo_url})"

        if item:
            seen_repos.add(repo_full)
            items.append(item)

    return items


# ── posts ─────────────────────────────────────────────────────────────────────

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


# ── main ──────────────────────────────────────────────────────────────────────

token = os.environ.get("GITHUB_TOKEN")

# Posts
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
post_lines = [f'- [{p["title"]}]({p["link"]})' for p in all_posts[:MAX_POSTS]]

# GitHub activity
activity_lines = [f"- {item}" for item in fetch_github_activity(token)]

# Now
now_text = fetch_now()

# Write README
with open(README) as f:
    content = f.read()

content = re.sub(
    rf"{re.escape(START)}.*?{re.escape(END)}",
    f"{START}\n" + "\n".join(post_lines) + f"\n{END}",
    content, flags=re.DOTALL,
)

if activity_lines:
    content = re.sub(
        rf"{re.escape(REC_START)}.*?{re.escape(REC_END)}",
        f"{REC_START}\n" + "\n".join(activity_lines) + f"\n{REC_END}",
        content, flags=re.DOTALL,
    )
    print(f"updated recently: {len(activity_lines)} events")
else:
    print("skipped recently (no activity fetched)")

if now_text:
    content = re.sub(
        rf"{re.escape(NOW_START)}.*?{re.escape(NOW_END)}",
        f"{NOW_START}\n{now_text}\n{NOW_END}",
        content, flags=re.DOTALL,
    )
    print("updated now section")
else:
    print("skipped now section (api unavailable)")

with open(README, "w") as f:
    f.write(content)

print(f"wrote {len(post_lines)} posts to README")
