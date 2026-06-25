"""Create short issue bodies that include Markdown files from static/posts.

Default mode is a dry run. Set GITHUB_TOKEN and pass --apply to update issues.
The script intentionally does not delete issue content; it moves the long body
into versioned Markdown files in the repository, then points the issue at them.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC_POSTS = ROOT / "static" / "posts"
BLOG_BASE = ROOT / "blogBase.json"


def request_json(url: str, token: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer {}".format(token),
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "OmnisyR-blog-migrator",
    }

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def read_issue_numbers() -> list[int]:
    data = json.loads(BLOG_BASE.read_text(encoding="utf-8"))
    numbers: list[int] = []

    for section in ("postListJson", "singeListJson"):
        for key in data.get(section, {}):
            if key.startswith("P") and key[1:].isdigit():
                numbers.append(int(key[1:]))

    return sorted(set(numbers))


def legacy_to_pipe(value: str) -> str:
    match = re.fullmatch(r";;;e([\s\S]*?);;;e;;;c([\s\S]*?);;;c", value or "")
    if not match:
        return value
    return "{} || {}".format(match.group(1).strip(), match.group(2).strip())


def build_body(number: int) -> str:
    return "<!-- gmeek:include static/posts/{}.md -->\n".format(number)


def migrate_issue(repo: str, token: str, number: int, apply: bool, clean_title: bool) -> None:
    post_file = STATIC_POSTS / "{}.md".format(number)
    if not post_file.exists():
        print("skip #{}: missing {}".format(number, post_file.relative_to(ROOT)))
        return

    issue_url = "https://api.github.com/repos/{}/issues/{}".format(repo, number)
    issue = request_json(issue_url, token)
    new_body = build_body(number)
    payload = {"body": new_body}

    if clean_title:
        clean = legacy_to_pipe(issue.get("title", ""))
        if clean and clean != issue.get("title"):
            payload["title"] = clean

    if issue.get("body") == new_body and "title" not in payload:
        print("skip #{}: already migrated".format(number))
        return

    if apply:
        request_json(issue_url, token, method="PATCH", payload=payload)
        print("updated #{}".format(number))
    else:
        action = "would update #{} body -> {}".format(number, new_body.strip())
        if "title" in payload:
            action += ", title -> {}".format(payload["title"])
        print(action)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="OmnisyR/OmnisyR.github.io")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--clean-title", action="store_true", help="convert legacy bilingual issue titles to 'English || 中文'")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("Set GITHUB_TOKEN or GH_TOKEN first.", file=sys.stderr)
        return 2

    for number in read_issue_numbers():
        migrate_issue(args.repo, token, number, args.apply, args.clean_title)

    if not args.apply:
        print("Dry run only. Re-run with --apply to update GitHub issues.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8"), file=sys.stderr)
        raise
