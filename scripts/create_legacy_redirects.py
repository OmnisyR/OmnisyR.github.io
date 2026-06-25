"""Create static redirect pages for old Gmeek post URLs."""

from __future__ import annotations

import html
import json
import pathlib
import sys


def resolve_output_path(docs_dir: pathlib.Path, relative_path: str) -> pathlib.Path:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    target = (docs_dir / normalized).resolve()
    if docs_dir.resolve() not in target.parents and target != docs_dir.resolve():
        raise ValueError(f"redirect path escapes docs dir: {relative_path}")
    if target.suffix.lower() != ".html":
        raise ValueError(f"redirect path must be an HTML file: {relative_path}")
    return target


def render_redirect(to_url: str) -> str:
    escaped_url = html.escape(to_url, quote=True)
    js_url = json.dumps(to_url, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={escaped_url}">
  <link rel="canonical" href="{escaped_url}">
  <title>Redirecting</title>
  <script>location.replace({js_url});</script>
</head>
<body>
  <p><a href="{escaped_url}">Redirecting</a></p>
</body>
</html>
"""


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: create_legacy_redirects.py /path/to/docs redirects.json", file=sys.stderr)
        return 2

    docs_dir = pathlib.Path(sys.argv[1]).resolve()
    redirects_path = pathlib.Path(sys.argv[2])
    redirects = json.loads(redirects_path.read_text(encoding="utf-8"))

    for item in redirects:
        output_path = resolve_output_path(docs_dir, item["from"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_redirect(item["to"]), encoding="utf-8")
        print(f"{item['from']} -> {item['to']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
